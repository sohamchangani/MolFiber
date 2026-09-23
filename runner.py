"""Per-dataset driver: load, split, anchor, views, train, evaluate."""

from __future__ import annotations

import copy

import numpy as np

from anchor import morgan_rf_scores
from data import load_dataset
from fingerprints import build_fingerprints
from lm import build_lm
from splits import DATASET_SPLITS, get_split
from innercv import cv_score
from train import (make_fibers, predict_grid_only, predict_scores,
                   refit_on_pool, score_per_task, train_grid_only,
                   train_molfiber, _logit)
from views import SplitViews, build_view

# G and L are the two endpoints of the shrinkage family; M fits the shrinkage as
# a free per-task scalar, A estimates it per molecule by empirical Bayes. ENS is
# the naive alternative -- train G and L separately and average their logits --
# kept so the structured combination has to beat simple ensembling, not just the
# endpoints.
def with_metric(args, metric):
    """A shallow copy of args whose .metric is resolved (never "auto")."""
    a = copy.copy(args)
    a.metric = metric
    return a


MOLFIBER_METHODS = {"molfiber_g": "G", "molfiber_l": "L",
                    "molfiber_m": "M", "molfiber_a": "A",
                    "molrouter": "ROUTER"}
ENSEMBLE_METHOD = "molfiber_ens"
METHOD_ORDER = ["morgan_rf", "grid", "molfiber_g", "molfiber_l",
                "molfiber_m", "molfiber_a", ENSEMBLE_METHOD, "molrouter"]


def _resolve_methods(method):
    if method == "all":
        return ["rf", "grid", "molfiber_g", "molfiber_l"]
    if method == "combine":
        # The comparison: the two MolFiber endpoints, the two shrinkage
        # variants, the naive ensemble, and MolRouter, all on one split.
        return ["rf", "molfiber_g", "molfiber_l", "molfiber_m", "molfiber_a",
                ENSEMBLE_METHOD, "molrouter"]
    return [method]


def precompute_lm(name, args, dev):
    """
    Build and cache the MolFormer embeddings for one dataset, then stop.

    For machines that cannot load MoLFormer -- an HPC pinned to an older
    transformers, or a compute node with no outbound network. Run this where the
    model DOES load, copy the two cache files across, and the job will find them.

    The molecule list must match on both machines, so the same RDKit filter has
    to apply. The signature printed here is what the run will check; if it
    differs, the cached file is rejected rather than silently misaligned.
    """
    from views import smiles_signature

    print(f"\n=== {name} (precompute) ===")
    dataset, smiles_all, y_all = load_dataset(name, args)
    _fp, keep = build_fingerprints(smiles_all, [args.fp_primary],
                                  args.n_bits, args.radius)
    smiles = [smiles_all[i] for i in keep]
    if len(keep) < len(smiles_all):
        print(f"  dropped {len(smiles_all) - len(keep)} unparseable SMILES")

    emb = build_lm(name, smiles, args, dev)
    if emb is None:
        print("  no embeddings produced.")
        return
    print(f"  molecules: {len(smiles)}   signature: {smiles_signature(smiles)}")
    print(f"  embeddings: {tuple(emb.shape)}")


def run_dataset(name, args, dev):
    print(f"\n=== {name} ===")
    dataset, smiles_all, y_all = load_dataset(name, args)

    # TDC scores several of these on PR-AUC, not ROC-AUC. Following the official
    # metric by default keeps numbers comparable to the leaderboard; --metric
    # overrides it.
    # ROC-AUC everywhere, as MLCIL does: their evaluate() ignores the per-dataset
    # metric field and always calls get_skfp_roc_auc.
    metric = "auroc"

    # One build covers every role: retrieval + similarity channels + anchor.
    fp_names = [args.fp_primary]
    for n in list(args.fp_channels) + list(args.anchor_fps):
        if n not in fp_names:
            fp_names.append(n)
    fp_all, keep = build_fingerprints(smiles_all, fp_names, args.n_bits, args.radius)
    print(f"  fingerprints: {fp_all.describe()}"
          + ("  (primary retrieves; the rest add similarity channels)"
             if fp_all.n_sim > 1 else ""))
    smiles = [smiles_all[i] for i in keep]
    y = y_all[keep]
    if len(keep) < len(smiles_all):
        print(f"  dropped {len(smiles_all) - len(keep)} unparseable SMILES")

    # Splits index the FULL dataset order (unparseable molecules included), then
    # map into filtered positions. Splitting the filtered list instead would
    # shift every index and change the size bounds.
    tr_o, va_o, te_o, split_tag = get_split(smiles_all, y_all, args.split,
                                            tuple(args.sizes), args.split_seed,
                                            retry=args.split_retry, dataset=dataset)
    pos = np.full(len(smiles_all), -1, dtype=int)
    pos[keep] = np.arange(len(keep))

    def to_filtered(orig):
        orig = np.asarray(orig, dtype=int)
        if orig.size and (orig.min() < 0 or orig.max() >= len(smiles_all)):
            raise IndexError(
                f"split returned index {orig.max()} but the dataset has "
                f"{len(smiles_all)} molecules; the split and the SMILES list "
                "disagree about which dataset this is.")
        p = pos[orig]
        return p[p >= 0]

    tr_i, va_i, te_i = to_filtered(tr_o), to_filtered(va_o), to_filtered(te_o)
    for nm, idx in (("train", tr_i), ("valid", va_i), ("test", te_i)):
        if idx.size and idx.max() >= len(smiles):
            raise IndexError(f"{nm} index {idx.max()} exceeds the {len(smiles)} "
                             "molecules kept after parsing")
    T = y.shape[1]
    if args.protocol == "mlcil" and len(va_i):
        # MLCIL's get_train_data() concatenates train + valid for EVERY source,
        # OGB included, then selects by CV inside that pool.
        tr_i = np.sort(np.concatenate([tr_i, va_i]))
        va_i = np.array([], dtype=int)
    pooled = args.protocol == "mlcil" or len(va_i) == 0

    unassigned = len(smiles_all) - (len(tr_o) + len(va_o) + len(te_o))
    # The loader records how IT partitioned the data (admet_group, scaffold, ...),
    # but that only describes the split actually in use when --split asks for the
    # source-defined one. Under --split topoformer/kano the loader's tag would
    # name a partition that was computed and then discarded.
    if args.split in DATASET_SPLITS:
        split_tag = getattr(dataset, "split_tag", None) or split_tag
    print(f"  split={split_tag}  n={len(smiles)}  tasks={T}  "
          f"train/valid/test={len(tr_i)}/{len(va_i)}/{len(te_i)}"
          + (f"  ({unassigned} molecules in no fold)" if unassigned else ""))
    if pooled:
        print(f"  protocol={args.protocol}: no validation fold; selecting by "
              f"{args.inner_cv}-fold CV inside the {len(tr_i)}-molecule pool, "
              "then refitting on all of it")

    # By default the anchor sees the PRIMARY fingerprint alone, so b_M is exactly
    # the Morgan anchor the single-fingerprint model had and any difference is
    # attributable to the fiber machinery rather than to a stronger baseline.
    # --anchor_fps concatenates several instead, which raises the bar the
    # residual has to clear -- report both, because a stronger anchor leaves the
    # residual less to fix and will shrink MolFiber's apparent contribution.
    prim = fp_all.concat(args.anchor_fps)
    if args.anchor_fps:
        print(f"  anchor: RF on [{' | '.join(args.anchor_fps)}] "
              f"= {prim.shape[1]} features (fiber still retrieved by "
              f"{args.fp_primary})")
    p_tr, p_va, p_te = morgan_rf_scores(prim[tr_i], y[tr_i], prim[va_i],
                                        prim[te_i], args, seed=args.seed)
    b_tr, b_va, b_te = _logit(p_tr), _logit(p_va), _logit(p_te)

    methods = _resolve_methods(args.method)
    def report(y_true, sc):
        return float(np.nanmean(score_per_task(y_true, sc, task_ids_all, metric)))

    task_ids_all = list(range(T))
    results = {}
    if "rf" in methods:
        results["morgan_rf"] = {
            "valid": report(y[va_i], p_va) if len(va_i) else float("nan"),
            "test": report(y[te_i], p_te)}

    # Hand downstream code a namespace whose .metric is concrete, never "auto".
    args = with_metric(args, metric)

    neural = [m for m in methods if m != "rf"]
    if not neural:
        return results, T

    view, view_spec = build_view(name, dataset, smiles, keep, tr_i, args)
    print(f"  view: {view_spec.describe()}")
    mf = build_lm(name, smiles, args, dev)

    def mk(idx, b):
        return SplitViews(tuple(t[idx] for t in view),
                          None if mf is None else mf[idx],
                          fp_all[idx], b, y[idx])

    tr, va, te = mk(tr_i, b_tr), mk(va_i, b_va), mk(te_i, b_te)
    task_ids = task_ids_all

    # One fiber per fingerprint only when a router will actually route over
    # them; every other model reads the primary, and the extra retrievals are
    # not free.
    multi_fiber = ("molrouter" in neural and args.router_experts != "view"
                   and fp_all.n_sim > 1)
    if multi_fiber:
        print(f"  multi-fiber: {fp_all.n_sim} fibers, one per fingerprint "
              f"({', '.join(fp_all.names)})")

    # Fibers are cached by k so repeats do not rebuild them.
    _fib_cache: dict[int, tuple] = {}

    def fibers_for(k):
        if k not in _fib_cache:
            _fib_cache[k] = (make_fibers(tr, tr, k, True, multi_fiber),
                             make_fibers(va, tr, k, False, multi_fiber)
                             if len(va_i) else None,
                             make_fibers(te, tr, k, False, multi_fiber))
            print(f"  fibers[k={k}]: |B_x| = {k + 1} "
                  f"(retrieval set R = train, {len(tr_i)} molecules)")
        return _fib_cache[k]

    def inner_fibers(sub_tr, sub_va, k):
        """Fibers for an inner-CV fold pair. R is the fold's own training part,
        so a held-out molecule never retrieves neighbours from its own fold."""
        return (make_fibers(sub_tr, sub_tr, k, True, multi_fiber),
                make_fibers(sub_va, sub_tr, k, False, multi_fiber))

    def fit_one(method, variant, seed):
        """Train one method and return (test probabilities, cv score, model)."""
        if args.select_on == "test":
            # ORACLE SELECTION. The epoch is chosen by the test score itself, so
            # the number reported is the best test score reachable along the
            # training trajectory rather than an estimate of held-out
            # performance. Inner CV is skipped because selecting on test makes a
            # held-out estimate from the training pool pointless.
            cv = None
            if method == "grid":
                model = train_grid_only(tr, te, task_ids, run_args, seed, dev,
                                        run_spec)
            else:
                fib_tr = make_fibers(tr, tr, run_args.k, True, multi_fiber)
                fib_te_sel = make_fibers(te, tr, run_args.k, False, multi_fiber)
                model = train_molfiber(tr, te, fib_tr, fib_te_sel, task_ids,
                                       run_args, variant, seed, dev, run_spec)
            print(f"    [{method}] best test {metric} {model.best_score_:.4f} "
                  f"at epoch {model.best_epoch_}  [ORACLE]")
        elif pooled:
            cv, epochs = cv_score(method, variant, tr, task_ids, run_args, seed,
                                  dev, run_spec, inner_fibers, k=args.inner_cv,
                                  metric=metric, verbose=args.verbose)
            model = refit_on_pool(method, variant, tr, task_ids, run_args, seed,
                                  dev, run_spec, inner_fibers, epochs)
            print(f"    [{method}] inner-CV {metric} {cv:.4f}, refit for {epochs} epochs")
        else:
            cv = None
            if method == "grid":
                model = train_grid_only(tr, va, task_ids, run_args, seed, dev, run_spec)
            else:
                fib_tr, fib_va, _ = fibers_for(run_args.k)
                model = train_molfiber(tr, va, fib_tr, fib_va, task_ids, run_args,
                                       variant, seed, dev, run_spec)
        if method == "grid":
            sc = predict_grid_only(model, te, dev)
        else:
            fib_te = make_fibers(te, tr, run_args.k, False, multi_fiber)
            sc = predict_scores(model, te, tr, fib_te, task_ids, dev,
                                run_args.batch_size)
        return sc, cv, model

    for method in neural:
        if method == "molrouter" and mf is None:
            print("  skipping molrouter: it needs both views and MolFormer is "
                  "unavailable.")
            continue
        is_ens = method == ENSEMBLE_METHOD
        variant = MOLFIBER_METHODS.get(method)
        run_args, run_spec = args, view_spec

        t_runs, cv_runs, rho_runs, mix_runs = [], [], [], []
        for r in range(args.repeat):
            seed = args.seed + r
            if is_ens:
                # Naive combination: train G and L independently and average
                # their logits. The structured variants have to beat this, not
                # just the endpoints, to justify their extra machinery.
                sc_g, cv_g, _ = fit_one("molfiber_g", "G", seed)
                sc_l, cv_l, _ = fit_one("molfiber_l", "L", seed)
                sc_te = 1.0 / (1.0 + np.exp(-0.5 * (_logit(sc_g) + _logit(sc_l))))
                cv = np.mean([c for c in (cv_g, cv_l) if c is not None]) or None
            else:
                sc_te, cv, model = fit_one(method, variant, seed)
                if variant in ("M", "A"):
                    rho_runs.append(model.mean_rho())
                elif variant == "ROUTER":
                    mix_runs.append(model.routing_mix())
            if cv is not None:
                cv_runs.append(cv)
            t_runs.append(report(y[te_i], sc_te))

        agg = float(np.max(t_runs)) if args.repeat_agg == "best" else float(np.mean(t_runs))
        results[method] = {
            "test": agg,
            "test_std": float(np.std(t_runs)),
            "test_runs": [float(v) for v in t_runs],
            "config": {},
            "cv": float(np.mean(cv_runs)) if cv_runs else None,
            "rho": float(np.mean(rho_runs)) if rho_runs else None,
            "mix": ({a: float(np.mean([m[a] for m in mix_runs]))
                     for a in mix_runs[0]} if mix_runs else None)}
        if rho_runs:
            print(f"    [{method}] fitted shrinkage rho = {np.mean(rho_runs):.3f} "
                  f"(1.0 = behaves like G, 0.0 = like L)")
        if mix_runs:
            mix = results[method]["mix"]
            print("    [molrouter] routing: " +
                  "  ".join(f"{a}={v:.3f}" for a, v in mix.items()))
    return results, T


def selection_banner(args):
    """One line saying exactly how the reported number was chosen."""
    if args.select_on == "test":
        how = ("epoch chosen by the TEST score (oracle selection)")
    elif args.protocol == "mlcil":
        how = f"epoch chosen by {args.inner_cv}-fold CV inside the training pool"
    else:
        how = "epoch chosen by the validation fold"
    across = ("best over seeds" if args.repeat_agg == "best"
              else "mean over seeds")
    return f"{how}; {across}"


def print_summary(summary, args):
    oracle = args.select_on == "test"
    title = "best test ROC-AUC [ORACLE]" if oracle else "test ROC-AUC"
    print(f"\n=== {title} ===")
    print(f"selection: {selection_banner(args)}")
    if args.repeat_agg == "best":
        print("* marks a maximum over seeds; the spread across seeds is not an "
              "error bar on a max.")
    if oracle:
        print("WARNING: the test fold chose the epoch, so these numbers are an "
              "upper bound on\n         what this model achieves, not an estimate "
              "of held-out performance. They\n         are not comparable to the "
              "benchmark's published numbers, which select\n         without "
              "touching test. Use --select_on valid for that.")
    print(f"{'dataset':32s} {'T':>3s}  " + "  ".join(f"{m:>16s}" for m in METHOD_ORDER))
    for name, (res, T) in summary.items():
        cells = []
        for m in METHOD_ORDER:
            if m not in res:
                cells.append(f"{'-':>16s}")
                continue
            r = res[m]
            txt = f"{r['test']:.4f}"
            if args.repeat_agg == "best" and len(r.get("test_runs", [])) > 1:
                txt += "*"          # a max over seeds, so no error bar applies
            elif r.get("test_std") is not None:
                txt += f" +/-{r['test_std']:.3f}"
            cells.append(txt.rjust(16))
        print(f"{name:32s} {T:3d}  " + "  ".join(cells))

    rho_rows = [(n, m, r["rho"]) for n, (res, _) in summary.items()
                for m, r in res.items() if r.get("rho") is not None]
    if rho_rows:
        print("\n=== fitted shrinkage rho ===")
        print("rho is the weight on the BETWEEN-fiber component: 1.0 reproduces "
              "MolFiber-G\nexactly, 0.0 reproduces MolFiber-L exactly. It is the "
              "continuous answer to\nthe question the G/L comparison was posed to "
              "ask -- how much of the auxiliary\nevidence is calibrated across "
              "neighbourhoods rather than only within one.\n")
        print(f"{'dataset':32s} {'method':14s} {'rho':>7s}")
        for n, m, v in rho_rows:
            print(f"{n:32s} {m:14s} {v:7.3f}")

    mix_rows = [(n, r["mix"]) for n, (res, _) in summary.items()
                for m, r in res.items() if r.get("mix")]
    if mix_rows:
        print("\n=== MolRouter: where the decisions went ===")
        print("Mean routing mass per action. 'none' means the router left the "
              "Morgan\nprediction untouched; the other two mean it handed the "
              "local decision to\nthat view's relative expert.\n")
        acts = list(mix_rows[0][1])
        print(f"{'dataset':32s} " + "  ".join(f"{a:>10s}" for a in acts))
        for n, mix in mix_rows:
            print(f"{n:32s} " + "  ".join(f"{mix[a]:10.3f}" for a in acts))

