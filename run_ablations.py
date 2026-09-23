#!/usr/bin/env python3
"""
Fusion ablations. STANDALONE ENTRY POINT.

    python run_ablations.py --suite tdc --repeat 5
    python run_ablations.py --dataset AMES --methods early late gated anchor_mlp

This file and ablation_models.py are the whole ablation package. Neither is
imported by the main pipeline, and deleting both leaves MolFiber untouched --
main.py, runner.py, train.py, model.py and cli.py are unmodified.

It reuses the pipeline's data preparation rather than reimplementing it, so the
ablations and MolFiber see byte-identical splits, fingerprints, anchor logits,
GraphGrid tensors and MolFormer embeddings. That is the point: any difference in
the reported score is then attributable to how the views are combined, and not to
a different split, a differently fitted anchor, or a different training budget.

The training loop below deliberately mirrors train_grid_only -- same optimiser,
same masked BCE, same early stopping, same epoch budget -- for the same reason.
"""

from __future__ import annotations

import argparse
import copy
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ablation_models import FUSION_METHODS, build_fusion
from anchor import morgan_rf_scores
from cli import build_parser as pipeline_parser
from data import load_dataset
from deps import HAS_RDKIT, SUITES, TDC_SUITE
from fingerprints import build_fingerprints
from innercv import make_folds
from lm import build_lm
from splits import DATASET_SPLITS, get_split
from train import _logit, score_per_task
from views import SplitViews, build_view


# ---------------------------------------------------------------------------
# CLI: the pipeline's parser plus the fusion-specific options
# ---------------------------------------------------------------------------

def build_ablation_parser():
    p = pipeline_parser()
    p.prog = "run_ablations"
    p.description = ("Fusion ablations over GraphGrid, MolFormer and Morgan. "
                     "Shares every data-preparation flag with main.py.")
    g = p.add_argument_group("fusion ablations")
    g.add_argument("--methods", nargs="+", default=list(FUSION_METHODS),
                   choices=list(FUSION_METHODS) + ["rf"],
                   help="Which ablations to run. 'rf' adds the Morgan anchor on "
                        "its own as the floor.")
    g.add_argument("--fusion_morgan", default="anchor", choices=["anchor", "fp"],
                   help="How the Morgan view enters. 'anchor' reuses the frozen "
                        "random-forest logit MolFiber is built on, so the "
                        "baselines are not handicapped; 'fp' uses a trainable "
                        "head on the raw fingerprint, the pure multimodal "
                        "setting. Report the two separately.")
    g.add_argument("--late_mode", default="learned", choices=["learned", "mean"],
                   help="Decision-level combiner: fixed equal weights, or "
                        "per-target convex weights that strictly contain them.")
    g.add_argument("--gate_hidden", type=int, default=64)
    g.add_argument("--gate_temp", type=float, default=1.0,
                   help="Softmax temperature for the gated mixture.")
    return p


# ---------------------------------------------------------------------------
# Batching and training
# ---------------------------------------------------------------------------

def fusion_batch(sv, rows, dev):
    """(grid, mf, fp, blogit) for a row subset. No fibers are built."""
    grid = sv.gather(rows)[0].to(dev)
    mf = sv.mf[rows].to(dev)
    fp = torch.from_numpy(np.asarray(sv.fp.primary[rows], dtype=np.float32)).to(dev)
    blogit = torch.from_numpy(np.asarray(sv.blogit[rows], dtype=np.float32)).to(dev)
    return grid, mf, fp, blogit


@torch.no_grad()
def predict_fusion(model, sv, dev, bs=512):
    model.eval()
    n = sv.view[0].shape[0]
    out = []
    for s in range(0, n, bs):
        rows = np.arange(s, min(s + bs, n))
        out.append(torch.sigmoid(model(*fusion_batch(sv, rows, dev))).cpu())
    return torch.cat(out).numpy() if out else np.zeros((0, 1))


def train_fusion(method, tr, va, task_ids, args, seed, dev, view_spec, metric):
    """Same optimiser, loss, early stopping and budget as the pipeline baselines."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_fusion(method, view_spec, len(task_ids), sv_fp_width(tr), args,
                         tr.mf.shape[1] if tr.mf is not None else 0).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=args.weight_decay)
    y_tr = torch.from_numpy(tr.y[:, task_ids]).float()

    best, best_state, patience, best_epoch = -np.inf, None, 0, 0
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(tr.view[0].shape[0])
        for s in range(0, perm.size, args.batch_size):
            rows = perm[s:s + args.batch_size]
            logits = model(*fusion_batch(tr, rows, dev))
            target = y_tr[rows].to(dev)
            m = ~torch.isnan(target)
            if m.sum() == 0:
                continue
            loss = F.binary_cross_entropy_with_logits(
                logits[m], torch.nan_to_num(target)[m])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

        sc = predict_fusion(model, va, dev)
        auc = np.nanmean(score_per_task(va.y, sc, task_ids, metric))
        auc = -np.inf if np.isnan(auc) else auc
        if args.verbose:
            print(f"      [{method}] epoch {epoch:3d}  "
                  f"{args.select_on} {metric} {auc:.4f}")
        if auc > best + 1e-5:
            best, patience, best_epoch = auc, 0, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.best_epoch_ = max(best_epoch, 1)
    model.best_score_ = best
    model.eval()
    return model


def sv_fp_width(sv):
    return sv.fp.primary.shape[1]


def subset(sv, rows):
    """Row-subset a SplitViews, keeping every tensor aligned."""
    return SplitViews(tuple(t[rows] for t in sv.view),
                      None if sv.mf is None else sv.mf[rows],
                      sv.fp[rows], sv.blogit[rows], sv.y[rows])


def cv_select(method, pool, task_ids, args, seed, dev, view_spec, metric, k):
    """
    k-fold CV inside the training pool, mirroring the pipeline's protocol.

    Returns (mean fold score, median best epoch). The winner is then refitted on
    the whole pool for that many epochs with early stopping disabled, which is
    the analogue of sklearn's refit=True used everywhere else in this project.
    """
    n = len(pool.blogit)
    scores, epochs = [], []
    for held in make_folds(n, k, seed):
        keep = np.setdiff1d(np.arange(n), held)
        tr_f, va_f = subset(pool, keep), subset(pool, held)
        if np.unique(va_f.y[~np.isnan(va_f.y)]).size < 2:
            continue
        m = train_fusion(method, tr_f, va_f, task_ids, args, seed, dev,
                         view_spec, metric)
        scores.append(m.best_score_)
        epochs.append(m.best_epoch_)
    if not scores:
        return -np.inf, args.epochs
    return float(np.mean(scores)), int(np.median(epochs))


# ---------------------------------------------------------------------------
# Per-dataset driver -- data preparation mirrors runner.run_dataset exactly
# ---------------------------------------------------------------------------

def run_dataset(name, args, dev):
    print(f"\n=== {name} ===")
    dataset, smiles_all, y_all = load_dataset(name, args)
    metric = "auroc"

    fp_names = [args.fp_primary]
    for n in list(args.fp_channels) + list(args.anchor_fps):
        if n not in fp_names:
            fp_names.append(n)
    fp_all, keep = build_fingerprints(smiles_all, fp_names, args.n_bits, args.radius)
    smiles = [smiles_all[i] for i in keep]
    y = y_all[keep]
    if len(keep) < len(smiles_all):
        print(f"  dropped {len(smiles_all) - len(keep)} unparseable SMILES")

    tr_o, va_o, te_o, split_tag = get_split(smiles_all, y_all, args.split,
                                            tuple(args.sizes), args.split_seed,
                                            retry=args.split_retry, dataset=dataset)
    pos = np.full(len(smiles_all), -1, dtype=int)
    pos[keep] = np.arange(len(keep))

    def to_filtered(orig):
        orig = np.asarray(orig, dtype=int)
        p = pos[orig]
        return p[p >= 0]

    tr_i, va_i, te_i = to_filtered(tr_o), to_filtered(va_o), to_filtered(te_o)
    T = y.shape[1]
    if args.protocol == "mlcil" and len(va_i):
        tr_i = np.sort(np.concatenate([tr_i, va_i]))
        va_i = np.array([], dtype=int)
    pooled = args.protocol == "mlcil" or len(va_i) == 0
    if args.split in DATASET_SPLITS:
        split_tag = getattr(dataset, "split_tag", None) or split_tag
    print(f"  split={split_tag}  n={len(smiles)}  tasks={T}  "
          f"train/valid/test={len(tr_i)}/{len(va_i)}/{len(te_i)}")

    prim = fp_all.concat(args.anchor_fps)
    p_tr, p_va, p_te = morgan_rf_scores(prim[tr_i], y[tr_i], prim[va_i],
                                        prim[te_i], args, seed=args.seed)
    b_tr, b_va, b_te = _logit(p_tr), _logit(p_va), _logit(p_te)

    task_ids = list(range(T))
    results = {}
    if "rf" in args.methods:
        results["morgan_rf"] = {
            "test": float(np.nanmean(score_per_task(y[te_i], p_te, task_ids, metric))),
            "test_std": 0.0, "test_runs": [0.0], "mix": None}

    neural = [m for m in args.methods if m != "rf"]
    if not neural:
        return results, T

    view, view_spec = build_view(name, dataset, smiles, keep, tr_i, args)
    mf = build_lm(name, smiles, args, dev)
    if mf is None:
        print("  skipping fusion ablations: they need the MolFormer view.")
        return results, T

    def mk(idx, b):
        return SplitViews(tuple(t[idx] for t in view), mf[idx],
                          fp_all[idx], b, y[idx])

    tr, va, te = mk(tr_i, b_tr), mk(va_i, b_va), mk(te_i, b_te)

    for method in neural:
        t_runs, mix_runs, cv_runs = [], [], []
        for r in range(args.repeat):
            seed = args.seed + r
            if args.select_on == "test":
                model = train_fusion(method, tr, te, task_ids, args, seed, dev,
                                     view_spec, metric)
                print(f"    [{method}] best test {metric} "
                      f"{model.best_score_:.4f} at epoch {model.best_epoch_} [ORACLE]")
            elif pooled:
                cv, epochs = cv_select(method, tr, task_ids, args, seed, dev,
                                       view_spec, metric, args.inner_cv)
                a = copy.copy(args)
                a.epochs, a.patience = max(epochs, 1), 10 ** 9
                model = train_fusion(method, tr, tr, task_ids, a, seed, dev,
                                     view_spec, metric)
                cv_runs.append(cv)
                print(f"    [{method}] inner-CV {metric} {cv:.4f}, "
                      f"refit for {epochs} epochs")
            else:
                model = train_fusion(method, tr, va, task_ids, args, seed, dev,
                                     view_spec, metric)
            sc_te = predict_fusion(model, te, dev)
            t_runs.append(float(np.nanmean(score_per_task(y[te_i], sc_te,
                                                          task_ids, metric))))
            if hasattr(model, "view_weights"):
                mix_runs.append(model.view_weights())

        agg = float(np.max(t_runs)) if args.repeat_agg == "best" else float(np.mean(t_runs))
        results[method] = {
            "test": agg, "test_std": float(np.std(t_runs)),
            "test_runs": [float(v) for v in t_runs],
            "cv": float(np.mean(cv_runs)) if cv_runs else None,
            "mix": ({k: float(np.mean([m[k] for m in mix_runs]))
                     for k in mix_runs[0]} if mix_runs else None)}
        if mix_runs:
            print("    [%s] view weights: " % method +
                  "  ".join(f"{k}={v:.3f}" for k, v in results[method]["mix"].items()))
    return results, T


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

ORDER = ["morgan_rf"] + list(FUSION_METHODS)


def print_summary(summary, args):
    oracle = args.select_on == "test"
    print(f"\n=== fusion ablations: test ROC-AUC"
          + (" [ORACLE]" if oracle else "") + " ===")
    print(f"morgan branch: {args.fusion_morgan}"
          + ("  (frozen RF logit, same anchor MolFiber uses)"
             if args.fusion_morgan == "anchor"
             else "  (trainable head on the raw fingerprint)"))
    if oracle:
        print("WARNING: the test fold chose the epoch; these are an upper bound, "
              "not held-out estimates.")
    print(f"\n{'dataset':30s} {'T':>3s}  " + "  ".join(f"{m:>16s}" for m in ORDER))
    for name, (res, T) in summary.items():
        cells = []
        for m in ORDER:
            if m not in res:
                cells.append(f"{'-':>16s}")
                continue
            txt = f"{res[m]['test']:.4f}"
            if args.repeat_agg == "best" and len(res[m].get("test_runs", [])) > 1:
                txt += "*"
            elif res[m].get("test_std") is not None:
                txt += f" +/-{res[m]['test_std']:.3f}"
            cells.append(txt.rjust(16))
        print(f"{name:30s} {T:3d}  " + "  ".join(cells))

    rows = [(n, m, r["mix"]) for n, (res, _) in summary.items()
            for m, r in res.items() if r.get("mix")]
    if rows:
        print("\n=== which view each scheme leaned on ===")
        print("Mean weight on each view. late uses one weight per target; gated "
              "reweights\nper molecule. Neither can abstain -- only MolRouter "
              "can leave the anchor alone.\n")
        keys = sorted({k for _n, _m, mix in rows for k in mix})
        print(f"{'dataset':26s} {'method':12s} " + "  ".join(f"{k:>11s}" for k in keys))
        for n, m, mix in rows:
            print(f"{n:26s} {m:12s} " +
                  "  ".join(f"{mix.get(k, float('nan')):11.3f}" for k in keys))

    print("\nanchor_mlp is MolFiber with the fiber deleted and everything else "
          "held fixed;\nthe gap between it and MolFiber-L is the fiber's "
          "contribution. gated is MolRouter\nwithout the fiber and without the "
          "abstain action.")


def main(argv=None):
    args = build_ablation_parser().parse_args(argv)
    if not HAS_RDKIT:
        sys.exit("rdkit is required: pip install rdkit")
    dev = (("cuda" if torch.cuda.is_available() else "cpu")
           if args.device == "auto" else args.device)
    print(f"device: {dev}")

    datasets = SUITES[args.suite] if args.suite else (args.dataset or TDC_SUITE)
    print(f"datasets: {len(datasets)}"
          + (f" ({args.suite} suite)" if args.suite else ""))

    summary = {}
    for name in datasets:
        summary[name] = run_dataset(name, args, dev)
    print_summary(summary, args)
    return summary


if __name__ == "__main__":
    main()
