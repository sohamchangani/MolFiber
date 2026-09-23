"""Command-line interface."""

from __future__ import annotations

import argparse
import sys

import torch

from deps import HAS_RDKIT, SUITES, check_versions, resolve_datasets
from fingerprints import BIT_FPS, FP_KINDS
from runner import precompute_lm, print_summary, run_dataset
from splits import SPLIT_CHOICES


def build_parser():
    p = argparse.ArgumentParser(
        prog="molfiber",
        description="MolFiber: Morgan-anchored fiber residuals on the OGB molecule datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    p.add_argument("--suite", default="ogb", choices=list(SUITES),
                   help="Run a whole benchmark: 'tdc' is the 18 TDC datasets, "
                        "'ogb' the 8 OGB ones, 'all' both. Overrides --dataset.")
    p.add_argument("--dataset", nargs="+", default=None,
                   help="Explicit dataset names. OGB (molhiv, moltox21, ...) or "
                        "TDC (AMES, CYP3A4_Veith, hERG, ...); TDC names are "
                        "case-insensitive. Defaults to the TDC suite.")
    p.add_argument("--source", default="auto", choices=["auto", "ogb", "tdc"],
                   help="auto routes each name by the TDC registry, falling back "
                        "to OGB.")
    p.add_argument("--root", default="dataset")
    p.add_argument("--cache_dir", default="cache")
    p.add_argument("--precompute_lm", action="store_true",
                   help="Build and cache the MolFormer embeddings, then exit. Run "
                        "this where MoLFormer loads, copy cache/*.npy and "
                        "cache/*.npy.sig to the cluster, and the job will pick "
                        "them up without ever importing transformers.")
    p.add_argument("--rebuild_cache", action="store_true",
                   help="Ignore cached features and embeddings and recompute them. "
                        "Caches are validated against a signature of the molecule "
                        "list, so this is rarely needed.")
    p.add_argument("--protocol", default="mlcil", choices=["mlcil", "standard"],
                   help="mlcil follows MLCIL/benchmarking_molecular_models: the 12 "
                        "ADMET-benchmark datasets use admet_group's train_val/test "
                        "split, the rest a scaffold get_split; train and valid are "
                        "merged into one pool for EVERY source (OGB included); "
                        "selection is by k-fold CV inside the pool with a refit on "
                        "all of it; and the metric is ROC-AUC everywhere. standard "
                        "keeps the validation fold and early-stops on it.")
    p.add_argument("--inner_cv", type=int, default=5,
                   help="Folds for selection when there is no validation fold. 5 "
                        "matches their GridSearchCV(cv=5); 1 is a single holdout, "
                        "much cheaper and noisier.")
    p.add_argument("--canon_smiles", action="store_true",
                   help="Apply Chem.CanonSmiles to every molecule, as MLCIL's "
                        "build_dataset does. Only affects the SMILES-consuming LM "
                        "view; fingerprints and AtomGrid read the mol graph.")
    p.add_argument("--illegal_smiles", default=None, metavar="PATH",
                   help="Drop the molecules listed in this file from every fold. "
                        "MLCIL ships config/illegal_smiles.txt (46 molecules that "
                        "break some embedders); a copy is bundled here. This "
                        "changes the test set, so use it only when comparing "
                        "directly against their numbers.")
    p.add_argument("--method", default="combine",
                   choices=["rf", "grid", "molfiber_g", "molfiber_l",
                            "molfiber_m", "molfiber_a", "molfiber_ens",
                            "molrouter", "all", "combine"],
                   help="molfiber_g/l are the two endpoints of the shrinkage "
                        "family; molfiber_m fits the shrinkage as a free per-task "
                        "scalar; molfiber_a estimates it per molecule by empirical "
                        "Bayes from fiber coherence; molfiber_ens averages the "
                        "logits of separately trained G and L; molrouter keeps "
                        "GraphGrid and MolFormer as separate relative experts and "
                        "routes per molecule between them and leaving Morgan "
                        "alone. 'combine' runs the whole comparison.")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--select_on", default="test", choices=["valid", "test"],
                   help="Which split chooses the epoch (and, when pooled, whether "
                        "inner CV runs at all). 'valid' is the honest protocol and "
                        "matches the benchmark. 'test' reports the BEST TEST SCORE "
                        "along the training trajectory: the test fold selects the "
                        "checkpoint, so the result is an upper bound on the model, "
                        "not an estimate of held-out performance, and is not "
                        "comparable to published numbers.")
    p.add_argument("--repeat_agg", default="mean", choices=["mean", "best"],
                   help="How to combine --repeat seeds. 'best' takes the maximum, "
                        "which is also optimistic: the spread across seeds stops "
                        "being an error bar once you report the max.")
    p.add_argument("--repeat", type=int, default=5,
                   help="Repeats vary the NEURAL seed; the Morgan anchor is fit once.")
    p.add_argument("--verbose", action="store_true")

    g = p.add_argument_group("split")
    g.add_argument("--split", default="dataset", choices=SPLIT_CHOICES,
                   help="dataset = the split the source defines (OGB's official "
                        "scaffold split, or TDC's get_split); ogb is an alias. "
                        "kano_balanced = TOPOFORMER's train_KANO_scffold.py "
                        "(seed 23); topoformer = its train_scffold.py (seed 123, "
                        "isomeric scaffolds); kano = deterministic, unbalanced. "
                        "These are DIFFERENT partitions -- numbers only compare "
                        "within one.")
    g.add_argument("--no_admet_group", action="store_true",
                   help="Do not use tdc.benchmark_group.admet_group for the 12 "
                        "datasets that belong to it; use a plain scaffold split "
                        "instead. Departs from MLCIL.")
    g.add_argument("--tdc_split_method", default="scaffold",
                   choices=["scaffold", "random", "cold_drug"],
                   help="Passed to TDC's get_split. TDC's own default is random; "
                        "scaffold is what the ADMET benchmark group uses and is "
                        "the harder, more realistic setting.")
    g.add_argument("--tdc_seed", type=int, default=42,
                   help="Seed for TDC's split. The ADMET benchmark reports over "
                        "seeds 1-5; vary this to reproduce that protocol.")
    g.add_argument("--split_seed", type=int, default=None,
                   help="Defaults to the reference seed for the chosen split "
                        "(kano 0, kano_balanced 23, topoformer 123). Ignored for ogb.")
    g.add_argument("--split_retry", action="store_true",
                   help="Bump the seed if valid/test comes out single-class. Off by "
                        "default: a bump means you are no longer on the paper's split.")
    g.add_argument("--sizes", type=float, nargs=3, default=[0.8, 0.1, 0.1])

    g = p.add_argument_group("fingerprints")
    g.add_argument("--fp_primary", default="ecfp4", choices=BIT_FPS,
                   help="The fingerprint that trains the anchor AND retrieves "
                        "the fiber. Retrieval needs a matmul-form similarity, so "
                        "count fingerprints are not eligible here.")
    g.add_argument("--fp_channels", nargs="*", default=[],
                   choices=sorted(FP_KINDS),
                   help="Extra fingerprints used ONLY as similarity channels "
                        "inside the fiber, and as router agreement features. The "
                        "fiber's membership and weighting are unchanged, so this "
                        "isolates what a richer notion of similarity buys. "
                        "Count fingerprints (morganc, rdkc) are allowed here "
                        "because within-fiber MinMax is cheap. Try: rdk maccs")
    g.add_argument("--anchor_fps", nargs="*", default=[],
                   choices=sorted(FP_KINDS),
                   help="Fingerprints CONCATENATED to train the anchor b_M. "
                        "Empty means the primary alone. Count fingerprints are "
                        "fine here because the anchor computes no similarity, "
                        "only threshold splits. The fiber is still retrieved by "
                        "--fp_primary, so this changes the baseline and nothing "
                        "else. Try: morganc rdkc maccs")
    g.add_argument("--radius", type=int, default=2)
    g.add_argument("--n_bits", type=int, default=2048)
    g.add_argument("--n_estimators", type=int, default=200)
    g.add_argument("--min_samples_leaf", type=int, default=1)
    g.add_argument("--n_jobs", type=int, default=-1,
                   help="Threads for the random forest. Set to 1 if the run "
                        "segfaults: joblib's pool alongside torch's OpenMP "
                        "runtime is a common cause of native crashes.")
    g.add_argument("--cv_folds", type=int, default=5,
                   help="Folds for the out-of-fold anchor logits b_M on train.")
    g.add_argument("--anchor_inference", choices=["refit", "foldmean"], default="refit",
                   help="refit = f_M refit on all of train; foldmean = average the CV "
                        "fold models, so the train and inference anchors come from "
                        "models fit on the same amount of data.")

    g = p.add_argument_group("fibers")
    g.add_argument("--k", type=int, default=24, help="|F_k(x)|; |B_x| = k + 1.")

    g = p.add_argument_group("GraphGrid view (purely topological)")
    g.add_argument("--grid_k", type=int, default=10)
    g.add_argument("--grid_channels", choices=["adj", "adj_occ"], default="adj",
                   help="adj = the plain 1 x k x k edge-density image; adj_occ adds a "
                        "block-occupancy channel (matters for molecules with fewer "
                        "heavy atoms than grid_k).")
    g.add_argument("--n_bins", type=int, default=10,
                   help="Quantile bins per descriptor before the lexicographic sort.")
    g.add_argument("--hks_t", type=float, default=0.1,
                   help="HKS diffusion time. At small t, HKS = 1 - t*deg + O(t^2), so "
                        "it nearly duplicates the degree key.")
    g.add_argument("--tie_break", choices=["node_id", "canonical"], default="node_id",
                   help="node_id matches the reference but makes the grid depend on "
                        "SMILES parse order; canonical uses RDKit canonical atom "
                        "ranks and is isomorphism-invariant.")
    g.add_argument("--blocks", choices=["equal", "threshold"], default="equal")
    g.add_argument("--bond_types", type=int, default=1, choices=[1, 4],
                   help="4 splits the adjacency by bond type, adding chemistry to an "
                        "otherwise purely topological view.")
    g.add_argument("--grid_width", type=int, default=32)

    g = p.add_argument_group("MolFormer view")
    g.add_argument("--lm", default="molformer", choices=["molformer", "none"],
                   help="none disables the language-model view; MolFiber then "
                        "runs on GraphGrid alone and MolRouter is skipped, since "
                        "it needs two views to route between.")
    g.add_argument("--lm_model", default="ibm/MoLFormer-XL-both-10pct")
    g.add_argument("--lm_revision", default=None, metavar="SHA",
                   help="Pin MoLFormer's Hub revision. trust_remote_code=True "
                        "downloads modelling code that imports transformers "
                        "internals, so an unpinned revision can stop matching "
                        "your transformers version without warning.")
    g.add_argument("--lm_optional", action="store_true",
                   help="Continue with the GraphGrid view alone if MolFormer "
                        "cannot be loaded, instead of stopping. Off by default: a "
                        "batch job that silently drops a view produces numbers "
                        "for a different experiment than the one you asked for.")
    g.add_argument("--lm_batch", type=int, default=64)

    g = p.add_argument_group("model / optimisation")
    g.add_argument("--tau", type=float, default=0.1,
                   help="Temperature for alpha_zw. Tanimoto within a fiber spans only "
                        "about 0.14, so tau=1 makes alpha effectively uniform.")
    g.add_argument("--tau_c", type=float, default=0.1, help="Temperature for beta_xz.")
    g.add_argument("--lam", type=float, default=4.0,
                   help="Bound on the residual: |S - b_M| <= lambda.")
    g.add_argument("--learn_lambda", action="store_true")
    g.add_argument("--rho_lr_mult", type=float, default=10.0,
                   help="Learning-rate multiplier for the shrinkage parameters. "
                        "They sit downstream of a zero-initialised psi, so at the "
                        "shared rate they barely move and the fitted rho stays "
                        "pinned at its initial value.")
    g.add_argument("--router_experts", default="cross",
                   choices=["cross", "fiber", "view"],
                   help="MolRouter's action space. cross = one expert per "
                        "(fingerprint fiber, view) pair; fiber = one per fiber, "
                        "each seeing both views; view = one per view on the "
                        "primary fiber only (the single-fiber router). With one "
                        "fingerprint, cross and view coincide.")
    g.add_argument("--router_hidden", type=int, default=64,
                   help="Width of MolRouter's routing MLP.")
    g.add_argument("--router_temp", type=float, default=1.0,
                   help="Softmax temperature for the router. Below 1 sharpens "
                        "toward a hard choice between the three actions.")
    g.add_argument("--router_none_prior", type=float, default=0.0,
                   help="Initial logit added to the 'leave Morgan alone' action. "
                        "Positive values make the experts earn the right to "
                        "intervene.")
    g.add_argument("--router_entropy", type=float, default=0.0,
                   help="Penalty on routing entropy. Positive values push the "
                        "router to commit to one action instead of hedging.")
    g.add_argument("--rho_init", type=float, default=0.5,
                   help="Initial shrinkage for molfiber_m. 0.5 sits midway "
                        "between G and L so neither endpoint is favoured.")
    g.add_argument("--d_view", type=int, default=128)
    g.add_argument("--d_pair", type=int, default=128)
    g.add_argument("--hidden", type=int, default=256)
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--lr", type=float, default=1e-3)
    g.add_argument("--weight_decay", type=float, default=1e-5)
    g.add_argument("--batch_size", type=int, default=64)
    g.add_argument("--epochs", type=int, default=100)
    g.add_argument("--patience", type=int, default=15)

    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if not HAS_RDKIT:
        sys.exit("rdkit is required: pip install rdkit")
    check_versions()
    dev = (("cuda" if torch.cuda.is_available() else "cpu")
           if args.device == "auto" else args.device)
    print(f"device: {dev}")

    # Names are passed through untouched: load_dataset routes them, and only the
    # OGB branch applies the ogbg- prefix. Normalising here would turn "AMES"
    # into "ogbg-AMES" and silently route every TDC dataset to OGB.
    datasets = resolve_datasets(args.suite, args.dataset)
    print(f"datasets: {len(datasets)}"
          + (f" ({args.suite} suite)" if args.suite else ""))

    if args.precompute_lm:
        if not args.cache_dir:
            raise SystemExit("--precompute_lm needs a --cache_dir to write into.")
        for name in datasets:
            precompute_lm(name, args, dev)
        print(f"\nDone. Copy {args.cache_dir}/ to the cluster and run there "
              "without --precompute_lm.")
        return None

    summary = {}
    for name in datasets:
        summary[name] = run_dataset(name, args, dev)
    print_summary(summary, args)
    return summary
