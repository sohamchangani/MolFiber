"""
Scaffold splits.

Four options, and they are genuinely different partitions -- not variants of one
algorithm. Numbers are only comparable within a split.

  ogb             OGB's official scaffold split, shipped with the dataset. Use
                  this to compare against published OGB leaderboard numbers.
  kano_balanced   train_KANO_scffold.py -> split_data('scaffold_balanced', seed=23)
                  -> scaffold.py::scaffold_split(balanced=True). Non-isomeric
                  Murcko scaffolds, groups as SETS, float size bounds.
  topoformer      train_scffold.py::scaffold_split(seed=123). ISOMERIC scaffolds
                  (so stereoisomers can separate), groups as LISTS, int() bounds.
                  In the repo this branch is only wired up for BBBP/bace/toxcast.
  kano            scaffold.py::scaffold_split(balanced=False), deterministic.

The three ported splits use python's `random` (Mersenne Twister), NOT np.random;
swapping them silently produces a different split from the same seed. Molecules
whose SMILES will not parse are dropped from every fold, but still counted in
n_total for the size bounds -- as in the originals.
"""

from __future__ import annotations

import random as _pyrandom
from collections import defaultdict

import numpy as np

from deps import Chem, MurckoScaffold


# =====================================================================
# 2. Scaffold splits -- faithful ports from joshem163/TOPOFORMER
# =====================================================================
#
# The repo contains TWO different scaffold splits, and they are not the same
# algorithm. Both are ported here exactly, including the details that change
# which molecule lands where:
#
#   "kano_balanced"  train_KANO_scffold.py ->
#                    split_data(split_type='scaffold_balanced', seed=23)
#                    -> scaffold.py::scaffold_split(balanced=True, seed=23)
#       * scaffold = MurckoScaffoldSmiles(includeChirality=False)  [NON-isomeric]
#       * groups are python SETS, so `train += index_set` extends in set order
#       * big/small partition at len > val_size/2 or len > test_size/2,
#         then random.shuffle(big); random.shuffle(small); index_sets = big+small
#       * size bounds are FLOATS: 0.8*n, 0.1*n (no truncation)
#       * used for all 7 datasets
#
#   "topoformer"     train_scffold.py::scaffold_split(seed=123)
#       * scaffold = Chem.MolToSmiles(GetScaffoldForMol(mol))  [ISOMERIC -- the
#         default isomericSmiles=True, so stereoisomers get DIFFERENT scaffolds
#         than under the KANO version]
#       * groups are LISTS in first-appearance order
#       * sorted by size desc, then random.shuffle -- the sort is undone by the
#         shuffle but still fixes the pre-shuffle permutation, so it must stay
#       * size bounds are int()-TRUNCATED: int(0.8*n), int(0.1*n)
#       * in the repo this branch is only wired up for BBBP / bace / toxcast;
#         the `else` for the other datasets is commented out
#
#   "kano"           scaffold.py::scaffold_split(balanced=False) -- deterministic,
#                    largest-scaffold-first. Kept for continuity with earlier runs.
#
# Both use python's `random` (Mersenne Twister), NOT np.random -- swapping them
# silently produces a different split from the same seed.
#
# Molecules whose SMILES will not parse are dropped from every fold (as in the
# original), but n_total for the size bounds still counts them, also as in the
# original.

def scaffold_kano(smiles: str, include_chirality: bool = False):
    """scaffold.py::generate_scaffold -- MurckoScaffoldSmiles, non-isomeric."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=include_chirality)


def scaffold_topoformer(smiles: str):
    """train_scffold.py::generate_scaffold -- MolToSmiles(GetScaffoldForMol), ISOMERIC."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))


# kept as an alias for older call sites
generate_scaffold = scaffold_kano


def kano_scaffold_split(smiles_list, sizes=(0.8, 0.1, 0.1), balanced=False, seed=0):
    """Port of scaffold.py::scaffold_split (KANO). Groups are SETS, bounds FLOAT."""
    n = len(smiles_list)
    train_size, val_size, test_size = sizes[0] * n, sizes[1] * n, sizes[2] * n

    scaffold_to_indices = defaultdict(set)
    for i, smi in enumerate(smiles_list):
        sc = scaffold_kano(smi)
        if sc is not None:
            scaffold_to_indices[sc].add(i)

    if balanced:
        index_sets = list(scaffold_to_indices.values())
        big, small = [], []
        for s in index_sets:
            if len(s) > val_size / 2 or len(s) > test_size / 2:
                big.append(s)
            else:
                small.append(s)
        _pyrandom.seed(seed)
        _pyrandom.shuffle(big)
        _pyrandom.shuffle(small)
        index_sets = big + small
    else:
        index_sets = sorted(list(scaffold_to_indices.values()),
                            key=lambda s: len(s), reverse=True)

    train, val, test = [], [], []
    for s in index_sets:
        if len(train) + len(s) <= train_size:
            train += s
        elif len(val) + len(s) <= val_size:
            val += s
        else:
            test += s
    return np.array(train, dtype=int), np.array(val, dtype=int), np.array(test, dtype=int)


def topoformer_scaffold_split(smiles_list, sizes=(0.8, 0.1, 0.1), seed=123):
    """Port of train_scffold.py::scaffold_split. Groups are LISTS, bounds int()."""
    _pyrandom.seed(seed)

    scaffolds = {}
    for idx, smi in enumerate(smiles_list):
        sc = scaffold_topoformer(smi)
        if sc is not None:
            scaffolds.setdefault(sc, []).append(idx)

    scaffold_sets = sorted(scaffolds.values(), key=lambda x: len(x), reverse=True)
    _pyrandom.shuffle(scaffold_sets)

    n_total = len(smiles_list)
    n_train, n_valid = int(sizes[0] * n_total), int(sizes[1] * n_total)

    train, val, test = [], [], []
    for s in scaffold_sets:
        if len(train) + len(s) <= n_train:
            train.extend(s)
        elif len(val) + len(s) <= n_valid:
            val.extend(s)
        else:
            test.extend(s)
    return np.array(train, dtype=int), np.array(val, dtype=int), np.array(test, dtype=int)


def dataset_split(dataset):
    """
    The split the data source itself defines, via get_idx_split().

      OGB : the scaffold split shipped with the dataset
            (dataset/<name>/split/scaffold/{train,valid,test}.csv.gz)
      TDC : get_split(method=..., seed=..., frac=...), mapped back to row indices

    This is the split that makes numbers comparable to the respective public
    leaderboards. It is a DIFFERENT partition from the TOPOFORMER/KANO ports
    above, and for TDC it need not cover every molecule -- TDC's scaffold
    splitter omits SMILES that RDKit cannot parse.
    """
    idx = dataset.get_idx_split()
    return tuple(np.asarray(idx[k]).reshape(-1).astype(int)
                 for k in ("train", "valid", "test"))


# retained so older call sites keep working
ogb_split = dataset_split


SPLIT_DEFAULT_SEED = {"kano": 0, "kano_balanced": 23, "topoformer": 123,
                      "dataset": None, "ogb": None}
SPLIT_CHOICES = list(SPLIT_DEFAULT_SEED)
DATASET_SPLITS = ("dataset", "ogb")          # "ogb" retained for backwards compat


def _run_split(smiles_list, mode, sizes, seed, dataset=None):
    if mode in DATASET_SPLITS:
        if dataset is None or not hasattr(dataset, "get_idx_split"):
            raise ValueError(f"--split {mode} needs a dataset exposing get_idx_split().")
        return dataset_split(dataset)
    if mode == "topoformer":
        return topoformer_scaffold_split(smiles_list, sizes, seed=seed)
    return kano_scaffold_split(smiles_list, sizes,
                               balanced=(mode == "kano_balanced"), seed=seed)


def fold_scorable(y_fold: np.ndarray, allow_empty: bool = False) -> bool:
    """
    True if at least one task in this fold has both classes present.

    An intentionally empty validation fold (the MLCIL protocol pools it into
    train) is not a defect, so allow_empty suppresses the warning for it.
    """
    if y_fold.size == 0:
        return allow_empty
    for t in range(y_fold.shape[1]):
        col = y_fold[:, t]
        col = col[~np.isnan(col)]
        if col.size and np.unique(col).size >= 2:
            return True
    return False


def get_split(smiles_list, y, mode="kano_balanced", sizes=(0.8, 0.1, 0.1),
              seed=None, retry=False, dataset=None):
    """
    Indices are into `smiles_list` as given -- the FULL dataset order, with
    unparseable molecules included -- matching every reference implementation.

    The dataset-defined split is fixed and cannot be re-seeded (for TDC, vary
    --tdc_seed instead).

    TOPOFORMER's own guard against single-class folds (`validate_test_set`) is
    referenced but never defined anywhere in that repository, so the reshuffle
    loop was dead code. `retry` reinstates what it was evidently meant to do,
    and says loudly when it fires: a bumped seed means the numbers are no longer
    on the paper's split. The OGB split is fixed and cannot be re-seeded.
    """
    if seed is None:
        seed = SPLIT_DEFAULT_SEED.get(mode, 0)

    tr, va, te = _run_split(smiles_list, mode, sizes, seed, dataset)
    tag = mode if mode in DATASET_SPLITS else f"{mode}[seed={seed}]"
    # An empty valid fold is expected when the source defines only train/test.
    if fold_scorable(y[va], allow_empty=True) and fold_scorable(y[te]):
        return tr, va, te, tag

    if mode in DATASET_SPLITS or not retry:
        print(f"  WARNING: {tag} gives a single-class valid/test fold; reporting on "
              "it anyway" + ("." if mode in DATASET_SPLITS
                             else " (--split_retry to bump the seed)."))
        return tr, va, te, tag + "[UNSCORABLE]"

    for bump in range(1, 51):
        tr, va, te = _run_split(smiles_list, mode, sizes, seed + bump, dataset)
        if fold_scorable(y[va]) and fold_scorable(y[te]):
            print(f"  WARNING: {mode}[seed={seed}] gave a single-class valid/test fold. "
                  f"Retried to seed={seed + bump}. This is NOT the paper's split.")
            return tr, va, te, f"{mode}[seed={seed + bump}, RETRIED]"
    print(f"  WARNING: no seed in [{seed}, {seed + 50}] produced a scorable fold.")
    return tr, va, te, f"{mode}[degenerate]"
