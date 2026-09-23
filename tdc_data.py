"""
Therapeutics Data Commons (TDC) loader.

Provides the same (dataset, smiles, labels) contract as the OGB loader, so the
rest of the pipeline is unchanged. Two things need care.

1. TDC's split does not return indices.
   `get_split()` ends every fold with `.reset_index(drop=True)`, so the original
   row positions are gone. This module recovers them by matching each fold row
   back to the full dataframe on (Drug_ID, Drug), consuming matches so that
   duplicated molecules are assigned deterministically rather than all mapping
   to the first occurrence.

   TDC's scaffold splitter also silently DROPS molecules whose SMILES RDKit
   cannot parse -- they are never added to any scaffold set, so they appear in
   no fold. Those rows are reported rather than quietly lost.

2. TDC has no graph objects.
   OGB ships PyG graphs; TDC ships SMILES only. `RDKitGraphs` builds the minimal
   attributes GraphGrid reads (x, edge_index, edge_attr) from RDKit on demand,
   using OGB's encoding conventions so the two sources are interchangeable.
   AtomGrid needs only SMILES and is unaffected.

Metric note: TDC's own leaderboard scores several of these on PR-AUC, not
ROC-AUC (see TDC_OFFICIAL_METRIC). Reporting ROC-AUC alone on those is not
comparable to published numbers, so the runner reports both.
"""

from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import torch

from deps import Chem

# name -> TDC task class. Names are matched case-insensitively.
TDC_DATASETS = {
    # ADME
    "bioavailability_ma": "ADME",
    "cyp1a2_veith": "ADME",
    "cyp2c19_veith": "ADME",
    "cyp2c9_substrate_carbonmangels": "ADME",
    "cyp2c9_veith": "ADME",
    "cyp2d6_substrate_carbonmangels": "ADME",
    "cyp2d6_veith": "ADME",
    "cyp3a4_substrate_carbonmangels": "ADME",
    "cyp3a4_veith": "ADME",
    "hia_hou": "ADME",
    "pampa_ncats": "ADME",
    "pgp_broccatelli": "ADME",
    # Tox
    "ames": "Tox",
    "dili": "Tox",
    "herg": "Tox",
    "herg_karim": "Tox",
    # HTS
    "sarscov2_3clpro_diamond": "HTS",
    "sarscov2_vitro_touret": "HTS",
}

# Datasets that MLCIL/benchmarking_molecular_models routes through
# tdc.benchmark_group.admet_group (config/dataset/clf_*.yaml with
# `benchmark: admet`). The rest go through a plain scaffold get_split.
ADMET_GROUP_DATASETS = {
    "ames", "bioavailability_ma", "cyp2c9_substrate_carbonmangels",
    "cyp2c9_veith", "cyp2d6_substrate_carbonmangels", "cyp2d6_veith",
    "cyp3a4_substrate_carbonmangels", "cyp3a4_veith", "dili", "hia_hou",
    "pgp_broccatelli", "herg",
}

# The metric TDC's ADMET leaderboard uses. Datasets outside the benchmark group
# have no official metric; they default to ROC-AUC here.
#
# NOTE: MLCIL scores every dataset with ROC-AUC regardless (their evaluate()
# ignores dataset_config.metric and calls get_skfp_roc_auc), so --protocol mlcil
# forces auroc. This registry only drives --metric auto.
TDC_OFFICIAL_METRIC = {
    "bioavailability_ma": "auroc",
    "cyp2c9_veith": "auprc",
    "cyp2d6_veith": "auprc",
    "cyp3a4_veith": "auprc",
    "cyp2c9_substrate_carbonmangels": "auprc",
    "cyp2d6_substrate_carbonmangels": "auprc",
    "cyp3a4_substrate_carbonmangels": "auroc",
    "hia_hou": "auroc",
    "pgp_broccatelli": "auroc",
    "ames": "auroc",
    "dili": "auroc",
    "herg": "auroc",
}

# OGB bond-type encoding, so GraphGrid's --bond_types 4 means the same thing here.
_BOND_TYPE = {
    Chem.rdchem.BondType.SINGLE: 0,
    Chem.rdchem.BondType.DOUBLE: 1,
    Chem.rdchem.BondType.TRIPLE: 2,
    Chem.rdchem.BondType.AROMATIC: 3,
} if Chem is not None else {}


def is_tdc(name: str) -> bool:
    return canonical_tdc_name(name) is not None


def canonical_tdc_name(name: str):
    """TDC names are case-insensitive; return the registry key or None."""
    key = name.strip().lower()
    if key.startswith("tdc-"):
        key = key[4:]
    return key if key in TDC_DATASETS else None


def official_metric(name: str):
    key = canonical_tdc_name(name)
    return TDC_OFFICIAL_METRIC.get(key) if key else None


class RDKitGraphs:
    """
    Lazy graph provider for TDC, matching the attributes GraphGrid reads.

    x is (N, 1) holding atomic_number - 1 in column 0, which is OGB's convention;
    GraphGrid only uses x for the atom count, so nothing else is needed.
    """

    def __init__(self, smiles):
        self.smiles = list(smiles)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, i):
        mol = Chem.MolFromSmiles(self.smiles[int(i)])
        if mol is None:
            mol = Chem.MolFromSmiles("C")
        n = mol.GetNumAtoms()
        x = np.array([[a.GetAtomicNum() - 1] for a in mol.GetAtoms()],
                     dtype=np.int64).reshape(n, 1)
        src, dst, etype = [], [], []
        for b in mol.GetBonds():
            i0, j0 = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
            t = _BOND_TYPE.get(b.GetBondType(), 0)
            src += [i0, j0]
            dst += [j0, i0]
            etype += [t, t]
        ei = torch.tensor([src, dst], dtype=torch.long).reshape(2, -1)
        ea = torch.tensor(etype, dtype=torch.long).reshape(-1, 1)
        return _Graph(torch.from_numpy(x), ei, ea, n)


class _Graph:
    __slots__ = ("x", "edge_index", "edge_attr", "num_nodes")

    def __init__(self, x, edge_index, edge_attr, num_nodes):
        self.x, self.edge_index = x, edge_index
        self.edge_attr, self.num_nodes = edge_attr, num_nodes


class TDCDataset(RDKitGraphs):
    """RDKit graphs plus the split TDC itself defines, exposed as get_idx_split()."""

    def __init__(self, smiles, split_idx):
        super().__init__(smiles)
        self._split = split_idx
        self.split_tag = "tdc"

    def get_idx_split(self):
        return self._split


def _fold_to_indices(full_df, fold_df, key_cols):
    """
    Recover row positions of `fold_df` within `full_df`.

    Matches are consumed, so N duplicate molecules in the full frame map to N
    distinct positions instead of all collapsing onto the first.
    """
    buckets = defaultdict(deque)
    for pos, key in enumerate(zip(*(full_df[c] for c in key_cols))):
        buckets[key].append(pos)

    out, missing = [], 0
    for key in zip(*(fold_df[c] for c in key_cols)):
        if buckets[key]:
            out.append(buckets[key].popleft())
        else:
            missing += 1
    return np.array(sorted(out), dtype=int), missing


def _admet_group_split(name, full, key_cols, root):
    """
    The ADMET benchmark-group split, as MLCIL uses it.

    group.get(name) returns train_val / test. There is no validation fold: the
    benchmark defines one training pool and a held-out test set, and MLCIL
    selects hyperparameters by k-fold CV inside the pool.
    """
    from tdc.benchmark_group import admet_group
    group = admet_group(path=root)
    bench = group.get(name)
    tr, miss_tr = _fold_to_indices(full, bench["train_val"], key_cols)
    te, miss_te = _fold_to_indices(full, bench["test"], key_cols)
    if miss_tr or miss_te:
        print(f"  WARNING: {miss_tr + miss_te} benchmark rows did not match the "
              f"full frame on {key_cols}; excluded.")
    return {"train": tr, "valid": np.array([], dtype=int), "test": te}, "admet_group"


def _scaffold_group_split(data, full, key_cols, method, seed, frac,
                          merge_train_valid=True):
    """
    Plain TDC get_split for datasets outside the ADMET benchmark group.

    MLCIL merges train and valid into one pool here too, so the returned valid
    fold is empty by construction rather than by later merging.
    """
    split = data.get_split(method=method, seed=seed, frac=list(frac))
    parts, missing = {}, 0
    for fold in ("train", "valid", "test"):
        ids, miss = _fold_to_indices(full, split[fold], key_cols)
        parts[fold] = ids
        missing += miss
    if missing:
        print(f"  WARNING: {missing} split rows did not match the full frame on "
              f"{key_cols}; excluded.")
    if not merge_train_valid:
        return parts, method
    pool = np.sort(np.concatenate([parts["train"], parts["valid"]]))
    return ({"train": pool, "valid": np.array([], dtype=int), "test": parts["test"]},
            f"{method}+train_val_merged")


def load_tdc(name, root="data", method="scaffold", seed=42, frac=(0.7, 0.1, 0.2),
             use_admet_group=True, merge_train_valid=True, canon_smiles=False,
             illegal_smiles=None):
    """
    Returns (TDCDataset, smiles list, labels (N, 1)).

    Follows MLCIL/benchmarking_molecular_models: the 12 datasets in the ADMET
    benchmark group take that group's train_val/test split; the other 6 take a
    scaffold get_split with train and valid merged. Either way there is NO
    validation fold -- selection happens by CV inside the training pool.
    """
    key = canonical_tdc_name(name)
    if key is None:
        raise ValueError(f"{name} is not a known TDC dataset in this registry")

    try:
        from tdc.single_pred import ADME, HTS, Tox
    except ImportError as exc:                                 # pragma: no cover
        raise ImportError("PyTDC is required for TDC datasets: pip install PyTDC") from exc

    cls = {"ADME": ADME, "Tox": Tox, "HTS": HTS}[TDC_DATASETS[key]]
    data = cls(name=key, path=root)

    full = data.get_data(format="df").reset_index(drop=True)
    smiles = full["Drug"].astype(str).tolist()
    y = full["Y"].to_numpy(dtype=float).reshape(-1, 1)
    key_cols = [c for c in ("Drug_ID", "Drug") if c in full.columns] or ["Drug"]

    if use_admet_group and key in ADMET_GROUP_DATASETS:
        try:
            idx, tag = _admet_group_split(key, full, key_cols, root)
        except Exception as exc:
            print(f"  WARNING: admet_group unavailable for {key} ({exc}); "
                  "falling back to a scaffold split.")
            idx, tag = _scaffold_group_split(data, full, key_cols, method, seed,
                                             frac, merge_train_valid)
    else:
        idx, tag = _scaffold_group_split(data, full, key_cols, method, seed, frac,
                                         merge_train_valid)

    assigned = sum(len(v) for v in idx.values())
    if assigned < len(full):
        print(f"  note: {len(full) - assigned} molecules are in no fold "
              f"({tag} omits them)")

    if canon_smiles:
        smiles = canonicalize(smiles)
    if illegal_smiles:
        idx, n_dropped = drop_illegal(idx, smiles, illegal_smiles)
        if n_dropped:
            print(f"  dropped {n_dropped} molecules on the illegal-SMILES list")

    ds = TDCDataset(smiles, idx)
    ds.split_tag = tag
    return ds, smiles, y


def canonicalize(smiles):
    """Chem.CanonSmiles on every molecule, as MLCIL's build_dataset does."""
    out = []
    for smi in smiles:
        try:
            out.append(Chem.CanonSmiles(smi))
        except Exception:
            out.append(smi)
    return out


def load_illegal_smiles(path):
    if not path:
        return None
    with open(path) as fh:
        return {ln.strip() for ln in fh if ln.strip()}


def drop_illegal(idx, smiles, illegal):
    """Remove listed molecules from every fold, leaving row positions intact."""
    bad = {i for i, s in enumerate(smiles) if s in illegal}
    if not bad:
        return idx, 0
    return ({k: np.array([i for i in v if i not in bad], dtype=int)
             for k, v in idx.items()}, len(bad))
