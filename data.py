"""OGB dataset loading, Morgan fingerprints, and Tanimoto similarity."""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import torch

from deps import (Chem, HAS_OGB, PygGraphPropPredDataset, normalize_name,
                  rdFingerprintGenerator)
from tdc_data import is_tdc, load_tdc


# =====================================================================
# 1. Dataset loading
# =====================================================================

def load_dataset(name, args):
    """
    Dispatch on source. Returns (dataset, smiles, labels (N, T)).

    Both loaders return an object exposing get_idx_split(), so `--split dataset`
    means "whatever split this source defines" -- OGB's scaffold split files, or
    TDC's get_split() -- and every other split option works off the SMILES list
    identically for both sources.
    """
    if args.source == "tdc" or (args.source == "auto" and is_tdc(name)):
        from tdc_data import load_illegal_smiles
        return load_tdc(
            name, args.root, args.tdc_split_method, args.tdc_seed,
            tuple(args.sizes),
            use_admet_group=not getattr(args, "no_admet_group", False),
            merge_train_valid=(args.protocol == "mlcil"),
            canon_smiles=args.canon_smiles,
            illegal_smiles=load_illegal_smiles(args.illegal_smiles))
    return load_ogb(normalize_name(name), args.root)


def _stacked_labels(dataset) -> np.ndarray:
    """(N, T) float array of labels with NaN for missing, robust across PyG versions."""
    for attr in ("_data", "data"):
        obj = getattr(dataset, attr, None)
        y = getattr(obj, "y", None) if obj is not None else None
        if y is not None and torch.is_tensor(y) and y.dim() == 2 and y.shape[0] == len(dataset):
            return y.detach().cpu().numpy().astype(float)
    return np.stack([dataset[i].y.detach().cpu().numpy().reshape(-1)
                     for i in range(len(dataset))]).astype(float)


def load_ogb(name: str, root: str):
    """Returns (dataset, smiles list, (N,T) label array with NaN for missing)."""
    if not HAS_OGB:
        raise ImportError("ogb is required: pip install ogb torch_geometric")
    dataset = PygGraphPropPredDataset(name=name, root=root)
    y = _stacked_labels(dataset)

    csv_path = os.path.join(root, name.replace("-", "_"), "mapping", "mol.csv.gz")
    df = pd.read_csv(csv_path)
    smi_col = "smiles" if "smiles" in df.columns else df.columns[-1]
    smiles = df[smi_col].astype(str).tolist()
    if len(smiles) != len(dataset):
        raise RuntimeError(f"{name}: {len(smiles)} SMILES vs {len(dataset)} graphs")
    return dataset, smiles, y

# =====================================================================
# 3. Morgan fingerprints + Tanimoto
# =====================================================================

def morgan_matrix(smiles_list, radius=2, n_bits=2048):
    """Binary ECFP matrix. Returns (fp uint8 (M, n_bits), valid_idx into smiles_list)."""
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    rows, keep = [], []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        arr = np.zeros((n_bits,), dtype=np.uint8)
        for b in gen.GetFingerprint(mol).GetOnBits():
            arr[b] = 1
        rows.append(arr)
        keep.append(i)
    return np.stack(rows) if rows else np.zeros((0, n_bits), np.uint8), np.array(keep)


def tanimoto_matrix(A: np.ndarray, B: np.ndarray | None = None) -> np.ndarray:
    """Binary Tanimoto between rows of A and rows of B (default B = A)."""
    Af = A.astype(np.float32)
    Bf = Af if B is None else B.astype(np.float32)
    inter = Af @ Bf.T
    a = Af.sum(1)[:, None]
    b = Bf.sum(1)[None, :]
    union = a + b - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
