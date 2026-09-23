"""
Auxiliary views: GraphGrid (structural) and MolFormer (language model).

GraphGrid is built here and cached; MolFormer lives in lm.py. Both are frozen --
only the projections P_G and P_F on top of them are trained.

SplitViews carries one split's tensors, all indexed by the same row ids. `view`
is a 1-tuple so the fiber gather, the encoders and the caches share one code path
and adding a second structural view later would not require touching them.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass

import torch

from graphgrid import build_all_grids, fit_pr_edges


@dataclass
class ViewSpec:
    """What the encoder needs to know about the structural view."""
    in_ch: int
    grid_width: int = 32

    def describe(self):
        return f"graphgrid: {self.in_ch} x k x k image"


class SplitViews:
    """Per-split tensors, kept row-aligned."""

    __slots__ = ("view", "mf", "fp", "blogit", "y")

    def __init__(self, view, mf, fp, blogit, y):
        self.view = tuple(view)
        self.mf, self.fp, self.blogit, self.y = mf, fp, blogit, y
        # Every tensor here is indexed by the same row ids. A silent length
        # mismatch shows up much later as an opaque out-of-bounds error deep in
        # the model, so it is caught at construction with the culprit named.
        n = len(blogit)
        sizes = {f"view[{i}]": len(t) for i, t in enumerate(self.view)}
        sizes["fp"] = len(fp)
        sizes["y"] = len(y)
        if mf is not None:
            sizes["mf"] = len(mf)
        bad = {k: v for k, v in sizes.items() if v != n}
        if bad:
            raise ValueError(
                f"SplitViews row-count mismatch: anchor logits have {n} rows but "
                f"{bad} differ. Usually a stale cache in --cache_dir built on a "
                f"different molecule set; delete it or pass --rebuild_cache.")

    def gather(self, rows):
        return tuple(t[rows] for t in self.view)


def smiles_signature(smiles):
    """Short digest of the exact molecule list a cache was built from."""
    h = hashlib.sha1()
    h.update(str(len(smiles)).encode())
    for smi in smiles:
        h.update(smi.encode("utf-8", "replace"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _cache_path(name, args):
    if not args.cache_dir:
        return None
    os.makedirs(args.cache_dir, exist_ok=True)
    return os.path.join(
        args.cache_dir,
        f"{name}_graphgrid_k{args.grid_k}_{args.grid_channels}_{args.blocks}"
        f"_b{args.bond_types}_bins{args.n_bins}_t{args.hks_t}_{args.tie_break}.pt")


def _load_cached(cache, smiles, rebuild):
    """
    Cached grids, or None if absent, stale or unreadable.

    Row count alone is not enough -- two different molecule lists of equal length
    would pass -- so the signature pins the exact SMILES the cache was built
    from, and a cache never silently outlives the molecule set it describes.
    """
    if rebuild or not cache or not os.path.exists(cache):
        return None
    try:
        blob = torch.load(cache)
    except Exception as exc:
        print(f"  WARNING: could not read {os.path.basename(cache)} ({exc}); rebuilding.")
        return None

    if isinstance(blob, dict) and "sig" in blob:
        if blob.get("n") == len(smiles) and blob["sig"] == smiles_signature(smiles):
            return blob["data"]
        print(f"  WARNING: {os.path.basename(cache)} was built on a different "
              f"molecule set ({blob.get('n')} molecules vs {len(smiles)} now); "
              "rebuilding.")
        return None

    rows = blob.shape[0]
    if rows != len(smiles):
        print(f"  WARNING: {os.path.basename(cache)} holds {rows} molecules but "
              f"this run has {len(smiles)}; rebuilding.")
        return None
    print("  note: cache predates signature checking; rewriting it with one.")
    return blob


def _save_cached(cache, smiles, data):
    if not cache:
        return
    os.makedirs(os.path.dirname(cache) or ".", exist_ok=True)
    torch.save({"data": data, "n": len(smiles), "sig": smiles_signature(smiles)}, cache)


def build_view(name, dataset, smiles, keep, train_pos, args):
    """(view_tensors, ViewSpec). Z-score statistics are fit on TRAIN only."""
    pr_edges = (fit_pr_edges(dataset, smiles, keep, train_pos, args.grid_k, args)
                if args.blocks == "threshold" else None)
    cache = _cache_path(name, args)

    grids = _load_cached(cache, smiles, getattr(args, "rebuild_cache", False))
    if grids is not None:
        print(f"  graphgrid: cached {tuple(grids.shape)}")
    else:
        grids = build_all_grids(dataset, smiles, keep, args, pr_edges)
        _save_cached(cache, smiles, grids)
        print(f"  graphgrid: {tuple(grids.shape)}")
    assert grids.shape[0] == len(smiles), (
        f"graphgrid rows {grids.shape[0]} != {len(smiles)} molecules")

    mean = grids[train_pos].mean(dim=(0, 2, 3), keepdim=True)
    std = grids[train_pos].std(dim=(0, 2, 3), keepdim=True)
    grids = (grids - mean) / (std + 1e-8)
    return (grids,), ViewSpec(in_ch=grids.shape[1], grid_width=args.grid_width)
