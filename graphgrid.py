"""GraphGrid: a purely topological k x k image per molecule."""

from __future__ import annotations

import numpy as np
import torch

from deps import Chem, tqdm


# =====================================================================
# 6. GraphGrid  (purely topological)
# =====================================================================
#
# Per molecule:
#   1. four node descriptors -- HKS(t), degree centrality, k-core number,
#      PageRank -- computed on the UNWEIGHTED, UNLABELLED molecular graph.
#      No atom types, no bond orders, no charges: this view is shape only.
#   2. quantile-bin each descriptor into n_bins bins, within the molecule
#   3. lexicographic sort of nodes by (hks, deg, kcore, pagerank) bins,
#      tie-broken by node id
#   4. permute the adjacency into that order, block-pool into a k x k image of
#      mean edge density between blocks
#
# The four numpy descriptors were checked against networkx (nx.laplacian_matrix
# + eigh, nx.degree_centrality, nx.core_number, nx.pagerank) over 125 molecules
# including disconnected salts and 1-2 atom graphs: max abs difference 0 for
# HKS / degree / k-core and 2.8e-17 for PageRank.
#
# Two properties of this ordering are worth keeping in mind (both measurable
# with the checks in the repo notes):
#   * HKS(t) = 1 - t*deg + O(t^2), so at small t the first two sort keys are
#     near-duplicates and k-core is nearly constant on molecules. Use --hks_t
#     to push HKS to a genuinely different scale.
#   * where the four binned keys tie, the order falls through to raw node id,
#     which is SMILES parse order -- so the grid is deterministic but not
#     isomorphism-invariant. --tie_break canonical swaps in RDKit's canonical
#     atom ranking, which makes it invariant at the cost of departing from the
#     reference implementation.


def quantile_bin_1d(x, n_bins=10):
    """Quantile-bin into [0, n_bins-1]; a constant column collapses to bin 0."""
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        return x.astype(np.int64)
    if np.allclose(x.min(), x.max()):
        return np.zeros_like(x, dtype=np.int64)
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = np.quantile(x, qs, method="linear")
    return np.searchsorted(edges, x, side="right").astype(np.int64)


def _dense_adj(edge_index: np.ndarray, n: int, edge_type=None, n_types: int = 1):
    """(n_types, n, n) dense adjacency. Channel 0 is all bonds when n_types == 1."""
    A = np.zeros((n_types, n, n), dtype=np.float32)
    if edge_index.size == 0:
        return A
    src, dst = edge_index[0], edge_index[1]
    if n_types == 1:
        A[0, src, dst] = 1.0
    else:
        A[np.clip(edge_type, 0, n_types - 1), src, dst] = 1.0
    return A


def compute_hks(A: np.ndarray, t: float = 0.1) -> np.ndarray:
    """diag(exp(-t L)) with L the unnormalised Laplacian."""
    if A.shape[0] == 0:
        return np.zeros(0)
    w, V = np.linalg.eigh(np.diag(A.sum(1)) - A)
    return np.diag((V * np.exp(-t * w)) @ V.T).copy()


def compute_degree_centrality(A: np.ndarray) -> np.ndarray:
    n = A.shape[0]
    return np.ones(n) if n <= 1 else A.sum(1) / (n - 1)


def compute_kcore(A: np.ndarray) -> np.ndarray:
    """Batagelj-Zaversnik peeling; agrees with nx.core_number."""
    n = A.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.int64)
    deg = A.sum(1).astype(np.int64).copy()
    removed = np.zeros(n, dtype=bool)
    core = np.zeros(n, dtype=np.int64)
    k, BIG = 0, np.iinfo(np.int64).max
    for _ in range(n):
        v = int(np.argmin(np.where(removed, BIG, deg)))
        k = max(k, int(deg[v]))
        core[v] = k
        removed[v] = True
        deg[np.where((A[v] > 0) & ~removed)[0]] -= 1
    return core


def compute_pagerank(A, alpha=0.85, max_iter=100, tol=1e-6) -> np.ndarray:
    """Matches nx.pagerank: L1 convergence test, dangling mass spread uniformly."""
    n = A.shape[0]
    if n == 0:
        return np.zeros(0)
    deg = A.sum(1)
    P = np.zeros_like(A, dtype=np.float64)
    nz = deg > 0
    P[nz] = A[nz] / deg[nz, None]
    dangling = ~nz
    r = np.full(n, 1.0 / n)
    for _ in range(max_iter):
        r_new = alpha * (P.T @ r + r[dangling].sum() / n) + (1 - alpha) / n
        err = np.abs(r_new - r).sum()
        r = r_new
        if err < n * tol:
            break
    return r


def _canonical_ranks(smiles, n):
    """RDKit canonical atom ranks; falls back to node index if atoms don't align."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None and mol.GetNumAtoms() == n:
            return np.array(list(Chem.CanonicalRankAtoms(mol, breakTies=True)), dtype=np.int64)
    except Exception:
        pass
    return np.arange(n, dtype=np.int64)


def graphgrid_order(data, smiles=None, n_bins=10, col_order=(0, 1, 2, 3),
                    tie_break="node_id", hks_t=0.1):
    """Canonical node order + the raw descriptor matrix (N, 4)."""
    n = int(data.x.shape[0])
    ei = data.edge_index.detach().cpu().numpy()
    # float64 is load-bearing, not defensive. The descriptors are quantile-binned,
    # so a 1e-8 difference from a float32 eigendecomposition can push an atom
    # across a bin edge and change the entire node ordering -- and therefore the
    # grid. The round() likewise collapses values that are equal up to numerical
    # noise into genuine ties, so the ordering does not depend on the BLAS.
    A = _dense_adj(ei, n)[0].astype(np.float64)
    A = np.maximum(A, A.T)

    S = np.stack([compute_hks(A, hks_t),
                  compute_degree_centrality(A),
                  compute_kcore(A).astype(float),
                  compute_pagerank(A)], axis=1)                 # (N, 4)

    binned = np.stack([quantile_bin_1d(np.round(S[:, c], 10), n_bins)
                       for c in range(4)], axis=1)
    tb = (_canonical_ranks(smiles, n) if tie_break == "canonical"
          else np.arange(n, dtype=np.int64))

    # np.lexsort: LAST key is most significant, so the tie-break goes first.
    keys = [tb] + [binned[:, c] for c in reversed(col_order)]
    return np.lexsort(keys), S


def _blocks_equal(n: int, k: int) -> np.ndarray:
    blk = np.zeros(n, dtype=np.int64)
    for bi, idx in enumerate(np.array_split(np.arange(n), k)):
        blk[idx] = bi
    return blk


def _blocks_threshold(pr_sorted, edges) -> np.ndarray:
    return np.clip(np.searchsorted(edges, pr_sorted, side="right"),
                   0, len(edges)).astype(np.int64)


def build_grid(data, smiles, k, mode="adj", block_mode="equal", pr_edges=None,
               n_bond_types=1, n_bins=10, tie_break="node_id", hks_t=0.1):
    """(C, k, k) float32 grid. mode 'adj' is the plain 1-channel edge-density image."""
    n = int(data.x.shape[0])
    ei = data.edge_index.detach().cpu().numpy()
    sort_idx, S = graphgrid_order(data, smiles, n_bins, tie_break=tie_break, hks_t=hks_t)

    if n_bond_types > 1 and getattr(data, "edge_attr", None) is not None and ei.size:
        et = data.edge_attr.detach().cpu().numpy()[:, 0]
    else:
        et = None
    A = _dense_adj(ei, n, et, n_bond_types)
    A = np.maximum(A, A.transpose(0, 2, 1))
    A = A[:, sort_idx][:, :, sort_idx]

    if block_mode == "threshold" and pr_edges is not None:
        blk = _blocks_threshold(S[sort_idx, 3], pr_edges)
    else:
        blk = _blocks_equal(n, k)

    B = np.zeros((n, k), dtype=np.float32)
    B[np.arange(n), blk] = 1.0
    counts = B.sum(0)
    safe = np.maximum(counts, 1.0)

    chans = [(B.T @ A[c] @ B) / (safe[:, None] * safe[None, :]) for c in range(A.shape[0])]
    if mode == "adj_occ":
        occ = (counts > 0).astype(np.float32)
        chans.append(np.outer(occ, occ))       # tells "no atoms here" from "no bonds here"
    return np.stack(chans).astype(np.float32)


def n_grid_channels(mode, n_bond_types):
    return n_bond_types + (1 if mode == "adj_occ" else 0)


def fit_pr_edges(dataset, smiles, ds_index, positions, k, args=None):
    """PageRank cut points for threshold blocks, pooled over TRAIN nodes only."""
    assert len(smiles) == len(ds_index), "smiles and ds_index must be aligned"
    kw = dict(n_bins=getattr(args, "n_bins", 10),
              tie_break=getattr(args, "tie_break", "node_id"),
              hks_t=getattr(args, "hks_t", 0.1))
    vals = []
    for p in positions:
        p = int(p)
        _, S = graphgrid_order(dataset[int(ds_index[p])], smiles[p], **kw)
        vals.append(S[:, 3])
    allv = np.concatenate(vals) if vals else np.zeros(1)
    return np.quantile(allv, np.linspace(0, 1, k + 1)[1:-1])


def build_all_grids(dataset, smiles, ds_index, args, pr_edges=None):
    """(M, C, k, k) tensor, row p aligned to smiles[p] / ds_index[p]."""
    assert len(smiles) == len(ds_index), (
        f"smiles ({len(smiles)}) and ds_index ({len(ds_index)}) must be aligned; "
        "pass the FILTERED smiles list together with the kept dataset indices")
    grids = [build_grid(dataset[int(ds_index[p])], smiles[p], args.grid_k,
                        mode=args.grid_channels, block_mode=args.blocks,
                        pr_edges=pr_edges, n_bond_types=args.bond_types,
                        n_bins=args.n_bins, tie_break=args.tie_break, hks_t=args.hks_t)
             for p in tqdm(range(len(ds_index)), desc="    graphgrid", leave=False)]
    return torch.from_numpy(np.stack(grids))


def zscore_fit(t: torch.Tensor):
    """Per-channel mean/std, fit on TRAIN grids only."""
    return t.mean(dim=(0, 2, 3), keepdim=True), t.std(dim=(0, 2, 3), keepdim=True)


def zscore_apply(t, mean, std):
    return (t - mean) / (std + 1e-8)
