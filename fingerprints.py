"""
Multiple fingerprints, used as multiple GEOMETRIES rather than as more features.

The Morgan fingerprint does three separate jobs in this pipeline: it trains the
anchor b_M, it decides which molecules land in the fiber, and it supplies the
similarity scalar that the pair encoder and both attention weightings run on.
Concatenating fingerprints the way CLAMP does (`morganc+rdkc`, folded by
summation into one 8192-dim vector) only improves the first, and folding
actively destroys the ability to compute a per-fingerprint similarity at all.

Here the fingerprints are kept SEPARATE, and used for the other two jobs:

  * s_M(z,w) becomes a VECTOR of similarities, one per fingerprint, so the pair
    encoder can tell "substructure-similar but pharmacophore-dissimilar" from
    "similar in every sense". A single Tanimoto collapses that distinction.
  * the neighbour sets under different fingerprints DISAGREE -- measured over
    1500 drug-like molecules, ECFP4 and the RDKit path fingerprint share only
    ~3.8 of their 8 nearest neighbours, with mean Jaccard 0.34, and the
    disagreement is per-molecule rather than uniform. That disagreement is a
    label-free signal for exactly the local ambiguity MolRouter exists to
    resolve, so it is exposed as a router feature.

Retrieval still happens on the PRIMARY fingerprint alone, so the fiber, the
anchor, and the "Morgan-anchored" story are unchanged. Only the descriptors of
the fiber gain channels.

COUNT vs BINARY. morganc and rdkc are count vectors. Binary Tanimoto computed
via A @ B.T is a dot product, not Tanimoto, and silently misreads counts as a
different (unbounded-shape) similarity. Count fingerprints therefore use MinMax
(Ruzicka) similarity, sum(min) / sum(max), which is the correct generalisation
and reduces to Tanimoto on binary input. MinMax has no matmul shortcut, so it is
only affordable for the within-fiber K x K blocks -- which is all this needs,
since retrieval stays on the primary fingerprint.
"""

from __future__ import annotations

import numpy as np

from deps import Chem, tqdm

# name -> (kind, builder). "binary" uses matmul Tanimoto; "count" uses MinMax.
FP_KINDS = {
    "ecfp4": "binary",
    "fcfp4": "binary",
    "rdk": "binary",
    "maccs": "binary",
    "atompair": "binary",
    "torsion": "binary",
    "pattern": "binary",
    "morganc": "count",
    "rdkc": "count",
}
DEFAULT_FPS = ["ecfp4", "rdk", "maccs"]

# Only binary fingerprints can retrieve: MinMax has no matmul form, so a
# full-dataset kNN under a count fingerprint is O(N*M*D) and impractical.
BIT_FPS = sorted(n for n, k in FP_KINDS.items() if k == "binary")
MACCS_BITS = 167


def _generator(name, n_bits, radius):
    from rdkit.Chem import rdFingerprintGenerator as rfg
    if name == "ecfp4":
        return rfg.GetMorganGenerator(radius=radius, fpSize=n_bits)
    if name == "fcfp4":
        inv = rfg.GetMorganFeatureAtomInvGen()
        return rfg.GetMorganGenerator(radius=radius, fpSize=n_bits,
                                      atomInvariantsGenerator=inv)
    if name == "morganc":
        return rfg.GetMorganGenerator(radius=radius, fpSize=n_bits,
                                      countSimulation=False)
    if name == "rdk":
        return rfg.GetRDKitFPGenerator(fpSize=n_bits, maxPath=6)
    if name == "rdkc":
        return rfg.GetRDKitFPGenerator(fpSize=n_bits, maxPath=6)
    if name == "atompair":
        return rfg.GetAtomPairGenerator(fpSize=n_bits)
    if name == "torsion":
        return rfg.GetTopologicalTorsionGenerator(fpSize=n_bits)
    return None


def _one(name, mol, gen, n_bits):
    from rdkit.Chem import MACCSkeys
    if name == "maccs":
        arr = np.zeros(MACCS_BITS, dtype=np.float32)
        for b in MACCSkeys.GenMACCSKeys(mol).GetOnBits():
            arr[b] = 1.0
        return arr
    if name == "pattern":
        arr = np.zeros(n_bits, dtype=np.float32)
        for b in Chem.PatternFingerprint(mol, fpSize=n_bits).GetOnBits():
            arr[b] = 1.0
        return arr
    if FP_KINDS[name] == "count":
        arr = np.zeros(n_bits, dtype=np.float32)
        for k, v in gen.GetCountFingerprint(mol).GetNonzeroElements().items():
            arr[k % n_bits] += v
        return arr
    arr = np.zeros(n_bits, dtype=np.float32)
    for b in gen.GetFingerprint(mol).GetOnBits():
        arr[b] = 1.0
    return arr


class FingerprintSet:
    """
    Row-aligned fingerprint matrices, one per name. names[0] is the PRIMARY:
    the anchor is trained on it and the fiber is retrieved with it.
    """

    def __init__(self, names, mats, kinds):
        self.names, self.mats, self.kinds = list(names), list(mats), list(kinds)

    @property
    def primary(self):
        return self.mats[0]

    @property
    def n_sim(self):
        return len(self.names)

    def subset(self, rows):
        return FingerprintSet(self.names, [m[rows] for m in self.mats], self.kinds)

    def concat(self, names=None):
        """
        (N, sum D) horizontal stack of the named fingerprints, for the anchor.

        The anchor is a random forest, which splits on thresholds one feature at
        a time, so blocks of different width and scale can simply sit side by
        side -- no folding, no normalisation, and count fingerprints need no
        MinMax because no similarity is computed here. That is exactly why the
        anchor can use fingerprints the fiber cannot.
        """
        if not names:
            return self.primary
        idx = []
        for n in names:
            if n not in self.names:
                raise KeyError(f"{n} was not built; add it to the fingerprint set")
            idx.append(self.names.index(n))
        if len(idx) == 1:
            return self.mats[idx[0]]
        return np.hstack([np.asarray(self.mats[i], dtype=np.float32) for i in idx])

    def rotated(self, i):
        """
        The same fingerprints with index i moved to the front.

        Used to build one fiber per fingerprint: fiber i is retrieved by, and
        weighted by, fingerprint i, while still carrying every other
        fingerprint as a similarity channel over its own members.
        """
        order = [i] + [j for j in range(len(self.names)) if j != i]
        return FingerprintSet([self.names[j] for j in order],
                              [self.mats[j] for j in order],
                              [self.kinds[j] for j in order])

    # Behave like the plain array this replaces, so row-indexing and length
    # checks elsewhere keep working unchanged.
    def __getitem__(self, rows):
        return self.subset(rows)

    def __len__(self):
        return self.mats[0].shape[0]

    @property
    def shape(self):
        return self.mats[0].shape

    def describe(self):
        return ", ".join(f"{n}[{k},{m.shape[1]}]"
                         for n, m, k in zip(self.names, self.mats, self.kinds))


def build_fingerprints(smiles, names=None, n_bits=2048, radius=2):
    """
    Returns (FingerprintSet, keep) where keep indexes the parseable molecules.

    Every fingerprint is computed on the same kept molecules, so all matrices
    stay row-aligned with each other and with the SMILES list.
    """
    names = list(names or DEFAULT_FPS)
    for n in names:
        if n not in FP_KINDS:
            raise ValueError(f"unknown fingerprint {n}; known: {sorted(FP_KINDS)}")

    gens = {n: _generator(n, n_bits, radius) for n in names}
    rows = {n: [] for n in names}
    keep = []
    for i, smi in enumerate(tqdm(smiles, desc="    fingerprints", leave=False)):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for n in names:
            rows[n].append(_one(n, mol, gens[n], n_bits))
        keep.append(i)

    mats, kinds = [], []
    for n in names:
        m = np.stack(rows[n]) if rows[n] else np.zeros((0, n_bits), np.float32)
        mats.append(m)
        kinds.append(FP_KINDS[n])
    return FingerprintSet(names, mats, kinds), np.array(keep, dtype=int)


# ---------------------------------------------------------------------------
# Similarities
# ---------------------------------------------------------------------------

def tanimoto_matrix(A, B=None):
    """Binary Tanimoto between rows of A and rows of B (default B = A)."""
    Af = A.astype(np.float32)
    Bf = Af if B is None else B.astype(np.float32)
    inter = Af @ Bf.T
    a = Af.sum(1)[:, None]
    b = Bf.sum(1)[None, :]
    union = a + b - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)


def similarity_matrix(A, B, kind):
    """
    Query-by-reference similarity. Count fingerprints fall back to binarised
    Tanimoto here: the full Q x N MinMax has no matmul form and is not
    affordable at retrieval scale. Within fibers, where the blocks are K x K,
    the exact MinMax is used instead (see pairwise_block).
    """
    if kind == "count":
        return tanimoto_matrix((A > 0).astype(np.float32), (B > 0).astype(np.float32))
    return tanimoto_matrix(A, B)


def pairwise_block(mem, kind, chunk=64):
    """
    (B, K, d) fiber members -> (B, K, K) similarities.

    Exact MinMax for counts, Tanimoto for binary. Chunked because the MinMax
    intermediate is (chunk, K, K, d).
    """
    B, K, _ = mem.shape
    out = np.zeros((B, K, K), dtype=np.float32)
    for s in range(0, B, chunk):
        e = min(s + chunk, B)
        blk = mem[s:e].astype(np.float32)
        if kind == "count":
            lo = np.minimum(blk[:, :, None, :], blk[:, None, :, :]).sum(-1)
            hi = np.maximum(blk[:, :, None, :], blk[:, None, :, :]).sum(-1)
            out[s:e] = np.divide(lo, hi, out=np.zeros_like(lo), where=hi > 0)
        else:
            inter = np.einsum("bik,bjk->bij", blk, blk, optimize=True)
            pop = blk.sum(2)
            union = pop[:, :, None] + pop[:, None, :] - inter
            out[s:e] = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return out


def knn_indices(fp_q, fp_ref, k, kind, self_rows=None, chunk=512):
    """Top-k neighbour indices into fp_ref, under one fingerprint."""
    Q, P = fp_q.shape[0], fp_ref.shape[0]
    k_eff = int(min(k, max(P - (1 if self_rows is not None else 0), 1)))
    idx = np.zeros((Q, k_eff), dtype=np.int64)
    for s in range(0, Q, chunk):
        e = min(s + chunk, Q)
        S = similarity_matrix(fp_q[s:e], fp_ref, kind)
        if self_rows is not None:
            rows = np.arange(s, e)
            hit = self_rows[rows] >= 0
            S[np.where(hit)[0], self_rows[rows][hit]] = -1.0
        idx[s:e] = np.argsort(-S, axis=1, kind="stable")[:, :k_eff]
    return idx


def knn_all(fps, fp_ref_set, k, self_rows=None, cache=None):
    """
    Top-k indices under EVERY fingerprint, keyed by fingerprint name.

    Retrieval is the expensive step (a full query-by-reference similarity per
    fingerprint), and it does not depend on which fingerprint is currently
    "primary". Building one fiber per fingerprint would otherwise repeat the
    same F retrievals F times; the cache makes it F.
    """
    cache = {} if cache is None else cache
    out = {}
    for i, name in enumerate(fps.names):
        key = (name, k)
        if key not in cache:
            cache[key] = knn_indices(fps.mats[i], fp_ref_set.mats[i], k,
                                     fps.kinds[i], self_rows)
        out[name] = cache[key]
    return out


def neighbourhood_agreement(fps, fp_ref_set, k, self_rows=None, cache=None):
    """
    How much the fingerprints agree about who x's neighbours are.

    Returns (jaccard (Q, A), alt_idx list) where A = number of non-primary
    fingerprints and jaccard[:, a] is |primary_knn & alt_knn| / |union|.

    A value near 1 means every notion of similarity picks the same neighbours,
    so the local neighbourhood is well defined and the anchor is probably on
    firm ground. A value near 0 means x sits on a boundary between similarity
    notions -- which is where an auxiliary view has something to add.
    """
    knn = knn_all(fps, fp_ref_set, k, self_rows, cache)
    prim = knn[fps.names[0]]
    prim_sets = [set(r.tolist()) for r in prim]

    jac, alt_idx = [], []
    for a in range(1, len(fps.names)):
        alt = knn[fps.names[a]]
        alt_idx.append(alt)
        col = np.empty(len(prim_sets), dtype=np.float32)
        for i, ps in enumerate(prim_sets):
            as_ = set(alt[i].tolist())
            u = len(ps | as_)
            col[i] = (len(ps & as_) / u) if u else 0.0
        jac.append(col)
    j = (np.stack(jac, axis=1) if jac
         else np.zeros((len(prim_sets), 0), dtype=np.float32))
    return j, alt_idx


def anchor_fiber_spread(blogit_ref, prim_idx, alt_idx_list):
    """
    (Q, T) std, across the alternative fibers, of each fiber's mean anchor logit.

    Small means every similarity notion places x among molecules the anchor
    scores alike, so which fiber you pick does not matter. Large means the
    fingerprints disagree about x in a way that changes what the anchor says
    about its neighbourhood -- ambiguity the router can act on. Uses only b_M,
    which is out-of-fold on train, so no label reaches this.
    """
    means = [blogit_ref[prim_idx].mean(axis=1)]
    for alt in alt_idx_list:
        means.append(blogit_ref[alt].mean(axis=1))
    if len(means) == 1:
        return np.zeros_like(means[0], dtype=np.float32)
    return np.stack(means, axis=0).std(axis=0).astype(np.float32)
