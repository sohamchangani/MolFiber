"""Approximate Morgan fibers B_x = {x} u F_k(x)."""

from __future__ import annotations

import numpy as np

from fingerprints import (anchor_fiber_spread, neighbourhood_agreement,
                          pairwise_block, similarity_matrix)


# =====================================================================
# 5. Approximate Morgan fibers  B_x = {x} u F_k(x)
# =====================================================================
#
# C_M = D (no candidate restriction), and the retrieval set is R = D_train for
# EVERY molecule, train or test. The task is inductive: a test molecule pulls
# neighbours only from labelled training data, never from other test molecules.
#
# Member 0 of every fiber is the centre x itself. Neighbour labels are never
# read -- r_zw contains only projected views and s_M -- so the only path from a
# label to a score is b_M, which is out-of-fold on train.

class Fibers:
    """
    centre : (Q,)             rows into the query split
    nbr    : (Q, k)           rows into the TRAIN split
    simmat : (Q, k+1, k+1, S) s_M(z, w) per fingerprint. Channel 0 is the
                              PRIMARY fingerprint -- the one that retrieved the
                              fiber, and the only one the attention weightings
                              read. The rest are extra descriptors of the same
                              pairs, so the fiber geometry is unchanged.
    mask   : (Q, k+1)         valid members (index 0, the centre, always True)
    agree  : (Q, A)           Jaccard overlap between the primary neighbour set
                              and each alternative fingerprint's, per molecule
    spread : (Q, T)           std across fingerprints of each alternative
                              fiber's mean anchor logit
    """

    __slots__ = ("centre", "nbr", "simmat", "mask", "agree", "spread")

    def __init__(self, centre, nbr, simmat, mask, agree=None, spread=None):
        self.centre, self.nbr, self.simmat, self.mask = centre, nbr, simmat, mask
        Q = centre.shape[0]
        self.agree = np.zeros((Q, 0), np.float32) if agree is None else agree
        self.spread = np.zeros((Q, 0), np.float32) if spread is None else spread

    def __len__(self):
        return self.centre.shape[0]

    @property
    def n_sim(self):
        return self.simmat.shape[-1]


def build_fibers(fps_query, fps_ref, k, self_rows=None, chunk=512,
                 blogit_ref=None, knn_cache=None):
    """
    F_k(x) = kNN_M(x; R) under the PRIMARY fingerprint, with every fingerprint
    contributing a similarity channel over the resulting fiber.

    fps_query / fps_ref are FingerprintSet objects (or bare arrays, treated as a
    single binary fingerprint). Retrieval uses only the primary, so the fiber is
    exactly the one the single-fingerprint version would have built; the extra
    fingerprints add channels, never members.

    self_rows: (Q,) row of each query inside fps_ref, or -1. Used so a training
    molecule is not retrieved as its own neighbour.

    blogit_ref: (N_ref, T) anchor logits of the reference split. When given, the
    cross-fingerprint anchor spread is computed here, since it depends only on
    b_M and the alternative neighbour sets -- no labels, no model.
    """
    fps_query = _as_set(fps_query)
    fps_ref = _as_set(fps_ref)
    S = fps_query.n_sim

    Q, P = fps_query.primary.shape[0], fps_ref.primary.shape[0]
    k_eff = int(min(k, max(P - (1 if self_rows is not None else 0), 1)))

    nbr = np.zeros((Q, k), dtype=np.int64)
    mask = np.zeros((Q, k + 1), dtype=bool)
    mask[:, 0] = True
    simmat = np.zeros((Q, k + 1, k + 1, S), dtype=np.float32)

    for s in range(0, Q, chunk):
        e = min(s + chunk, Q)
        sim = similarity_matrix(fps_query.primary[s:e], fps_ref.primary,
                                fps_query.kinds[0])
        if self_rows is not None:
            rows = np.arange(s, e)
            hit = self_rows[rows] >= 0
            sim[np.where(hit)[0], self_rows[rows][hit]] = -1.0
        order = np.argsort(-sim, axis=1, kind="stable")[:, :k_eff]
        top = np.take_along_axis(sim, order, axis=1)
        nbr[s:e, :k_eff] = order
        mask[s:e, 1:k_eff + 1] = top >= 0.0

        for c in range(S):
            mem = np.concatenate(
                [fps_query.mats[c][s:e][:, None, :].astype(np.float32),
                 fps_ref.mats[c][nbr[s:e]].astype(np.float32)], axis=1)
            simmat[s:e, :, :, c] = pairwise_block(mem, fps_query.kinds[c])

    bad = ~mask
    simmat[bad[:, :, None].repeat(k + 1, 2)] = 0.0
    simmat[bad[:, None, :].repeat(k + 1, 1)] = 0.0

    agree = spread = None
    if S > 1:
        agree, alt_idx = neighbourhood_agreement(fps_query, fps_ref, k_eff,
                                                 self_rows, knn_cache)
        if blogit_ref is not None:
            spread = anchor_fiber_spread(blogit_ref, nbr[:, :k_eff], alt_idx)
    return Fibers(np.arange(Q), nbr, simmat, mask, agree, spread)


def build_multi_fibers(fps_query, fps_ref, k, self_rows=None, chunk=512,
                       blogit_ref=None, per_fingerprint=True):
    """
    One fiber per fingerprint, or just the primary fiber.

    A molecule's "local neighbourhood" is not a canonical object: it depends on
    which similarity defines it, and the definitions disagree. Over drug-like
    molecules ECFP4 and the RDKit path fingerprint share fewer than half of
    their top-8 neighbours. Two molecules that sit in the same ECFP4 fiber but
    different RDK fibers differ in a way ECFP4 structurally cannot see, which is
    exactly the local ambiguity an auxiliary view should be asked to resolve.

    Fiber i is retrieved AND weighted by fingerprint i (the set is rotated so
    that fingerprint sits in channel 0), while still carrying every other
    fingerprint as a descriptor channel over its own members. Returns a list of
    Fibers, primary first.
    """
    fps_query, fps_ref = _as_set(fps_query), _as_set(fps_ref)
    n = fps_query.n_sim if per_fingerprint else 1
    # Retrieval is keyed by fingerprint, not by which one is currently primary,
    # so the F fibers share one cache instead of repeating F retrievals F times.
    cache, out = {}, []
    for i in range(n):
        q = fps_query.rotated(i) if i else fps_query
        r = fps_ref.rotated(i) if i else fps_ref
        out.append(build_fibers(q, r, k, self_rows, chunk, blogit_ref, cache))
    return out


def _as_set(x):
    """Accept a FingerprintSet or a bare array (single binary fingerprint)."""
    if hasattr(x, "mats"):
        return x
    from fingerprints import FingerprintSet
    return FingerprintSet(["primary"], [x], ["binary"])
