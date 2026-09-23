"""
Fusion ablation models. STANDALONE.

Nothing in the main pipeline imports this file, and deleting it leaves the
pipeline working unchanged. It reads from the pipeline (GraphGridCNN, _mlp) but
never modifies it, so the MolFiber results and these ablations can be run from
the same checkout without one affecting the other.

These exist to answer the question a reviewer asks first: is the fiber doing any
work, or would ordinary multimodal fusion of GraphGrid, MolFormer and Morgan get
there too? Each baseline sees exactly the same three views as MolFiber, the same
frozen encoders and the same training protocol, and differs only in how the views
are combined. None of them uses a fiber.

    early        feature-level. Concatenate the three view embeddings and feed
                 one MLP. The standard strong multimodal baseline.

    late         decision-level. One head per view, each producing its own
                 logits, combined afterwards -- by a fixed mean or by learned
                 per-target convex weights. Views never interact before the
                 decision, so this measures how much of the signal is simply
                 additive across views.

    gated        intermediate. Per-view logits as in late fusion, but the mixture
                 weights are produced per molecule by a gate conditioned on the
                 view embeddings. This is deliberately MolRouter with the fiber
                 removed: same mixture-over-views idea, but gating on ABSOLUTE
                 view embeddings rather than fiber-relative contrasts, and with
                 no abstain action. It is the sharpest of the three, because
                 beating early or late fusion could be credited to gating alone,
                 whereas beating this isolates the fiber.

    anchor_mlp   not a fusion scheme but the decisive ablation of MolFiber:
                 S = b_M + lambda*tanh(MLP([g_x, m_x])). The fiber is deleted and
                 everything else held fixed -- frozen anchor, same lambda bound,
                 zero-initialised readout so S == b_M at step zero. The gap to
                 MolFiber-L is attributable to the fiber and nothing else.

FAIRNESS. The Morgan branch defaults to the same frozen random-forest logit b_M
that MolFiber anchors on. Giving the baselines a weak trainable head on raw
fingerprints while MolFiber gets the tuned forest would manufacture a win that
has nothing to do with fibers. morgan_mode="fp" selects the trainable-head
version for a pure multimodal comparison; report the two separately.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import GraphGridCNN, _mlp          # read-only import

FUSION_METHODS = ("early", "late", "gated", "anchor_mlp")
VIEW_NAMES = ("graphgrid", "molformer", "morgan")


class MorganBranch(nn.Module):
    """
    The Morgan view as either the frozen anchor logit or a trainable head.

    In "anchor" mode the expert logit IS b_M, passed through untouched, so the
    baselines inherit exactly the anchor quality MolFiber is built on; the
    embedding offered to the gate and to early fusion is a small projection of
    it. In "fp" mode both come from a trainable projection of the fingerprint.
    """

    def __init__(self, d_fp, n_tasks, d_view, dropout=0.1, mode="anchor"):
        super().__init__()
        self.mode = mode
        d_in = n_tasks if mode == "anchor" else d_fp
        self.proj = nn.Sequential(nn.Linear(d_in, d_view), nn.ReLU())
        self.head = None if mode == "anchor" else _mlp([d_view, n_tasks], dropout)

    def forward(self, fp, blogit):
        src = blogit if self.mode == "anchor" else fp
        emb = self.proj(src)
        logit = blogit if self.mode == "anchor" else self.head(emb)
        return emb, logit


class ViewTrunk(nn.Module):
    """Shared encoders: GraphGrid CNN, MolFormer projection, Morgan branch."""

    def __init__(self, view_spec, n_tasks, d_fp, d_view=128, mf_dim=0,
                 dropout=0.1, morgan_mode="anchor"):
        super().__init__()
        if mf_dim <= 0:
            raise ValueError("fusion baselines need the MolFormer view; "
                             "run without --lm none.")
        self.P_G = GraphGridCNN(view_spec.in_ch, d_view, view_spec.grid_width, dropout)
        self.P_F = nn.Sequential(nn.Linear(mf_dim, d_view), nn.ReLU())
        self.morgan = MorganBranch(d_fp, n_tasks, d_view, dropout, morgan_mode)

    def embed(self, grid, mf, fp, blogit):
        b_emb, b_logit = self.morgan(fp, blogit)
        return self.P_G(grid), self.P_F(mf), b_emb, b_logit


class EarlyFusion(nn.Module):
    """Concatenate the three view embeddings, then one joint MLP."""

    def __init__(self, view_spec, n_tasks, d_fp, d_view=128, hidden=256,
                 mf_dim=0, dropout=0.1, morgan_mode="anchor", **_):
        super().__init__()
        self.trunk = ViewTrunk(view_spec, n_tasks, d_fp, d_view, mf_dim,
                               dropout, morgan_mode)
        self.head = _mlp([3 * d_view, hidden, n_tasks], dropout)

    def forward(self, grid, mf, fp, blogit):
        g, m, b_emb, _ = self.trunk.embed(grid, mf, fp, blogit)
        return self.head(torch.cat([g, m, b_emb], dim=-1))


class LateFusion(nn.Module):
    """
    One head per view; logits combined afterwards.

    mode="mean"    fixed equal weights, the classic decision-level baseline.
    mode="learned" per-target convex weights via softmax of a free parameter --
                   stacking with the combiner trained jointly. It strictly
                   contains the mean, so it cannot be worse except through
                   optimisation noise.
    """

    def __init__(self, view_spec, n_tasks, d_fp, d_view=128, hidden=256,
                 mf_dim=0, dropout=0.1, morgan_mode="anchor", mode="learned", **_):
        super().__init__()
        self.trunk = ViewTrunk(view_spec, n_tasks, d_fp, d_view, mf_dim,
                               dropout, morgan_mode)
        self.head_g = _mlp([d_view, hidden, n_tasks], dropout)
        self.head_m = _mlp([d_view, hidden, n_tasks], dropout)
        self.mode = mode
        if mode == "learned":
            self.w = nn.Parameter(torch.zeros(n_tasks, 3))

    def forward(self, grid, mf, fp, blogit):
        g, m, _, b_logit = self.trunk.embed(grid, mf, fp, blogit)
        z = torch.stack([self.head_g(g), self.head_m(m), b_logit], dim=-1)
        if self.mode == "mean":
            return z.mean(-1)
        return (torch.softmax(self.w, dim=-1).unsqueeze(0) * z).sum(-1)

    def view_weights(self):
        if self.mode != "learned":
            return {k: 1 / 3 for k in VIEW_NAMES}
        w = torch.softmax(self.w, dim=-1).mean(0)
        return dict(zip(VIEW_NAMES, (float(v) for v in w)))


class GatedFusion(nn.Module):
    """
    Per-molecule mixture over the three view logits.

    The gate reads the concatenated view embeddings and emits per-target weights
    over the three views. Relative to MolRouter this keeps the mixture-of-experts
    idea but drops the two things that define MolRouter: the experts here score a
    molecule in absolute terms rather than relative to its Morgan neighbours, and
    there is no abstain action, so the anchor can be down-weighted but never left
    untouched.
    """

    def __init__(self, view_spec, n_tasks, d_fp, d_view=128, hidden=256,
                 mf_dim=0, dropout=0.1, morgan_mode="anchor", gate_hidden=64,
                 gate_temp=1.0, **_):
        super().__init__()
        self.trunk = ViewTrunk(view_spec, n_tasks, d_fp, d_view, mf_dim,
                               dropout, morgan_mode)
        self.head_g = _mlp([d_view, hidden, n_tasks], dropout)
        self.head_m = _mlp([d_view, hidden, n_tasks], dropout)
        self.gate = _mlp([3 * d_view, gate_hidden, 3 * n_tasks], dropout)
        self.n_tasks, self.gate_temp = n_tasks, gate_temp
        self.last_pi = None

    def forward(self, grid, mf, fp, blogit):
        g, m, b_emb, b_logit = self.trunk.embed(grid, mf, fp, blogit)
        z = torch.stack([self.head_g(g), self.head_m(m), b_logit], dim=-1)
        logits = self.gate(torch.cat([g, m, b_emb], dim=-1))
        pi = torch.softmax(logits.view(-1, self.n_tasks, 3) / self.gate_temp, dim=-1)
        self.last_pi = pi.detach()
        return (pi * z).sum(-1)

    def view_weights(self):
        if self.last_pi is None:
            return {k: float("nan") for k in VIEW_NAMES}
        w = self.last_pi.mean(dim=(0, 1))
        return dict(zip(VIEW_NAMES, (float(v) for v in w)))


class AnchoredResidual(nn.Module):
    """
    MolFiber with the fiber removed: S = b_M + lambda*tanh(MLP([g_x, m_x])).

    Frozen anchor, same bound, zero-initialised readout so S == b_M at step zero.
    The auxiliary views act on x alone instead of through comparisons with its
    Morgan neighbours, so the gap to MolFiber-L measures the fiber and nothing
    else.
    """

    def __init__(self, view_spec, n_tasks, d_fp, d_view=128, hidden=256,
                 mf_dim=0, dropout=0.1, lam=4.0, morgan_mode="anchor", **_):
        super().__init__()
        self.trunk = ViewTrunk(view_spec, n_tasks, d_fp, d_view, mf_dim,
                               dropout, morgan_mode)
        self.head = _mlp([2 * d_view, hidden, n_tasks], dropout)
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.register_buffer("raw_lam", torch.tensor(float(np.log(np.expm1(lam)))))

    @property
    def lam(self):
        return F.softplus(self.raw_lam)

    def forward(self, grid, mf, fp, blogit):
        g, m, _, _ = self.trunk.embed(grid, mf, fp, blogit)
        return blogit + self.lam * torch.tanh(self.head(torch.cat([g, m], dim=-1)))


def build_fusion(method, view_spec, n_tasks, d_fp, args, mf_dim):
    common = dict(view_spec=view_spec, n_tasks=n_tasks, d_fp=d_fp,
                  d_view=args.d_view, hidden=args.hidden, mf_dim=mf_dim,
                  dropout=args.dropout, morgan_mode=args.fusion_morgan)
    if method == "early":
        return EarlyFusion(**common)
    if method == "late":
        return LateFusion(mode=args.late_mode, **common)
    if method == "gated":
        return GatedFusion(gate_hidden=args.gate_hidden,
                           gate_temp=args.gate_temp, **common)
    if method == "anchor_mlp":
        return AnchoredResidual(lam=args.lam, **common)
    raise ValueError(f"unknown fusion method {method}")
