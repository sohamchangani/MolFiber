"""
Neural modules.

Two views per molecule, both frozen upstream of a trainable projection:

    g_x = P_G(GraphGrid(x))    topological, a CNN over the k x k block image
    m_x = P_F(MolFormer(x))    a linear projection of the frozen embedding

Two models over them, with different theories of how auxiliary evidence should
act on the Morgan anchor.

MolFiber treats the two views JOINTLY. One pair encoder sees both, so a single
contextual score u carries their combined verdict, and the model learns

  * a bounded residual  lambda * tanh(.)  -- how far it may move b_M at all;
  * a centring coefficient rho -- whether the correction is read absolutely
    (rho = 1, "MolFiber-G") or relative to the local Morgan neighbourhood
    (rho = 0, "MolFiber-L"), with M and A estimating rho instead of fixing it.

MolRouter keeps the two views SEPARATE, as competing relative experts. Each
expert compares x only against its Morgan neighbours and returns a centred
correction; a task-specific router then decides, per molecule, whether to leave
b_M alone or to hand the decision to one expert or the other.

So MolFiber asks whether and how much auxiliary evidence should refine Morgan;
MolRouter asks whether and WHICH view should resolve a local Morgan ambiguity.
In both, psi's output layer is zero-initialised, so at step 0 the correction is
identically zero and the model starts exactly at the Morgan anchor.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphGridCNN(nn.Module):
    """
    P_G: a small CNN over the (C, k, k) GraphGrid image.

    The ReLU output matters: both models form h_z * h_w as a co-activation term,
    and on unconstrained activations that product mixes sign agreement with
    magnitude and stops being readable as agreement.
    """

    def __init__(self, in_ch, d_out, width=32, dropout=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1), nn.BatchNorm2d(width), nn.ReLU(),
            nn.Conv2d(width, 2 * width, 3, padding=1), nn.BatchNorm2d(2 * width), nn.ReLU(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(4 * width, d_out), nn.ReLU())

    def forward(self, x):
        h = self.body(x)
        return self.head(torch.cat([h.mean(dim=(2, 3)), h.amax(dim=(2, 3))], dim=1))


def _mlp(sizes, dropout=0.1, out_relu=False):
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or out_relu:
            layers += [nn.ReLU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


# simmat is (B, K, K, S). Channel 0 is the PRIMARY fingerprint -- the one that
# retrieved the fiber. Both attention weightings read only that channel, so the
# fiber's geometry and weighting are exactly what the single-fingerprint model
# had; the extra channels enter only as descriptors inside the pair encoder.

def _primary(simmat):
    return simmat[..., 0]


def _beta_weights(simmat, mask, tau_c):
    """beta_xz over the neighbours of the centre: softmax(s_M(x,z) / tau_c)."""
    logits = (_primary(simmat)[:, 0, 1:] / tau_c).masked_fill(~mask[:, 1:], -1e9)
    return torch.softmax(logits, dim=1)


def _alpha_weights(simmat, mask, tau):
    """alpha_zw over w != z, restricted to valid fiber members."""
    K = mask.shape[1]
    eye = torch.eye(K, dtype=torch.bool, device=mask.device).unsqueeze(0)
    valid_w = mask.unsqueeze(1).expand(-1, K, -1) & ~eye
    logits = (_primary(simmat) / tau).masked_fill(~valid_w, -1e9)
    return torch.softmax(logits, dim=2)


class MolFiber(nn.Module):
    """
    Joint treatment of both views.

      r_zw = [g_z - g_w, g_z (*) g_w, m_z - m_w, m_z (*) m_w, s_M(z,w)]
      e_zw = phi(r_zw)
      c_xz = sum_{w != z} alpha_zw e_zw
      u_xz = psi(c_xz)
      S    = b_M + lambda * tanh( u_xx - (1 - rho) * ubar_x )

    psi sees c_xz and nothing else -- no g_x, no m_x, no b_M -- which is what
    stops the model collapsing into an ordinary global classifier on [g, m].

    Writing u_xx = (u_xx - ubar_x) + ubar_x splits the score into a WITHIN-fiber
    deviation and the BETWEEN-fiber level, so G keeps both and L keeps only the
    first: they are the endpoints of one shrinkage family, not two methods. If
    the contextual score carries a neighbourhood-level bias eta shared by every
    fiber member, centring cancels eta but also discards mean_z f(z), which is
    real signal because fiber members are near-neighbours. Hence rho.

      G  rho = 1 (fixed)          L  rho = 0 (fixed)
      M  rho = sigmoid(param), one per task -- the Mundlak form, with the within
         coefficient normalised to 1 so rho alone is identified
      A  rho(x) = tau^2 / (tau^2 + s^2(x) / n_eff(x)), the empirical-Bayes
         reliability of ubar_x, where s^2 is the within-fiber variance and
         n_eff = exp(entropy(beta)). tau^2 = kappa * Var_x(ubar_x) is estimated
         from a running between-fiber variance rather than fitted freely, which
         is what makes rho invariant to psi's arbitrary output scale.
    """

    VARIANTS = ("G", "L", "M", "A")

    def __init__(self, view_spec, n_tasks, d_view=128, d_pair=128, hidden=256,
                 mf_dim=0, variant="G", tau=0.1, tau_c=0.1, lam=4.0,
                 learn_lambda=False, dropout=0.1, rho_init=0.5, n_sim=1):
        super().__init__()
        self.variant = variant.upper()
        if self.variant not in self.VARIANTS:
            raise ValueError(f"variant must be one of {self.VARIANTS}")
        self.use_mf = mf_dim > 0
        self.tau, self.tau_c = tau, tau_c

        self.P_G = GraphGridCNN(view_spec.in_ch, d_view, view_spec.grid_width, dropout)
        self.P_F = nn.Sequential(nn.Linear(mf_dim, d_view), nn.ReLU()) if self.use_mf else None

        n_terms = 4 if self.use_mf else 2          # (diff, prod) per available view
        # n_sim similarity channels rather than one scalar: two molecules can be
        # substructure-similar and pharmacophore-dissimilar, and a single
        # Tanimoto collapses that distinction.
        self.n_sim = n_sim
        self.phi = _mlp([n_terms * d_view + n_sim, hidden, d_pair], dropout,
                        out_relu=True)
        self.psi = _mlp([d_pair, hidden, n_tasks], dropout)
        nn.init.zeros_(self.psi[-1].weight)
        nn.init.zeros_(self.psi[-1].bias)

        if learn_lambda:
            self.raw_lam = nn.Parameter(torch.tensor(float(np.log(np.expm1(lam)))))
        else:
            self.register_buffer("raw_lam", torch.tensor(float(np.log(np.expm1(lam)))))

        if self.variant == "M":
            p = float(np.log(rho_init / (1 - rho_init)))
            self.rho_logit = nn.Parameter(torch.full((n_tasks,), p))
        if self.variant == "A":
            self.raw_kappa = nn.Parameter(torch.zeros(n_tasks))
            self.register_buffer("between_var", torch.ones(n_tasks))
            self.between_momentum = 0.1
        self.last_rho = None

    @property
    def lam(self):
        return F.softplus(self.raw_lam)

    def forward(self, view_mem, mf_mem, simmat, mask, b_mem=None,
                agree=None, spread=None):
        """
        view_mem is a 1-tuple of GraphGrid images, each (B, K, C, k, k);
        mf_mem (B, K, D); simmat (B, K, K, S); mask (B, K). K = k + 1, member 0 is
        the centre. b_mem, agree and spread are accepted for interface parity
        with MolRouter and are unused here. Returns the correction (B, T).
        """
        B, K = mask.shape
        g = self.P_G(view_mem[0].flatten(0, 1)).view(B, K, -1)
        gz = g.unsqueeze(2).expand(-1, -1, K, -1)
        gw = g.unsqueeze(1).expand(-1, K, -1, -1)
        parts = [gz - gw, gz * gw]

        if self.use_mf:
            m = self.P_F(mf_mem.flatten(0, 1)).view(B, K, -1)
            mz = m.unsqueeze(2).expand(-1, -1, K, -1)
            mw = m.unsqueeze(1).expand(-1, K, -1, -1)
            parts += [mz - mw, mz * mw]
        parts.append(simmat)                               # (B, K, K, S)

        e = self.phi(torch.cat(parts, dim=-1))             # (B, K, K, d_pair)
        alpha = _alpha_weights(simmat, mask, self.tau)
        c = (alpha.unsqueeze(-1) * e).sum(dim=2)           # (B, K, d_pair)
        u = self.psi(c) * mask.unsqueeze(-1)               # (B, K, T)
        u_centre = u[:, 0]

        if self.variant == "G":
            self.last_rho = None
            return self.lam * torch.tanh(u_centre)

        beta = _beta_weights(simmat, mask, self.tau_c)
        ubar = (beta.unsqueeze(-1) * u[:, 1:]).sum(dim=1)  # (B, T)

        if self.variant == "L":
            rho = torch.zeros((), device=u.device)
        elif self.variant == "M":
            rho = torch.sigmoid(self.rho_logit)
        else:
            dev = u[:, 1:] - ubar.unsqueeze(1)
            s2 = (beta.unsqueeze(-1) * dev.pow(2)).sum(dim=1)
            ent = -(beta.clamp_min(1e-12) * beta.clamp_min(1e-12).log()).sum(1)
            n_eff = ent.exp().clamp(min=1.0).unsqueeze(-1)
            if self.training and ubar.shape[0] > 1:
                with torch.no_grad():
                    self.between_var.mul_(1 - self.between_momentum).add_(
                        self.between_momentum * ubar.detach().var(dim=0, unbiased=False))
            tau2 = F.softplus(self.raw_kappa) * self.between_var + 1e-12
            rho = tau2 / (tau2 + s2 / n_eff)

        self.last_rho = rho.detach()
        self._accumulate_rho(self.last_rho, B)
        return self.lam * torch.tanh(u_centre - (1.0 - rho) * ubar)

    def reset_report_stats(self):
        """Clear the accumulated rho so a report covers exactly one pass."""
        self._rho_sum, self._rho_n = 0.0, 0

    def _accumulate_rho(self, rho, batch):
        """
        Running mean of rho over a whole pass.

        rho is per-molecule for variant A, so reading it off the final minibatch
        would describe only the tail of the split. Accumulating makes the
        reported value the average over every molecule actually scored.
        """
        if not hasattr(self, "_rho_n"):
            self.reset_report_stats()
        val = rho.mean().item() if rho.dim() else float(rho)
        self._rho_sum += val * batch
        self._rho_n += batch

    def mean_rho(self):
        """Fitted shrinkage, for reporting. 1.0 behaves like G, 0.0 like L."""
        if self.variant == "G":
            return 1.0
        if self.variant == "L":
            return 0.0
        if self.variant == "M":
            return float(torch.sigmoid(self.rho_logit).mean())
        if getattr(self, "_rho_n", 0):
            return self._rho_sum / self._rho_n
        return float(self.last_rho.mean()) if self.last_rho is not None else float("nan")

    def shrinkage_parameters(self):
        """Parameters governing rho. They sit downstream of a near-zero psi early
        in training, so at the shared learning rate they barely move; the caller
        gives them their own, larger one."""
        names = {"M": ["rho_logit"], "A": ["raw_kappa"]}.get(self.variant, [])
        return [getattr(self, n) for n in names]


class _RelativeExpert(nn.Module):
    """
    One view, one centred verdict.

        r_zw   = [h_z - h_w, h_z (*) h_w, s_M(z,w)]
        e_zw   = phi(r_zw)
        c_xz   = sum_{w != z} alpha_zw e_zw
        u_xz   = psi(c_xz)
        delta  = u_xx - ubar_x

    The output is centred by construction, which is what makes it RELATIVE: the
    expert never reports an absolute opinion about x, only how x differs from
    the Morgan neighbours it is being compared against. Any bias the encoder
    carries uniformly across a region of chemical space cancels in that
    subtraction, so the router chooses between two views' local contrasts rather
    than between two miscalibrated absolute scales.
    """

    def __init__(self, d_view, d_pair, hidden, n_tasks, tau, tau_c, dropout, n_sim=1):
        super().__init__()
        self.tau, self.tau_c = tau, tau_c
        self.phi = _mlp([2 * d_view + n_sim, hidden, d_pair], dropout, out_relu=True)
        self.psi = _mlp([d_pair, hidden, n_tasks], dropout)
        nn.init.zeros_(self.psi[-1].weight)
        nn.init.zeros_(self.psi[-1].bias)

    def forward(self, h, simmat, mask):
        B, K, _ = h.shape
        hz = h.unsqueeze(2).expand(-1, -1, K, -1)
        hw = h.unsqueeze(1).expand(-1, K, -1, -1)
        e = self.phi(torch.cat([hz - hw, hz * hw, simmat], dim=-1))
        alpha = _alpha_weights(simmat, mask, self.tau)
        c = (alpha.unsqueeze(-1) * e).sum(dim=2)
        u = self.psi(c) * mask.unsqueeze(-1)               # (B, K, T)
        beta = _beta_weights(simmat, mask, self.tau_c)
        ubar = (beta.unsqueeze(-1) * u[:, 1:]).sum(dim=1)
        return u[:, 0] - ubar                              # (B, T)


class MolRouter(nn.Module):
    """
    Relative experts over fibers and views, with a task-specific router.

        S(x) = b_M(x) + lambda * sum_e pi_e(x) * tanh(delta_e(x))

    plus a "none" action contributing exactly zero, so routing to it leaves the
    Morgan prediction untouched rather than merely shrinking the correction --
    the router can genuinely abstain, which is the difference between "how much"
    and "whether".

    THE ACTION SPACE. An expert is a (fiber, view) pair.

      fiber  which fingerprint defines x's neighbourhood. A molecule's local
             neighbourhood is not canonical: ECFP4 and the RDKit path
             fingerprint share fewer than half of their top-8 neighbours, so two
             molecules can sit in the same ECFP4 fiber and different RDK fibers,
             differing in a way ECFP4 structurally cannot see.
      view   which representation states the difference, GraphGrid or MolFormer.

    --router_experts picks how much of that product to spend parameters on:

      cross  every (fiber, view) pair          1 + F*V actions
      fiber  one expert per fiber, both views  1 + F actions
      view   one expert per view, primary fiber only   1 + V actions

    With a single fingerprint, `cross` reduces to exactly the two-view router,
    so the multi-fiber model is a strict generalisation rather than a different
    method.

    THE ROUTER is conditioned on descriptors of the local Morgan ambiguity it
    exists to resolve: fiber geometry, how confident the anchor is at x, how far
    x departs from its neighbourhood, how much the neighbourhood's own anchor
    scores disagree, how much the fingerprints disagree about who the neighbours
    are, and how much the anchor's verdict changes when the neighbourhood is
    redefined. The experts' correction magnitudes enter as a mean and a max, so
    the router input width does not grow with the number of experts. No labels
    enter -- b_M is out-of-fold on train.
    """

    N_FIBER_FEATS = 6
    N_TASK_FEATS = 7
    VIEWS = ("graphgrid", "molformer")

    def __init__(self, view_spec, n_tasks, d_view=128, d_pair=128, hidden=256,
                 mf_dim=0, tau=0.1, tau_c=0.1, lam=4.0, learn_lambda=False,
                 dropout=0.1, router_hidden=64, router_temp=1.0,
                 router_none_prior=0.0, n_sim=1, n_fibers=1,
                 router_experts="cross", fiber_names=None, **_ignored):
        super().__init__()
        if mf_dim <= 0:
            raise ValueError(
                "MolRouter needs both views, but the MolFormer embedding is "
                "missing. Run without --lm none.")
        self.n_tasks, self.tau, self.tau_c = n_tasks, tau, tau_c
        self.router_temp = router_temp
        self.mode = router_experts
        self.n_fibers = 1 if router_experts == "view" else n_fibers
        names = list(fiber_names or [f"fp{i}" for i in range(self.n_fibers)])
        names = names[:self.n_fibers]

        self.P_G = GraphGridCNN(view_spec.in_ch, d_view, view_spec.grid_width, dropout)
        self.P_F = nn.Sequential(nn.Linear(mf_dim, d_view), nn.ReLU())

        # experts[e] = (fiber index, view index or None for "both views")
        self.expert_spec, self.actions = [], ["none"]
        for fi in range(self.n_fibers):
            if router_experts == "fiber":
                self.expert_spec.append((fi, None))
                self.actions.append(names[fi])
            else:
                for vi, v in enumerate(self.VIEWS):
                    self.expert_spec.append((fi, vi))
                    self.actions.append(v if router_experts == "view"
                                        else f"{names[fi]}/{v}")

        # An expert seeing both views takes their concatenation, hence 2*d_view.
        self.experts = nn.ModuleList([
            _RelativeExpert(d_view if vi is not None else 2 * d_view,
                            d_pair, hidden, n_tasks, tau, tau_c, dropout, n_sim)
            for _, vi in self.expert_spec])

        self.n_actions = len(self.actions)
        n_in = self.N_FIBER_FEATS + self.N_TASK_FEATS
        self.router = _mlp([n_in, router_hidden, self.n_actions], dropout)
        # A per-task bias makes the router task-specific without giving every
        # task its own weight matrix -- toxcast has ~600 of them.
        self.task_bias = nn.Parameter(torch.zeros(n_tasks, self.n_actions))
        with torch.no_grad():
            self.task_bias[:, 0] = router_none_prior

        if learn_lambda:
            self.raw_lam = nn.Parameter(torch.tensor(float(np.log(np.expm1(lam)))))
        else:
            self.register_buffer("raw_lam", torch.tensor(float(np.log(np.expm1(lam)))))
        self.last_pi = None

    @property
    def lam(self):
        return F.softplus(self.raw_lam)

    def _project(self, fb):
        """Per-member view embeddings for one fiber batch."""
        B, K = fb.mask.shape
        g = self.P_G(fb.view_mem[0].flatten(0, 1)).view(B, K, -1)
        m = self.P_F(fb.mf_mem.flatten(0, 1)).view(B, K, -1)
        return g, m

    def _ambiguity(self, fb, d_stack):
        """(B, T, N_FIBER_FEATS + N_TASK_FEATS) router input, from the primary fiber."""
        simmat, mask, b_mem = fb.simmat, fb.mask, fb.b_mem
        B, K = mask.shape
        valid = mask[:, 1:].float()
        n_valid = valid.sum(1, keepdim=True).clamp(min=1.0)
        s_raw = _primary(simmat)[:, 0, 1:]

        mean_s = (s_raw * valid).sum(1, keepdim=True) / n_valid
        var_s = ((s_raw - mean_s) ** 2 * valid).sum(1, keepdim=True) / n_valid
        beta = _beta_weights(simmat, mask, self.tau_c)
        ent = -(beta.clamp_min(1e-12) * beta.clamp_min(1e-12).log()).sum(1, keepdim=True)
        n_eff = ent.exp() / max(K - 1, 1)

        # Cross-fingerprint neighbourhood agreement, aggregated to mean and min so
        # the width does not depend on how many fingerprints are configured.
        # Absent alternatives give 1.0 -- "every fingerprint agrees" -- the
        # correct no-information default.
        agree = fb.agree
        if agree is None or agree.shape[-1] == 0:
            a_mean = torch.ones(B, 1, device=mask.device)
            a_min = torch.ones(B, 1, device=mask.device)
        else:
            a_mean = agree.mean(dim=1, keepdim=True)
            a_min = agree.min(dim=1, keepdim=True).values
        fiber = torch.cat([mean_s, var_s.clamp_min(0).sqrt(), n_eff,
                           n_valid / max(K - 1, 1), a_mean, a_min], dim=1)

        if b_mem is None:
            b_mem = torch.zeros(B, K, self.n_tasks, device=mask.device)
        b_x = b_mem[:, 0]
        b_nb = b_mem[:, 1:]
        w = (valid / n_valid).unsqueeze(-1)
        b_bar = (b_nb * w).sum(1)
        b_sd = (((b_nb - b_bar.unsqueeze(1)) ** 2) * w).sum(1).clamp_min(0).sqrt()

        mag = torch.tanh(d_stack).abs()                    # (E, B, T)
        sp = (torch.zeros_like(b_x) if fb.spread is None or fb.spread.shape[-1] == 0
              else fb.spread)
        task = torch.stack([b_x, b_x.abs(), b_x - b_bar, b_sd,
                            mag.mean(0), mag.amax(0), sp], dim=-1)

        return torch.cat([fiber.unsqueeze(1).expand(-1, self.n_tasks, -1), task], dim=-1)

    def forward(self, batches):
        """
        batches: list of FiberBatch, one per fingerprint fiber (primary first).
        Returns the correction (B, T) to add to b_M.
        """
        if not isinstance(batches, (list, tuple)):
            batches = [batches]
        proj = [self._project(fb) for fb in batches[:self.n_fibers]]

        deltas = []
        for (fi, vi), expert in zip(self.expert_spec, self.experts):
            g, m = proj[fi]
            h = g if vi == 0 else m if vi == 1 else torch.cat([g, m], dim=-1)
            fb = batches[fi]
            deltas.append(expert(h, fb.simmat, fb.mask))
        d_stack = torch.stack(deltas, dim=0)               # (E, B, T)

        feats = self._ambiguity(batches[0], d_stack)
        logits = (self.router(feats) + self.task_bias.unsqueeze(0)) / self.router_temp
        pi = torch.softmax(logits, dim=-1)                 # (B, T, 1 + E)
        self.last_pi = pi.detach()
        self._pi_live = pi
        self._accumulate_pi(self.last_pi, pi.shape[0])

        # pi[..., 0] is "none" and multiplies nothing.
        corr = (pi[..., 1:].permute(2, 0, 1) * torch.tanh(d_stack)).sum(0)
        return self.lam * corr

    def router_entropy_term(self):
        """
        Mean routing entropy for the current batch. Penalising it makes the
        router commit to an action rather than hedge across all of them.

        Reads the LIVE pi, not the detached copy kept for reporting -- a penalty
        computed on the detached tensor would silently carry no gradient.
        """
        p = getattr(self, "_pi_live", None)
        if p is None:
            return torch.zeros((), device=self.task_bias.device)
        p = p.clamp_min(1e-12)
        return -(p * p.log()).sum(-1).mean()

    def reset_report_stats(self):
        """Clear the accumulated routing mix so a report covers one pass."""
        self._pi_sum, self._pi_n = None, 0

    def _accumulate_pi(self, pi, batch):
        """
        Running mean of the routing distribution over a whole pass.

        Routing is per molecule, so reading the mix off the final minibatch
        would describe only the tail of the split rather than the dataset.
        """
        if not hasattr(self, "_pi_n"):
            self.reset_report_stats()
        m = pi.mean(dim=(0, 1)) * batch
        self._pi_sum = m if self._pi_sum is None else self._pi_sum + m
        self._pi_n += batch

    def routing_mix(self):
        """Mean probability mass on each action over the last full pass."""
        if getattr(self, "_pi_n", 0):
            mix = self._pi_sum / self._pi_n
        elif self.last_pi is not None:
            mix = self.last_pi.mean(dim=(0, 1))
        else:
            return {a: float("nan") for a in self.actions}
        return {a: float(mix[i]) for i, a in enumerate(self.actions)}

    def shrinkage_parameters(self):
        """Router parameters sit downstream of zero-initialised experts, so like
        MolFiber's rho they need their own, larger learning rate."""
        return [self.task_bias] + list(self.router.parameters())


class GridOnly(nn.Module):
    """GraphGrid alone: no fibers, no Morgan anchor. The ablation baseline."""

    def __init__(self, view_spec, n_tasks, d_view=128, hidden=256, dropout=0.1):
        super().__init__()
        self.E_G = GraphGridCNN(view_spec.in_ch, d_view, view_spec.grid_width, dropout)
        self.head = _mlp([d_view, hidden, n_tasks], dropout)

    def forward(self, *view):
        return self.head(self.E_G(view[0]))
