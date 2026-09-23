"""Training and evaluation loops for MolFiber and the GraphGrid-CNN baseline."""

from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from fibers import build_multi_fibers
from model import GridOnly, MolFiber, MolRouter


def _logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


METRICS = {"auroc": roc_auc_score, "auprc": average_precision_score}


def score_per_task(y, scores, task_ids, metric="auroc"):
    """
    Per-task score; NaN where the fold is unscorable.

    auprc (average precision) is what TDC's leaderboard uses for the CYP
    datasets, where positives are rare enough that ROC-AUC flatters a model.
    """
    fn = METRICS[metric]
    out = np.full(len(task_ids), np.nan)
    for j, t in enumerate(task_ids):
        col = y[:, t]
        m = ~np.isnan(col)
        if m.sum() and np.unique(col[m]).size == 2:
            out[j] = fn(col[m], scores[m, j])
    return out


def auc_per_task(y, scores, task_ids):
    """ROC-AUC per task. Kept for call sites that want it explicitly."""
    return score_per_task(y, scores, task_ids, "auroc")


def make_fibers(sv, train_sv, k, is_train=False, per_fingerprint=False):
    """
    R = D_train always. Training molecules exclude themselves from their own fiber.

    Returns a LIST of fibers -- one per fingerprint when per_fingerprint, else
    just the primary. Every entry shares the same centre rows.
    """
    self_rows = np.arange(len(sv.fp)) if is_train else None
    return build_multi_fibers(sv.fp, train_sv.fp, k, self_rows,
                              blogit_ref=train_sv.blogit,
                              per_fingerprint=per_fingerprint)


class FiberBatch:
    """One fiber's slice of a minibatch, in the layout both models consume."""

    __slots__ = ("view_mem", "mf_mem", "simmat", "mask", "b_mem", "agree", "spread")

    def __init__(self, view_mem, mf_mem, simmat, mask, b_mem, agree, spread):
        self.view_mem, self.mf_mem = view_mem, mf_mem
        self.simmat, self.mask = simmat, mask
        self.b_mem, self.agree, self.spread = b_mem, agree, spread


def _batch_one(model, sv, train_sv, fib, pos, dev):
    """Assemble one FiberBatch: centre from `sv`, neighbours from `train_sv`."""
    c_rows = fib.centre[pos]
    n_rows = fib.nbr[pos]
    view_mem = tuple(
        torch.cat([c.unsqueeze(1), n], dim=1).to(dev)
        for c, n in zip(sv.gather(c_rows), train_sv.gather(n_rows)))
    if getattr(model, "use_mf", True):
        mf = torch.cat([sv.mf[c_rows].unsqueeze(1),
                        train_sv.mf[n_rows]], dim=1).to(dev)
    else:
        mf = torch.zeros(1, device=dev)
    # Anchor logits of every fiber member. The router reads the neighbourhood's
    # Morgan disagreement from these; MolFiber ignores them. No labels are
    # involved -- b_M is out-of-fold on train.
    b_mem = torch.from_numpy(
        np.concatenate([sv.blogit[c_rows][:, None, :],
                        train_sv.blogit[n_rows]], axis=1)).float().to(dev)
    agree = (torch.from_numpy(fib.agree[pos]).float().to(dev)
             if fib.agree.shape[1] else None)
    spread = (torch.from_numpy(fib.spread[pos]).float().to(dev)
              if fib.spread.shape[1] else None)
    return FiberBatch(view_mem, mf,
                      torch.from_numpy(fib.simmat[pos]).to(dev),
                      torch.from_numpy(fib.mask[pos]).to(dev),
                      b_mem, agree, spread)


def _gather(model, sv, train_sv, fibs, pos, dev):
    """
    Run the model on one minibatch.

    `fibs` is a list of Fibers, one per fingerprint fiber, primary first. Every
    fiber shares the same centre rows, so a molecule's several neighbourhoods
    line up; models that use only one fiber take the primary.
    """
    if not isinstance(fibs, (list, tuple)):
        fibs = [fibs]
    n_needed = getattr(model, "n_fibers", 1)
    batches = [_batch_one(model, sv, train_sv, f, pos, dev)
               for f in fibs[:max(n_needed, 1)]]
    if isinstance(model, MolRouter):
        return model(batches)
    b = batches[0]
    return model(b.view_mem, b.mf_mem, b.simmat, b.mask, b.b_mem, b.agree, b.spread)


@torch.no_grad()
def predict_scores(model, sv, train_sv, fib, task_ids, dev, bs):
    """S(x) = b_M(x) + correction, returned as probabilities."""
    model.eval()
    # Reported rho / routing mix must describe this pass, not whatever batch ran
    # last, so the accumulators start clean here.
    if hasattr(model, "reset_report_stats"):
        model.reset_report_stats()
    n = len(fib[0]) if isinstance(fib, (list, tuple)) else len(fib)
    out = np.zeros((n, len(task_ids)))
    for s in range(0, n, bs):
        pos = np.arange(s, min(s + bs, n))
        out[pos] = _gather(model, sv, train_sv, fib, pos, dev).cpu().numpy()
    S = sv.blogit[:, task_ids] + out
    return 1.0 / (1.0 + np.exp(-S))


def train_molfiber(tr, va, fib_tr, fib_va, task_ids, args, variant, seed, dev,
                   view_spec):
    torch.manual_seed(seed)
    np.random.seed(seed)

    common = dict(view_spec=view_spec, n_tasks=len(task_ids), d_view=args.d_view,
                  d_pair=args.d_pair, hidden=args.hidden,
                  mf_dim=(tr.mf.shape[1] if tr.mf is not None else 0),
                  tau=args.tau, tau_c=args.tau_c, lam=args.lam,
                  learn_lambda=args.learn_lambda, dropout=args.dropout,
                  n_sim=fib_tr[0].n_sim)
    if variant == "ROUTER":
        model = MolRouter(router_hidden=args.router_hidden,
                          router_temp=args.router_temp,
                          router_none_prior=args.router_none_prior,
                          n_fibers=len(fib_tr),
                          router_experts=args.router_experts,
                          fiber_names=getattr(tr.fp, "names", None),
                          **common).to(dev)
    else:
        model = MolFiber(variant=variant,
                         rho_init=getattr(args, "rho_init", 0.5), **common).to(dev)

    shrink = getattr(model, "shrinkage_parameters", list)()
    shrink_ids = {id(p) for p in shrink}
    groups = [{"params": [p for p in model.parameters() if id(p) not in shrink_ids],
               "lr": args.lr}]
    if shrink:
        groups.append({"params": shrink,
                       "lr": args.lr * getattr(args, "rho_lr_mult", 10.0),
                       "weight_decay": 0.0})
    opt = torch.optim.Adam(groups, lr=args.lr, weight_decay=args.weight_decay)
    y_tr = torch.from_numpy(tr.y[:, task_ids]).float()
    b_tr = torch.from_numpy(tr.blogit[:, task_ids]).float()
    n = len(fib_tr[0])

    best, best_state, patience, best_epoch = -np.inf, None, 0, 0
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(n)
        tot, nb = 0.0, 0
        for s in range(0, n, args.batch_size):
            pos = perm[s:s + args.batch_size]
            corr = _gather(model, tr, tr, fib_tr, pos, dev)
            rows = fib_tr[0].centre[pos]
            S = b_tr[rows].to(dev) + corr
            target = y_tr[rows].to(dev)
            m = ~torch.isnan(target)
            if m.sum() == 0:
                continue
            loss = F.binary_cross_entropy_with_logits(S[m], torch.nan_to_num(target)[m])
            coef = getattr(args, "router_entropy", 0.0)
            if coef and hasattr(model, "router_entropy_term"):
                # Penalising routing entropy pushes the router to commit to one
                # action instead of hedging across all three.
                loss = loss + coef * model.router_entropy_term()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss)
            nb += 1

        sc = predict_scores(model, va, tr, fib_va, task_ids, dev, args.batch_size)
        auc = np.nanmean(score_per_task(va.y, sc, task_ids,
                                        getattr(args, "metric", "auroc")))
        auc = -np.inf if np.isnan(auc) else auc
        if args.verbose:
            print(f"      [{variant}] epoch {epoch:3d}  loss {tot / max(nb,1):.4f}  "
                  f"{getattr(args, 'select_on', 'valid')} "
                  f"{getattr(args, 'metric', 'auroc')} {auc:.4f}  "
                  f"lambda {float(model.lam):.3f}")
        if auc > best + 1e-5:
            best, patience, best_epoch = auc, 0, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.best_epoch_ = max(best_epoch, 1)
    model.best_score_ = best
    return model


def refit_on_pool(method, variant, pool, task_ids, args, seed, dev, view_spec,
                  make_fibers_fn, epochs):
    """
    Train on the WHOLE pool for a fixed number of epochs, no early stopping.

    The analogue of sklearn's refit=True: every label in the pool contributes to
    the final model, and the epoch budget comes from the inner CV rather than
    from a fold held out here.
    """
    a = copy.copy(args)
    a.epochs, a.patience = max(int(epochs), 1), 10 ** 9
    if method == "grid":
        return train_grid_only(pool, pool, task_ids, a, seed, dev, view_spec)
    fib, _ = make_fibers_fn(pool, pool, a.k)
    return train_molfiber(pool, pool, fib, fib, task_ids, a, variant, seed, dev,
                          view_spec)


def train_grid_only(tr, va, task_ids, args, seed, dev, view_spec):
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = GridOnly(view_spec, len(task_ids), args.d_view, args.hidden,
                     args.dropout).to(dev)
    shrink = getattr(model, "shrinkage_parameters", list)()
    shrink_ids = {id(p) for p in shrink}
    groups = [{"params": [p for p in model.parameters() if id(p) not in shrink_ids],
               "lr": args.lr}]
    if shrink:
        groups.append({"params": shrink,
                       "lr": args.lr * getattr(args, "rho_lr_mult", 10.0),
                       "weight_decay": 0.0})
    opt = torch.optim.Adam(groups, lr=args.lr, weight_decay=args.weight_decay)
    y_tr = torch.from_numpy(tr.y[:, task_ids]).float()

    best, best_state, patience, best_epoch = -np.inf, None, 0, 0
    for epoch in range(args.epochs):
        model.train()
        perm = np.random.permutation(tr.view[0].shape[0])
        for s in range(0, perm.size, args.batch_size):
            rows = perm[s:s + args.batch_size]
            logits = model(*[t.to(dev) for t in tr.gather(rows)])
            target = y_tr[rows].to(dev)
            m = ~torch.isnan(target)
            if m.sum() == 0:
                continue
            loss = F.binary_cross_entropy_with_logits(logits[m], torch.nan_to_num(target)[m])
            opt.zero_grad()
            loss.backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            sc = torch.sigmoid(torch.cat(
                [model(*[t.to(dev) for t in va.gather(slice(s, s + 512))]).cpu()
                 for s in range(0, va.view[0].shape[0], 512)])).numpy()
        auc = np.nanmean(score_per_task(va.y, sc, task_ids,
                                        getattr(args, "metric", "auroc")))
        auc = -np.inf if np.isnan(auc) else auc
        if auc > best + 1e-5:
            best, patience, best_epoch = auc, 0, epoch + 1
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            patience += 1
            if patience >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.best_epoch_ = max(best_epoch, 1)
    model.best_score_ = best
    model.eval()
    return model


@torch.no_grad()
def predict_grid_only(model, sv, dev):
    model.eval()
    return torch.sigmoid(torch.cat(
        [model(*[t.to(dev) for t in sv.gather(slice(s, s + 512))]).cpu()
         for s in range(0, sv.view[0].shape[0], 512)])).numpy()
