"""
Inner cross-validation over the training pool.

MLCIL/benchmarking_molecular_models gives its models no validation fold: the
benchmark hands over one training pool plus a held-out test set, and selection
happens via `GridSearchCV(cv=5, scoring='roc_auc', refit=True)` inside the pool,
after which the winner is refit on the whole pool and scored once on test.

Reproducing that for a neural model needs one extra decision that sklearn does
not: when to stop training. `refit=True` has no early-stopping notion, so this
module follows the standard analogue --

    1. run K folds; within each, hold out one fold for early stopping and record
       both the fold score and the epoch at which it peaked;
    2. report the mean fold score (the selection signal);
    3. refit on the FULL pool for the median best epoch, with no early stopping.

That keeps every label in the final fit, as `refit=True` does, while taking the
epoch budget from held-out data rather than from the training loss.

`--inner_cv 1` degrades this to a single held-out fold, which is much cheaper and
usually adequate for ranking hyperparameters, at the cost of a noisier estimate.
"""

from __future__ import annotations

import numpy as np

from train import (predict_grid_only, predict_scores, score_per_task,
                   train_grid_only, train_molfiber)
from views import SplitViews


def _subset(sv, rows):
    """Row-subset every tensor in a SplitViews, keeping them aligned."""
    return SplitViews(tuple(t[rows] for t in sv.view),
                      None if sv.mf is None else sv.mf[rows],
                      sv.fp[rows], sv.blogit[rows], sv.y[rows])


def make_folds(n, k, seed):
    """Deterministic k-fold row partition of the training pool."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(n)
    return [np.sort(f) for f in np.array_split(perm, max(k, 1))]


def cv_score(method, variant, pool, task_ids, args, seed, dev, view_spec,
             make_fibers_fn, k=5, metric="auroc", verbose=False):
    """
    Returns (mean_fold_score, median_best_epoch).

    Each fold trains on the other K-1 folds and early-stops on the held-out one,
    so no fold's labels influence its own score.
    """
    n = len(pool.blogit)
    folds = make_folds(n, k, seed)
    scores, epochs = [], []

    for i, held in enumerate(folds):
        keep = np.setdiff1d(np.arange(n), held)
        tr, va = _subset(pool, keep), _subset(pool, held)
        if np.unique(va.y[~np.isnan(va.y)]).size < 2:
            continue                          # unscorable fold, skip it
        if method == "grid":
            model = train_grid_only(tr, va, task_ids, args, seed, dev, view_spec)
            sc = predict_grid_only(model, va, dev)
        else:
            fib_tr, fib_va = make_fibers_fn(tr, va, args.k)
            model = train_molfiber(tr, va, fib_tr, fib_va, task_ids, args,
                                   variant, seed, dev, view_spec)
            sc = predict_scores(model, va, tr, fib_va, task_ids, dev, args.batch_size)
        s = float(np.nanmean(score_per_task(va.y, sc, task_ids, metric)))
        if not np.isnan(s):
            scores.append(s)
            epochs.append(getattr(model, "best_epoch_", args.epochs))
        if verbose:
            print(f"      fold {i + 1}/{len(folds)}  {metric} {s:.4f}")

    if not scores:
        return -np.inf, args.epochs
    return float(np.mean(scores)), int(np.median(epochs))
