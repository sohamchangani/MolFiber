"""Stage 1: the frozen Morgan random-forest anchor b_M."""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from deps import tqdm


# =====================================================================
# 4. Stage 1 -- Morgan-RF with cross-fitted train scores
# =====================================================================

def _make_rf(seed, args):
    return RandomForestClassifier(
        # n_jobs is exposed because -1 is a common source of native crashes:
        # joblib's thread pool on top of an OpenMP runtime that torch has also
        # loaded can abort the process. --n_jobs 1 costs speed, buys stability.
        n_estimators=args.n_estimators, n_jobs=getattr(args, "n_jobs", -1),
        random_state=seed,
        min_samples_leaf=args.min_samples_leaf, class_weight=None)


def _proba1(clf, X):
    """P(y=1); tolerates an empty fold and a model that saw only one class."""
    if X.shape[0] == 0:
        return np.zeros(0, dtype=float)
    classes = list(clf.classes_)
    if 1 not in classes:
        return np.zeros(X.shape[0], dtype=float)
    return clf.predict_proba(X)[:, classes.index(1)]


def morgan_rf_scores(fp_tr, y_tr, fp_va, fp_te, args, seed=0):
    """
    Returns (b_tr, b_va, b_te), each (N_split, T).

    b_tr is OUT-OF-FOLD: every train molecule is scored by a model that did not
    see it. Without this the train pool C_M would be defined by near-perfect
    in-sample scores and would not resemble the test pool at all.
    """
    T = y_tr.shape[1]
    b_tr = np.full((fp_tr.shape[0], T), 0.5)
    b_va = np.full((fp_va.shape[0], T), 0.5)
    b_te = np.full((fp_te.shape[0], T), 0.5)

    rng = np.random.RandomState(seed)
    folds = rng.randint(0, args.cv_folds, size=fp_tr.shape[0])

    for t in tqdm(range(T), desc="    stage-1 RF", leave=False):
        col = y_tr[:, t]
        lab = ~np.isnan(col)
        if np.unique(col[lab]).size < 2:
            continue

        # --- cross-fitted (out-of-fold) train scores ---
        for f in range(args.cv_folds):
            fit_m = lab & (folds != f)
            if np.unique(col[fit_m]).size < 2:
                continue
            clf = _make_rf(seed + f, args).fit(fp_tr[fit_m], col[fit_m].astype(int))
            b_tr[folds == f, t] = _proba1(clf, fp_tr[folds == f])

        # --- inference anchor ---
        if getattr(args, "anchor_inference", "refit") == "foldmean":
            # Average the CV fold models. Keeps train and inference anchors on the
            # same footing: b_M at train time comes from models fit on (K-1)/K of
            # the data, so a full refit at inference is systematically sharper,
            # and the residual was never trained against that sharper anchor.
            acc_va, acc_te, nfit = 0.0, 0.0, 0
            for f in range(args.cv_folds):
                fit_m = lab & (folds != f)
                if np.unique(col[fit_m]).size < 2:
                    continue
                clf = _make_rf(seed + f, args).fit(fp_tr[fit_m], col[fit_m].astype(int))
                acc_va = acc_va + _proba1(clf, fp_va)
                acc_te = acc_te + _proba1(clf, fp_te)
                nfit += 1
            if nfit:
                b_va[:, t] = acc_va / nfit
                b_te[:, t] = acc_te / nfit
        else:
            clf = _make_rf(seed, args).fit(fp_tr[lab], col[lab].astype(int))
            b_va[:, t] = _proba1(clf, fp_va)
            b_te[:, t] = _proba1(clf, fp_te)

    return b_tr, b_va, b_te
