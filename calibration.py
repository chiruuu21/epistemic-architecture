"""
Calibration metrics and held-out recalibration, numpy only.

ECE alone is not evidence of calibration. A predictor that always outputs
the base accuracy has ECE ~0 and tells you nothing (AUROC 0.5). With GPT-2
small on TriviaQA, accuracy is a few percent, so "always say 3%" would
pass the proposal's ECE < 0.05 target. Every report here therefore pairs
ECE with accuracy, AUROC (does confidence rank right above wrong?), Brier
score, the constant-confidence baseline, and per-bin counts.

    ece(conf, correct)                 equal-width 10-bin ECE (the usual one)
    ece(conf, correct, adaptive=True)  equal-mass bins (robust when bins are thin)
    reliability(conf, correct)         per-bin count / mean confidence / accuracy
    PlattScaler().fit(X, y)            logistic map from raw features to probability,
                                       fit on a calibration split only
"""

from __future__ import annotations

import numpy as np


def _arr(conf, correct):
    c = np.asarray(conf, dtype=np.float64).ravel()
    y = np.asarray(correct, dtype=np.float64).ravel()
    if c.shape != y.shape:
        raise ValueError(f"conf {c.shape} and correct {y.shape} differ")
    if c.size == 0:
        raise ValueError("empty input")
    if np.any((c < 0) | (c > 1)) or not np.all(np.isfinite(c)):
        raise ValueError("confidences must be finite and in [0, 1]")
    if not np.all((y == 0) | (y == 1)):
        raise ValueError("correct must be 0/1")
    return c, y


def _bins(c, n_bins, adaptive):
    if adaptive:
        order = np.argsort(c, kind="stable")
        bin_of = np.empty(c.size, dtype=int)
        bin_of[order] = np.minimum(np.arange(c.size) * n_bins // c.size, n_bins - 1)
        return bin_of
    # equal width; confidence 1.0 falls in the top bin
    return np.minimum((c * n_bins).astype(int), n_bins - 1)


def reliability(conf, correct, n_bins: int = 10, adaptive: bool = False):
    """List of (lo, hi, count, mean_conf, accuracy) for non-empty bins."""
    c, y = _arr(conf, correct)
    b = _bins(c, n_bins, adaptive)
    rows = []
    for k in range(n_bins):
        m = b == k
        if not m.any():
            continue
        lo, hi = (c[m].min(), c[m].max()) if adaptive else (k / n_bins, (k + 1) / n_bins)
        rows.append((float(lo), float(hi), int(m.sum()), float(c[m].mean()), float(y[m].mean())))
    return rows


def ece(conf, correct, n_bins: int = 10, adaptive: bool = False) -> float:
    """sum_k (n_k / n) * |acc_k - conf_k|."""
    c, y = _arr(conf, correct)
    return float(sum(n * abs(acc - mc) for _, _, n, mc, acc in
                     reliability(c, y, n_bins, adaptive)) / c.size)


def brier(conf, correct) -> float:
    c, y = _arr(conf, correct)
    return float(np.mean((c - y) ** 2))


def auroc(pos_scores, neg_scores) -> float:
    """P(score of a random positive > random negative), ties = 1/2.
    Rank-based, O(n log n). nan if either side is empty."""
    p = np.asarray(pos_scores, dtype=np.float64).ravel()
    n = np.asarray(neg_scores, dtype=np.float64).ravel()
    if p.size == 0 or n.size == 0:
        return float("nan")
    allv = np.concatenate([p, n])
    order = np.argsort(allv, kind="mergesort")
    ranks = np.empty(allv.size)
    sorted_v = allv[order]
    # average ranks over ties
    i = 0
    while i < allv.size:
        j = i
        while j + 1 < allv.size and sorted_v[j + 1] == sorted_v[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    r_pos = ranks[: p.size].sum()
    return float((r_pos - p.size * (p.size + 1) / 2.0) / (p.size * n.size))


def auroc_correct(conf, correct) -> float:
    c, y = _arr(conf, correct)
    return auroc(c[y == 1], c[y == 0])


def summary(conf, correct, n_bins: int = 10) -> dict:
    c, y = _arr(conf, correct)
    return {
        "n": int(c.size),
        "accuracy": float(y.mean()),
        "mean_conf": float(c.mean()),
        "ece": ece(c, y, n_bins),
        "ece_adaptive": ece(c, y, n_bins, adaptive=True),
        "brier": brier(c, y),
        "auroc": auroc_correct(c, y),
    }


def bootstrap(stat_fn, n: int, B: int = 2000, seed: int = 0):
    """95% percentile interval of stat_fn(idx) over B resamples of range(n).
    stat_fn takes an index array; nan results (e.g. AUROC on a resample
    with one class) are dropped."""
    rng = np.random.default_rng(seed)
    vals = np.array([stat_fn(rng.integers(0, n, n)) for _ in range(B)], dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(lo), float(hi)


def logit(p, eps: float = 1e-6):
    p = np.clip(np.asarray(p, dtype=np.float64), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(x):
    x = np.asarray(x, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-x))


class PlattScaler:
    """p = sigmoid(X @ w + b), fit by L2-regularised Newton's method.

    X: (n, d) raw features (e.g. logit of the LM's answer probability, or
    the belief match cosine). With one feature this is Platt scaling; with
    d = 1 and no bias it reduces to temperature scaling of the logit.
    The small ridge (l2) keeps the fit finite when a split is perfectly
    separable, which happens with small calibration sets."""

    def __init__(self, l2: float = 1e-3, max_iter: int = 100, smooth: bool = True):
        self.l2 = l2
        self.smooth = smooth
        self.max_iter = max_iter
        self.coef_ = None

    @staticmethod
    def _design(X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        return np.hstack([X, np.ones((X.shape[0], 1))])

    def fit(self, X, y):
        A = self._design(X)
        y = np.asarray(y, dtype=np.float64).ravel()
        if A.shape[0] != y.size:
            raise ValueError("X and y lengths differ")
        if self.smooth:
            # Platt (1999) target smoothing: 1 -> (N+ + 1)/(N+ + 2),
            # 0 -> 1/(N- + 2). Without it a calibration split where every
            # store-path answer happened to be right fits p = 1.0 exactly
            # (intercept -> infinity), and one wrong test answer is then
            # infinitely surprising.
            n_pos, n_neg = y.sum(), y.size - y.sum()
            y = np.where(y == 1, (n_pos + 1) / (n_pos + 2), 1 / (n_neg + 2))
        w = np.zeros(A.shape[1])
        reg = self.l2 * np.eye(A.shape[1])
        reg[-1, -1] = 0.0  # do not shrink the intercept
        for _ in range(self.max_iter):
            p = sigmoid(A @ w)
            grad = A.T @ (p - y) + reg @ w
            H = (A * (p * (1 - p))[:, None]).T @ A + reg + 1e-9 * np.eye(A.shape[1])
            step = np.linalg.solve(H, grad)
            w -= step
            if np.max(np.abs(step)) < 1e-8:
                break
        self.coef_ = w
        return self

    def predict(self, X):
        if self.coef_ is None:
            raise RuntimeError("fit() first")
        return sigmoid(self._design(X) @ self.coef_)
