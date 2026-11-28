"""Split-conformal absolute residual intervals, independently per field."""
import math
import numpy as np


def fit_quantiles(points, targets, coverage=0.9):
    p, y = np.asarray(points, dtype=float), np.asarray(targets, dtype=float)
    if p.shape != y.shape or p.ndim != 2 or not len(p):
        raise ValueError('Expected nonempty aligned [n, fields] arrays')
    if not 0 < coverage < 1 or not np.isfinite(p).all() or not np.isfinite(y).all():
        raise ValueError('Invalid coverage or nonfinite input')
    if (p < 0).any() or (y < 0).any():
        raise ValueError('Nutrition points and targets must be nonnegative')
    rank = math.ceil((len(p) + 1) * coverage)
    q = np.full(p.shape[1], np.inf) if rank > len(p) else np.partition(np.abs(y-p), rank-1, axis=0)[rank-1]
    return q, rank


def bounds(points, q):
    p, q = np.asarray(points, dtype=float), np.asarray(q, dtype=float)
    if not np.isfinite(p).all() or (p < 0).any() or np.isnan(q).any() or (q < 0).any():
        raise ValueError('Invalid points or quantiles')
    return np.maximum(0, p-q), p+q
