"""Train-only VAD-to-DINO regression and reconstruction diagnostics.

Inputs are scene VAD predicted from visual features, never test reaction labels.
The decoder can recover information retained by VAD; it cannot add information.
"""
import numpy as np
from scipy.spatial.distance import cdist


def scene_sample(offsets, maximum=8):
    """Evenly sample each clip, giving every clip equal total fitting weight."""
    offsets = np.asarray(offsets)
    if offsets[0] != 0 or (np.diff(offsets) <= 0).any() or maximum < 1:
        raise ValueError('Expected positive scene counts')
    indices, weights = [], []
    for a, b in zip(offsets[:-1], offsets[1:]):
        chosen = np.linspace(a, b - 1, min(maximum, b - a), dtype=int)
        indices.extend(chosen)
        weights.extend([1 / len(chosen)] * len(chosen))
    weights = np.asarray(weights, float)
    return np.asarray(indices), weights / weights.sum()


class Basis:
    def __init__(self, kind, seed=42, bandwidth=1., landmarks=128):
        if kind not in {'linear', 'quadratic', 'rbf'}:
            raise ValueError(kind)
        self.kind, self.seed = kind, seed
        self.bandwidth, self.landmarks = bandwidth, landmarks

    def fit(self, x, weights):
        x = np.asarray(x, float)
        self.mean = np.average(x, axis=0, weights=weights)
        self.scale = np.sqrt(np.average((x - self.mean) ** 2, axis=0, weights=weights)).clip(1e-6)
        z = (x - self.mean) / self.scale
        rng = np.random.default_rng(self.seed)
        self.centers = z[rng.choice(len(z), min(self.landmarks, len(z)), replace=False)]
        return self

    def transform(self, x):
        z = (np.asarray(x, float) - self.mean) / self.scale
        if self.kind == 'linear':
            return z
        if self.kind == 'quadratic':
            return np.column_stack([z] + [z[:, i] * z[:, j] for i in range(3) for j in range(i, 3)])
        return np.column_stack([z, np.exp(-cdist(z, self.centers, 'sqeuclidean') / (2 * self.bandwidth ** 2))])


class RidgeMap:
    """Minimize weighted mean squared error + alpha * squared coefficient norm."""
    def fit(self, phi, y, weights, alpha):
        phi, y = np.asarray(phi, float), np.asarray(y, float)
        w = np.asarray(weights, float)
        if alpha <= 0 or len(w) != len(y) or len(phi) != len(y) or (w <= 0).any():
            raise ValueError('Invalid fitting arrays or regularization')
        w = w / w.sum()
        self.xmean = np.average(phi, axis=0, weights=w)
        self.ymean = np.average(y, axis=0, weights=w)
        x = phi - self.xmean
        gram = x.T @ (x * w[:, None])
        self.coef = np.linalg.solve(gram + alpha * np.eye(phi.shape[1]), x.T @ ((y - self.ymean) * w[:, None]))
        return self

    def predict(self, phi):
        return (phi - self.xmean) @ self.coef + self.ymean


def cosine_rows(a, b):
    return np.sum(a * b, axis=1) / np.maximum(np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1), 1e-12)


def reconstruction_metrics(target, predicted, mean, offsets):
    """Equal-clip scene metrics; R2 compares with a training-mean predictor."""
    target, predicted = np.asarray(target, float), np.asarray(predicted, float)
    if target.shape != predicted.shape or not np.isfinite(predicted).all():
        raise ValueError('Invalid reconstruction')
    err = np.square(target - predicted).mean(1)
    null = np.square(target - mean).mean(1)
    cos = cosine_rows(target, predicted)
    centered_cos = cosine_rows(target - mean, predicted - mean)
    reduce = lambda v: np.array([v[a:b].mean() for a, b in zip(offsets[:-1], offsets[1:])])
    mse, baseline = reduce(err), reduce(null)
    return {'mse': float(mse.mean()), 'r2_vs_train_mean': float(1 - mse.mean() / baseline.mean()),
            'cosine': float(reduce(cos).mean()), 'centered_cosine': float(reduce(centered_cos).mean())}, mse


def clip_mean(values, offsets):
    return np.stack([np.asarray(values[a:b], float).mean(0) for a, b in zip(offsets[:-1], offsets[1:])])


def retrieval(target, predicted, mean):
    """Retrieve the corresponding test clip among all test clips, centered cosine."""
    a, b = predicted - mean, target - mean
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-12)
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-12)
    ranks = []
    for start in range(0, len(a), 256):
        sims = a[start:start + 256] @ b.T
        ids = np.arange(start, start + len(sims))
        diagonal = sims[np.arange(len(sims)), ids]
        # Average rank for exact ties; a constant decoder must not win by ordering.
        rank = 1 + (sims > diagonal[:, None] + 1e-12).sum(1)
        rank = rank + .5 * ((np.abs(sims - diagonal[:, None]) <= 1e-12).sum(1) - 1)
        ranks.extend(rank)
    ranks = np.asarray(ranks)
    return {'recall_at_1': float((ranks <= 1).mean()), 'recall_at_10': float((ranks <= 10).mean()),
            'mean_reciprocal_rank': float((1 / ranks).mean()), 'median_rank': float(np.median(ranks)),
            'tie_rule': 'average rank', 'candidates': len(a)}
