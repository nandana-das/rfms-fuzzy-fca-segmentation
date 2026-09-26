"""Minimal self-contained Fuzzy C-Means (Dunn 1973; Bezdek 1981) for the Retail II baseline.

Canonical alternate optimization on standardized data:
    memberships u_ik = 1 / sum_j (d_ik / d_ij)^(2/(m-1))
    centroids v_i = sum_k u_ik^m x_k / sum_k u_ik^m
with fuzzifier m (2.0 here), deterministic k-means++-style seeding, a fixed iteration
budget and an objective tolerance. New points (test customers) are assigned by
computing memberships against frozen train centroids - no refitting, so the
holdout protocol stays leak-free.

Note: empty-cluster handling is not implemented; with k-means++ seeds on continuous
standardized features an exactly-empty cluster is practically impossible, and a
degenerate centroid would simply stay at its seed.
"""

from __future__ import annotations

import numpy as np


class FuzzyCMeans:
    """Canonical fuzzy c-means with deterministic initialization."""

    def __init__(
        self,
        n_clusters: int,
        m: float = 2.0,
        max_iter: int = 300,
        tol: float = 1e-7,
        random_state: int = 42,
    ) -> None:
        if m <= 1.0:
            raise ValueError("fuzzifier m must be > 1")
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.centroids_: np.ndarray | None = None
        self.memberships_: np.ndarray | None = None
        self.objective_: list[float] = []
        self.n_iter_: int = 0

    # -- initialization ---------------------------------------------------
    def _kmeanspp_seeds(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        n = len(X)
        seeds = [int(rng.integers(n))]
        d2 = ((X - X[seeds[0]]) ** 2).sum(axis=1)
        for _ in range(1, self.n_clusters):
            total = float(d2.sum())
            if total <= 0:
                seeds.append(int(rng.integers(n)))
            else:
                seeds.append(int(rng.choice(n, p=d2 / total)))
            d2 = np.minimum(d2, ((X - X[seeds[-1]]) ** 2).sum(axis=1))
        return X[seeds].copy()

    # -- alternate updates --------------------------------------------------
    def _update_memberships(self, X: np.ndarray, centroids: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(X[:, None, :] - centroids[None, :, :], axis=2)  # (n, k)
        eps = 1e-12
        power = 2.0 / (self.m - 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            # ratio[i, k, j] = d_ik / d_ij -> u_ik = 1 / sum_j ratio[i, k, j]^power.
            # A point exactly on a centroid produces inf ratios / NaN there; the
            # hard-assignment branch below fixes those rows, so warnings are suppressed.
            ratio = dist[:, :, None] / np.maximum(dist[:, None, :], eps)
            u = 1.0 / (ratio**power).sum(axis=2)
        # Points sitting exactly on a centroid get a hard assignment there.
        close = dist.min(axis=1) < eps
        if close.any():
            rows = np.where(close)[0]
            u[rows] = 0.0
            u[rows, dist.argmin(axis=1)[rows]] = 1.0
        return u

    def _update_centroids(self, X: np.ndarray, u: np.ndarray) -> np.ndarray:
        um = u**self.m
        return (um.T @ X) / um.sum(axis=0)[:, None]

    # -- public API ---------------------------------------------------------
    def fit(self, X: np.ndarray) -> "FuzzyCMeans":
        X = np.asarray(X, dtype=float)
        rng = np.random.default_rng(self.random_state)
        centroids = self._kmeanspp_seeds(X, rng)
        u = self._update_memberships(X, centroids)
        prev_obj = np.inf
        for it in range(self.max_iter):
            centroids = self._update_centroids(X, u)
            u = self._update_memberships(X, centroids)
            dist2 = ((X[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2)
            obj = float(((u**self.m) * dist2).sum())
            self.objective_.append(obj)
            self.n_iter_ = it + 1
            if abs(prev_obj - obj) < self.tol:
                break
            prev_obj = obj
        self.centroids_ = centroids
        self.memberships_ = u
        return self

    def assign(self, X: np.ndarray) -> np.ndarray:
        """Memberships of new points against the frozen fitted centroids."""
        if self.centroids_ is None:
            raise RuntimeError("fit must be called before assign")
        return self._update_memberships(np.asarray(X, dtype=float), self.centroids_)
