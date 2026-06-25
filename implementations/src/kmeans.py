"""K-Means clustering — from scratch.

Companion notes: ../../ml-theory/06-kmeans-pca.md

STATUS: 🟡 scaffold. To be implemented in Week 6.
Target: clean OOP, fit/predict, no sklearn for the core algorithm.

Checklist before calling this "done without notes":
  - k-means++ init (or random restarts via n_init)
  - assign step (nearest centroid) + update step (mean)
  - convergence check + track inertia (within-cluster SSE)
"""

from __future__ import annotations

import numpy as np


class KMeans:
    """K-Means clustering.

    TODO (Week 6): implement.
    """

    def __init__(self, n_clusters: int = 3, max_iter: int = 300, n_init: int = 10, random_state: int | None = None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "KMeans":
        raise NotImplementedError("Implement in Week 6 — see notes.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implement in Week 6 — see notes.")
