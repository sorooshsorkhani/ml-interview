"""Principal Component Analysis (stretch) — from scratch.

Companion notes: ../../ml-theory/06-kmeans-pca.md

STATUS: 🟡 scaffold. To be implemented in Week stretch (W6).
Target: clean OOP, fit/predict, no sklearn for the core algorithm.

Checklist before calling this "done without notes":
  - center data, covariance matrix (or SVD)
  - eigendecomposition, sort by eigenvalue
  - project onto top-k components, explained variance
"""

from __future__ import annotations

import numpy as np


class PCA:
    """Principal Component Analysis (stretch).

    TODO (Week stretch (W6)): implement.
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ratio_ = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "PCA":
        raise NotImplementedError("Implement in Week stretch (W6) — see notes.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implement in Week stretch (W6) — see notes.")
