"""K-Nearest Neighbors classifier — from scratch.

Companion notes: ../../ml-theory/02-knn.md

STATUS: 🟡 scaffold. To be implemented in Week 2.
Target: clean OOP, fit/predict, no sklearn for the core algorithm.

Checklist before calling this "done without notes":
  - store training data in fit (lazy learner)
  - distance computation (euclidean/manhattan)
  - argpartition for top-k, majority vote, tie-breaking
"""

from __future__ import annotations

import numpy as np


class KNNClassifier:
    """K-Nearest Neighbors classifier.

    TODO (Week 2): implement.
    """

    def __init__(self, n_neighbors: int = 5, metric: str = "euclidean"):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.X_ = None
        self.y_ = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "KNNClassifier":
        raise NotImplementedError("Implement in Week 2 — see notes.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implement in Week 2 — see notes.")
