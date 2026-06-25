"""K-Nearest Neighbors classifier from scratch.

Companion notes: ../../ml-theory/02-knn.md

KNN is a *lazy*, *non-parametric* learner: fit() just stores the training data;
all the work happens at predict() time, where each query point is classified by
majority vote of its k nearest training points.

Design goals (AMLS "code without notes" target):
  - clean OOP, fit / predict / predict_proba
  - euclidean & manhattan distances, vectorized
  - deterministic tie-breaking (lower class label wins)
  - optional feature standardization (KNN is scale-sensitive!)
"""

from __future__ import annotations

import numpy as np


class KNNClassifier:
    """K-Nearest Neighbors classifier.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors (k) to vote.
    metric : {"euclidean", "manhattan"}
        Distance metric.
    standardize : bool
        Standardize features using training stats (strongly recommended:
        distance is dominated by large-scale features otherwise).
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        metric: str = "euclidean",
        standardize: bool = True,
    ):
        if metric not in ("euclidean", "manhattan"):
            raise ValueError("metric must be 'euclidean' or 'manhattan'")
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.standardize = standardize

        self.X_: np.ndarray | None = None
        self.y_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        self._sigma[self._sigma == 0] = 1.0
        return (X - self._mu) / self._sigma

    def _standardize_apply(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mu) / self._sigma

    def _distances(self, X: np.ndarray) -> np.ndarray:
        """Pairwise distances: rows = query points, cols = training points.

        Returns an (n_query, n_train) matrix.
        """
        if self.metric == "euclidean":
            # ||a - b||^2 = ||a||^2 - 2 a·b + ||b||^2  (vectorized, no python loop)
            q_sq = np.sum(X**2, axis=1)[:, None]          # (n_query, 1)
            t_sq = np.sum(self.X_**2, axis=1)[None, :]    # (1, n_train)
            cross = X @ self.X_.T                          # (n_query, n_train)
            sq = np.maximum(q_sq - 2 * cross + t_sq, 0.0)  # clip tiny negatives
            return np.sqrt(sq)
        else:  # manhattan
            # |a - b| summed over features; broadcast (n_query, 1, p) - (1, n_train, p)
            return np.abs(X[:, None, :] - self.X_[None, :, :]).sum(axis=2)

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNNClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        if self.standardize:
            X = self._standardize_fit(X)
        self.X_ = X
        self.y_ = y
        self.classes_ = np.unique(y)
        return self

    def _neighbor_labels(self, X: np.ndarray) -> np.ndarray:
        """Return the (n_query, k) array of nearest-neighbor labels."""
        if self.standardize:
            X = self._standardize_apply(X)
        dist = self._distances(X)
        k = min(self.n_neighbors, self.X_.shape[0])
        # argpartition is O(n) vs O(n log n) for a full sort — we only need top-k.
        nn_idx = np.argpartition(dist, kth=k - 1, axis=1)[:, :k]
        return self.y_[nn_idx]

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        neighbor_labels = self._neighbor_labels(X)
        preds = []
        for row in neighbor_labels:
            # Count votes per class; tie -> lowest class label (np.unique is sorted).
            vals, counts = np.unique(row, return_counts=True)
            preds.append(vals[np.argmax(counts)])
        return np.array(preds)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Fraction of the k neighbors belonging to each class (column order = classes_)."""
        X = np.asarray(X, dtype=float)
        neighbor_labels = self._neighbor_labels(X)
        k = neighbor_labels.shape[1]
        proba = np.zeros((X.shape[0], len(self.classes_)))
        class_to_col = {c: j for j, c in enumerate(self.classes_)}
        for i, row in enumerate(neighbor_labels):
            vals, counts = np.unique(row, return_counts=True)
            for v, c in zip(vals, counts):
                proba[i, class_to_col[v]] = c / k
        return proba

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y).ravel()
        return float(np.mean(self.predict(X) == y))


if __name__ == "__main__":
    # Two well-separated blobs -> KNN should nail it.
    rng = np.random.default_rng(0)
    Xa = rng.normal(loc=[0, 0], scale=0.5, size=(100, 2))
    Xb = rng.normal(loc=[3, 3], scale=0.5, size=(100, 2))
    X = np.vstack([Xa, Xb])
    y = np.array([0] * 100 + [1] * 100)

    for metric in ("euclidean", "manhattan"):
        model = KNNClassifier(n_neighbors=5, metric=metric).fit(X, y)
        print(f"metric={metric:>9}  train acc={model.score(X, y):.3f}")
