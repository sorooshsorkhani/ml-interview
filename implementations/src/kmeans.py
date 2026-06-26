"""K-Means clustering — from scratch.

Companion notes: ../../ml-theory/06-kmeans-pca.md

K-Means is an *unsupervised*, *iterative* algorithm (Lloyd's algorithm) that
partitions n points into k clusters by alternating two steps until convergence:
  - ASSIGN:  each point -> its nearest centroid (by squared Euclidean distance)
  - UPDATE:  each centroid -> the mean of the points assigned to it
This is coordinate descent on the *inertia* objective (within-cluster SSE), so
inertia decreases monotonically and the algorithm converges to a LOCAL optimum.

Because the result depends on initialization, we:
  - seed centroids with **k-means++** (spread-out, probabilistic init), and
  - run `n_init` independent restarts and keep the lowest-inertia one.

Design goals (AMLS "code without notes" target):
  - clean OOP, fit / predict / fit_predict
  - k-means++ init + n_init restarts
  - vectorized assign step, inertia tracking, convergence check
"""

from __future__ import annotations

import numpy as np


class KMeans:
    """K-Means clustering (Lloyd's algorithm with k-means++ init).

    Parameters
    ----------
    n_clusters : int
        Number of clusters (k).
    max_iter : int
        Max assign/update iterations per run.
    n_init : int
        Number of random restarts; the lowest-inertia run is kept.
    tol : float
        Convergence threshold on centroid shift (Frobenius norm).
    random_state : int | None
        Seed for reproducible initialization.

    Learned attributes (sklearn-style trailing underscore)
    ------------------------------------------------------
    cluster_centers_ : (k, n_features) final centroids
    labels_          : (n,) cluster index per training point
    inertia_         : float, within-cluster SSE of the best run
    n_iter_          : iterations the best run took to converge
    """

    def __init__(
        self,
        n_clusters: int = 3,
        max_iter: int = 300,
        n_init: int = 10,
        tol: float = 1e-4,
        random_state: int | None = None,
    ):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state

        self.cluster_centers_: np.ndarray | None = None
        self.labels_: np.ndarray | None = None
        self.inertia_: float | None = None
        self.n_iter_: int | None = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _sq_dists(X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """Squared Euclidean distances, (n_samples, k).

        ||x - c||^2 = ||x||^2 - 2 x·c + ||c||^2  (vectorized, no python loop).
        """
        x_sq = np.sum(X**2, axis=1)[:, None]           # (n, 1)
        c_sq = np.sum(centers**2, axis=1)[None, :]     # (1, k)
        cross = X @ centers.T                          # (n, k)
        return np.maximum(x_sq - 2 * cross + c_sq, 0.0)  # clip tiny negatives

    def _kmeanspp_init(self, X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """k-means++: pick centroids spread out, with probability ∝ D(x)^2.

        The first centroid is uniform-random; each next one is drawn with
        probability proportional to its squared distance to the nearest
        already-chosen centroid. This dramatically reduces bad local optima.
        """
        n = X.shape[0]
        centers = np.empty((self.n_clusters, X.shape[1]), dtype=float)
        centers[0] = X[rng.integers(n)]
        # closest squared distance to any chosen centroid, so far
        closest_sq = self._sq_dists(X, centers[:1]).ravel()
        for j in range(1, self.n_clusters):
            total = closest_sq.sum()
            if total == 0:  # all points coincide with a centroid -> pick uniformly
                centers[j] = X[rng.integers(n)]
            else:
                probs = closest_sq / total
                centers[j] = X[rng.choice(n, p=probs)]
            new_sq = self._sq_dists(X, centers[j : j + 1]).ravel()
            closest_sq = np.minimum(closest_sq, new_sq)
        return centers

    def _single_run(self, X: np.ndarray, rng: np.random.Generator):
        """One full Lloyd's run from a k-means++ init. Returns (centers, labels, inertia, n_iter)."""
        centers = self._kmeanspp_init(X, rng)
        labels = np.zeros(X.shape[0], dtype=int)
        n_iter = 0
        for n_iter in range(1, self.max_iter + 1):
            # ASSIGN: nearest centroid
            sq = self._sq_dists(X, centers)
            labels = np.argmin(sq, axis=1)

            # UPDATE: centroid = mean of its members
            new_centers = centers.copy()
            for c in range(self.n_clusters):
                members = X[labels == c]
                if len(members) > 0:
                    new_centers[c] = members.mean(axis=0)
                else:
                    # Empty cluster: re-seed it on the point farthest from its centroid
                    # (a standard fix that keeps k clusters alive).
                    farthest = np.argmax(np.min(sq, axis=1))
                    new_centers[c] = X[farthest]

            shift = np.linalg.norm(new_centers - centers)
            centers = new_centers
            if shift <= self.tol:  # converged
                break

        # final inertia under the converged centers
        sq = self._sq_dists(X, centers)
        labels = np.argmin(sq, axis=1)
        inertia = float(np.min(sq, axis=1).sum())
        return centers, labels, inertia, n_iter

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "KMeans":
        X = np.asarray(X, dtype=float)
        if self.n_clusters > X.shape[0]:
            raise ValueError("n_clusters cannot exceed number of samples")
        rng = np.random.default_rng(self.random_state)

        best = None
        for _ in range(self.n_init):
            centers, labels, inertia, n_iter = self._single_run(X, rng)
            if best is None or inertia < best[2]:
                best = (centers, labels, inertia, n_iter)

        self.cluster_centers_, self.labels_, self.inertia_, self.n_iter_ = best
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.cluster_centers_ is None:
            raise ValueError("call fit() before predict()")
        return np.argmin(self._sq_dists(X, self.cluster_centers_), axis=1)

    def fit_predict(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return self.fit(X).labels_


if __name__ == "__main__":
    # Three well-separated blobs -> K-Means with k=3 should recover them.
    rng = np.random.default_rng(0)
    centers_true = np.array([[0, 0], [6, 6], [0, 6]])
    X = np.vstack([rng.normal(c, 0.5, size=(80, 2)) for c in centers_true])

    km = KMeans(n_clusters=3, random_state=0).fit(X)
    print(f"inertia={km.inertia_:.2f}  converged in {km.n_iter_} iters")
    print("centroids (sorted):")
    print(np.array(sorted(km.cluster_centers_.tolist())))
