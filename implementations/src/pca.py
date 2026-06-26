"""Principal Component Analysis (stretch) — from scratch.

Companion notes: ../../ml-theory/06-kmeans-pca.md

PCA finds an orthogonal basis (the *principal components*) ordered by how much
of the data's variance each direction captures, then projects onto the top k.
Two equivalent routes:
  - eigendecomposition of the covariance matrix Xc^T Xc / (n-1), or
  - SVD of the centered data Xc = U S V^T  (more numerically stable).
This implementation uses SVD. The components are the rows of V^T; the explained
variance of component i is proportional to the singular value squared, s_i^2.

Key discipline: PCA is variance-based, so you MUST center (and usually scale)
the features first — otherwise large-scale features dominate the components.

Checklist before calling this "done without notes":
  - center data, SVD (or covariance + eigendecomposition)
  - sort components by explained variance (SVD already does)
  - project onto top-k; expose explained_variance_ratio_; inverse_transform
"""

from __future__ import annotations

import numpy as np


class PCA:
    """Principal Component Analysis via SVD.

    Parameters
    ----------
    n_components : int
        Number of principal components to keep.

    Learned attributes
    ------------------
    mean_                      : (n_features,) feature means used for centering
    components_                : (n_components, n_features) the top components (rows)
    explained_variance_        : (n_components,) variance captured per component
    explained_variance_ratio_  : (n_components,) fraction of total variance per component
    """

    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.mean_: np.ndarray | None = None
        self.components_: np.ndarray | None = None
        self.explained_variance_: np.ndarray | None = None
        self.explained_variance_ratio_: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "PCA":
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_                       # center: PCA is about variance, not location

        # SVD of the centered data: Xc = U S V^T. Rows of Vt are the components.
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        explained = (S**2) / (n - 1)             # variance along each component
        total = explained.sum()

        k = self.n_components
        self.components_ = Vt[:k]
        self.explained_variance_ = explained[:k]
        self.explained_variance_ratio_ = explained[:k] / total
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Project centered data onto the top-k components -> (n_samples, k)."""
        X = np.asarray(X, dtype=float)
        if self.components_ is None:
            raise ValueError("call fit() before transform()")
        return (X - self.mean_) @ self.components_.T

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return self.fit(X).transform(X)

    def inverse_transform(self, Z: np.ndarray) -> np.ndarray:
        """Map projected points back to the original feature space (lossy if k < d)."""
        Z = np.asarray(Z, dtype=float)
        return Z @ self.components_ + self.mean_

    # alias so the file matches the repo's fit/predict convention loosely
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)


if __name__ == "__main__":
    # Data that lives mostly along a tilted line -> PC1 should capture most variance.
    rng = np.random.default_rng(0)
    t = rng.normal(size=(300, 1))
    X = t @ np.array([[2.0, 1.0]]) + 0.05 * rng.normal(size=(300, 2))

    pca = PCA(n_components=2).fit(X)
    print("explained variance ratio:", np.round(pca.explained_variance_ratio_, 4))
    print("PC1 direction:", np.round(pca.components_[0], 3))
