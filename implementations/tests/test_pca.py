"""Tests for PCA (SVD-based, from scratch).

Run from the repo root:
    pytest implementations/tests/ -q
or directly:
    python implementations/tests/test_pca.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pca import PCA  # noqa: E402


def _tilted_line(n=300, seed=0):
    # Points along a tilted line + tiny noise -> almost all variance on PC1.
    rng = np.random.default_rng(seed)
    t = rng.normal(size=(n, 1))
    return t @ np.array([[2.0, 1.0]]) + 0.02 * rng.normal(size=(n, 2))


def test_first_component_captures_most_variance():
    X = _tilted_line()
    pca = PCA(n_components=2).fit(X)
    assert pca.explained_variance_ratio_[0] > 0.98


def test_explained_variance_ratio_sums_to_one_with_all_components():
    X = _tilted_line()
    pca = PCA(n_components=2).fit(X)
    assert np.isclose(pca.explained_variance_ratio_.sum(), 1.0)


def test_components_are_orthonormal():
    X = _tilted_line()
    pca = PCA(n_components=2).fit(X)
    gram = pca.components_ @ pca.components_.T
    assert np.allclose(gram, np.eye(2), atol=1e-8)


def test_transform_shape_and_reconstruction():
    X = _tilted_line()
    pca = PCA(n_components=2).fit(X)
    Z = pca.transform(X)
    assert Z.shape == (X.shape[0], 2)
    # full-rank projection (k = d) reconstructs exactly
    assert np.allclose(pca.inverse_transform(Z), X, atol=1e-8)


def test_one_component_reconstruction_is_close():
    X = _tilted_line()
    pca = PCA(n_components=1).fit(X)
    recon = pca.inverse_transform(pca.transform(X))
    # since PC1 holds >98% of variance, 1-D reconstruction loses very little
    rel_err = np.linalg.norm(recon - X) / np.linalg.norm(X - X.mean(axis=0))
    assert rel_err < 0.2


def test_matches_sklearn_variance_ratio_if_available():
    try:
        from sklearn.decomposition import PCA as SkPCA
    except ImportError:
        return  # sklearn optional
    X = _tilted_line()
    mine = PCA(n_components=2).fit(X).explained_variance_ratio_
    theirs = SkPCA(n_components=2).fit(X).explained_variance_ratio_
    assert np.allclose(mine, theirs, atol=1e-6)


if __name__ == "__main__":
    test_first_component_captures_most_variance()
    test_explained_variance_ratio_sums_to_one_with_all_components()
    test_components_are_orthonormal()
    test_transform_shape_and_reconstruction()
    test_one_component_reconstruction_is_close()
    test_matches_sklearn_variance_ratio_if_available()
    print("All pca tests passed.")
