"""Tests for KNNClassifier.

Run from the repo root:
    pytest implementations/tests/ -q
or directly:
    python implementations/tests/test_knn.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from knn import KNNClassifier  # noqa: E402


def _two_blobs(n=100, seed=0):
    rng = np.random.default_rng(seed)
    Xa = rng.normal(loc=[0, 0], scale=0.5, size=(n, 2))
    Xb = rng.normal(loc=[4, 4], scale=0.5, size=(n, 2))
    X = np.vstack([Xa, Xb])
    y = np.array([0] * n + [1] * n)
    return X, y


def test_separable_blobs_accurate():
    X, y = _two_blobs()
    model = KNNClassifier(n_neighbors=5).fit(X, y)
    assert model.score(X, y) > 0.98


def test_k1_memorizes_training_set():
    # With k=1 and no duplicate points, each point is its own nearest neighbor.
    X, y = _two_blobs()
    model = KNNClassifier(n_neighbors=1).fit(X, y)
    assert model.score(X, y) == 1.0


def test_euclidean_and_manhattan_agree_on_easy_data():
    X, y = _two_blobs()
    e = KNNClassifier(n_neighbors=5, metric="euclidean").fit(X, y)
    m = KNNClassifier(n_neighbors=5, metric="manhattan").fit(X, y)
    assert np.array_equal(e.predict(X), m.predict(X))


def test_predict_proba_sums_to_one():
    X, y = _two_blobs()
    model = KNNClassifier(n_neighbors=5).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_standardization_matters_on_skewed_scales():
    # Make feature 1 dominate by scaling it up 1000x. The informative feature is feature 0.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 2))
    y = (X[:, 0] > 0).astype(int)          # label depends ONLY on feature 0
    X[:, 1] *= 1000.0                       # noise feature on a huge scale
    with_std = KNNClassifier(n_neighbors=5, standardize=True).fit(X, y).score(X, y)
    without_std = KNNClassifier(n_neighbors=5, standardize=False).fit(X, y).score(X, y)
    assert with_std > without_std


def test_tie_breaking_is_deterministic():
    # k=2 with one neighbor of each class -> tie -> lowest label (0) should win.
    X = np.array([[0.0], [10.0]])
    y = np.array([0, 1])
    model = KNNClassifier(n_neighbors=2, standardize=False).fit(X, y)
    # A query at 5.0 is equidistant; both neighbors used -> tie -> class 0.
    assert model.predict(np.array([[5.0]]))[0] == 0


def test_invalid_metric_raises():
    try:
        KNNClassifier(metric="cosine")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unsupported metric")


if __name__ == "__main__":
    test_separable_blobs_accurate()
    test_k1_memorizes_training_set()
    test_euclidean_and_manhattan_agree_on_easy_data()
    test_predict_proba_sums_to_one()
    test_standardization_matters_on_skewed_scales()
    test_tie_breaking_is_deterministic()
    test_invalid_metric_raises()
    print("All knn tests passed.")
