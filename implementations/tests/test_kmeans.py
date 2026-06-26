"""Tests for KMeans (Lloyd's algorithm + k-means++, from scratch).

Run from the repo root:
    pytest implementations/tests/ -q
or directly:
    python implementations/tests/test_kmeans.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from kmeans import KMeans  # noqa: E402


def _three_blobs(n=80, seed=0):
    rng = np.random.default_rng(seed)
    centers = np.array([[0, 0], [6, 6], [0, 6]])
    X = np.vstack([rng.normal(c, 0.4, size=(n, 2)) for c in centers])
    return X, centers


def _purity(labels, true_blocks):
    """Fraction of points whose cluster's majority-true-block matches their own."""
    correct = 0
    for c in np.unique(labels):
        members = true_blocks[labels == c]
        if len(members):
            maj = np.bincount(members).argmax()
            correct += np.sum(members == maj)
    return correct / len(labels)


def test_recovers_well_separated_blobs():
    X, centers = _three_blobs()
    true_blocks = np.repeat(np.arange(3), 80)
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    assert _purity(km.labels_, true_blocks) > 0.99


def test_centroids_near_true_centers():
    X, centers = _three_blobs()
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    found = np.array(sorted(km.cluster_centers_.tolist()))
    expected = np.array(sorted(centers.tolist()))
    # each recovered centroid should sit close to a true center
    assert np.allclose(found, expected, atol=0.5)


def test_inertia_decreases_with_more_clusters():
    X, _ = _three_blobs()
    i2 = KMeans(n_clusters=2, random_state=0).fit(X).inertia_
    i3 = KMeans(n_clusters=3, random_state=0).fit(X).inertia_
    i5 = KMeans(n_clusters=5, random_state=0).fit(X).inertia_
    # more clusters -> lower (or equal) within-cluster SSE, always
    assert i2 > i3 > i5


def test_predict_matches_labels_on_training_data():
    X, _ = _three_blobs()
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    assert np.array_equal(km.predict(X), km.labels_)


def test_reproducible_with_seed():
    X, _ = _three_blobs()
    a = KMeans(n_clusters=3, random_state=42).fit(X)
    b = KMeans(n_clusters=3, random_state=42).fit(X)
    assert np.allclose(a.cluster_centers_, b.cluster_centers_)
    assert a.inertia_ == b.inertia_


def test_k_equals_n_gives_zero_inertia():
    # One centroid per point -> every point sits on its centroid -> inertia 0.
    X = np.array([[0.0, 0.0], [1.0, 1.0], [5.0, 5.0]])
    km = KMeans(n_clusters=3, random_state=0).fit(X)
    assert km.inertia_ < 1e-9


def test_too_many_clusters_raises():
    X = np.zeros((2, 2))
    try:
        KMeans(n_clusters=5).fit(X)
    except ValueError:
        return
    raise AssertionError("expected ValueError when n_clusters > n_samples")


if __name__ == "__main__":
    test_recovers_well_separated_blobs()
    test_centroids_near_true_centers()
    test_inertia_decreases_with_more_clusters()
    test_predict_matches_labels_on_training_data()
    test_reproducible_with_seed()
    test_k_equals_n_gives_zero_inertia()
    test_too_many_clusters_raises()
    print("All kmeans tests passed.")
