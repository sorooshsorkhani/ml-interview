"""Tests for DecisionTreeClassifier (CART, from scratch).

Run from the repo root:
    pytest implementations/tests/ -q
or directly:
    python implementations/tests/test_decision_tree.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from decision_tree import DecisionTreeClassifier  # noqa: E402


def _xor(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)
    return X, y


def _two_blobs(n=100, seed=0):
    rng = np.random.default_rng(seed)
    Xa = rng.normal(loc=[0, 0], scale=0.5, size=(n, 2))
    Xb = rng.normal(loc=[4, 4], scale=0.5, size=(n, 2))
    X = np.vstack([Xa, Xb])
    y = np.array([0] * n + [1] * n)
    return X, y


def test_solves_xor_when_linear_cannot():
    # XOR is not linearly separable; a tree of depth >= 2 should nail it.
    X, y = _xor()
    model = DecisionTreeClassifier(max_depth=4).fit(X, y)
    assert model.score(X, y) > 0.98


def test_unrestricted_tree_memorizes_training_set():
    # With no depth limit and no contradictory rows, a CART tree reaches 0 train error.
    X, y = _two_blobs()
    model = DecisionTreeClassifier().fit(X, y)
    assert model.score(X, y) == 1.0


def test_max_depth_limits_growth():
    X, y = _xor()
    model = DecisionTreeClassifier(max_depth=1).fit(X, y)
    assert model.depth() <= 1
    # A depth-1 stump cannot solve XOR -> well below perfect.
    assert model.score(X, y) < 0.9


def test_gini_and_entropy_both_separate_easy_data():
    X, y = _two_blobs()
    g = DecisionTreeClassifier(criterion="gini").fit(X, y)
    e = DecisionTreeClassifier(criterion="entropy").fit(X, y)
    assert g.score(X, y) == 1.0
    assert e.score(X, y) == 1.0


def test_predict_proba_is_valid_distribution():
    X, y = _two_blobs()
    model = DecisionTreeClassifier(max_depth=3).fit(X, y)
    proba = model.predict_proba(X)
    assert proba.shape == (X.shape[0], 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert (proba >= 0).all()


def test_pure_node_is_a_single_leaf():
    # All one class -> root should be a leaf immediately, depth 0.
    X = np.array([[0.0], [1.0], [2.0]])
    y = np.array([1, 1, 1])
    model = DecisionTreeClassifier().fit(X, y)
    assert model.depth() == 0
    assert np.array_equal(model.predict(np.array([[5.0]])), np.array([1]))


def test_contradictory_rows_stop_gracefully():
    # Identical features, different labels: no split can improve purity ->
    # must terminate as a (majority) leaf rather than recurse forever.
    X = np.array([[0.0], [0.0], [0.0]])
    y = np.array([0, 0, 1])
    model = DecisionTreeClassifier().fit(X, y)
    assert model.depth() == 0
    assert model.predict(np.array([[0.0]]))[0] == 0  # majority class


def test_invalid_criterion_raises():
    try:
        DecisionTreeClassifier(criterion="mse")
    except ValueError:
        return
    raise AssertionError("expected ValueError for unsupported criterion")


if __name__ == "__main__":
    test_solves_xor_when_linear_cannot()
    test_unrestricted_tree_memorizes_training_set()
    test_max_depth_limits_growth()
    test_gini_and_entropy_both_separate_easy_data()
    test_predict_proba_is_valid_distribution()
    test_pure_node_is_a_single_leaf()
    test_contradictory_rows_stop_gracefully()
    test_invalid_criterion_raises()
    print("All decision tree tests passed.")
