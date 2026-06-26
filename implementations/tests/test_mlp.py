"""Tests for MLP (one-hidden-layer, numpy, from scratch).

Run from the repo root:
    pytest implementations/tests/ -q
or directly:
    python implementations/tests/test_mlp.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from mlp import MLP  # noqa: E402


def _xor():
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    return X, y


def _two_blobs(n=100, seed=0):
    rng = np.random.default_rng(seed)
    Xa = rng.normal([-2, -2], 0.5, size=(n, 2))
    Xb = rng.normal([2, 2], 0.5, size=(n, 2))
    X = np.vstack([Xa, Xb])
    y = np.array([0] * n + [1] * n)
    return X, y


def test_learns_xor():
    # XOR is not linearly separable -> only solvable *because* of the hidden layer.
    X, y = _xor()
    clf = MLP(hidden_size=8, learning_rate=0.5, n_epochs=3000, random_state=0).fit(X, y)
    assert np.array_equal(clf.predict(X), y)


def test_separates_blobs():
    X, y = _two_blobs()
    clf = MLP(hidden_size=8, learning_rate=0.1, n_epochs=400, random_state=0).fit(X, y)
    acc = np.mean(clf.predict(X) == y)
    assert acc > 0.98


def test_loss_decreases():
    X, y = _two_blobs()
    clf = MLP(hidden_size=8, learning_rate=0.1, n_epochs=200, random_state=0).fit(X, y)
    # final loss should be well below the start; overall trend strictly downward
    assert clf.loss_curve_[-1] < clf.loss_curve_[0]
    assert clf.loss_curve_[-1] < 0.1


def test_predict_proba_is_valid_distribution():
    X, y = _two_blobs()
    clf = MLP(hidden_size=8, n_epochs=50, random_state=0).fit(X, y)
    P = clf.predict_proba(X)
    assert P.shape == (X.shape[0], 2)
    assert np.allclose(P.sum(axis=1), 1.0)      # rows are probabilities
    assert np.all(P >= 0.0)


def test_handles_multiclass():
    rng = np.random.default_rng(1)
    centers = np.array([[0, 0], [5, 5], [0, 5]])
    X = np.vstack([rng.normal(c, 0.4, size=(60, 2)) for c in centers])
    y = np.repeat(np.arange(3), 60)
    clf = MLP(hidden_size=16, learning_rate=0.1, n_epochs=400, random_state=0).fit(X, y)
    assert set(clf.classes_) == {0, 1, 2}
    assert np.mean(clf.predict(X) == y) > 0.97


def test_reproducible_with_seed():
    X, y = _two_blobs()
    a = MLP(hidden_size=8, n_epochs=100, random_state=42).fit(X, y)
    b = MLP(hidden_size=8, n_epochs=100, random_state=42).fit(X, y)
    assert np.allclose(a.W1, b.W1) and np.allclose(a.W2, b.W2)
    assert a.loss_curve_[-1] == b.loss_curve_[-1]


def test_gradient_check():
    """Numerical gradient of the loss matches the analytical backprop gradient.

    This is the real proof the backward pass is correct.
    """
    rng = np.random.default_rng(0)
    X = rng.normal(size=(5, 3))
    y = np.array([0, 1, 2, 1, 0])
    clf = MLP(hidden_size=4, random_state=0)
    clf.classes_, y_idx = np.unique(y, return_inverse=True)
    Y = clf._one_hot(y_idx, 3)
    clf._init_params(3, 3, rng)

    def loss():
        probs = clf._forward(X)[3]
        return -np.sum(Y * np.log(probs + 1e-12)) / X.shape[0]

    z1, a1, _, probs = clf._forward(X)
    dW1, db1, dW2, db2 = clf._backward(X, Y, z1, a1, probs)

    eps = 1e-5
    # check a handful of W1 entries via central differences
    for (i, j) in [(0, 0), (1, 2), (2, 3)]:
        clf.W1[i, j] += eps
        lp = loss()
        clf.W1[i, j] -= 2 * eps
        lm = loss()
        clf.W1[i, j] += eps
        numerical = (lp - lm) / (2 * eps)
        assert abs(numerical - dW1[i, j]) < 1e-6


if __name__ == "__main__":
    test_learns_xor()
    test_separates_blobs()
    test_loss_decreases()
    test_predict_proba_is_valid_distribution()
    test_handles_multiclass()
    test_reproducible_with_seed()
    test_gradient_check()
    print("All mlp tests passed.")
