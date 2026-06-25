"""Tests for LinearRegressionGD.

Run from the repo root:
    pytest implementations/tests/ -q
or without pytest:
    python implementations/tests/test_linear_regression.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from linear_regression import LinearRegressionGD  # noqa: E402


def _make_data(n=500, noise=0.1, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 3))
    true_w = np.array([3.0, -2.0, 0.5])
    y = X @ true_w + 5.0 + rng.normal(scale=noise, size=n)
    return X, y, true_w


def test_recovers_linear_signal():
    X, y, _ = _make_data()
    model = LinearRegressionGD(learning_rate=0.05, n_epochs=400, random_state=0)
    model.fit(X, y)
    assert model.score(X, y) > 0.99


def test_loss_decreases_monotonically_ish():
    X, y, _ = _make_data()
    model = LinearRegressionGD(learning_rate=0.05, n_epochs=200, random_state=0)
    model.fit(X, y)
    hist = model.loss_history_
    assert hist[-1] < hist[0]
    # End should be much lower than start.
    assert hist[-1] < 0.5 * hist[0]


def test_l2_shrinks_weights():
    X, y, _ = _make_data(noise=1.0)
    ols = LinearRegressionGD(learning_rate=0.05, n_epochs=400, penalty=None, random_state=0).fit(X, y)
    ridge = LinearRegressionGD(
        learning_rate=0.05, n_epochs=400, penalty="l2", lambda_=1.0, random_state=0
    ).fit(X, y)
    # L2 should pull the weight norm down relative to OLS.
    assert np.linalg.norm(ridge.w_) < np.linalg.norm(ols.w_)


def test_predict_shape():
    X, y, _ = _make_data()
    model = LinearRegressionGD(n_epochs=50, random_state=0).fit(X, y)
    preds = model.predict(X)
    assert preds.shape == (X.shape[0],)


def test_full_batch_matches_minibatch_closely():
    X, y, _ = _make_data()
    mb = LinearRegressionGD(learning_rate=0.05, n_epochs=400, batch_size=32, random_state=0).fit(X, y)
    fb = LinearRegressionGD(learning_rate=0.05, n_epochs=400, batch_size=None, random_state=0).fit(X, y)
    assert abs(mb.score(X, y) - fb.score(X, y)) < 0.02


if __name__ == "__main__":
    test_recovers_linear_signal()
    test_loss_decreases_monotonically_ish()
    test_l2_shrinks_weights()
    test_predict_shape()
    test_full_batch_matches_minibatch_closely()
    print("All linear_regression tests passed.")
