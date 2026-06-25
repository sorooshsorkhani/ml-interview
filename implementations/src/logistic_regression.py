"""Logistic Regression from scratch (gradient descent).

Companion notes: ../../ml-theory/00-foundations.md

STATUS: done prior to this plan — PORT your existing implementation here.
Mirror the structure of linear_regression.py:
  - sigmoid, binary cross-entropy loss
  - gradient: (1/n) Xᵀ(p - y)
  - fit/predict_proba/predict, optional L2 penalty
  - internal standardization, loss history
"""

from __future__ import annotations

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


class LogisticRegressionGD:
    """TODO: port your existing logistic regression here."""

    def __init__(self, learning_rate: float = 0.01, n_epochs: int = 1000,
                 penalty: str | None = None, lambda_: float = 0.0,
                 random_state: int | None = None):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.penalty = penalty
        self.lambda_ = lambda_
        self.random_state = random_state

    def fit(self, X, y):
        raise NotImplementedError("Port your existing implementation here.")

    def predict_proba(self, X):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
