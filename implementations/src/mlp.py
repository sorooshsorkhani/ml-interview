"""One-hidden-layer MLP (numpy) — from scratch.

Companion notes: ../../ml-theory/07-neural-networks.md
Canonical from-scratch build #6.

Architecture:
    input (d) --[W1,b1]--> hidden (h) --ReLU--> --[W2,b2]--> logits (c) --softmax--> probs

Training is full-batch (or mini-batch) gradient descent. The only real machinery
beyond logistic regression is **backpropagation**: one forward pass that caches the
intermediate activations, then one backward pass that applies the chain rule layer
by layer to get every gradient.

The backward pass is the thing to be able to write without notes:
    delta2 = probs - Y                 # softmax + cross-entropy => clean residual
    dW2    = a1.T @ delta2
    delta1 = (delta2 @ W2.T) * (z1 > 0)  # propagate back, gate by ReLU' = 1[z>0]
    dW1    = X.T  @ delta1

Design goals (AMLS "code without notes" target):
  - clean OOP, fit / predict / predict_proba
  - random (He-style) init to break symmetry
  - forward with cached activations, backprop, GD update loop
  - numerically stable softmax + cross-entropy
"""

from __future__ import annotations

import numpy as np


class MLP:
    """One-hidden-layer multilayer perceptron (numpy), for multiclass classification.

    Parameters
    ----------
    hidden_size : int
        Number of hidden units h.
    learning_rate : float
        Gradient-descent step size.
    n_epochs : int
        Number of full passes over the data.
    batch_size : int | None
        Mini-batch size; None => full-batch gradient descent.
    random_state : int | None
        Seed for reproducible initialization / shuffling.

    Learned attributes (sklearn-style trailing underscore)
    ------------------------------------------------------
    classes_     : (c,) sorted unique class labels
    loss_curve_  : list of per-epoch mean cross-entropy losses
    W1, b1, W2, b2 : the parameters
    """

    def __init__(
        self,
        hidden_size: int = 16,
        learning_rate: float = 0.1,
        n_epochs: int = 1000,
        batch_size: int | None = None,
        random_state: int | None = None,
    ):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.random_state = random_state

        self.classes_: np.ndarray | None = None
        self.loss_curve_: list[float] = []
        self.W1 = self.b1 = self.W2 = self.b2 = None

    # ------------------------------------------------------------------ #
    @staticmethod
    def _softmax(Z: np.ndarray) -> np.ndarray:
        """Row-wise softmax, shifted by the max for numerical stability."""
        Z = Z - Z.max(axis=1, keepdims=True)
        E = np.exp(Z)
        return E / E.sum(axis=1, keepdims=True)

    @staticmethod
    def _one_hot(y_idx: np.ndarray, n_classes: int) -> np.ndarray:
        Y = np.zeros((y_idx.shape[0], n_classes))
        Y[np.arange(y_idx.shape[0]), y_idx] = 1.0
        return Y

    def _init_params(self, d: int, c: int, rng: np.random.Generator) -> None:
        # He initialization for the ReLU layer: std = sqrt(2/fan_in). Random (not
        # zero) is what breaks symmetry so hidden units can differentiate.
        self.W1 = rng.normal(0.0, np.sqrt(2.0 / d), size=(d, self.hidden_size))
        self.b1 = np.zeros(self.hidden_size)
        self.W2 = rng.normal(0.0, np.sqrt(2.0 / self.hidden_size), size=(self.hidden_size, c))
        self.b2 = np.zeros(c)

    def _forward(self, X: np.ndarray):
        z1 = X @ self.W1 + self.b1
        a1 = np.maximum(0.0, z1)            # ReLU
        z2 = a1 @ self.W2 + self.b2
        probs = self._softmax(z2)
        return z1, a1, z2, probs            # cache z1/a1 for the backward pass

    def _backward(self, X, Y, z1, a1, probs):
        n = X.shape[0]
        delta2 = (probs - Y) / n            # softmax + cross-entropy residual
        dW2 = a1.T @ delta2
        db2 = delta2.sum(axis=0)
        delta1 = (delta2 @ self.W2.T) * (z1 > 0)   # gate by ReLU'
        dW1 = X.T @ delta1
        db1 = delta1.sum(axis=0)
        return dW1, db1, dW2, db2

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "MLP":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        n, d = X.shape
        c = self.classes_.shape[0]
        Y = self._one_hot(y_idx, c)

        rng = np.random.default_rng(self.random_state)
        self._init_params(d, c, rng)
        self.loss_curve_ = []

        bs = self.batch_size or n
        for _ in range(self.n_epochs):
            order = rng.permutation(n)
            Xs, Ys = X[order], Y[order]
            for start in range(0, n, bs):
                Xb, Yb = Xs[start : start + bs], Ys[start : start + bs]
                z1, a1, z2, probs = self._forward(Xb)
                dW1, db1, dW2, db2 = self._backward(Xb, Yb, z1, a1, probs)
                self.W1 -= self.learning_rate * dW1
                self.b1 -= self.learning_rate * db1
                self.W2 -= self.learning_rate * dW2
                self.b2 -= self.learning_rate * db2

            # epoch loss on the full set (mean cross-entropy)
            probs_full = self._forward(X)[3]
            eps = 1e-12
            self.loss_curve_.append(float(-np.sum(Y * np.log(probs_full + eps)) / n))
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.W1 is None:
            raise ValueError("call fit() before predict_proba()")
        return self._forward(X)[3]

    def predict(self, X: np.ndarray) -> np.ndarray:
        idx = np.argmax(self.predict_proba(X), axis=1)
        return self.classes_[idx]


if __name__ == "__main__":
    # XOR: the textbook proof that a hidden layer beats a linear model.
    X = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1, 1, 0])
    clf = MLP(hidden_size=8, learning_rate=0.5, n_epochs=2000, random_state=0).fit(X, y)
    print("XOR predictions:", clf.predict(X), "  truth:", y)
    print(f"final loss: {clf.loss_curve_[-1]:.4f}")
