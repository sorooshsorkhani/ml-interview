"""Linear Regression from scratch (gradient descent), with optional L2 (Ridge) penalty.

Companion notes: ../../ml-theory/01-regularization-and-optimization.md

Design goals (the AMLS "code without notes" target):
  - clean OOP with fit / predict
  - mini-batch gradient descent
  - optional L2 regularization (penalty="l2"), bias term NOT penalized
  - internal standardization so the single learning rate behaves
  - loss history for convergence inspection
"""

from __future__ import annotations

import numpy as np


class LinearRegressionGD:
    """Ordinary / Ridge linear regression trained by (mini-batch) gradient descent.

    Parameters
    ----------
    learning_rate : float
        Step size eta.
    n_epochs : int
        Number of passes over the training data.
    batch_size : int | None
        Mini-batch size. None => full-batch gradient descent.
    penalty : {None, "l2"}
        None for OLS, "l2" for Ridge.
    lambda_ : float
        Regularization strength (only used when penalty="l2").
    standardize : bool
        If True, standardize features internally (fit stats on training data only).
    shuffle : bool
        Shuffle samples each epoch.
    random_state : int | None
        Seed for reproducibility.
    """

    def __init__(
        self,
        learning_rate: float = 0.01,
        n_epochs: int = 1000,
        batch_size: int | None = 32,
        penalty: str | None = None,
        lambda_: float = 0.0,
        standardize: bool = True,
        shuffle: bool = True,
        random_state: int | None = None,
    ):
        if penalty not in (None, "l2"):
            raise ValueError("penalty must be None or 'l2'")
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.penalty = penalty
        self.lambda_ = lambda_
        self.standardize = standardize
        self.shuffle = shuffle
        self.random_state = random_state

        self.w_: np.ndarray | None = None
        self.b_: float = 0.0
        self.loss_history_: list[float] = []
        self._mu: np.ndarray | None = None
        self._sigma: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        self._mu = X.mean(axis=0)
        self._sigma = X.std(axis=0)
        self._sigma[self._sigma == 0] = 1.0  # guard constant features
        return (X - self._mu) / self._sigma

    def _standardize_apply(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mu) / self._sigma

    def _mse(self, X: np.ndarray, y: np.ndarray) -> float:
        err = X @ self.w_ + self.b_ - y
        loss = np.mean(err**2)
        if self.penalty == "l2":
            loss += self.lambda_ * np.sum(self.w_**2)
        return float(loss)

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegressionGD":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).ravel()
        n, p = X.shape

        if self.standardize:
            X = self._standardize_fit(X)

        rng = np.random.default_rng(self.random_state)
        self.w_ = np.zeros(p)
        self.b_ = 0.0
        self.loss_history_ = []

        bs = self.batch_size or n
        for _ in range(self.n_epochs):
            idx = rng.permutation(n) if self.shuffle else np.arange(n)
            for start in range(0, n, bs):
                batch = idx[start : start + bs]
                Xb, yb = X[batch], y[batch]
                m = len(batch)

                err = Xb @ self.w_ + self.b_ - yb          # (m,)
                grad_w = (2.0 / m) * (Xb.T @ err)          # MSE gradient
                grad_b = (2.0 / m) * err.sum()

                if self.penalty == "l2":
                    grad_w += 2.0 * self.lambda_ * self.w_  # L2 term, bias excluded

                self.w_ -= self.learning_rate * grad_w
                self.b_ -= self.learning_rate * grad_b

            self.loss_history_.append(self._mse(X, y))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if self.standardize:
            X = self._standardize_apply(X)
        return X @ self.w_ + self.b_

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """R^2 coefficient of determination."""
        y = np.asarray(y, dtype=float).ravel()
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        return 1.0 - ss_res / ss_tot


if __name__ == "__main__":
    # Tiny smoke test: y = 3*x1 - 2*x2 + 5 + noise
    rng = np.random.default_rng(0)
    X = rng.normal(size=(500, 2))
    y = 3 * X[:, 0] - 2 * X[:, 1] + 5 + rng.normal(scale=0.1, size=500)

    for pen, lam in [(None, 0.0), ("l2", 0.1)]:
        model = LinearRegressionGD(
            learning_rate=0.05, n_epochs=300, penalty=pen, lambda_=lam, random_state=0
        )
        model.fit(X, y)
        print(f"penalty={pen!s:>4}  R^2={model.score(X, y):.4f}  final_loss={model.loss_history_[-1]:.4f}")
