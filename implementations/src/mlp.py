"""One-hidden-layer MLP (numpy) — from scratch.

Companion notes: ../../ml-theory/07-neural-networks.md

STATUS: 🟡 scaffold. To be implemented in Week 7.
Target: clean OOP, fit/predict, no sklearn for the core algorithm.

Checklist before calling this "done without notes":
  - forward pass (affine -> activation -> output)
  - loss (cross-entropy/MSE)
  - backprop via chain rule (the key reps)
  - parameter init (avoid symmetry), GD update loop
"""

from __future__ import annotations

import numpy as np


class MLP:
    """One-hidden-layer MLP (numpy).

    TODO (Week 7): implement.
    """

    def __init__(self, hidden_size: int = 16, learning_rate: float = 0.01, n_epochs: int = 1000, random_state: int | None = None):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.random_state = random_state
        self.params_ = {}

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "MLP":
        raise NotImplementedError("Implement in Week 7 — see notes.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implement in Week 7 — see notes.")
