"""Decision Tree classifier (CART) — from scratch.

Companion notes: ../../ml-theory/03-decision-trees.md

STATUS: 🟡 scaffold. To be implemented in Week 3.
Target: clean OOP, fit/predict, no sklearn for the core algorithm.

Checklist before calling this "done without notes":
  - impurity (gini/entropy) + information gain
  - best-split search over features & thresholds
  - recursive build + stopping criteria + predict by tree traversal
"""

from __future__ import annotations

import numpy as np


class DecisionTreeClassifier:
    """Decision Tree classifier (CART).

    TODO (Week 3): implement.
    """

    def __init__(self, max_depth: int | None = None, min_samples_split: int = 2, criterion: str = "gini"):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root_ = None

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "DecisionTreeClassifier":
        raise NotImplementedError("Implement in Week 3 — see notes.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implement in Week 3 — see notes.")
