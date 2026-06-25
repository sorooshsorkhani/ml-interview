"""Gaussian Naive Bayes (stretch) — from scratch.

Companion notes: ../../ml-theory/11-naive-bayes.md

STATUS: 🟡 scaffold. To be implemented in Week stretch.
Target: clean OOP, fit/predict, no sklearn for the core algorithm.

Checklist before calling this "done without notes":
  - class priors, per-class feature mean/variance
  - log-likelihood under Gaussian, argmax of posterior
  - log-space to avoid underflow
"""

from __future__ import annotations

import numpy as np


class GaussianNB:
    """Gaussian Naive Bayes (stretch).

    TODO (Week stretch): implement.
    """

    def __init__(self):
        self.class_priors_ = None
        self.theta_ = None  # per-class feature means
        self.var_ = None    # per-class feature variances

    def fit(self, X: np.ndarray, y: np.ndarray = None) -> "GaussianNB":
        raise NotImplementedError("Implement in Week stretch — see notes.")

    def predict(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError("Implement in Week stretch — see notes.")
