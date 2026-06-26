"""Decision Tree classifier (CART) — from scratch.

Companion notes: ../../ml-theory/03-decision-trees.md

A CART classification tree recursively partitions the feature space with
axis-aligned splits (feature j <= threshold t) chosen to maximize the drop in
impurity (information gain). Prediction walks a query down the tree to a leaf
and returns that leaf's majority class.

Design goals (AMLS "code without notes" target):
  - clean OOP, fit / predict / predict_proba
  - gini OR entropy impurity, information gain to pick splits
  - exhaustive best-split search over features & candidate thresholds
  - recursive build with stopping criteria (max_depth, min_samples_split, purity)
  - predict by tree traversal

The split-search is the heart of the algorithm and the part interviewers ask
you to reproduce end to end.
"""

from __future__ import annotations

import numpy as np


class _Node:
    """A single tree node — either an internal split or a leaf.

    Internal node: `feature` and `threshold` set; routes left if
    x[feature] <= threshold, else right.
    Leaf: `value` set to the class-probability vector (over classes_).
    """

    __slots__ = ("feature", "threshold", "left", "right", "value")

    def __init__(self, *, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value  # class-probability vector for leaves

    @property
    def is_leaf(self) -> bool:
        return self.value is not None


class DecisionTreeClassifier:
    """Decision Tree classifier (CART, binary axis-aligned splits).

    Parameters
    ----------
    max_depth : int | None
        Maximum depth of the tree. None = grow until other stopping rules hit.
    min_samples_split : int
        A node with fewer than this many samples becomes a leaf (no split tried).
    criterion : {"gini", "entropy"}
        Impurity measure used to score splits.
    """

    def __init__(
        self,
        max_depth: int | None = None,
        min_samples_split: int = 2,
        criterion: str = "gini",
    ):
        if criterion not in ("gini", "entropy"):
            raise ValueError("criterion must be 'gini' or 'entropy'")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion

        self.root_: _Node | None = None
        self.classes_: np.ndarray | None = None
        self._n_classes: int | None = None

    # ----------------------- impurity ----------------------- #
    def _class_probs(self, y: np.ndarray) -> np.ndarray:
        """Probability vector over classes_ for the labels in this node."""
        counts = np.array([(y == c).sum() for c in self.classes_], dtype=float)
        total = counts.sum()
        return counts / total if total > 0 else counts

    def _impurity(self, y: np.ndarray) -> float:
        p = self._class_probs(y)
        p = p[p > 0]  # 0*log0 = 0, and p=0 contributes nothing to gini either
        if self.criterion == "gini":
            return 1.0 - np.sum(p**2)
        return -np.sum(p * np.log2(p))  # entropy (bits)

    # ----------------------- split search ----------------------- #
    def _best_split(self, X: np.ndarray, y: np.ndarray):
        """Find the (feature, threshold) with the largest information gain.

        Returns (feature, threshold, gain) or (None, None, 0.0) if no split
        improves purity.
        """
        n_samples, n_features = X.shape
        parent_impurity = self._impurity(y)
        best_gain = 0.0
        best_feat, best_thr = None, None

        for feat in range(n_features):
            values = X[:, feat]
            # Candidate thresholds = midpoints between consecutive unique values.
            uniq = np.unique(values)
            if uniq.size == 1:
                continue  # constant feature: no split possible
            thresholds = (uniq[:-1] + uniq[1:]) / 2.0

            for thr in thresholds:
                left_mask = values <= thr
                n_left = left_mask.sum()
                n_right = n_samples - n_left
                if n_left == 0 or n_right == 0:
                    continue
                # Weighted child impurity, then gain = parent - weighted children.
                imp_left = self._impurity(y[left_mask])
                imp_right = self._impurity(y[~left_mask])
                child_impurity = (n_left * imp_left + n_right * imp_right) / n_samples
                gain = parent_impurity - child_impurity
                if gain > best_gain:
                    best_gain, best_feat, best_thr = gain, feat, thr

        return best_feat, best_thr, best_gain

    # ----------------------- recursive build ----------------------- #
    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> _Node:
        n_samples = X.shape[0]

        # Stopping criteria → make a leaf.
        pure = np.unique(y).size == 1
        too_small = n_samples < self.min_samples_split
        too_deep = self.max_depth is not None and depth >= self.max_depth
        if pure or too_small or too_deep:
            return _Node(value=self._class_probs(y))

        feat, thr, gain = self._best_split(X, y)
        if feat is None or gain <= 0.0:
            # No split improves purity (e.g. duplicate rows with mixed labels).
            return _Node(value=self._class_probs(y))

        left_mask = X[:, feat] <= thr
        left = self._build(X[left_mask], y[left_mask], depth + 1)
        right = self._build(X[~left_mask], y[~left_mask], depth + 1)
        return _Node(feature=feat, threshold=thr, left=left, right=right)

    # ----------------------- public API ----------------------- #
    def fit(self, X: np.ndarray, y: np.ndarray) -> "DecisionTreeClassifier":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y).ravel()
        self.classes_ = np.unique(y)
        self._n_classes = self.classes_.size
        self.root_ = self._build(X, y, depth=0)
        return self

    def _leaf_proba(self, x: np.ndarray) -> np.ndarray:
        node = self.root_
        while not node.is_leaf:
            node = node.left if x[node.feature] <= node.threshold else node.right
        return node.value

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        return np.array([self._leaf_proba(x) for x in X])

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y = np.asarray(y).ravel()
        return float(np.mean(self.predict(X) == y))

    def depth(self) -> int:
        """Actual depth of the fitted tree (a single leaf has depth 0)."""
        def _d(node: _Node) -> int:
            if node.is_leaf:
                return 0
            return 1 + max(_d(node.left), _d(node.right))

        return _d(self.root_) if self.root_ is not None else 0


if __name__ == "__main__":
    # XOR-ish problem: linearly inseparable, but a depth-2 tree solves it.
    rng = np.random.default_rng(0)
    n = 200
    X = rng.uniform(-1, 1, size=(n, 2))
    y = ((X[:, 0] > 0) ^ (X[:, 1] > 0)).astype(int)  # XOR of the two quadrant signs

    for crit in ("gini", "entropy"):
        model = DecisionTreeClassifier(criterion=crit, max_depth=4).fit(X, y)
        print(f"criterion={crit:>8}  train acc={model.score(X, y):.3f}  depth={model.depth()}")
