# Decision Trees

> **Week:** 3 · **Status:** ✅ written · **From-scratch build:** `DecisionTreeClassifier` (CART split logic end to end)
> **One-line:** A model that recursively partitions feature space with axis-aligned splits chosen to maximize purity (information gain / Gini reduction), then predicts the majority class of the leaf a point falls into.

## Table of contents
- [1. Intuition](#1-intuition)
- [2. The math](#2-the-math)
  - [2.1 Impurity: Gini & entropy](#21-impurity-gini--entropy)
  - [2.2 Information gain](#22-information-gain)
- [3. The algorithm](#3-the-algorithm)
- [4. Overfitting, pruning & regularization](#4-overfitting-pruning--regularization)
- [5. From scratch](#5-from-scratch)
- [6. With scikit-learn](#6-with-scikit-learn)
- [7. Assumptions](#7-assumptions)
- [8. When to use / not use](#8-when-to-use--not-use)
- [9. Pitfalls & gotchas](#9-pitfalls--gotchas)
- [10. Interview questions](#10-interview-questions)
- [11. Connections](#11-connections)

---

## 1. Intuition

A decision tree is a flowchart of yes/no questions about features. Starting at the root, each internal node asks one question (`is feature_j <= t?`), routing a point left or right, until it reaches a **leaf** that holds a prediction. Training is the act of *choosing the questions*: at each node, pick the single feature-and-threshold split that best separates the classes.

"Best separates" means **most pure**: after the split, each child should be as close as possible to containing a single class. The tree greedily, recursively, repeats this on each child.

What makes trees distinctive:
- **Non-parametric & non-linear:** no fixed functional form; the boundary is built from stacked **axis-aligned** rectangles, so it can fit highly non-linear targets (it solves XOR, which a linear model can't).
- **Greedy & recursive:** each split is chosen locally to maximize immediate purity gain — no look-ahead, no global optimization (finding the optimal tree is NP-hard).
- **Interpretable:** a shallow tree is a readable set of rules — a big reason they're loved in practice and in interviews.
- **Scale-invariant:** splits are thresholds on one feature at a time, so monotonic rescaling of a feature doesn't change the tree. **No standardization needed** — the opposite of KNN ([Week 2](02-knn.md)).

> Trees are the base learner behind Random Forests ([Week 4](04-ensembles-bagging.md)) and gradient boosting ([Week 5](05-ensembles-boosting.md)) — the most important practical models in classical ML. Understanding the single tree is the foundation for both.

## 2. The math

A node holds a set of training samples. We score how "mixed" its labels are with an **impurity** measure, and we choose the split that reduces (weighted) impurity the most.

### 2.1 Impurity: Gini & entropy

Let $p_c$ be the fraction of samples in a node belonging to class $c$ (so $\sum_c p_c = 1$).

**Gini impurity** — expected error if you labeled a random sample by the node's class distribution:
$$
G = \sum_{c} p_c (1 - p_c) = 1 - \sum_{c} p_c^2
$$

**Entropy** — average information (bits) needed to encode the class:
$$
H = -\sum_{c} p_c \log_2 p_c
$$

Both are **0 when the node is pure** (one class) and **maximal when classes are uniform**. For binary classes the maximum is $G = 0.5$ and $H = 1$ bit (both at $p = 0.5$).

| | Gini | Entropy |
|---|---|---|
| Range (binary) | 0 → 0.5 | 0 → 1 |
| Cost | cheaper (no logs) | logarithms |
| Behavior | very similar splits in practice | slightly more sensitive to changes in class probs |

> **Practical takeaway:** Gini and entropy almost always pick the same/similar splits. Gini is sklearn's default because it's cheaper. Don't over-think the choice in an interview — say "they're nearly equivalent, Gini is faster."

### 2.2 Information gain

A candidate split partitions a parent node's $n$ samples into left ($n_L$) and right ($n_R$) children. The **information gain** is the parent impurity minus the *sample-weighted* average child impurity:
$$
\text{Gain} = I(\text{parent}) - \left( \frac{n_L}{n} I(\text{left}) + \frac{n_R}{n} I(\text{right}) \right)
$$

where $I$ is Gini or entropy. The weighting by $n_L/n$, $n_R/n$ is essential — a split that peels off one pure sample shouldn't beat a split that cleanly halves the data. We pick the $(j, t)$ maximizing Gain. (With entropy this is literally "information gain"; with Gini it's "Gini reduction," but the principle is identical.)

## 3. The algorithm

This is **CART** (Classification And Regression Trees) — binary, axis-aligned splits.

```
BUILD(node samples X, y, depth):
    if stopping criterion met:               # pure / too deep / too few samples
        return Leaf(class distribution of y)

    best_gain, best_(feature, threshold) = -inf, None
    for each feature j:
        for each candidate threshold t:      # midpoints of sorted unique values of feature j
            split X,y by (x[j] <= t) into left/right
            gain = impurity(parent) - weighted impurity(left, right)
            track the best (j, t)

    if best_gain <= 0:                        # no split helps -> leaf
        return Leaf(class distribution of y)

    left  = BUILD(samples where x[j] <= t, depth+1)
    right = BUILD(samples where x[j] >  t, depth+1)
    return InternalNode(best_feature, best_threshold, left, right)

PREDICT(x):
    walk from root: go left if x[feature] <= threshold else right, until a leaf
    return the leaf's majority class (or its class-probability vector)
```

**Candidate thresholds:** for a numeric feature, sort the unique values and try the **midpoints between consecutive values** — every distinct partition of the data is covered with the fewest candidates.

**Complexity** (n samples, p features, tree depth d):
- A naive split search at one node scans all features × all thresholds: $O(n \cdot p)$ to evaluate (with sorting, $O(p \cdot n \log n)$ per node; the from-scratch build re-derives unique values each time, which is fine for learning).
- Total training $\approx O(p \cdot n \log n \cdot d)$ for a balanced tree.
- **Prediction is cheap:** $O(d)$ — just walk root-to-leaf. This is the mirror image of KNN (cheap train, expensive predict).

## 4. Overfitting, pruning & regularization

A fully grown tree keeps splitting until every leaf is pure → it **memorizes the training set** (often 100% train accuracy, poor generalization). This is the central weakness: **single trees are high-variance.**

**Pre-pruning (early stopping)** — stop growing via hyperparameters:
- `max_depth` — hard cap on depth (the main knob).
- `min_samples_split` — don't split a node with too few samples.
- `min_samples_leaf` — require each leaf to keep at least this many samples.
- `min_impurity_decrease` — only split if gain exceeds a threshold.

**Post-pruning** — grow a large tree, then prune back subtrees that don't help on validation. The classic method is **cost-complexity pruning** (a.k.a. weakest-link pruning): minimize $R_\alpha(T) = R(T) + \alpha |T|$, trading training error $R(T)$ against the number of leaves $|T|$. The penalty $\alpha$ (sklearn's `ccp_alpha`) is tuned by cross-validation; larger $\alpha$ → smaller tree.

> **The real fix in practice is ensembling.** A single tree is a high-variance learner, so we average many of them (Random Forest, [Week 4](04-ensembles-bagging.md)) or boost weak shallow trees (gradient boosting, [Week 5](05-ensembles-boosting.md)). The bias–variance lens from [Week 1](01-regularization-and-optimization.md)/[statistics/02](../statistics/02-bias-variance-mle.md): depth ↑ → bias ↓, variance ↑.

## 5. From scratch

Implementation: [`implementations/src/decision_tree.py`](../implementations/src/decision_tree.py) — `DecisionTreeClassifier` with `fit` / `predict` / `predict_proba` / `score` / `depth`. Tests: [`test_decision_tree.py`](../implementations/tests/test_decision_tree.py).

Design decisions mapped to the theory:
- **`_impurity`** computes Gini or entropy from a node's class-probability vector; zero-probability classes are dropped so `0·log0` is handled (§2.1).
- **`_best_split`** is the heart: it loops over every feature, builds candidate thresholds as **midpoints of consecutive unique values**, and keeps the $(j,t)$ with the largest weighted information gain (§2.2, §3). Constant features and all-left/all-right splits are skipped.
- **`_build`** recurses, with explicit **stopping criteria**: node is pure, `n < min_samples_split`, `depth >= max_depth`, or `gain <= 0` (§3, §4). The `gain <= 0` guard is what makes **contradictory rows** (identical features, different labels) terminate gracefully as a majority leaf instead of recursing forever — there's a test for exactly this.
- **Leaves store the class-probability vector**, so `predict_proba` is free and `predict` is `argmax` over it.
- **Prediction** walks root→leaf comparing `x[feature] <= threshold` — $O(\text{depth})$ (§3).

Tests demonstrate the key theory points: it **solves XOR** (non-linear, §1), an unrestricted tree **memorizes** the training set (§4), `max_depth=1` can't solve XOR (the bias of a stump, §4), and Gini vs. entropy both separate easy data (§2.1).

## 6. With scikit-learn

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import GridSearchCV

tree = DecisionTreeClassifier(
    criterion="gini",        # or "entropy" / "log_loss"
    max_depth=None,          # the main regularizer — tune this
    min_samples_leaf=1,
    ccp_alpha=0.0,           # cost-complexity pruning strength
    random_state=0,
)

# Tune the regularizers with CV:
grid = GridSearchCV(
    tree,
    param_grid={
        "max_depth": [3, 5, 8, None],
        "min_samples_leaf": [1, 5, 20],
        "ccp_alpha": [0.0, 0.001, 0.01],
    },
    cv=5, scoring="f1",
)
# No StandardScaler needed — trees are scale-invariant.

tree.feature_importances_   # impurity-based importance per feature
```

Key hyperparameters: `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `ccp_alpha` (§4). `feature_importances_` reports the total impurity decrease each feature contributes — useful but **biased toward high-cardinality features** (a known gotcha).

## 7. Assumptions

Trees are refreshingly assumption-light:
- **No distributional assumptions**, no linearity, no feature independence.
- **No feature scaling required** (scale-invariant splits, §1).
- **Implicitly assume the boundary is well-approximated by axis-aligned rectangles** — a diagonal boundary needs a deep staircase of splits to approximate (a real limitation).
- Greedy splitting assumes locally-good splits lead to a globally-good tree (not guaranteed — it's a heuristic).

## 8. When to use / not use

**Use when:**
- You want **interpretability** / explainable rules (shallow tree).
- **Mixed feature types**, non-linear interactions, no desire to scale/encode heavily.
- As the **base learner** for ensembles (the dominant real use).
- Missing values / monotonic transforms shouldn't matter.

**Avoid / prefer alternatives when:**
- You need a **single high-accuracy model** — a lone tree is high-variance; use a Random Forest or gradient boosting instead.
- The true boundary is **smooth/diagonal** — linear/kernel models capture it with far less complexity.
- **Very high-dimensional sparse data** (e.g., text) — linear models usually win.
- You need stable predictions — small data changes can produce a very different tree (high variance).

## 9. Pitfalls & gotchas
- **Overfitting by default** — an unpruned tree memorizes training data. Always regularize (`max_depth`, pruning) or ensemble. ⭐
- **High variance / instability** — a small change in the data can flip early splits and reshape the whole tree.
- **Greedy, not optimal** — the best *sequence* of local splits isn't the globally best tree.
- **Biased feature importances** — impurity-based importance favors high-cardinality / continuous features; prefer permutation importance when it matters.
- **Axis-aligned only** — diagonal boundaries need many staircase splits (oblique trees exist but are rare).
- **Class imbalance** — pure-class chasing can ignore the minority class; use `class_weight` or resampling.
- **Confusing "information gain" with requiring entropy** — Gini reduction is the same idea; both are "impurity decrease."
- **Extrapolation** — like KNN, a tree predicts a constant outside the range of training data (leaves are flat).

## 10. Interview questions
Must answer cold (⭐):
- ⭐ **How does a decision tree decide where to split?** It exhaustively tries each feature and candidate threshold, computes the (sample-weighted) impurity decrease — information gain — and picks the split with the largest gain; then recurses. (§2.2, §3)
- ⭐ **Gini vs. entropy — what's the difference?** Both measure node impurity (0 = pure, max = uniform); entropy uses logs, Gini doesn't. In practice they pick nearly identical splits; Gini is cheaper and is sklearn's default. (§2.1)
- ⭐ **Why do decision trees overfit, and how do you prevent it?** They keep splitting until leaves are pure → memorize the data (high variance). Prevent with pre-pruning (`max_depth`, `min_samples_leaf`), post-pruning (cost-complexity / `ccp_alpha`), or — best — ensembling (Random Forest, boosting). (§4)
- ⭐ **Do you need to scale features for a decision tree?** No — splits are thresholds on one feature at a time, so the tree is invariant to monotonic rescaling. (Contrast KNN/linear models.) (§1, §7)
- ⭐ **What's the bias–variance profile of a single tree?** Low bias, high variance — flexible enough to fit anything, but unstable. Depth ↑ → bias ↓, variance ↑. (§4)
- **What is information gain, formally?** Parent impurity minus the weighted average of child impurities. (§2.2)
- **Train vs. predict complexity?** Training ~$O(p \cdot n \log n \cdot \text{depth})$; prediction $O(\text{depth})$ — cheap. (§3)
- **Why is the greedy split search not optimal?** Finding the globally optimal tree is NP-hard; CART makes locally-greedy splits with no look-ahead. (§1, §3)
- **How does a tree handle non-linear / XOR problems?** By stacking axis-aligned splits — a depth-2 tree solves XOR, unlike a linear model. (§1, §5)
- **What's the problem with impurity-based feature importance?** It's biased toward high-cardinality/continuous features; use permutation importance instead. (§9)
- **How does a regression tree differ?** Same recursion, but impurity = variance/MSE of targets in a node, and a leaf predicts the mean target. (§3, conceptually)

## 11. Connections
- **Base learner for ensembles:** bagging/Random Forests ([04](04-ensembles-bagging.md)) and boosting ([05](05-ensembles-boosting.md)) exist specifically to tame the single tree's high variance.
- **Bias–variance** framing shared with [regularization](01-regularization-and-optimization.md) and [KNN](02-knn.md#4-choosing-k); formalized in [statistics/02](../statistics/02-bias-variance-mle.md).
- **No scaling needed** — the explicit contrast with [KNN](02-knn.md#21-distance-metrics) (scale-sensitive).
- **Pruning ≈ regularization** — the complexity penalty $\alpha|T|$ mirrors L1/L2 penalties from [Week 1](01-regularization-and-optimization.md).
- Choosing `max_depth`/pruning strength via [cross-validation](../evaluation/03-cross-validation-leakage.md).
- Flashcards: [flashcards/03-decision-trees.md](../flashcards/03-decision-trees.md).

---
### Sources
- *ISLR* — Ch. 8 (Tree-based methods: trees, pruning, then bagging/RF/boosting).
- Géron, *Hands-On ML* — Decision Trees chapter (CART, Gini vs. entropy, regularization).
