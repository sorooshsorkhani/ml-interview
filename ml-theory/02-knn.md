# K-Nearest Neighbors (KNN)

> **Week:** 2 · **Status:** ✅ written · **From-scratch build:** `KNNClassifier`
> **One-line:** A lazy, non-parametric method that classifies a point by majority vote of its k closest training points — there is no "training," all the work happens at prediction time.

## Table of contents
- [1. Intuition](#1-intuition)
- [2. The math](#2-the-math)
  - [2.1 Distance metrics](#21-distance-metrics)
  - [2.2 The decision rule](#22-the-decision-rule)
- [3. The algorithm](#3-the-algorithm)
- [4. Choosing k](#4-choosing-k)
- [5. The curse of dimensionality](#5-the-curse-of-dimensionality)
- [6. From scratch](#6-from-scratch)
- [7. With scikit-learn](#7-with-scikit-learn)
- [8. Assumptions & preconditions](#8-assumptions--preconditions)
- [9. When to use / not use](#9-when-to-use--not-use)
- [10. Pitfalls & gotchas](#10-pitfalls--gotchas)
- [11. Interview questions](#11-interview-questions)
- [12. Connections](#12-connections)

---

## 1. Intuition

"You are the company you keep." To classify a new point, look at the **k** training points nearest to it and let them vote. Nearby points are assumed to share a label, so the local neighborhood is a good predictor.

What makes KNN distinctive:
- **Lazy learning:** `fit()` does essentially nothing — it just memorizes the training set. There is no model, no parameters fit, no loss minimized. All computation is deferred to `predict()`.
- **Non-parametric:** it makes no assumption about the form of the decision boundary (no "the boundary is linear"). The number of "parameters" effectively grows with the data — the whole training set *is* the model.
- **Instance-based:** predictions come directly from stored instances, not a learned summary.

The decision boundary KNN induces is **locally adaptive and arbitrarily wiggly** — it can carve out complex regions a linear model never could. That flexibility is its strength (low bias) and its danger (high variance when k is small).

> Contrast with linear/logistic regression (Week 0–1): those are **parametric** and **eager** — they fit a fixed set of weights at training time, then prediction is cheap. KNN flips that: cheap training, expensive prediction.

## 2. The math

### 2.1 Distance metrics
"Nearest" requires a distance. For points $x, x' \in \mathbb{R}^p$:

**Euclidean (L2)** — the default; straight-line distance:
$$
d_2(x, x') = \sqrt{\sum_{j=1}^{p} (x_j - x'_j)^2} = \lVert x - x' \rVert_2
$$

**Manhattan (L1)** — city-block distance; more robust to outliers, sometimes better in high dimensions:
$$
d_1(x, x') = \sum_{j=1}^{p} |x_j - x'_j|
$$

**Minkowski** generalizes both with order $q$ (q=2 → Euclidean, q=1 → Manhattan):
$$
d_q(x, x') = \left( \sum_j |x_j - x'_j|^q \right)^{1/q}
$$

**Cosine distance** ($1 - \cos\theta$) for text/embeddings where direction matters more than magnitude.

> **Scale is everything here.** Distance sums over features, so a feature measured in the thousands (e.g., salary) drowns out one in the units (e.g., age). **You must standardize/normalize features before KNN.** This is the single most important practical point and a guaranteed interview probe.

### 2.2 The decision rule
Let $N_k(x)$ be the set of the $k$ training points nearest to query $x$.

**Classification** (majority vote):
$$
\hat{y}(x) = \arg\max_{c} \sum_{i \in N_k(x)} \mathbb{1}(y_i = c)
$$

**Class probability estimate** = fraction of neighbors in each class: $\hat{p}(c \mid x) = \frac{1}{k}\sum_{i \in N_k(x)} \mathbb{1}(y_i = c)$.

**Regression** variant (KNN regression): average the neighbors' targets, $\hat{y}(x) = \frac{1}{k}\sum_{i \in N_k(x)} y_i$.

**Distance weighting** (optional): weight each neighbor's vote by $1/d$ so closer points count more — helps when k is larger.

## 3. The algorithm

```
TRAIN (fit):
    store X_train, y_train      # that's the whole "training" — O(1) work, O(n) memory

PREDICT one query x:
    for each training point xi:  compute d(x, xi)      # O(n·p)
    find the k smallest distances                       # O(n) with partial selection
    return the majority label among those k neighbors   # O(k)
```

**Complexity** (n train points, p features, one query):
- Train: $O(1)$ time, $O(np)$ space (store the data).
- Predict (brute force): $O(np)$ per query for distances + $O(n)$ to select top-k → **$O(np)$ per query**. Expensive at inference, especially with large n — the core tradeoff.
- **Speedups:** KD-trees / ball-trees reduce neighbor search to ~$O(\log n)$ per query in low dimensions (but degrade to brute force in high dimensions); approximate nearest neighbors (ANN, e.g. HNSW/FAISS) for large-scale/embedding search.

> **Implementation detail worth knowing:** to get the k smallest distances you don't need a full sort ($O(n \log n)$) — use **partial selection** (`np.argpartition`, a quickselect) which is $O(n)$. The from-scratch build uses this.

## 4. Choosing k

`k` controls the **bias–variance tradeoff** (ties back to [Week 1](01-regularization-and-optimization.md)):

| k | Boundary | Bias | Variance | Risk |
|---|---|---|---|---|
| small (k=1) | very wiggly, follows every point | low | **high** | overfits noise |
| large | smooth, averages big regions | **high** | low | underfits; ignores local structure |
| k = n | predicts the global majority always | max | min | useless |

- **k=1** classifies each training point as itself → 0 training error but typically poor generalization (memorizes noise).
- Pick k by **cross-validation** ([evaluation/03](../evaluation/03-cross-validation-leakage.md)) — sweep k and pick the best validation score.
- **Use an odd k for binary classification** to avoid tie votes.
- Rule of thumb (weak): $k \approx \sqrt{n}$ as a starting point, then tune.

## 5. The curse of dimensionality

KNN's Achilles' heel. As the number of features $p$ grows:
- **Distances concentrate:** the ratio between the nearest and farthest point's distance approaches 1 — "nearest" becomes meaningless because *everything* is roughly equidistant.
- **Data sparsity:** to keep the same density of neighbors you need exponentially more data. In high dimensions, your "k nearest" neighbors may actually be far away and unrepresentative.
- **Irrelevant features hurt:** every noisy feature adds to the distance and dilutes the signal (unlike tree models, which ignore unused features).

**Mitigations:** dimensionality reduction (PCA — [Week 6](06-kmeans-pca.md)), feature selection, learned embeddings, or simply prefer a different model when p is large.

## 6. From scratch

Implementation: [`implementations/src/knn.py`](../implementations/src/knn.py) — `KNNClassifier` with `fit` / `predict` / `predict_proba`. Tests: [`test_knn.py`](../implementations/tests/test_knn.py).

Design decisions mapped to the theory:
- **`fit` just stores** `X_`, `y_`, and `classes_` — embodies lazy learning (§1).
- **Vectorized Euclidean distance** uses the identity $\lVert a-b\rVert^2 = \lVert a\rVert^2 - 2a\!\cdot\!b + \lVert b\rVert^2$ to compute the full (n_query × n_train) distance matrix with one matrix multiply — no Python loops (§2.1).
- **`np.argpartition(..., k-1)`** grabs the k nearest in $O(n)$ instead of sorting (§3).
- **Majority vote with deterministic tie-breaking:** on a tie, the lowest class label wins (because `np.unique` returns sorted classes) — reproducible predictions (§2.2).
- **`standardize=True` by default** — bakes in the #1 practical rule (§2.1). A test (`test_standardization_matters_on_skewed_scales`) demonstrates accuracy collapsing when a noise feature is on a 1000× scale and standardization is off.

## 7. With scikit-learn

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

knn = make_pipeline(
    StandardScaler(),                                   # always scale first
    KNeighborsClassifier(n_neighbors=5, weights="uniform", metric="minkowski", p=2)
)

# Tune k (and weighting) with cross-validation:
grid = GridSearchCV(
    knn,
    param_grid={
        "kneighborsclassifier__n_neighbors": [3, 5, 7, 9, 15],
        "kneighborsclassifier__weights": ["uniform", "distance"],
    },
    cv=5, scoring="accuracy",
)
```

Key hyperparameters:
- `n_neighbors` (k) — the bias–variance knob (§4).
- `weights` — `"uniform"` (equal vote) vs `"distance"` (closer = stronger).
- `metric` / `p` — `p=2` Euclidean, `p=1` Manhattan.
- `algorithm` — `"auto"`, `"kd_tree"`, `"ball_tree"`, `"brute"` (search-speed structure, §3).

## 8. Assumptions & preconditions
- **Meaningful distance:** features must be on **comparable scales** (standardize!) and the metric must reflect true similarity for the problem.
- **Local smoothness:** nearby points tend to share labels (the core assumption). Fails on highly non-smooth targets.
- **Enough data relative to dimensionality** (§5) — KNN is data-hungry in high dimensions.
- **No strong assumption on boundary shape** — that's the upside of being non-parametric.

## 9. When to use / not use
**Use when:**
- Small-to-medium datasets, low-to-moderate dimensionality.
- A strong, simple **baseline** is wanted (it's trivial to implement).
- The decision boundary is irregular/local and you have enough data.
- Inference latency isn't critical, or you can afford an index (KD-tree/ANN).

**Avoid / prefer alternatives when:**
- **High dimensionality** (curse of dimensionality) → linear models, trees, or reduce dimensions first.
- **Large training sets with tight latency** (prediction is $O(np)$ per query) → eager models or ANN.
- Many irrelevant/noisy features → tree-based models handle these gracefully.
- You need an interpretable parametric model or extrapolation beyond the data range (KNN can't extrapolate).

## 10. Pitfalls & gotchas
- **Not scaling features** → distance dominated by large-scale features. The #1 mistake. ⭐
- **Even k in binary classification** → tie votes; use odd k (or distance weighting).
- **Including the query point itself** when "predicting" on training data → artificially perfect with k=1 (a leakage-flavored trap in evaluation).
- **Treating it as cheap at inference** — it's the opposite; storage + per-query cost scale with n.
- **Ignoring the curse of dimensionality** — KNN quietly degrades as p grows.
- **Class imbalance** skews the majority vote toward the frequent class → consider distance weighting or resampling.
- **Memory:** the whole training set must be kept around at serving time.

## 11. Interview questions
Must answer cold (⭐):
- ⭐ **Why is KNN called a "lazy" / non-parametric learner?** No training-time model fit; stores data and defers all work to prediction; no fixed parameter count, complexity grows with data. (§1)
- ⭐ **How does k affect bias and variance?** Small k = low bias/high variance (wiggly, overfits); large k = high bias/low variance (smooth, underfits). (§4)
- ⭐ **Why must you scale features before KNN?** Distance sums over features; large-scale features dominate otherwise. (§2.1)
- ⭐ **What is the curse of dimensionality and why does it hurt KNN specifically?** Distances concentrate / data gets sparse → "nearest" loses meaning; noisy features dilute distance. (§5)
- ⭐ **What's the time complexity of KNN at training vs. prediction?** Train O(1) (store data); predict O(n·p) per query (brute force). (§3)
- **How do you choose k?** Cross-validation; odd k for binary; ~√n as a start. (§4)
- **How would you speed up KNN prediction?** KD-tree/ball-tree (low-dim), approximate NN (ANN/HNSW/FAISS) for large/embedding data. (§3)
- **KNN vs. K-Means — totally different things, right?** Yes: KNN is *supervised classification* (k neighbors vote); K-Means is *unsupervised clustering* (k centroids). Common naming trap. (See [06](06-kmeans-pca.md).)
- **Can KNN do regression?** Yes — average (optionally distance-weighted) the neighbors' targets. (§2.2)
- **How do you handle ties / class imbalance?** Odd k, distance weighting, resampling, or deterministic tie-break. (§2.2, §10)

## 12. Connections
- Bias–variance framing shared with [regularization](01-regularization-and-optimization.md#1-intuition) and formalized in [statistics/02](../statistics/02-bias-variance-mle.md).
- Standardization is the same discipline used before [GD/regularization](01-regularization-and-optimization.md#34-learning-rate--convergence).
- Curse of dimensionality motivates [PCA](06-kmeans-pca.md).
- **Don't confuse with K-Means** ([06](06-kmeans-pca.md)) — supervised vote vs. unsupervised centroids.
- Cross-validation for choosing k: [evaluation/03](../evaluation/03-cross-validation-leakage.md).
- Flashcards: [flashcards/02-knn.md](../flashcards/02-knn.md).

---
### Sources
- *ISLR* — Ch. 2 (KNN intro, bias–variance), Ch. 4 (KNN classification, curse of dimensionality).
- Géron, *Hands-On ML* — KNN and the importance of feature scaling.
