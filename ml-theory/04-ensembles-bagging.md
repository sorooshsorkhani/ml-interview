# Ensembles 1: Bagging & Random Forests

> **Week:** 4 · **Status:** ✅ written · **From-scratch build:** no (builds on the Week 3 tree)
> **One-line:** Train many high-variance models on bootstrap samples and average them to cut variance; Random Forests add per-split feature subsampling to *decorrelate* the trees, which makes the averaging work much better.

## Table of contents
- [1. Intuition](#1-intuition)
- [2. The math](#2-the-math)
  - [2.1 Why averaging reduces variance](#21-why-averaging-reduces-variance)
  - [2.2 Bootstrap & the OOB set](#22-bootstrap--the-oob-set)
- [3. The algorithm](#3-the-algorithm)
  - [3.1 Bagging](#31-bagging)
  - [3.2 Random Forest](#32-random-forest)
- [4. Feature importance](#4-feature-importance)
- [5. From scratch (sketch)](#5-from-scratch-sketch)
- [6. With scikit-learn](#6-with-scikit-learn)
- [7. Assumptions](#7-assumptions)
- [8. When to use / not use](#8-when-to-use--not-use)
- [9. Pitfalls & gotchas](#9-pitfalls--gotchas)
- [10. Interview questions](#10-interview-questions)
- [11. Connections](#11-connections)

---

## 1. Intuition

A single decision tree ([Week 3](03-decision-trees.md)) is **low-bias, high-variance**: it can fit almost anything, but small changes in the data reshape it completely. The ensemble idea: **don't fix the tree — average many of them.** If each tree is a noisy estimate of the truth and their errors are partly independent, the *average* keeps the signal and cancels the noise. That's variance reduction at (almost) no cost to bias.

Two ingredients build up to Random Forests:
- **Bagging (Bootstrap AGGregatING):** train each model on a different **bootstrap sample** (sample n points with replacement) of the training set, then average their predictions (regression) or majority-vote (classification).
- **Random Forest = bagged trees + a twist:** at *every split*, consider only a **random subset of features**. This stops all trees from keying off the same one or two dominant features, so the trees become **less correlated** — and averaging correlated estimators barely reduces variance, while averaging decorrelated ones reduces it a lot.

> **The headline:** bagging attacks variance; Random Forests attack the *correlation between trees* so the variance reduction actually pays off. Contrast with boosting ([Week 5](05-ensembles-boosting.md)), which attacks **bias** by adding trees sequentially.

## 2. The math

### 2.1 Why averaging reduces variance

Take $B$ estimators each with variance $\sigma^2$. If they were **independent**, the variance of their average is
$$
\operatorname{Var}\!\left(\frac{1}{B}\sum_{b=1}^{B} X_b\right) = \frac{\sigma^2}{B}
$$
— more trees → lower variance, bias unchanged. But bagged trees are **not** independent (they share most of the data). With pairwise correlation $\rho$, the variance of the average is:
$$
\rho\,\sigma^2 + \frac{1-\rho}{B}\,\sigma^2
$$
As $B \to \infty$ the second term vanishes but the first, $\rho\sigma^2$, **does not**. So the floor on variance is set by how correlated the trees are. **This single formula is the entire justification for Random Forests:** to drive variance down you must lower $\rho$ — which is exactly what random feature subsampling does. ⭐

### 2.2 Bootstrap & the OOB set

A **bootstrap sample** draws $n$ points from the training set *with replacement*. The probability a given point is **not** picked in one draw is $(1 - 1/n)$, so the probability it's missing from the whole sample is
$$
\left(1 - \tfrac{1}{n}\right)^n \xrightarrow{n\to\infty} e^{-1} \approx 0.368
$$
So each tree sees ~63% of the unique points and **~37% are "out-of-bag" (OOB)** for that tree. Those OOB points are a free held-out set: predict each point using only the trees that didn't train on it → the **OOB error**, an unbiased generalization estimate *without a separate validation split or CV*. ⭐

## 3. The algorithm

### 3.1 Bagging
```
BAGGING(training set D, B):
    for b = 1..B:
        D_b = bootstrap sample of D (n draws with replacement)
        train model M_b on D_b              # usually an unpruned (deep) tree
    PREDICT(x):
        regression:     average  M_b(x)
        classification: majority vote of M_b(x)   (or average predicted probs — "soft voting")
```
Bagging works best with **high-variance, low-bias** base learners (deep trees) — there's lots of variance to average away. Grow the trees **deep / unpruned** on purpose; the ensemble handles overfitting.

### 3.2 Random Forest
Same as bagging, **plus** at each node consider only a random subset of `max_features` features when searching for the best split:
```
RANDOM FOREST = BAGGING with deep trees, AND
    at every split: pick m of the p features at random, split only among those.
    typical m: sqrt(p) for classification, p/3 for regression.
```
That one change decorrelates the trees (§2.1). Everything else — bootstrap, deep trees, average/vote, OOB — is inherited from bagging.

> **Extra-Trees (Extremely Randomized Trees):** go further — also pick the split *threshold* at random (not the best one). Even more decorrelation/variance reduction, slightly more bias, faster to train.

## 4. Feature importance

Random Forests give two importance measures:
- **Impurity-based (MDI / "Gini importance"):** total weighted impurity decrease attributed to each feature across all trees. Fast (free at training) but **biased toward high-cardinality / continuous features** — same caveat as the single tree ([Week 3 §9](03-decision-trees.md#9-pitfalls--gotchas)).
- **Permutation importance:** shuffle one feature's values in held-out (or OOB) data and measure the drop in accuracy. Model-agnostic and more trustworthy, but costs extra passes. **Prefer this when importance actually drives a decision.** ⭐

Caveat for both: with **correlated features**, importance gets *split* between them, so a genuinely important feature can look weak because its twin absorbs the credit.

## 5. From scratch (sketch)

No standalone file — a Random Forest is "many [Week 3 trees](../implementations/src/decision_tree.py) + bootstrap + feature subsampling + vote." If you were to build it on top of `DecisionTreeClassifier`:

```python
class RandomForest:
    def fit(self, X, y):
        self.trees = []
        n = len(X)
        for _ in range(self.n_estimators):
            idx = np.random.choice(n, n, replace=True)        # bootstrap sample (§2.2)
            tree = DecisionTreeClassifier(max_depth=None)      # deep, unpruned (§3.1)
            tree.fit(X[idx], y[idx])                            # (feature subsampling would live INSIDE the tree's split search)
            self.trees.append(tree)
        return self

    def predict(self, X):
        preds = np.array([t.predict(X) for t in self.trees])   # (B, n_samples)
        # majority vote per column
        from scipy.stats import mode
        return mode(preds, axis=0).mode.ravel()
```
The one piece our Week 3 tree lacks is **per-split `max_features` subsampling** (§3.2) — that would go inside `_best_split`. Worth knowing you *could* build it; not on the canonical from-scratch checklist.

## 6. With scikit-learn

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=300,        # more trees = better & more stable, with diminishing returns; never hurts accuracy
    max_features="sqrt",     # the decorrelation knob (§3.2)
    max_depth=None,          # let trees grow deep; the ensemble controls variance
    min_samples_leaf=1,
    bootstrap=True,
    oob_score=True,          # free OOB generalization estimate (§2.2)
    n_jobs=-1,               # embarrassingly parallel — trees are independent
    random_state=0,
)
rf.fit(X_train, y_train)
rf.oob_score_                 # ~ validation accuracy, no CV needed
rf.feature_importances_       # impurity-based (§4) — pair with permutation_importance for trust
```
Key hyperparameters: `n_estimators` (more is safe, just slower), `max_features` (the RF-specific knob), `max_depth` / `min_samples_leaf` (mild regularization), `bootstrap` + `oob_score`. **Unlike a single tree, you rarely need to prune** — the ensemble averages variance away.

## 7. Assumptions

Inherits the tree's "assumption-light" nature, plus:
- **Base learners must be unstable / high-variance** for bagging to help (deep trees ✓; a linear model bagged barely changes — low variance to begin with).
- **Errors should be at least partly independent** — if every tree makes the same mistakes ($\rho \approx 1$), averaging does nothing (§2.1). Feature subsampling is what buys the independence.
- No feature scaling needed (tree-based, [Week 3 §7](03-decision-trees.md#7-assumptions)).

## 8. When to use / not use

**Use when:**
- You want a **strong, low-effort baseline** that "just works" — RF is the classic go-to on tabular data with minimal tuning.
- **Tabular / mixed-type** features, non-linear interactions, moderate dimensionality.
- You value the **OOB estimate** and **feature importances** out of the box.
- Robustness to outliers/noise and little hyperparameter fuss matter.

**Avoid / prefer alternatives when:**
- You need the **last few points of accuracy** on structured data → **gradient boosting** ([Week 5](05-ensembles-boosting.md)) usually wins.
- **Tight inference latency / memory** — hundreds of deep trees are big and slowish at predict time.
- **Very high-dimensional sparse** data (text) → linear models.
- You need an **interpretable** single model or smooth **extrapolation** (forests, like trees, predict flat outside the training range).

## 9. Pitfalls & gotchas
- **Thinking more trees can overfit** — they can't (much); RF accuracy plateaus as `n_estimators` grows, it doesn't degrade. The real overfitting knob is tree depth, and the ensemble already tames that. ⭐
- **Confusing bagging with boosting** — bagging = parallel, independent trees, reduces **variance**; boosting = sequential, each tree fixes the last, reduces **bias**. (Classic interview probe.) ⭐
- **Trusting impurity feature importances blindly** — biased to high-cardinality features; split among correlated features. Use permutation importance. (§4)
- **Forgetting why feature subsampling exists** — it's to *decorrelate* trees (§2.1), not just for speed.
- **Using OOB *and* a separate test set redundantly** — OOB already estimates generalization; still keep a final untouched test set for the honest number.
- **Treating it as a black box for extrapolation** — flat predictions beyond the data range.
- **Memory/latency at serving** — many deep trees; consider fewer/shallower trees or boosting if constrained.

## 10. Interview questions
Must answer cold (⭐):
- ⭐ **What problem does bagging solve, and how?** It reduces **variance** by training models on bootstrap samples and averaging/voting; independent errors cancel, signal survives. Bias roughly unchanged. (§1, §2.1)
- ⭐ **Why does a Random Forest randomly subsample features at each split?** To **decorrelate** the trees. Averaging correlated estimators leaves a variance floor of $\rho\sigma^2$; lowering $\rho$ is the only way past it. (§2.1, §3.2)
- ⭐ **Bagging vs. boosting?** Bagging: parallel, independent, deep trees, cuts **variance**. Boosting: sequential, each learner corrects the previous, shallow trees, cuts **bias**. (§1, [Week 5](05-ensembles-boosting.md))
- ⭐ **What is OOB error and why is it useful?** Each bootstrap omits ~37% of points; predict those with the trees that didn't see them → an **unbiased generalization estimate for free**, no separate CV. (§2.2)
- ⭐ **Why ~37%?** $(1-1/n)^n \to e^{-1} \approx 0.368$ — chance a point is left out of a bootstrap sample. (§2.2)
- **Can a Random Forest overfit if you add more trees?** No (essentially) — accuracy plateaus; overfitting comes from tree depth, which the ensemble averages out. (§9)
- **What's the variance of an average of B correlated trees?** $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$ — the $\rho\sigma^2$ floor motivates decorrelation. (§2.1)
- **Default `max_features`?** ~$\sqrt{p}$ for classification, $p/3$ for regression. (§3.2)
- **How does RF give feature importance, and what's the catch?** Impurity decrease (MDI) — biased toward high-cardinality features and split among correlated ones; prefer permutation importance. (§4)
- **What are Extra-Trees?** Random Forest that *also* randomizes split thresholds → more decorrelation, a bit more bias, faster. (§3.2)
- **Why grow the trees deep in a forest but prune a standalone tree?** A lone deep tree overfits; in a forest the averaging removes that variance, so deep trees give the low bias you want. (§3.1)

## 11. Connections
- **Base learner is the [Week 3 decision tree](03-decision-trees.md)** — RF exists specifically to fix the single tree's high variance ([Week 3 §4](03-decision-trees.md#4-overfitting-pruning--regularization)).
- **The other ensemble family is [boosting (Week 5)](05-ensembles-boosting.md)** — bias reduction, the direct contrast to bagging's variance reduction.
- **Bias–variance** is the whole story here — see [statistics/02](../statistics/02-bias-variance-mle.md) and [regularization (Week 1)](01-regularization-and-optimization.md).
- **Feature-importance bias** mirrors the single tree's ([Week 3 §9](03-decision-trees.md#9-pitfalls--gotchas)).
- **OOB vs. CV** for model assessment: [evaluation/03](../evaluation/03-cross-validation-leakage.md).
- Flashcards: [flashcards/04-ensembles-bagging.md](../flashcards/04-ensembles-bagging.md).

---
### Sources
- *ISLR* — Ch. 8.2 (Bagging, Random Forests, OOB, variable importance).
- Géron, *Hands-On ML* — Ensemble Learning and Random Forests chapter (voting, bagging/pasting, RF, Extra-Trees, feature importance).
