# Support Vector Machines

> **Week:** 6 (light) · **Status:** ✅ written (read-through) · **From-scratch build:** no
> **One-line:** Find the **maximum-margin** separating hyperplane — the boundary as far as possible from the nearest points of each class; **soft margins** (hinge loss) tolerate some overlap, and the **kernel trick** gives nonlinear boundaries by computing inner products in a high-dimensional space *without ever building it*.

> 📖 **Read-through** (light slot). Goal: discuss SVMs intelligently — margin, hinge loss, kernel trick, support vectors — not implement from scratch.

## Table of contents
- [1. Intuition](#1-intuition)
- [2. The math](#2-the-math)
  - [2.1 Hard margin](#21-hard-margin)
  - [2.2 Soft margin & hinge loss](#22-soft-margin--hinge-loss)
  - [2.3 The kernel trick](#23-the-kernel-trick)
- [3. The algorithm](#3-the-algorithm)
- [4. With scikit-learn](#4-with-scikit-learn)
- [5. Assumptions](#5-assumptions)
- [6. When to use / not use](#6-when-to-use--not-use)
- [7. Pitfalls & gotchas](#7-pitfalls--gotchas)
- [8. Interview questions](#8-interview-questions)
- [9. Connections](#9-connections)

---

## 1. Intuition

Many lines can separate two classes; which is *best*? SVM's answer: the one with the **widest margin** — the largest empty "street" between the classes. A boundary jammed up against the points generalizes worse than one centered in the gap. The points that touch the edges of the street are the **support vectors**; they alone define the boundary, and **moving any other point doesn't change it**. That's the SVM's signature property: the solution is **sparse in the data**. ⭐

Real data overlaps, so the **soft-margin** version lets some points sit inside the margin or on the wrong side, paying a penalty. And when no straight line works at all, the **kernel trick** lifts the data into a higher-dimensional space where it *is* linearly separable — computing only inner products, never the lifted coordinates.

## 2. The math

### 2.1 Hard margin

With labels $y_i \in \{-1, +1\}$ and a hyperplane $w^\top x + b = 0$, the (functional) margin condition is $y_i(w^\top x_i + b) \ge 1$. The geometric margin width is $2/\lVert w \rVert$, so maximizing the margin = minimizing $\lVert w \rVert$:
$$
\min_{w,b} \tfrac{1}{2}\lVert w \rVert^2 \quad\text{s.t.}\quad y_i(w^\top x_i + b) \ge 1 \;\;\forall i
$$
A convex quadratic program → a **unique global optimum** (unlike, say, neural nets). Only the constraints that are *active* (the support vectors) matter.

### 2.2 Soft margin & hinge loss

Introduce slack $\xi_i \ge 0$ for violations, penalized by $C$:
$$
\min_{w,b,\xi} \tfrac{1}{2}\lVert w \rVert^2 + C\sum_i \xi_i \quad\text{s.t.}\quad y_i(w^\top x_i + b) \ge 1 - \xi_i,\; \xi_i \ge 0
$$
Equivalently, the unconstrained **hinge-loss + L2** form:
$$
\min_{w,b}\; \tfrac{1}{2}\lVert w\rVert^2 + C\sum_i \max\!\big(0,\; 1 - y_i(w^\top x_i + b)\big)
$$
- **Hinge loss** $\max(0, 1 - y\,f(x))$: zero once a point is correctly classified *beyond* the margin; linear penalty inside/over the margin. This is the source of sparsity — correctly-and-confidently classified points contribute nothing.
- **$C$ is the regularization knob (inverse strength):** large $C$ → punish violations hard → narrow margin, low bias / high variance (can overfit). Small $C$ → wider, more tolerant margin → more bias, less variance. ⭐

### 2.3 The kernel trick

In the **dual** formulation the data appears *only* through inner products $x_i^\top x_j$. Replace that dot product with a **kernel** $K(x_i, x_j) = \phi(x_i)^\top \phi(x_j)$ and you get a linear SVM in the lifted feature space $\phi(\cdot)$ — **without ever computing $\phi$**. That's the trick: nonlinear boundaries at the cost of a kernel evaluation. ⭐

Common kernels:
- **Linear:** $x_i^\top x_j$ — high-dimensional/sparse data (text).
- **Polynomial:** $(\gamma\, x_i^\top x_j + r)^d$.
- **RBF / Gaussian:** $\exp(-\gamma \lVert x_i - x_j \rVert^2)$ — the default flexible choice; $\gamma$ sets how local each point's influence is (large $\gamma$ → wiggly, overfit; small $\gamma$ → smooth).

## 3. The algorithm

Training solves the (dual) convex QP — classically via **SMO** (Sequential Minimal Optimization), which optimizes pairs of dual variables at a time. You don't need SMO's details for interviews; the takeaways:
- The solution is a weighted combination of **support vectors** only.
- Predict with $f(x) = \operatorname{sign}\!\big(\sum_{i\in SV} \alpha_i y_i K(x_i, x) + b\big)$.
- Kernelized training is roughly **O(n²)–O(n³)** in the number of samples → **doesn't scale to huge n** (use `LinearSVC`/SGD for that).

## 4. With scikit-learn

```python
from sklearn.svm import SVC, LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# RBF kernel — scale first; SVMs are distance-based
clf = make_pipeline(StandardScaler(),
                    SVC(kernel="rbf", C=1.0, gamma="scale"))
clf.fit(X_train, y_train)

# Large n / high-dim sparse (text): linear, much faster
lin = make_pipeline(StandardScaler(with_mean=False), LinearSVC(C=1.0))
```
Tune `C` and (for RBF) `gamma` together by cross-validation. `SVC(probability=True)` adds Platt-scaled probabilities (slower; see [calibration, evaluation/04](../evaluation/04-calibration-imbalance.md)). For regression, `SVR` uses an ε-insensitive tube.

## 5. Assumptions
- **Scaled features** — distances/inner products dominate, so standardize (the recurring variance-based-method rule, as with [KNN](02-knn.md) and [K-Means](06-kmeans-pca.md)).
- A **margin exists** in the (possibly kernel-lifted) space; soft margin handles overlap.
- **Binary** at core — multiclass via one-vs-rest / one-vs-one.

## 6. When to use / not use
**Use when:** small-to-medium $n$ with clear margins; **high-dimensional** data (text, genomics) where $d \gg n$ and linear SVMs shine; you want a strong, well-regularized, theoretically clean classifier. **Avoid when:** $n$ is very large (kernel SVMs scale poorly → trees/linear/SGD); you need **calibrated probabilities** out of the box (SVMs output signed distances, not probabilities); features are unscaled or the problem is better served by gradient boosting on tabular data.

## 7. Pitfalls & gotchas
- **Forgetting to scale** — wrecks RBF/poly kernels and margins. ⭐
- **Misreading C** — it's *inverse* regularization: big C overfits, small C underfits.
- **Slow on big n** — kernel SVMs are ~O(n²–n³); reach for `LinearSVC`/SGD instead.
- **Expecting probabilities** — SVMs give margins; probabilities need Platt scaling and can be poorly calibrated.
- **RBF `gamma` runaway** — too large → memorizes training points (overfit islands around each).
- **Thinking all points matter** — only the **support vectors** define the boundary.

## 8. Interview questions
Must answer cold (⭐):
- ⭐ **What does an SVM optimize?** The maximum-margin separating hyperplane — minimize $\lVert w\rVert$ subject to all points being correctly classified by a margin (soft margin allows penalized violations). (§2.1–2.2)
- ⭐ **What are support vectors?** The points on/inside the margin that define the boundary; removing any other point leaves the solution unchanged → sparse solution. (§1)
- ⭐ **What is the kernel trick?** The dual depends only on inner products; swap them for a kernel $K(x_i,x_j)=\phi(x_i)^\top\phi(x_j)$ to get a nonlinear boundary without computing the high-dimensional mapping $\phi$. (§2.3)
- ⭐ **What does C do?** Inverse regularization on margin violations: large C → narrow margin, overfit; small C → wide margin, underfit. (§2.2)
- ⭐ **What loss does a (soft-margin) SVM use?** Hinge loss $\max(0, 1 - y\,f(x))$ + L2 — zero penalty once correctly classified beyond the margin. (§2.2)
- **What does RBF `gamma` control?** The reach of each point's influence; large gamma → wiggly/overfit, small gamma → smooth. (§2.3)
- **Why scale features for an SVM?** It's distance/inner-product based; unscaled features dominate. (§5)
- **Why don't SVMs scale to huge datasets?** Kernel training is ~O(n²–n³); use linear SVM / SGD instead. (§3)
- **SVM vs. logistic regression?** Both linear classifiers; SVM uses hinge loss + max-margin (sparse in support vectors, no probabilities), logistic uses log loss (gives probabilities, all points contribute). (§9)
- **Do SVMs output probabilities?** Not natively — they give signed margin distances; Platt scaling adds probabilities. (§6)

## 9. Connections
- **Scaling discipline** shared with [KNN (Week 2)](02-knn.md) and [K-Means/PCA (Week 6)](06-kmeans-pca.md) — all distance/variance-based.
- **Hinge vs. log loss** contrast with [logistic regression (foundations)](00-foundations.md); both are linear-margin classifiers with different losses.
- **Regularization via C** echoes [Week 1](01-regularization-and-optimization.md) — same bias–variance dial, different parameterization.
- **Calibration** of SVM scores → [evaluation/04](../evaluation/04-calibration-imbalance.md).
- No flashcard deck yet (read-through); add cards if it becomes interview-relevant.

---
### Sources
- *ISLR* — Ch. 9 (Support Vector Machines).
- Géron, *Hands-On ML* — Support Vector Machines chapter (linear/nonlinear SVC, kernels, SVR).
