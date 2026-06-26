# Ensembles 2: Boosting

> **Week:** 5 · **Status:** ✅ written · **From-scratch build:** no (the base learner is the Week 3 tree)
> **One-line:** Fit weak learners **sequentially**, each one correcting the mistakes the current ensemble still makes — AdaBoost by **reweighting** hard examples, gradient boosting by **fitting the residuals** (the negative gradient of the loss). The headline effect is **bias reduction**.

## Table of contents
- [1. Intuition](#1-intuition)
- [2. The math](#2-the-math)
  - [2.1 AdaBoost](#21-adaboost)
  - [2.2 Gradient boosting: boosting as gradient descent in function space](#22-gradient-boosting-boosting-as-gradient-descent-in-function-space)
  - [2.3 XGBoost: the second-order refinement](#23-xgboost-the-second-order-refinement)
- [3. The algorithm](#3-the-algorithm)
- [4. The knobs & how to tune them](#4-the-knobs--how-to-tune-them)
- [5. From scratch (sketch)](#5-from-scratch-sketch)
- [6. With scikit-learn / XGBoost / LightGBM](#6-with-scikit-learn--xgboost--lightgbm)
- [7. Assumptions](#7-assumptions)
- [8. When to use / not use](#8-when-to-use--not-use)
- [9. Pitfalls & gotchas](#9-pitfalls--gotchas)
- [10. Interview questions](#10-interview-questions)
- [11. Connections](#11-connections)

---

## 1. Intuition

Bagging ([Week 4](04-ensembles-bagging.md)) takes a **low-bias, high-variance** learner (a deep tree) and averages the variance away. Boosting comes at it from the *opposite corner*: start with a **high-bias, low-variance** learner (a **shallow** tree — often a "stump" of depth 1), and add many of them **sequentially**, each new one focused on what the current ensemble still gets wrong. The ensemble's bias falls as you add learners.

> **The headline contrast:** bagging = **parallel**, independent deep trees, attacks **variance**. Boosting = **sequential**, dependent shallow trees, attacks **bias**. ⭐

Two flavors of "focus on the mistakes":
- **AdaBoost (Adaptive Boosting):** keep a weight on each *training example*. After each weak learner, **increase the weights of the misclassified points** so the next learner pays them more attention. Final prediction is a **weighted vote** of the learners, where accurate learners get more say.
- **Gradient Boosting:** at each step, fit the new learner to the **residual errors** of the current ensemble — more precisely, to the **negative gradient of the loss**. For squared error that gradient *is* the residual `y − F(x)`, so "fit the residuals" is literally gradient descent, one tree per step.

Because each tree depends on the ones before it, boosting is **inherently sequential** (you can't parallelize across trees the way a Random Forest does — though within a single tree's split-finding, modern libraries parallelize heavily).

## 2. The math

### 2.1 AdaBoost

Binary labels $y_i \in \{-1, +1\}$. Start with uniform weights $w_i = 1/n$. For $m = 1 \dots M$:

1. Fit a weak classifier $h_m(x) \in \{-1,+1\}$ to the data using the current weights.
2. Compute its **weighted error**:
$$
\varepsilon_m = \frac{\sum_i w_i \,\mathbb{1}[\,y_i \ne h_m(x_i)\,]}{\sum_i w_i}
$$
3. Compute the learner's **say** (amount of influence):
$$
\alpha_m = \frac{1}{2}\ln\!\frac{1 - \varepsilon_m}{\varepsilon_m}
$$
A learner with $\varepsilon_m \to 0$ gets large $\alpha_m$; one at $\varepsilon_m = 0.5$ (random) gets $\alpha_m = 0$; one *worse* than random gets a **negative** weight (its vote is flipped).
4. **Reweight** examples — up-weight the ones it got wrong, down-weight the rest, then renormalize:
$$
w_i \leftarrow w_i \, e^{\,\alpha_m \,\mathbb{1}[y_i \ne h_m(x_i)]}\quad\text{(equivalently } w_i e^{-\alpha_m y_i h_m(x_i)}\text{)}
$$

Final prediction: $\displaystyle F(x) = \operatorname{sign}\!\Big(\sum_{m} \alpha_m h_m(x)\Big)$.

> **The deep fact:** AdaBoost is exactly **forward stagewise additive modeling under the exponential loss** $L(y, F) = e^{-yF(x)}$. The reweighting and the $\alpha_m$ formula both fall out of greedily minimizing that loss one learner at a time. This is the bridge from AdaBoost to the general gradient-boosting view. ⭐

### 2.2 Gradient boosting: boosting as gradient descent in function space

Build the model additively: $F_m(x) = F_{m-1}(x) + \nu\, h_m(x)$, where $\nu$ is the **learning rate** (shrinkage). We want to minimize a differentiable loss $L(y, F)$ over the *function* $F$. The trick: treat the current predictions $F_{m-1}(x_i)$ as parameters and take a gradient-descent step. The negative gradient w.r.t. the prediction at each point,
$$
r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}},
$$
is the **"pseudo-residual"** — the direction that most reduces the loss. Fit the next tree $h_m$ to those pseudo-residuals, then add a shrunken version of it.

For **squared error** $L = \frac{1}{2}(y - F)^2$, the negative gradient is $r_{im} = y_i - F_{m-1}(x_i)$ — the **ordinary residual**. So "fit each tree to the residuals" is the squared-error special case of "fit each tree to the negative gradient." For log loss (classification) or absolute/Huber loss (robust regression), the pseudo-residuals differ, but the recipe is identical. That generality — **plug in any differentiable loss** — is the whole point of the gradient-boosting framework.

### 2.3 XGBoost: the second-order refinement

XGBoost (and LightGBM/CatBoost) sharpen gradient boosting in two ways worth knowing:

- **Second-order (Newton) step:** instead of using only the gradient $g_i$, use a **second-order Taylor expansion** of the loss, bringing in the Hessian $h_i$. The optimal leaf value and the split-gain formula are then closed-form in terms of $\sum g_i$ and $\sum h_i$. This converges faster and more accurately than first-order steps.
- **Regularized objective:** the training objective is *loss + a penalty on tree complexity*,
$$
\mathcal{L} = \sum_i L(y_i, F(x_i)) + \sum_m \Omega(h_m),\qquad \Omega(h) = \gamma T + \tfrac{1}{2}\lambda \sum_j w_j^2,
$$
where $T$ = number of leaves and $w_j$ = leaf weights. So **regularization is built into the split criterion itself**, not bolted on. Plus engineering: sparsity-aware splitting (native missing-value handling), column/row subsampling, cache-aware histograms.

You don't need the full derivation cold, but **"second-order gradient + a regularized objective baked into the split gain"** is the right one-sentence answer for *"why is XGBoost better than vanilla GBM?"* ⭐

## 3. The algorithm

```
GRADIENT BOOSTING (regression, squared error):
    F_0(x) = mean(y)                          # constant baseline
    for m = 1..M:
        r_i = y_i - F_{m-1}(x_i)              # residuals = negative gradient
        fit a SHALLOW tree h_m to (x_i, r_i)  # depth ~3-8, NOT a deep tree
        F_m(x) = F_{m-1}(x) + nu * h_m(x)     # nu = learning rate (shrinkage)
    return F_M

ADABOOST (classification):
    w_i = 1/n
    for m = 1..M:
        fit weak learner h_m using weights w
        eps_m = weighted error of h_m
        alpha_m = 0.5 * ln((1 - eps_m) / eps_m)
        w_i *= exp(alpha_m * [h_m(x_i) wrong]); renormalize
    predict: sign( sum_m alpha_m * h_m(x) )
```

The base learner is **deliberately weak** (shallow tree). A deep tree would have low bias already and nothing for subsequent learners to correct — and would overfit. Boosting's job is to *combine many weak, biased learners into one strong, low-bias one.*

## 4. The knobs & how to tune them

The two that matter most **trade off against each other** and define the regularization story:

- **`learning_rate` (shrinkage, $\nu$):** scales each tree's contribution. **Smaller = more robust, less overfitting, but needs more trees.** The classic recipe: set a small `learning_rate` (e.g., 0.05–0.1) and use **early stopping** to pick `n_estimators`.
- **`n_estimators` (number of boosting rounds):** unlike a Random Forest, **more trees CAN overfit** — the ensemble keeps reducing training loss and will eventually fit noise. Tune with **early stopping on a validation set**. ⭐
- **`max_depth` / number of leaves:** controls the interaction order each tree captures. Shallow (3–8) is standard. Deeper trees → fewer rounds but more overfitting risk.
- **`subsample` < 1.0 (Stochastic Gradient Boosting):** fit each tree on a random fraction of rows; adds bagging-style variance reduction and speed. `colsample_bytree` does the same for features.
- **`reg_lambda` / `reg_alpha` / `gamma`** (XGBoost): L2 / L1 on leaf weights / min-split-gain — the explicit regularizers from §2.3.

> **Mental model:** `learning_rate` and `n_estimators` are two ends of the same rope — lower one, raise the other. Early stopping ties the knot for you.

## 5. From scratch (sketch)

Not a canonical from-scratch build, but gradient boosting on squared error is strikingly short on top of the [Week 3 tree](../implementations/src/decision_tree.py) (a regression variant):

```python
class GradientBoostingRegressor:
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3):
        self.n_estimators, self.lr, self.max_depth = n_estimators, learning_rate, max_depth

    def fit(self, X, y):
        self.F0 = y.mean()                       # constant baseline (§3)
        Fm = np.full(len(y), self.F0)
        self.trees = []
        for _ in range(self.n_estimators):
            residual = y - Fm                    # negative gradient for squared error (§2.2)
            tree = DecisionTreeRegressor(max_depth=self.max_depth)
            tree.fit(X, residual)                # fit the NEXT tree to the residuals
            Fm += self.lr * tree.predict(X)      # shrunken additive update
            self.trees.append(tree)
        return self

    def predict(self, X):
        return self.F0 + self.lr * sum(t.predict(X) for t in self.trees)
```
The entire idea — *baseline → residual → fit tree to residual → add shrunken tree → repeat* — is right there. Swapping the loss only changes the `residual` line (to the negative gradient of that loss).

## 6. With scikit-learn / XGBoost / LightGBM

```python
# sklearn's modern, fast histogram-based GBM (LightGBM-like; the one to reach for)
from sklearn.ensemble import HistGradientBoostingClassifier
gbm = HistGradientBoostingClassifier(
    learning_rate=0.1,
    max_iter=1000,             # upper bound on rounds...
    early_stopping=True,       # ...actually stopped here on internal validation
    max_depth=None, max_leaf_nodes=31,
    l2_regularization=1.0,
)
gbm.fit(X_train, y_train)

# XGBoost — the competition/industry workhorse
import xgboost as xgb
clf = xgb.XGBClassifier(
    n_estimators=2000, learning_rate=0.05, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    reg_lambda=1.0, eval_metric="logloss",
    early_stopping_rounds=50,
)
clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# AdaBoost (mostly of historical/teaching interest now)
from sklearn.ensemble import AdaBoostClassifier
ada = AdaBoostClassifier(n_estimators=200, learning_rate=1.0)  # default base = depth-1 stump
```
In practice on tabular data, **gradient boosting (XGBoost / LightGBM / `HistGradientBoosting`) is the default high-accuracy choice.** AdaBoost is worth understanding for the theory and interviews but is rarely the production pick.

## 7. Assumptions

- **Base learner should be weak / high-bias** (shallow trees). The opposite of bagging, which wants strong/high-variance learners.
- **The loss should be differentiable** for gradient boosting (so a negative gradient exists). AdaBoost specifically minimizes exponential loss.
- **Relatively clean labels.** Because boosting *up-weights* hard examples, **mislabeled points and outliers can dominate** — the ensemble pours capacity into fitting noise. (AdaBoost is especially sensitive; robust losses like Huber mitigate this in GBM.)
- No feature scaling needed (tree-based, [Week 3 §7](03-decision-trees.md#7-assumptions)).

## 8. When to use / not use

**Use when:**
- You want **maximum accuracy on tabular / structured data** — this is the family that wins Kaggle and most real tabular benchmarks. ⭐
- Mixed feature types, non-linear interactions, moderate-to-large $n$.
- You can afford to **tune** (learning rate × rounds × depth) and to spend the training time.

**Avoid / prefer alternatives when:**
- You want a **strong baseline with almost no tuning** → Random Forest ([Week 4](04-ensembles-bagging.md)) is more forgiving.
- **Noisy labels / heavy outliers** and you can't clean them → boosting can chase the noise; RF is more robust.
- **Very high-dimensional sparse** data (text) → linear models.
- **Strict training-time budget** or you need trees trained in parallel → RF parallelizes across trees; boosting is sequential.
- Unstructured data (images, audio, long text) → deep nets, not trees.

## 9. Pitfalls & gotchas
- **"More trees never hurts" — FALSE for boosting.** That's a *bagging* property. In boosting, too many rounds overfits; control it with **early stopping**. ⭐
- **Using deep trees as the base learner** — boosting wants *weak* learners; deep base trees overfit fast and defeat the purpose.
- **Forgetting the learning-rate ↔ n_estimators tradeoff** — lowering the learning rate without raising the number of rounds just underfits.
- **Sensitivity to noisy labels / outliers** — boosting up-weights the hardest (often mislabeled) points; use robust losses or clean the data.
- **Tuning on the test set via early stopping** — the early-stopping validation set is part of training; keep a final untouched test set for the honest number. (See [evaluation/03](../evaluation/03-cross-validation-leakage.md).)
- **Calibration direction** — boosting tends to push probabilities toward **0/1 (over-confident)**, the opposite of bagging's pull toward the middle; recalibrate if you need true probabilities. (See [evaluation/04](../evaluation/04-calibration-imbalance.md).) ⭐
- **Expecting parallel training across trees** — you can't; the sequential dependence is fundamental (only intra-tree work parallelizes).
- **Confusing AdaBoost reweighting with gradient boosting residuals** — same goal (focus on mistakes), different mechanism (sample weights vs. fitting the negative gradient).

## 10. Interview questions
Must answer cold (⭐):
- ⭐ **Bagging vs. boosting?** Bagging: parallel, independent, **deep** trees, reduces **variance**. Boosting: sequential, dependent, **shallow** trees, each corrects the last, reduces **bias**. (§1)
- ⭐ **How does AdaBoost work?** Weight examples; fit a weak learner; up-weight the misclassified; give each learner a say $\alpha_m = \frac12\ln\frac{1-\varepsilon}{\varepsilon}$; final prediction is the weighted vote. It minimizes exponential loss via forward stagewise modeling. (§2.1)
- ⭐ **What does gradient boosting actually fit each round?** The **negative gradient of the loss** (the pseudo-residual). For squared error that's the ordinary residual $y - F_{m-1}(x)$ — so "fit the residuals" is the squared-error special case. (§2.2)
- ⭐ **Why is it called "gradient" boosting?** Each added tree is a gradient-descent step in *function space*: it points in the direction (negative gradient) that most reduces the loss at each training point. (§2.2)
- ⭐ **Can boosting overfit with too many trees?** **Yes** — unlike a Random Forest. Use early stopping on a validation set. (§4, §9)
- ⭐ **learning_rate vs. n_estimators?** Two ends of one rope: smaller learning rate needs more trees, generalizes better. Standard recipe: small rate + early stopping. (§4)
- ⭐ **Why is XGBoost better than vanilla GBM?** Second-order (Newton) updates using the Hessian, a regularized objective baked into the split gain ($\gamma T + \frac12\lambda\sum w_j^2$), plus engineering (histograms, missing-value handling, subsampling). (§2.3)
- **Why a weak (shallow) base learner?** A deep tree already has low bias and overfits; boosting's job is to combine many high-bias weak learners into one low-bias strong one. (§1, §3)
- **When would you pick Random Forest over boosting?** Want a robust, low-tuning baseline; noisy labels/outliers; need parallel training; or you just need a solid number fast. (§8)
- **Why is boosting sensitive to outliers/noisy labels?** It up-weights the hardest-to-fit points, which are often the mislabeled/outlier ones, so the ensemble spends capacity fitting noise. (§7, §9)
- **How are boosted probabilities calibrated?** Tend to be over-confident (pushed toward 0/1); recalibrate (Platt/isotonic) if you need real probabilities. (§9, [evaluation/04](../evaluation/04-calibration-imbalance.md))
- **What is stochastic gradient boosting?** Subsample rows (and/or columns) per tree — adds variance reduction and speed on top of boosting. (§4)

## 11. Connections
- **Base learner is the [Week 3 decision tree](03-decision-trees.md)** (shallow this time) — the direct contrast is that [bagging/RF (Week 4)](04-ensembles-bagging.md) uses *deep* trees.
- **The bias–variance contrast with bagging** is the spine of both ensemble notes — see [statistics/02](../statistics/02-bias-variance-mle.md) and [regularization (Week 1)](01-regularization-and-optimization.md) for shrinkage as regularization.
- **Gradient boosting = gradient descent** in function space — ties back to optimization in [Week 1](01-regularization-and-optimization.md).
- **Early stopping requires honest validation** — [evaluation/03 (cross-validation & leakage)](../evaluation/03-cross-validation-leakage.md).
- **Calibration of boosted probabilities** — [evaluation/04](../evaluation/04-calibration-imbalance.md).
- Flashcards: [flashcards/05-ensembles-boosting.md](../flashcards/05-ensembles-boosting.md).

---
### Sources
- *ISLR* — Ch. 8.2.3 (Boosting).
- Géron, *Hands-On ML* — Ensemble Learning and Random Forests chapter (AdaBoost, Gradient Boosting, stacking) + the gradient boosting / XGBoost discussion.
- Friedman (2001), *Greedy Function Approximation: A Gradient Boosting Machine*; Chen & Guestrin (2016), *XGBoost*.
