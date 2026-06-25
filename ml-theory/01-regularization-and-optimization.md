# Regularization & Optimization

> **Week:** 1 · **Status:** ✅ written · **From-scratch build:** add L2 to linear regression GD
> **One-line:** Regularization adds a penalty on model complexity to trade a little bias for a large reduction in variance; optimization (gradient descent and its variants) is how we actually find the parameters that minimize the (now penalized) loss.

## Table of contents
- [1. Intuition](#1-intuition)
- [2. The math](#2-the-math)
  - [2.1 Ridge (L2)](#21-ridge-l2)
  - [2.2 Lasso (L1)](#22-lasso-l1)
  - [2.3 Elastic Net](#23-elastic-net)
  - [2.4 Why L1 gives sparsity (the geometry)](#24-why-l1-gives-sparsity-the-geometry)
- [3. Optimization: gradient descent and friends](#3-optimization-gradient-descent-and-friends)
  - [3.1 Batch GD](#31-batch-gradient-descent)
  - [3.2 SGD](#32-stochastic-gradient-descent-sgd)
  - [3.3 Mini-batch GD](#33-mini-batch-gd)
  - [3.4 Learning rate & convergence](#34-learning-rate--convergence)
  - [3.5 Momentum, RMSProp, Adam (preview)](#35-momentum-rmsprop-adam-preview)
- [4. The algorithm](#4-the-algorithm)
- [5. From scratch](#5-from-scratch)
- [6. With scikit-learn](#6-with-scikit-learn)
- [7. Assumptions & preconditions](#7-assumptions--preconditions)
- [8. When to use / not use](#8-when-to-use--not-use)
- [9. Pitfalls & gotchas](#9-pitfalls--gotchas)
- [10. Interview questions](#10-interview-questions)
- [11. Connections](#11-connections)

---

## 1. Intuition

A model that fits the training data *too* closely memorizes noise and generalizes poorly — high **variance**. One of the cleanest ways to control variance is to discourage the model from using large coefficients. The reasoning: large coefficients mean the prediction swings violently when an input feature wiggles slightly, which is exactly the kind of sensitivity that overfits noise.

**Regularization** = "minimize the training error, *but* keep the weights small." We add a penalty term to the loss that grows as the weights grow. The strength of that penalty is a hyperparameter (commonly λ or, in sklearn, related to `alpha`/`C`).

The mental model:

```
total objective = (how badly I fit the data)  +  λ · (how complex/large my weights are)
                   └── fit term (bias↓)            └── penalty term (variance↓)
```

- Turn λ **up** → simpler model, more bias, less variance.
- Turn λ **down** → flexible model, less bias, more variance.

This is the **bias–variance tradeoff** made tunable by a single knob. (See [statistics/bias-variance.md](../statistics/02-bias-variance-mle.md).)

**Optimization** is the *separate* question of *how* we find the weights that minimize this objective. For linear/logistic regression we could sometimes solve in closed form, but the general-purpose, scalable answer is **gradient descent**: repeatedly step downhill along the negative gradient of the loss. Almost everything in modern ML is trained this way, so getting GD crisp here pays off for neural nets later.

> **Two outcomes, one hour:** the regularization penalty changes the *objective*; gradient descent is the *machinery*. Week 1 wires both into your existing linear regression.

---

## 2. The math

Setup: linear model $\hat{y} = X w + b$ with $n$ samples, $p$ features. Unregularized least-squares loss (MSE):

$$
J_{\text{MSE}}(w, b) = \frac{1}{n} \sum_{i=1}^{n} \left( y_i - \hat{y}_i \right)^2 = \frac{1}{n}\, \lVert y - (Xw + b) \rVert_2^2
$$

We add a penalty $R(w)$ on the **weights only** (we do *not* penalize the bias $b$ — penalizing the intercept would needlessly shrink predictions toward 0).

### 2.1 Ridge (L2)

$$
J_{\text{Ridge}}(w,b) = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2 \;+\; \lambda \lVert w \rVert_2^2,
\qquad \lVert w \rVert_2^2 = \sum_{j=1}^{p} w_j^2
$$

Gradient w.r.t. $w$ (this is what you'll code):

$$
\nabla_w J_{\text{Ridge}} = \underbrace{-\frac{2}{n} X^\top (y - \hat{y})}_{\text{MSE gradient}} \;+\; \underbrace{2\lambda w}_{\text{L2 term}}
$$

The L2 term simply adds $2\lambda w$ to each weight's gradient → at every step we pull each weight a bit toward zero, proportional to its current size. This is why L2 is called **weight decay**.

**Closed form** (Ridge has one, unlike Lasso):

$$
w = (X^\top X + \lambda' I)^{-1} X^\top y
$$

The $+\lambda' I$ also makes the matrix invertible even when $X^\top X$ is singular (e.g., more features than samples, or collinear features) — a practical bonus of Ridge.

### 2.2 Lasso (L1)

$$
J_{\text{Lasso}}(w,b) = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2 \;+\; \lambda \lVert w \rVert_1,
\qquad \lVert w \rVert_1 = \sum_{j=1}^{p} |w_j|
$$

The subgradient of $|w_j|$ is $\text{sign}(w_j)$, so:

$$
\nabla_w J_{\text{Lasso}} = -\frac{2}{n} X^\top (y - \hat{y}) \;+\; \lambda \,\text{sign}(w)
$$

Note the penalty gradient is **constant in magnitude** ($\pm\lambda$) regardless of how small $w_j$ is. That constant push is what can drive a weight *exactly* to 0 — producing **sparse** solutions (automatic feature selection). $|w|$ is non-differentiable at 0, so in practice Lasso is solved with **coordinate descent** or **proximal/soft-thresholding** methods rather than vanilla GD.

### 2.3 Elastic Net

A convex combination of both penalties:

$$
J_{\text{EN}} = \text{MSE} \;+\; \lambda \left( \alpha \lVert w \rVert_1 + \tfrac{1-\alpha}{2}\lVert w \rVert_2^2 \right)
$$

- $\alpha = 1$ → pure Lasso; $\alpha = 0$ → pure Ridge.
- **Why it exists:** Lasso struggles when features are highly correlated (it arbitrarily picks one and zeros the rest) and can select at most $n$ features when $p > n$. Elastic Net's L2 component encourages correlated features to be selected *together* (the "grouping effect") while L1 still does selection.

### 2.4 Why L1 gives sparsity (the geometry)

The classic picture: minimizing MSE subject to a budget on the penalty is equivalent to the penalized form. The MSE contours are ellipses around the OLS solution; the constraint region is:

- **L2:** a circle/ball $\{w : \lVert w \rVert_2 \le t\}$ — smooth, no corners.
- **L1:** a diamond $\{w : \lVert w \rVert_1 \le t\}$ — has **corners on the axes**.

The first point where an expanding ellipse touches the constraint region is the solution. The L1 diamond's corners lie on the axes (where some $w_j = 0$), so the contact point frequently lands exactly on an axis → that coefficient is zeroed. The L2 ball has no corners, so it shrinks coefficients toward 0 but essentially never makes them exactly 0.

```
        L2 (ridge)                     L1 (lasso)
         w2                              w2
         |   .--.                        |   /\
         | /      \                       |  /  \
   ------+(   OLS  )----- w1        ------+ <    > ----- w1   <- contact at a corner
         | \      /                       |  \  /            (w1 = 0  -> sparse)
         |   '--'                         |   \/
```

---

## 3. Optimization: gradient descent and friends

We want $\theta^\* = \arg\min_\theta J(\theta)$. Gradient descent iterates:

$$
\theta \leftarrow \theta - \eta \, \nabla_\theta J(\theta)
$$

where $\eta$ is the **learning rate** (step size). The gradient points uphill, so we subtract it to go downhill.

### 3.1 Batch gradient descent
Use **all** $n$ samples to compute the gradient at each step.
- ➕ Stable, smooth descent; deterministic.
- ➖ One step requires a full pass over the data — slow/expensive on large datasets; can't update online.

### 3.2 Stochastic gradient descent (SGD)
Use **one** randomly chosen sample per update.
- ➕ Very fast per step; can escape shallow local minima / saddle points thanks to noise; supports online learning.
- ➖ Noisy, jittery path; never fully settles (oscillates around the minimum) → usually needs a **decaying learning rate** to converge.

### 3.3 Mini-batch GD
Use a small batch (e.g., 32–256). The default in practice.
- Best of both: vectorized/GPU-friendly, less noisy than SGD, far cheaper than full batch.
- Batch size is itself a hyperparameter (affects noise, generalization, and hardware utilization).

| Variant | Samples/step | Speed/step | Noise | Typical use |
|---|---|---|---|---|
| Batch | n | slow | none | small data, convex problems |
| SGD | 1 | fast | high | online / streaming |
| Mini-batch | b | medium | medium | **the default**, deep learning |

### 3.4 Learning rate & convergence
The single most important hyperparameter.
- **Too small** → painfully slow convergence; may stall.
- **Too large** → overshoots, **diverges** (loss → ∞ / NaN), or oscillates.
- **Convergence checks:** stop when the loss change between iterations falls below a tolerance, when the gradient norm is tiny, or after a max number of epochs.
- **Schedules:** step decay, exponential decay, $1/t$ decay, cosine annealing, warmup. Decaying LR lets SGD actually settle into the minimum.
- **Feature scaling matters here:** if features have wildly different scales, the loss surface is a stretched bowl and a single global LR is too big in one direction and too small in another → slow, zig-zagging descent. **Standardize features before GD** (and Ridge/Lasso penalties also assume comparable scales). This is a top interview point.

### 3.5 Momentum, RMSProp, Adam (preview)
You'll go deep in Week 7, but the one-liners:
- **Momentum:** accumulate a velocity (EMA of past gradients) to power through flat regions and dampen oscillation. $v \leftarrow \beta v + (1-\beta)\nabla J;\ \theta \leftarrow \theta - \eta v$.
- **RMSProp:** divide the step by a running RMS of recent gradients → per-parameter adaptive LR; handles different feature scales.
- **Adam:** momentum + RMSProp combined (first and second moment estimates, with bias correction). The de-facto default for deep nets.

---

## 4. The algorithm

Mini-batch gradient descent for L2-regularized linear regression:

```
Input: X (n×p), y (n), learning_rate η, λ, n_epochs, batch_size b
Standardize X (per-feature: subtract mean, divide by std)   # critical
Initialize w = 0 (p), b = 0
for epoch in 1..n_epochs:
    shuffle the data
    for each mini-batch (Xb, yb) of size b:
        ŷ   = Xb · w + b
        err = ŷ - yb
        grad_w = (2/b) · Xbᵀ · err + 2λ·w        # L2 term; do NOT include b
        grad_b = (2/b) · sum(err)
        w = w - η · grad_w
        b = b - η · grad_b
    (optional) record loss; check convergence / decay η
return w, b
```

**Complexity:** per epoch $O(n p)$ time (one pass), $O(p)$ extra space. Closed-form Ridge is $O(p^3)$ for the inverse (fine for small $p$, bad for large $p$) — another reason GD scales better.

---

## 5. From scratch

Implementation: [`implementations/src/linear_regression.py`](../implementations/src/linear_regression.py) with a `penalty` parameter (`None` / `"l2"`), and tests in [`implementations/tests/test_linear_regression.py`](../implementations/tests/test_linear_regression.py).

Key design decisions mapped to the math:
- `penalty="l2"`, `lambda_` controls $\lambda$ → adds `2 * lambda_ * w` to `grad_w` only (bias excluded), exactly the Ridge gradient in §2.1.
- Features are standardized internally (or we assume the caller did it) — see §3.4.
- `fit(X, y)` runs the loop in §4; `predict(X)` returns $Xw + b$.
- The **loss history** is stored so we can plot convergence and sanity-check the learning rate.

> Week 1's concrete task from the plan: take your existing linear-regression GD and add the L2 term. That's a two-line change in the gradient plus a hyperparameter — the value is in *understanding* why those two lines reduce variance.

## 6. With scikit-learn

```python
from sklearn.linear_model import Ridge, Lasso, ElasticNet, SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Always scale before regularizing:
ridge = make_pipeline(StandardScaler(), Ridge(alpha=1.0))     # alpha == λ strength
lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.1))
enet  = make_pipeline(StandardScaler(), ElasticNet(alpha=0.1, l1_ratio=0.5))  # l1_ratio == α

# Generic GD optimizer for linear models (lets you pick the penalty & schedule):
sgd = make_pipeline(
    StandardScaler(),
    SGDRegressor(penalty="l2", alpha=1e-4, learning_rate="invscaling", eta0=0.01, max_iter=1000)
)
```

Hyperparameter notes:
- `alpha` ↑ ⇒ more regularization (more bias, less variance). Tune via `GridSearchCV`/`RidgeCV`/`LassoCV` with **cross-validation** ([evaluation/03-cross-validation.md](../evaluation/03-cross-validation-leakage.md)).
- `l1_ratio` is Elastic Net's $\alpha$ in §2.3.
- For logistic regression in sklearn the knob is `C = 1/λ` (inverse strength) — *smaller* `C` = *stronger* regularization. Easy to get backwards in interviews.

## 7. Assumptions & preconditions
- **Features should be on comparable scales** before applying L1/L2 (the penalty treats all coefficients with one λ). Standardize.
- Linear-model assumptions still apply for the base fit (linearity in parameters, etc.).
- Regularization assumes that **smaller weights generalize better** — true when overfitting is the problem, not when the model is underfit.

## 8. When to use / not use
- **Use Ridge** when you have many correlated features and want to keep all of them but shrink — the default safe regularizer.
- **Use Lasso** when you want **feature selection** / a sparse, interpretable model, and suspect many features are irrelevant.
- **Use Elastic Net** when $p \gg n$ or features are highly correlated and you still want some selection.
- **Don't crank λ** if the model is underfitting (high bias already) — you'll make it worse.
- **GD vs. closed form:** use closed-form (normal equation / `Ridge`) for small $p$; use GD/SGD for large datasets, streaming, or when no closed form exists (logistic regression, neural nets).

## 9. Pitfalls & gotchas
- **Forgetting to scale** before regularizing → the penalty unfairly hammers large-scale features. #1 mistake.
- **Penalizing the bias term** → shrinks predictions toward 0 for no good reason. Exclude $b$.
- **Scaling using test-set statistics** → data leakage. Fit the scaler on train only, inside CV folds (use a `Pipeline`).
- **`C` vs. `alpha` confusion** in sklearn (logistic uses inverse strength `C`).
- **Learning rate too high** → NaNs/divergence; too low → looks "stuck." Always plot the loss curve.
- **Comparing Lasso coefficients across runs** without fixing scaling/seed — they can flip among correlated features.
- **Expecting Lasso to be smooth/differentiable** — it isn't at 0; that's why it needs coordinate descent, not plain GD.

## 10. Interview questions
Must answer cold (⭐):
- ⭐ **L1 vs. L2 — difference and when to use each?** L1 = sparse/feature-selection (corners on axes); L2 = shrinkage of correlated features, has closed form, differentiable everywhere. Mention the geometry.
- ⭐ **Why does L1 produce sparsity but L2 doesn't?** Diamond corners on the axes vs. smooth ball; constant-magnitude L1 gradient pushes weights to exactly 0. (§2.4)
- ⭐ **What's the bias–variance effect of increasing λ?** More bias, less variance.
- ⭐ **Batch vs. SGD vs. mini-batch — tradeoffs?** (§3.1–3.3 table.)
- ⭐ **Why must we scale features before regularization / GD?** Single λ across coefficients; stretched loss surface slows GD. (§3.4)
- **What happens if the learning rate is too high / too low?** Diverge vs. crawl.
- **Why don't we regularize the intercept?**
- **Does Ridge ever set coefficients to exactly zero?** No (asymptotically approaches but doesn't reach).
- **What is weight decay and how does it relate to L2?** Same thing for plain SGD: the $2\lambda w$ term decays weights each step.
- **When would you prefer Elastic Net over Lasso?** Correlated features / $p > n$ / grouping effect.
- **How do you choose λ?** Cross-validation; plot validation error vs. λ (the "regularization path").

## 11. Connections
- Built on [foundations](00-foundations.md) (linear & logistic regression).
- The optimizer machinery feeds directly into [neural networks](07-neural-networks.md) (Adam, momentum, batch norm as implicit regularization).
- Bias–variance formalism: [statistics/02-bias-variance-mle.md](../statistics/02-bias-variance-mle.md).
- λ selection via [cross-validation](../evaluation/03-cross-validation-leakage.md).
- Flashcards: [flashcards/01-regularization-optimization.md](../flashcards/01-regularization-optimization.md).

---
### Sources
- Géron, *Hands-On ML* — Ch. 4 (Training Models: GD variants, Ridge/Lasso/Elastic Net, learning curves).
- *ISLR* — Ch. 6 (Linear Model Selection & Regularization), §6.2 shrinkage methods + the constraint-region figure.
