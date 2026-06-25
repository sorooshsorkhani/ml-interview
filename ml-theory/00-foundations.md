# Foundations: Linear & Logistic Regression + Core ML Theory

> **Week:** done (prior to this plan) · **Status:** ✅ recap · **From-scratch builds:** Linear Regression, Logistic Regression
> **One-line:** The two workhorse linear models and the core vocabulary (supervised learning, loss, generalization, bias–variance) that everything else builds on.

This is a **consolidation/recap** note for material already covered, kept here so the repo is self-contained and so the canonical interview points are in one place. Deep regularization & optimization live in [01](01-regularization-and-optimization.md).

## Table of contents
- [1. The supervised learning setup](#1-the-supervised-learning-setup)
- [2. Linear Regression](#2-linear-regression)
- [3. Logistic Regression](#3-logistic-regression)
- [4. Core theory vocabulary](#4-core-theory-vocabulary)
- [5. Interview questions](#5-interview-questions)
- [6. Connections](#6-connections)

---

## 1. The supervised learning setup
- **Data:** $(x_i, y_i)$ pairs; learn $f: X \to Y$.
- **Regression** → continuous $y$; **classification** → discrete $y$.
- **Loss function** measures error on one example; **cost/objective** aggregates over the dataset (often + a regularization term).
- **Goal is generalization** — low error on *unseen* data, not just training data. Hence train/val/test splits and cross-validation.
- **Parametric** (fixed #params, e.g. linear regression) vs **non-parametric** (complexity grows with data, e.g. KNN).

## 2. Linear Regression
**Model:** $\hat{y} = w^\top x + b$.

**Loss (MSE):** $J(w,b) = \frac{1}{n}\sum_i (y_i - \hat{y}_i)^2$.

**Two ways to fit:**
- **Normal equation (closed form):** $w = (X^\top X)^{-1} X^\top y$. Exact, $O(p^3)$, breaks if $X^\top X$ singular (collinearity).
- **Gradient descent:** scalable, the general tool. Gradient: $\nabla_w J = -\frac{2}{n}X^\top(y-\hat{y})$.

**Key assumptions (the "LINE" mnemonic):** **L**inearity (in parameters), **I**ndependence of errors, **N**ormality of residuals (for inference/CI), **E**qual variance (homoscedasticity). Plus no perfect multicollinearity.

**From scratch:** [`implementations/src/linear_regression.py`](../implementations/src/linear_regression.py).

## 3. Logistic Regression
**Model:** $\hat{p} = \sigma(w^\top x + b)$, with sigmoid $\sigma(z) = \frac{1}{1+e^{-z}}$. Outputs a probability; predict class 1 if $\hat p \ge$ threshold (default 0.5).

**Loss (binary cross-entropy / log loss):**
$$J = -\frac{1}{n}\sum_i \big[ y_i \log \hat p_i + (1-y_i)\log(1-\hat p_i)\big]$$
Derived from **MLE** of a Bernoulli likelihood. It's **convex**, so GD finds the global optimum.

**Gradient (clean result):** $\nabla_w J = \frac{1}{n} X^\top(\hat p - y)$ — structurally identical to linear regression's, just with $\hat p$ from the sigmoid. Worth being able to derive.

**Multiclass:** softmax / multinomial logistic regression (cross-entropy generalizes).

**Why not MSE for classification?** Non-convex with sigmoid, and penalizes confident-correct/wrong predictions poorly; log loss is the principled (MLE) choice.

**From scratch:** [`implementations/src/logistic_regression.py`](../implementations/src/logistic_regression.py).

## 4. Core theory vocabulary
- **Bias–variance tradeoff:** total error ≈ bias² + variance + irreducible noise. Underfit = high bias; overfit = high variance. (Formal in [statistics/02](../statistics/02-bias-variance-mle.md).)
- **Overfitting / underfitting** and the role of model capacity & regularization.
- **Train / validation / test** discipline; **cross-validation**; **data leakage**. ([evaluation/03](../evaluation/03-cross-validation-leakage.md))
- **Parametric vs non-parametric; generative vs discriminative** (logistic = discriminative; Naive Bayes = generative).
- **The curse of dimensionality** (matters for KNN, distance metrics).
- **Evaluation metrics** beyond accuracy (precision/recall/F1/ROC-AUC). ([evaluation/01](../evaluation/01-classification-metrics.md))

## 5. Interview questions
- ⭐ Derive the logistic regression gradient from the log-loss.
- ⭐ Why log loss and not MSE for classification?
- ⭐ State the linear regression assumptions and how to check/violate each.
- ⭐ When is the normal equation preferable to gradient descent, and vice versa?
- What makes a model parametric vs non-parametric? Generative vs discriminative?
- How does the decision threshold trade off precision and recall?
- What is multicollinearity and why does it hurt linear regression (but less so prediction than inference)?

## 6. Connections
- Regularizing these models: [01-regularization-and-optimization.md](01-regularization-and-optimization.md).
- Logistic loss reappears in [neural networks](07-neural-networks.md) as the output-layer loss.
- MLE framing: [statistics/02-bias-variance-mle.md](../statistics/02-bias-variance-mle.md).

---
### Sources
- Géron, *Hands-On ML* — Ch. 4. · *ISLR* — Ch. 3 (linear), Ch. 4 (logistic).
