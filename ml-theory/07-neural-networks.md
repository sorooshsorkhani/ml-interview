# Neural Network Fundamentals

> **Week:** 7 · **Status:** ✅ written · **From-scratch build:** yes — [`MLP`](../implementations/src/mlp.py) (canonical #6, one hidden layer, numpy)
> **One-line:** Stack **affine transforms** ($Wx+b$) and **nonlinear activations** into layers, then train the weights by **backpropagation** — the chain rule applied layer-by-layer to get gradients — plus gradient descent. A one-hidden-layer net with a nonlinearity is already a **universal function approximator**; depth makes that approximation *efficient*.

## Table of contents
- [1. Intuition](#1-intuition)
- [2. The math](#2-the-math)
  - [2.1 Forward pass](#21-forward-pass)
  - [2.2 Activations](#22-activations)
  - [2.3 Loss functions](#23-loss-functions)
  - [2.4 Backpropagation](#24-backpropagation)
  - [2.5 Optimizers](#25-optimizers)
- [3. The algorithm](#3-the-algorithm)
- [4. From scratch](#4-from-scratch)
- [5. With scikit-learn / PyTorch](#5-with-scikit-learn--pytorch)
- [6. Deep-learning regularization (Week 7 light slot)](#6-deep-learning-regularization-week-7-light-slot)
- [7. Assumptions](#7-assumptions)
- [8. When to use / not use](#8-when-to-use--not-use)
- [9. Pitfalls & gotchas](#9-pitfalls--gotchas)
- [10. Interview questions](#10-interview-questions)
- [11. Connections](#11-connections)

---

## 1. Intuition

A neural network is just **logistic regression stacked and bent**. Logistic regression computes one affine score $w^\top x + b$ and squashes it. A neural net does that many times in parallel (a **layer** of units), feeds the outputs through a **nonlinearity**, and repeats. The nonlinearity is the whole point: stacking affine maps without it collapses to a single affine map ($W_2(W_1 x) = (W_2 W_1)x$), so the network could only ever draw straight boundaries. Insert a nonlinearity between layers and the network can **bend** decision boundaries and approximate essentially any function.

- **Hidden layer = learned features.** Earlier layers learn representations (edges → textures → parts → objects, in vision); later layers combine them. You're learning the feature engineering instead of hand-crafting it — that's the headline advantage over classical ML on unstructured data (text, images, audio).
- **Universal approximation theorem:** a single hidden layer with enough units can approximate any continuous function on a compact set arbitrarily well. But "enough units" can be exponential; **depth** lets you represent the same functions far more parameter-efficiently. ⭐
- **Training is just gradient descent** on a (non-convex) loss; the only new machinery vs. logistic regression is **backprop**, an efficient way to compute the gradient of the loss w.r.t. every weight.

## 2. The math

### 2.1 Forward pass

For a one-hidden-layer network (input $x \in \mathbb{R}^d$, hidden size $h$, $c$ outputs):

$$
\begin{aligned}
z^{(1)} &= W_1 x + b_1 & &(W_1 \in \mathbb{R}^{h\times d}) \quad\text{pre-activation} \\
a^{(1)} &= g(z^{(1)}) & &\text{activation (e.g. ReLU)} \\
z^{(2)} &= W_2 a^{(1)} + b_2 & &(W_2 \in \mathbb{R}^{c\times h}) \\
\hat{y} &= \sigma(z^{(2)})\ \text{or}\ \operatorname{softmax}(z^{(2)}) & &\text{output}
\end{aligned}
$$

In practice you process a **batch** $X \in \mathbb{R}^{n\times d}$ at once, so everything is matrix multiplies: $Z^{(1)} = X W_1^\top + b_1$, etc. — this is what makes nets GPU-friendly.

### 2.2 Activations

| Activation | Formula | Derivative | Notes |
|---|---|---|---|
| **Sigmoid** | $\sigma(z)=\frac1{1+e^{-z}}$ | $\sigma(z)(1-\sigma(z))$ | Squashes to (0,1); **saturates** → vanishing gradients; output-only now. |
| **Tanh** | $\tanh(z)$ | $1-\tanh^2(z)$ | Zero-centered (better than sigmoid) but still saturates. |
| **ReLU** | $\max(0,z)$ | $\mathbb{1}[z>0]$ | **Default** hidden activation. No saturation for $z>0$, cheap, sparse. Can "die" (stuck at 0). ⭐ |
| **Leaky ReLU** | $\max(\alpha z, z)$ | $1$ or $\alpha$ | Small slope $\alpha$ for $z<0$ fixes dying ReLU. |
| **GELU / SiLU** | smooth ReLU-likes | — | Default in transformers. |
| **Softmax** | $e^{z_i}/\sum_j e^{z_j}$ | (Jacobian) | Output layer for multiclass → a probability vector. |

> **Why nonlinearity at all:** without $g$, the whole stack is linear. **Why ReLU by default:** it doesn't saturate on the positive side, so gradients flow through deep stacks — the single biggest practical fix for vanishing gradients. ⭐

### 2.3 Loss functions

- **Regression:** mean squared error $\frac1n\sum(\hat y - y)^2$ (output is linear, no activation).
- **Binary classification:** binary cross-entropy (log loss), sigmoid output.
- **Multiclass:** categorical cross-entropy, softmax output: $L = -\sum_i y_i \log \hat y_i$.

> **Softmax + cross-entropy is a deliberate pairing.** The gradient of CE loss w.r.t. the **pre-softmax logits** collapses to the beautifully simple $\hat y - y$. (Same identity as sigmoid + BCE, and as the linear output + MSE.) That clean residual is why these output/loss pairs are canonical — and it's the first line of the backward pass. ⭐

### 2.4 Backpropagation

Backprop = **the chain rule, computed once from output to input, reusing intermediate results.** Define the error signal $\delta^{(l)} = \partial L / \partial z^{(l)}$ (gradient w.r.t. a layer's pre-activations). For our two-layer net with softmax/sigmoid output and matching loss:

$$
\begin{aligned}
\delta^{(2)} &= \hat y - y & &\text{(output error — the clean residual)} \\
\frac{\partial L}{\partial W_2} &= \delta^{(2)} (a^{(1)})^\top, \quad \frac{\partial L}{\partial b_2} = \delta^{(2)} \\
\delta^{(1)} &= (W_2^\top \delta^{(2)}) \odot g'(z^{(1)}) & &\text{(propagate back, gate by activation slope)} \\
\frac{\partial L}{\partial W_1} &= \delta^{(1)} x^\top, \quad \frac{\partial L}{\partial b_1} = \delta^{(1)}
\end{aligned}
$$

The pattern is mechanical and is the thing to memorize: **(1)** error at the output is $\hat y - y$; **(2)** a weight's gradient is `(error at the layer above) × (activation feeding into it)`; **(3)** to push error to the previous layer, multiply by $W^\top$ and **element-wise gate by the local activation derivative** $g'$. ⭐

> **Why it's efficient:** naively, computing each partial derivative separately is exponential in depth. Backprop is **reverse-mode automatic differentiation** — one forward pass to cache activations, one backward pass to get *all* gradients, in time linear in the number of parameters. This algorithm is the entire reason deep learning is trainable.
>
> **Vanishing/exploding gradients:** $\delta$ accumulates a product of $W^\top$ and $g'$ across layers. If those factors are $<1$ (e.g. sigmoid's $g' \le 0.25$), the signal **vanishes** in deep nets; if $>1$, it **explodes**. Fixes: ReLU, careful init (Xavier/He), residual connections, batch/layer norm, gradient clipping. (See [§6](#6-deep-learning-regularization-week-7-light-slot).) ⭐

### 2.5 Optimizers

All are variants of "step downhill," differing in *what* and *how much*:

- **(Batch) Gradient Descent:** gradient over the whole dataset → smooth but slow / memory-heavy.
- **SGD:** gradient on one example → noisy, cheap, the noise can help escape sharp minima.
- **Mini-batch SGD:** the practical default — gradient over a batch (32–256). One pass over all batches = one **epoch**.
- **Momentum:** accumulate a velocity ($v \leftarrow \beta v + g$, step by $v$) to glide through ravines and damp oscillations.
- **RMSProp / AdaGrad:** scale the step **per-parameter** by a running average of squared gradients — adapts the learning rate to each weight.
- **Adam:** momentum + RMSProp together (1st and 2nd moment estimates, bias-corrected). The robust default for most deep nets. ⭐

> **Learning rate is the most important hyperparameter.** Too high → divergence/oscillation; too low → crawls / stuck. Use LR schedules (warmup, cosine/step decay) and, ties to [Week 1](01-regularization-and-optimization.md), remember it's the same gradient-descent machinery — just non-convex now, so initialization and the optimizer matter much more.

## 3. The algorithm

```
TRAIN MLP (X, y):
    initialize W1,b1,W2,b2   # small random weights (break symmetry), zero biases
    repeat for n_epochs:
        for each mini-batch (Xb, yb):
            FORWARD:   z1 = Xb·W1ᵀ + b1;  a1 = g(z1)
                       z2 = a1·W2ᵀ + b2;  ŷ = softmax(z2)
            LOSS:      L = cross_entropy(ŷ, yb)
            BACKWARD:  δ2 = ŷ - yb
                       dW2 = δ2ᵀ·a1;  db2 = sum(δ2)
                       δ1 = (δ2·W2) ⊙ g'(z1)
                       dW1 = δ1ᵀ·Xb;  db1 = sum(δ1)
            UPDATE:    W -= lr·dW;  b -= lr·db        # (or Adam)
    return params
```

Two non-obvious essentials: **initialize weights randomly, not to zero** (zero weights make every hidden unit compute the identical gradient — they never differentiate; this is the *symmetry-breaking* requirement), and **cache the forward activations** ($z^{(1)}, a^{(1)}$) because backprop needs them.

## 4. From scratch

The canonical build #6 lives in [`mlp.py`](../implementations/src/mlp.py) — one hidden layer (ReLU) + softmax output, full-batch gradient descent, pure numpy. The whole thing is ~four matrix multiplies forward and the four-line backward pass from [§2.4](#24-backpropagation):

```python
def _forward(self, X):
    z1 = X @ self.W1 + self.b1
    a1 = np.maximum(0.0, z1)              # ReLU
    z2 = a1 @ self.W2 + self.b2
    return z1, a1, z2, self._softmax(z2)  # cache for backward

def _backward(self, X, Y, z1, a1, probs):
    n = X.shape[0]
    d2 = (probs - Y) / n                  # softmax+CE -> clean residual ⭐
    dW2, db2 = a1.T @ d2, d2.sum(0)
    d1 = (d2 @ self.W2.T) * (z1 > 0)      # gate by ReLU' = 1[z>0]
    dW1, db1 = X.T @ d1, d1.sum(0)
    return dW1, db1, dW2, db2
```

Tested: learns **XOR** (the textbook proof a hidden layer beats a linear model), separates blobs/moons, loss decreases monotonically, He-style init breaks symmetry, and a gradient-check matches numerical gradients. The "code without notes" essentials: **random init**, **forward with cached activations**, the **$\hat y - y$ output error**, **gate-by-$g'$ backprop**, and the **GD update loop**.

## 5. With scikit-learn / PyTorch

```python
# sklearn — fine for a tabular baseline, not for real deep learning
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu",
                    solver="adam", alpha=1e-4,           # alpha = L2 strength
                    early_stopping=True).fit(X, y)

# PyTorch — the real tool; you define the forward, autograd does backprop
import torch, torch.nn as nn
net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, c))
opt = torch.optim.Adam(net.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()                          # = logsoftmax + NLL
for xb, yb in loader:
    opt.zero_grad()
    loss = loss_fn(net(xb), yb)
    loss.backward()        # autograd computes every gradient (backprop)
    opt.step()
```

`CrossEntropyLoss` takes **raw logits** (it applies log-softmax internally for numerical stability) — a classic bug is to softmax twice.

## 6. Deep-learning regularization (Week 7 light slot)

Big nets overfit easily (millions of parameters); the toolkit for fighting it:

- **L2 / weight decay** — same $\lambda\lVert W\rVert^2$ penalty as [Week 1](01-regularization-and-optimization.md); shrinks weights, smooths the function. In sklearn it's `alpha`; in optimizers, `weight_decay`.
- **Dropout** — during training, randomly zero a fraction $p$ of activations each step; at test time use all units (scaled). Forces redundancy so no unit can rely on a specific other — like training an ensemble of subnetworks. Don't apply at test time. ⭐
- **Batch normalization** — normalize each layer's pre-activations to zero mean / unit variance **per mini-batch** (plus learnable scale/shift). Stabilizes and speeds training, allows higher LRs, and mildly regularizes. *Behaves differently train vs. eval* (eval uses running stats) — a common bug. **Layer norm** (normalize across features, not the batch) is the transformer/RNN counterpart and is batch-size-independent. ⭐
- **Early stopping** — watch validation loss, stop when it turns up. Cheap and effective; implicit regularization.
- **Data augmentation** — synthetic label-preserving transforms (crops/flips for images, paraphrase/back-translation for text) — the most effective regularizer when you have unstructured data.
- **Smaller architecture / parameter sharing** (CNNs, weight tying) — reduce capacity structurally.

> Lock the pairing cold: **dropout = random unit masking (train-only)**; **batch norm = per-batch normalization of activations (train/eval differ)**. They solve different problems (overfitting vs. optimization stability) and are routinely used together.

## 7. Assumptions

- **Enough data.** Nets are data-hungry; with little tabular data, GBMs/linear models usually win.
- **Features benefit from scaling/normalization** — unscaled inputs make optimization ill-conditioned (relates to BN).
- **The loss surface is non-convex** — no global-optimum guarantee; you accept a good local/flat minimum. Init and optimizer matter.
- **Differentiable everywhere it counts** — backprop needs gradients (ReLU's kink is handled by a subgradient).
- **i.i.d. mini-batches** for SGD's unbiased-gradient assumption (shuffle each epoch).

## 8. When to use / not use

**Use when:** **unstructured data** (images → CNN, text/sequences → transformers, audio), very large datasets, transfer learning from a pretrained model, or you need to learn representations end-to-end. This is where nets dominate everything else.

**Avoid / deprioritize when:** small-to-medium **tabular** data — **gradient-boosted trees** ([Week 5](05-ensembles-boosting.md)) typically match or beat nets with less tuning, less data, and more interpretability; when you need interpretability, low-latency CPU inference, or have a tiny dataset. "Try a GBM baseline first on tabular" is a credible interview stance. ⭐

## 9. Pitfalls & gotchas

- **Zero-initializing weights** → symmetry never breaks, all hidden units stay identical. Use small random / Xavier / He init. ⭐
- **No nonlinearity** (or only linear activations) → the whole network collapses to linear regression. The activation *is* the network.
- **Vanishing/exploding gradients** in deep nets → use ReLU, proper init, residuals, normalization, gradient clipping. (§2.4)
- **Learning rate wrong** → diverges (too high) or crawls/stalls (too low). Tune it first; use schedules.
- **Forgetting train/eval mode** → dropout active at test time, or batch-norm using batch stats at inference. Always switch modes. ⭐
- **Double-softmax** → applying softmax then feeding to a loss that expects logits.
- **Not scaling inputs** → ill-conditioned optimization, slow/unstable training.
- **Unshuffled or class-sorted batches** → biased gradients, poor convergence.
- **Reaching for a deep net on small tabular data** → overkill; a GBM is the stronger, simpler default.
- **Overfitting unnoticed** → always hold out validation; use dropout/early stopping/augmentation.

## 10. Interview questions

Must answer cold (⭐):
- ⭐ **What is backpropagation?** The chain rule applied from output to input in one backward pass, caching forward activations, to compute the gradient of the loss w.r.t. every parameter in time linear in the parameter count (reverse-mode autodiff). (§2.4)
- ⭐ **Why do we need a nonlinear activation?** Without it, stacked affine layers collapse to a single linear map — the net could only learn linear functions. The nonlinearity lets it approximate arbitrary functions. (§1, §2.2)
- ⭐ **Why ReLU over sigmoid in hidden layers?** ReLU doesn't saturate for positive inputs, so gradients flow through deep stacks (mitigates vanishing gradients); it's also cheap and induces sparsity. Sigmoid/tanh saturate ($g'\to 0$). (§2.2)
- ⭐ **Why can't you initialize all weights to zero?** Every hidden unit would receive an identical gradient and stay identical forever — symmetry never breaks. Use small random init. (§3, §9)
- ⭐ **What are vanishing/exploding gradients and how do you fix them?** The backprop signal is a product of $W^\top$ and $g'$ across layers; <1 factors vanish, >1 explode. Fixes: ReLU, Xavier/He init, residual connections, batch/layer norm, gradient clipping. (§2.4)
- ⭐ **What's the gradient of softmax + cross-entropy w.r.t. the logits?** Simply $\hat y - y$. (Same clean residual as sigmoid+BCE and linear+MSE.) (§2.3)
- ⭐ **Dropout vs. batch norm?** Dropout randomly zeros activations during training to prevent co-adaptation (an implicit ensemble), off at test time. Batch norm normalizes layer activations per mini-batch to stabilize/speed optimization; it uses running stats at eval. Different goals; often combined. (§6)
- ⭐ **Universal approximation — does it mean one layer is enough?** In theory one hidden layer can approximate any continuous function, but possibly needs exponentially many units; depth gives the same expressivity far more efficiently. (§1)
- **SGD vs. GD vs. Adam?** GD = full-data gradient (smooth, slow); SGD/mini-batch = noisy, cheap, scalable; Adam = momentum + per-parameter adaptive LR, the robust default. (§2.5)
- **What's an epoch vs. a batch vs. an iteration?** A batch is a subset used for one update (iteration); an epoch is one full pass over all batches. (§2.5)
- **When would you *not* use a neural net?** Small/medium tabular data — GBMs usually win with less tuning and more interpretability. (§8)
- **Why scale inputs?** Unscaled features make the loss surface ill-conditioned → slow/unstable optimization. (§7)

## 11. Connections
- **NN = stacked logistic regression** — one output unit with sigmoid *is* logistic regression ([Foundations](00-foundations.md)); the loss/gradient machinery is identical, just non-convex now.
- **Same gradient descent** as [Week 1: Regularization & Optimization](01-regularization-and-optimization.md) — L2/weight decay, LR, SGD all carry over; the new piece is backprop and non-convexity.
- **Softmax + cross-entropy residual** ($\hat y - y$) mirrors the [boosting](05-ensembles-boosting.md) "fit the negative gradient" idea — both are gradient-of-loss views.
- **Tabular default is still a GBM** ([Week 5](05-ensembles-boosting.md)) — know when *not* to reach for a net.
- **Builds the bridge to [Week 8: Transformers](08-nlp-transformers.md)** — attention blocks are MLPs + a content-based mixing layer; **layer norm** and residuals (introduced here) are core there.
- Implementation: [`mlp.py`](../implementations/src/mlp.py). Flashcards: [flashcards/07-neural-networks.md](../flashcards/07-neural-networks.md).

---
### Sources
- Géron, *Hands-On ML* — Introduction to Artificial Neural Networks with Keras; Training Deep Neural Networks (init, BN, dropout, optimizers).
- Goodfellow, Bengio, Courville, *Deep Learning* — Ch. 6 (feedforward nets, backprop), Ch. 7–8 (regularization, optimization).
- Nielsen, *Neural Networks and Deep Learning* — the clearest backprop derivation.
