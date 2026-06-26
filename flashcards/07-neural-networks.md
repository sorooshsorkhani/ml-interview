# Flashcards: Neural Networks

> One `Q:`/`A:` pair per card, atomic answers. Tags in `[brackets]` for later Anki export.
> ✅ Seeded from the topic note's "Interview questions" section.

---

**Q:** What is backpropagation?
**A:** The chain rule applied from output to input in one backward pass, caching forward activations, to compute the gradient of the loss w.r.t. every parameter in time linear in the parameter count. It's reverse-mode automatic differentiation. `[nn][backprop]`

---

**Q:** Why does a neural network need a nonlinear activation?
**A:** Without it, stacked affine layers collapse to a single linear map (W₂(W₁x) = (W₂W₁)x), so the net could only learn linear functions. The nonlinearity lets it bend boundaries / approximate arbitrary functions. `[nn][activations]`

---

**Q:** Why is ReLU the default hidden activation over sigmoid/tanh?
**A:** ReLU doesn't saturate for positive inputs, so gradients flow through deep stacks (mitigates vanishing gradients); it's cheap and induces sparsity. Sigmoid/tanh saturate (g'→0 at the tails). `[nn][activations]`

---

**Q:** Why can't you initialize all weights to zero?
**A:** Every hidden unit would receive an identical gradient and stay identical forever — symmetry never breaks. Use small random / Xavier / He init. `[nn][init]`

---

**Q:** What are vanishing and exploding gradients, and how do you fix them?
**A:** The backprop signal is a product of Wᵀ and g' across layers; factors <1 make it vanish, >1 make it explode. Fixes: ReLU, Xavier/He init, residual connections, batch/layer norm, gradient clipping. `[nn][backprop]`

---

**Q:** What's the gradient of softmax + cross-entropy w.r.t. the logits?
**A:** Simply ŷ − y (the clean residual). Same identity as sigmoid + BCE and linear output + MSE. `[nn][loss]`

---

**Q:** Why pair softmax with cross-entropy (and sigmoid with BCE)?
**A:** The gradient w.r.t. the pre-activation logits collapses to ŷ − y — numerically stable and the simple starting point for backprop. `[nn][loss]`

---

**Q:** Dropout vs. batch normalization?
**A:** Dropout randomly zeros activations during training to prevent co-adaptation (implicit ensemble), off at test time. Batch norm normalizes layer activations per mini-batch to stabilize/speed optimization, using running stats at eval. Different goals; often combined. `[nn][regularization]`

---

**Q:** What does the universal approximation theorem actually say?
**A:** A single hidden layer with enough units can approximate any continuous function on a compact set — but "enough" may be exponential. Depth gives the same expressivity far more parameter-efficiently. `[nn][theory]`

---

**Q:** SGD vs. (batch) GD vs. Adam?
**A:** GD = full-data gradient (smooth, slow, memory-heavy); SGD/mini-batch = noisy, cheap, scalable (the default); Adam = momentum + per-parameter adaptive learning rate (RMSProp), the robust default. `[nn][optimization]`

---

**Q:** Epoch vs. batch vs. iteration?
**A:** A batch is the subset used for one parameter update (one iteration); an epoch is one full pass over all batches. `[nn][optimization]`

---

**Q:** Why is the learning rate the most important hyperparameter?
**A:** Too high → divergence/oscillation; too low → crawls or stalls in a bad spot. Tune it first; use schedules (warmup, cosine/step decay). `[nn][optimization]`

---

**Q:** When would you NOT use a neural network?
**A:** Small-to-medium tabular data — gradient-boosted trees usually match or beat nets with less tuning, less data, and more interpretability. Also when you need interpretability or low-latency CPU inference. `[nn][model-selection]`

---

**Q:** Why scale/normalize inputs before training a net?
**A:** Unscaled features make the loss surface ill-conditioned → slow, unstable optimization. `[nn][preprocessing]`

---

**Q:** What two things must the forward pass cache for backprop, and why?
**A:** The pre-activations z and activations a of each layer — backprop needs them to compute gradients (e.g., a weight's gradient is errorᵀ·input, and the activation derivative g'(z) gates the propagated error). `[nn][backprop]`

---

**Q:** How does a neural net relate to logistic regression?
**A:** A single output unit with a sigmoid IS logistic regression. A net stacks many such units with nonlinearities between them; the loss/gradient machinery is identical, just non-convex. `[nn][connections]`

---

**Q:** What is the "dying ReLU" problem and a fix?
**A:** A unit stuck in the z<0 region always outputs 0 and gets zero gradient, so it never recovers. Fix with Leaky ReLU / GELU, lower learning rate, or better init. `[nn][activations]`

---

**Q:** What is early stopping and why does it regularize?
**A:** Monitor validation loss and stop when it turns up. It limits how far weights drift from their small init, implicitly constraining capacity — cheap and effective. `[nn][regularization]`
