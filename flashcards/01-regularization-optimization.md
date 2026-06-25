# Flashcards: Regularization & Optimization

> One `Q:`/`A:` pair per card, atomic answers. Tags in `[brackets]` for later Anki export.
> 🟡 Seeded from the topic note's "Interview questions" section as content is filled.

---

**Q:** What is the difference between L1 and L2 regularization in effect?
**A:** L1 (Lasso) drives some weights exactly to zero → sparse / feature selection. L2 (Ridge) shrinks all weights smoothly toward zero but rarely exactly zero. `[regularization]`

---

**Q:** Geometrically, why does L1 produce sparsity but L2 doesn't?
**A:** The L1 constraint region is a diamond with corners on the axes; the loss contour first touches it at a corner (a coordinate = 0). The L2 region is a smooth ball with no corners. `[regularization]`

---

**Q:** What is the effect of increasing the regularization strength λ on bias and variance?
**A:** More bias, less variance (simpler model). `[bias-variance]`

---

**Q:** Why must features be standardized before applying regularization or gradient descent?
**A:** A single λ penalizes all coefficients equally, so different feature scales distort the penalty; and unequal scales make the loss surface a stretched bowl, slowing GD. `[scaling]`

---

**Q:** Batch vs. SGD vs. mini-batch gradient descent — one tradeoff each?
**A:** Batch: stable but slow (full pass per step). SGD: fast & online but noisy, needs LR decay. Mini-batch: the practical default — vectorized, moderate noise. `[optimization]`

---

**Q:** In sklearn logistic regression, does a smaller `C` mean more or less regularization?
**A:** More. `C = 1/λ`, so smaller C = stronger regularization. `[sklearn]`

---
