# Flashcards: Bagging & Random Forests

> One `Q:`/`A:` pair per card, atomic answers. Tags in `[brackets]` for later Anki export.
> ✅ Seeded from the topic note's "Interview questions" section.

---

**Q:** What does bagging stand for, and what does it do?
**A:** Bootstrap AGGregating — train each model on a bootstrap sample (n draws with replacement) and average/vote the predictions. It reduces variance. `[ensembles][bagging]`

---

**Q:** Why does averaging many models reduce variance?
**A:** If errors are partly independent, they cancel in the average while the shared signal survives. For B independent estimators of variance σ², the average has variance σ²/B. `[ensembles][bias-variance]`

---

**Q:** What's the variance of an average of B *correlated* trees (correlation ρ)?
**A:** `ρσ² + (1−ρ)/B · σ²`. As B→∞ it approaches ρσ² — a floor set by the correlation, not removable by adding trees. `[ensembles][bias-variance]`

---

**Q:** Why does a Random Forest subsample features at each split?
**A:** To decorrelate the trees (lower ρ). The variance of a bagged average floors at ρσ², so reducing correlation is the only way to push variance lower. `[ensembles][random-forest]`

---

**Q:** What is a Random Forest, in one line?
**A:** Bagged deep decision trees plus per-split random feature subsampling to decorrelate them. `[ensembles][random-forest]`

---

**Q:** What is the out-of-bag (OOB) set?
**A:** The ~37% of training points not drawn into a given bootstrap sample. Predicting each point with only the trees that didn't see it gives an unbiased generalization estimate for free. `[ensembles][oob]`

---

**Q:** Why is ~37% of data out-of-bag per tree?
**A:** P(a point is missed in n draws with replacement) = (1 − 1/n)ⁿ → e⁻¹ ≈ 0.368. So ~63% of unique points are in-bag, ~37% out. `[ensembles][oob]`

---

**Q:** Bagging vs. boosting — the core difference?
**A:** Bagging = parallel, independent, deep trees, reduces variance. Boosting = sequential, each learner corrects the previous one, shallow trees, reduces bias. `[ensembles][boosting]`

---

**Q:** Can a Random Forest overfit by adding more trees?
**A:** Essentially no — accuracy plateaus as n_estimators grows, it doesn't degrade. Overfitting comes from tree depth, which the ensemble averages away. `[ensembles][random-forest]`

---

**Q:** What base learner does bagging need to be effective?
**A:** A high-variance, low-bias learner (e.g., deep unpruned trees). Bagging a low-variance model like linear regression barely helps. `[ensembles][bagging]`

---

**Q:** Default `max_features` for a Random Forest?
**A:** ~√p features per split for classification, ~p/3 for regression. `[ensembles][random-forest]`

---

**Q:** Why grow trees deep in a forest but prune a standalone tree?
**A:** A lone deep tree overfits; inside a forest the averaging removes that variance, so deep trees give the low bias you want with the variance controlled by the ensemble. `[ensembles][random-forest]`

---

**Q:** Two ways a Random Forest measures feature importance?
**A:** (1) Impurity decrease (MDI/Gini importance) — fast but biased to high-cardinality features; (2) permutation importance — shuffle a feature, measure accuracy drop; model-agnostic and more trustworthy. `[ensembles][feature-importance]`

---

**Q:** What goes wrong with feature importances when features are correlated?
**A:** The importance is split between the correlated features, so a genuinely important feature can look weak because its twin absorbs the credit. `[ensembles][feature-importance]`

---

**Q:** What are Extra-Trees (Extremely Randomized Trees)?
**A:** A Random Forest variant that also picks split thresholds at random rather than optimally — more decorrelation and variance reduction, slightly more bias, faster training. `[ensembles][random-forest]`

---

**Q:** Why is a Random Forest easy to parallelize?
**A:** The trees are trained independently (no sequential dependence, unlike boosting), so they can be built in parallel across cores. `[ensembles][random-forest]`
