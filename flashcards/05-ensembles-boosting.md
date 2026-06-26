# Flashcards: Boosting

> One `Q:`/`A:` pair per card, atomic answers. Tags in `[brackets]` for later Anki export.
> ✅ Seeded from the topic note's "Interview questions" section.

---

**Q:** Bagging vs. boosting — the core difference?
**A:** Bagging = parallel, independent, **deep** trees, reduces **variance**. Boosting = sequential, dependent, **shallow** trees, each corrects the previous one, reduces **bias**. `[ensembles][boosting]`

---

**Q:** How does AdaBoost work, in one breath?
**A:** Weight the examples; fit a weak learner; up-weight the misclassified points; give each learner a say α = ½·ln((1−ε)/ε); final prediction is the weighted vote of all learners. `[ensembles][adaboost]`

---

**Q:** What loss does AdaBoost minimize, and how?
**A:** Exponential loss e^(−yF(x)), via forward stagewise additive modeling — the reweighting and the α formula both fall out of greedily minimizing it one learner at a time. `[ensembles][adaboost]`

---

**Q:** In AdaBoost, what say does a learner with error ε get?
**A:** α = ½·ln((1−ε)/ε). ε→0 gives large positive say; ε=0.5 gives 0; ε>0.5 gives a negative say (its vote is flipped). `[ensembles][adaboost]`

---

**Q:** What does gradient boosting fit at each round?
**A:** The **negative gradient of the loss** (the "pseudo-residual") evaluated at the current predictions. `[ensembles][gradient-boosting]`

---

**Q:** For squared-error loss, what is the negative gradient?
**A:** The ordinary residual y − F_{m−1}(x). So "fit each tree to the residuals" is the squared-error special case of "fit each tree to the negative gradient." `[ensembles][gradient-boosting]`

---

**Q:** Why is it called *gradient* boosting?
**A:** Each added tree is a gradient-descent step in *function space* — it points in the negative-gradient direction that most reduces the loss at each training point. `[ensembles][gradient-boosting]`

---

**Q:** Can boosting overfit if you add too many trees?
**A:** Yes — unlike a Random Forest. The ensemble keeps reducing training loss and eventually fits noise. Control it with early stopping on a validation set. `[ensembles][boosting]`

---

**Q:** How do learning_rate and n_estimators trade off?
**A:** Two ends of one rope: a smaller learning rate needs more trees but generalizes better. Standard recipe: small learning rate + early stopping to pick the number of rounds. `[ensembles][boosting][tuning]`

---

**Q:** Why is XGBoost better than vanilla gradient boosting?
**A:** Second-order (Newton) updates using the Hessian, a regularized objective baked into the split-gain (γT + ½λ·Σw²), plus engineering: histograms, native missing-value handling, row/column subsampling. `[ensembles][xgboost]`

---

**Q:** Why use a *weak* (shallow) base learner in boosting?
**A:** A deep tree already has low bias and would overfit, leaving nothing to correct. Boosting's job is to combine many high-bias weak learners into one low-bias strong one. `[ensembles][boosting]`

---

**Q:** When would you choose Random Forest over boosting?
**A:** When you want a robust baseline with little tuning, have noisy labels/outliers, need parallel training, or just need a solid number fast. `[ensembles][boosting][random-forest]`

---

**Q:** Why is boosting sensitive to outliers and noisy labels?
**A:** It repeatedly up-weights the hardest-to-fit points, which are often the mislabeled/outlier ones, so the ensemble spends capacity fitting noise. Robust losses (Huber) help. `[ensembles][boosting]`

---

**Q:** How are boosted probabilities typically calibrated?
**A:** Over-confident — pushed toward 0/1 (opposite of bagging's pull toward the middle). Recalibrate with Platt/isotonic if you need true probabilities. `[ensembles][boosting][calibration]`

---

**Q:** What is stochastic gradient boosting?
**A:** Subsample the rows (and/or columns) used for each tree — adds bagging-style variance reduction and speed on top of boosting. `[ensembles][gradient-boosting]`

---

**Q:** Why can't you parallelize boosting across trees the way you can a Random Forest?
**A:** Each tree is fit to the residuals/reweighting produced by the previous trees — the dependence is sequential. Only the work *within* a single tree (split-finding) parallelizes. `[ensembles][boosting]`
