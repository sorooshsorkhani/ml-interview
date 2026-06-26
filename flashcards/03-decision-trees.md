# Flashcards: Decision Trees

> One `Q:`/`A:` pair per card, atomic answers. Tags in `[brackets]` for later Anki export.
> ✅ Seeded from the topic note's "Interview questions" section.

---

**Q:** How does a decision tree decide where to split a node?
**A:** It exhaustively tries each feature and candidate threshold, computes the sample-weighted impurity decrease (information gain), and picks the split with the largest gain — then recurses on each child. `[trees]`

---

**Q:** What is Gini impurity?
**A:** `G = 1 - Σ p_c²` — the expected error from labeling a random sample by the node's class distribution. 0 when pure, max (0.5 for binary) when classes are uniform. `[trees][impurity]`

---

**Q:** What is entropy as a node impurity?
**A:** `H = -Σ p_c log₂ p_c` — average bits to encode the class. 0 when pure, max (1 bit for binary) when uniform. `[trees][impurity]`

---

**Q:** Gini vs. entropy — does the choice matter?
**A:** Rarely. Both are 0 at purity and max at uniform; they pick nearly identical splits. Gini avoids logs so it's cheaper — sklearn's default. `[trees][impurity]`

---

**Q:** What is information gain (formally)?
**A:** Parent impurity minus the sample-weighted average of the child impurities: `I(parent) - (n_L/n)·I(left) - (n_R/n)·I(right)`. `[trees]`

---

**Q:** Why must information gain weight children by their sample counts?
**A:** Otherwise a split that peels off a single pure sample could beat a split that cleanly halves the data. Weighting by n_L/n, n_R/n rewards splits that purify *most* of the data. `[trees]`

---

**Q:** Why do decision trees overfit by default?
**A:** They keep splitting until leaves are pure, memorizing the training set → 0 train error but high variance and poor generalization. `[trees][overfitting]`

---

**Q:** How do you prevent a tree from overfitting?
**A:** Pre-pruning (max_depth, min_samples_leaf, min_impurity_decrease), post-pruning (cost-complexity / ccp_alpha tuned by CV), or — best in practice — ensembling (Random Forest, boosting). `[trees][overfitting]`

---

**Q:** What is cost-complexity (weakest-link) pruning?
**A:** Minimize `R_α(T) = R(T) + α·|T|` — training error plus a penalty α times the number of leaves. Larger α → smaller tree; α tuned by cross-validation (sklearn's ccp_alpha). `[trees][pruning]`

---

**Q:** Do decision trees need feature scaling?
**A:** No. Splits are thresholds on one feature at a time, so the tree is invariant to monotonic rescaling. (Opposite of KNN / linear models.) `[trees][scaling]`

---

**Q:** What is the bias–variance profile of a single decision tree?
**A:** Low bias, high variance — flexible enough to fit almost anything but unstable. Increasing depth lowers bias and raises variance. `[trees][bias-variance]`

---

**Q:** What's the train vs. predict complexity of a decision tree?
**A:** Training ≈ O(p · n log n · depth) (split search per node); prediction is O(depth) — just walk root to leaf. `[trees][complexity]`

---

**Q:** Why is the greedy split search not optimal?
**A:** Finding the globally optimal tree is NP-hard; CART picks the locally best split at each node with no look-ahead, so the result is a heuristic, not the global optimum. `[trees]`

---

**Q:** How does a tree solve a non-linear problem like XOR?
**A:** By stacking axis-aligned splits — a depth-2 tree partitions the plane into the four quadrants XOR needs, which a single linear boundary can't. `[trees]`

---

**Q:** What's the candidate threshold set for splitting a numeric feature?
**A:** The midpoints between consecutive sorted unique values of that feature — covers every distinct partition with the fewest candidates. `[trees]`

---

**Q:** What's the problem with impurity-based feature importance?
**A:** It's biased toward high-cardinality / continuous features (more split opportunities). Prefer permutation importance when importance matters. `[trees]`

---

**Q:** How does a regression tree differ from a classification tree?
**A:** Same recursive splitting, but impurity = variance/MSE of the targets in a node, and a leaf predicts the mean target instead of a majority class. `[trees]`

---

**Q:** Why are decision trees the base learner for Random Forests and boosting?
**A:** A single tree is high-variance; averaging many de-correlated trees (bagging/RF) or sequentially boosting shallow trees reduces variance/bias and gives state-of-the-art classical-ML accuracy. `[trees][ensembles]`
