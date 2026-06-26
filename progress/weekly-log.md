# Weekly Log

> One entry per week: what I studied, what stuck, what to revisit. Keeps the spaced-repetition honest.

## Template
```
### Week N (wk of <date>)
- ML topic: 
- Coding pattern + problems solved: 
- Light slot: 
- Wins: 
- Struggled with / revisit: 
- Hours actually spent: 
```

---

### Week 0 (setup)
- Repo structured; Week 1 ML note (Regularization & Optimization) written; linear regression + L2 implemented & tested; Arrays/hash-map pattern note written.
- Next: Week 1 — add L2 to my own linear regression (reference build in repo), finish Arrays lessons 3–4, probability refresh part 1.

### Week 2 (KNN)
- ML: KNN note written (lazy/non-parametric, distance metrics, choosing k & bias–variance, curse of dimensionality, complexity). `KNNClassifier` implemented from scratch (vectorized distances, argpartition top-k, deterministic tie-break, standardization) + 7 passing tests. KNN flashcard deck seeded (12 cards).
- To do on your side: read the note, then attempt the strings/hash-map coding problems ([dsa/01](../dsa/01-arrays-strings-hashmaps.md)); light slot — probability/stats part 2 (bias–variance formally, MLE intuition: [statistics/02](../statistics/02-bias-variance-mle.md), still a scaffold — say the word and I'll write it).
- Revisit later: the KNN vs. K-Means distinction (common interview trap).

### Week 3 (Decision Trees)
- ML: Decision Trees note written ([ml-theory/03](../ml-theory/03-decision-trees.md)) — Gini vs. entropy, information gain (weighted impurity decrease), CART greedy recursion, overfitting + pre/post-pruning (cost-complexity), scale-invariance, bias–variance, why trees are the base learner for RF/boosting. `DecisionTreeClassifier` implemented from scratch ([decision_tree.py](../implementations/src/decision_tree.py)): impurity (gini/entropy), exhaustive best-split over feature × midpoint thresholds, recursive build with stopping criteria, `gain<=0` guard for contradictory rows, predict by traversal — 8 passing tests (solves XOR, memorizes unrestricted, max_depth limits growth, graceful stop on duplicate-mixed rows). Flashcard deck seeded (18 cards). Note: this is the canonical from-scratch build #5 — done ahead of the "stretch goal" bar.
- Coding: two-pointers ([dsa/02](../dsa/02-two-pointers.md)) and sliding-window ([dsa/03](../dsa/03-sliding-window.md)) pattern notes written (triggers, templates, complexity, curated problem lists). To do on your side: solve 3–4 across both — suggested set is Valid Palindrome + Two Sum II + Container With Most Water (two pointers) and Longest Substring Without Repeating Characters (window).
- Light slot: classification metrics already written ([evaluation/01](../evaluation/01-classification-metrics.md)) — that satisfies Week 3's "evaluation part 1." Skim it to lock in confusion matrix / precision / recall / F1 / ROC-AUC.
- Revisit later: cost-complexity pruning (`ccp_alpha`) intuition; the "diagonal boundary needs a staircase of splits" limitation; impurity-based feature-importance bias.

### Week 4 (Ensembles 1: Bagging & Random Forests)
- ML: Bagging & Random Forests note written ([ml-theory/04](../ml-theory/04-ensembles-bagging.md)) — bagging cuts variance via bootstrap + averaging; the `ρσ² + (1−ρ)/B·σ²` correlated-average formula as the *whole* justification for RF's per-split feature subsampling (decorrelation); the `(1−1/n)ⁿ→e⁻¹≈37%` OOB derivation and OOB-as-free-validation; impurity vs. permutation feature importance; bagging-vs-boosting contrast; Extra-Trees. No from-scratch file — included a sketch showing RF = many [Week 3 trees](../implementations/src/decision_tree.py) + bootstrap + vote (the only missing piece is per-split `max_features`). Flashcard deck seeded (16 cards).
- Coding: binary-search pattern note written ([dsa/04](../dsa/04-binary-search.md)) — classic `lo<=hi`, the preferred first-True boundary template, binary-search-on-the-answer, rotated arrays, `bisect`; plus the sorting partner (Timsort O(n log n), "sort then search/two-pointer"). To do on your side: solve 3–4 — suggested Binary Search + Search Insert Position + Search in Rotated Sorted Array + Koko Eating Bananas (search-on-answer).
- Light slot (Evaluation part 2): calibration & class-imbalance deep dive written ([evaluation/04](../evaluation/04-calibration-imbalance.md)) — PR curves & log loss in depth, calibration vs. ranking (reliability diagrams, ECE), Platt vs. isotonic, the 3-front imbalance toolkit (resampling/SMOTE with the leakage warning, class weights, threshold tuning to cost).
- Revisit later: why RF probabilities are pulled to the middle (averaging) vs. boosting to the extremes — and what that means for calibration; the bagging-vs-boosting one-liner (parallel/variance vs. sequential/bias) — lock it cold for Week 5.

### Week 5 (Ensembles 2: Boosting)
- ML: Boosting note written ([ml-theory/05](../ml-theory/05-ensembles-boosting.md)) — the bias-reduction counterpart to Week 4's variance reduction: **sequential shallow** learners vs. bagging's **parallel deep** ones. AdaBoost (weighted error, say α=½·ln((1−ε)/ε), example reweighting, = forward-stagewise exponential-loss minimization); gradient boosting as **gradient descent in function space** — fit each tree to the **negative gradient** (pseudo-residual), which *is* the ordinary residual `y−F` for squared error; XGBoost's edge = **second-order (Newton) updates + a regularized objective baked into the split-gain** (γT+½λΣw²) + engineering. The learning_rate ↔ n_estimators "two ends of one rope" + early stopping story; why boosting *can* overfit (unlike RF), is outlier/noise-sensitive, and yields over-confident probabilities. From-scratch sketch (GBM = baseline→residual→fit tree→add shrunken tree) on top of the [Week 3 tree](../implementations/src/decision_tree.py). Flashcard deck seeded (16 cards).
- Coding: linked-lists pattern note written ([dsa/05](../dsa/05-linked-lists.md)) — the three-pointer reverse shuffle, fast/slow (middle, Floyd's cycle + cycle-start), dummy-head merge, two-pointer remove-nth-from-end; templates, complexity, common pointer-hygiene mistakes, curated 9-problem list. To do on your side: solve 3–4 — suggested **Reverse Linked List + Merge Two Sorted Lists + Linked List Cycle** + one Medium (**Remove Nth Node From End**). Overlearn reverse — it's a sub-step inside half the others.
- Light slot (Cross-validation, leakage & split discipline): note written ([evaluation/03](../evaluation/03-cross-validation-leakage.md)) — why one split isn't enough; k-fold / stratified / repeated, choosing k (bias–variance of the *estimator*); time-series forward-chaining (never shuffle) and GroupKFold; the **leakage catalog** (target, preprocessing/train-test contamination, temporal, group, tuning) and the Pipeline fix (transforms refit per fold) + nested CV; train/val/test discipline (test touched once).
- Revisit later: nested CV mechanics (inner-tune / outer-score); the AdaBoost-as-exponential-loss derivation if grilled; recalibrating boosted probabilities (ties Week 5 boosting → [evaluation/04](../evaluation/04-calibration-imbalance.md)).

### Week 6 (Unsupervised: K-Means & PCA)
- ML: K-Means & PCA note written ([ml-theory/06](../ml-theory/06-kmeans-pca.md)) — the through-line is **variance**, and the corollary that *both need scaled features*. K-Means: the **inertia** objective, why the update is the **mean** (and the implicit Euclidean assumption), Lloyd's = **coordinate descent → local optimum**, **k-means++** seeding (∝ D(x)²), choosing k via **elbow/silhouette** (not raw inertia), and the **K-Means≠KNN** trap. PCA: max-variance = min-reconstruction-error, eigendecomposition vs. **SVD of centered data**, explained-variance-ratio, choosing components, PCA is **unsupervised** (LDA if you want separation).
- From-scratch (TWO builds, both tested green): **`KMeans`** ([kmeans.py](../implementations/src/kmeans.py)) — the canonical build #4: vectorized squared-distance, k-means++ init, assign/update loop, convergence check, empty-cluster re-seed, `n_init` restarts keeping lowest inertia; 7 tests (recovers blobs, centroids near truth, **inertia strictly decreases with k**, seed-reproducible, k=n → 0 inertia, guards k>n). **`PCA`** ([pca.py](../implementations/src/pca.py), stretch) — center → SVD → top-k of Vᵀ, explained-variance-ratio, transform/inverse_transform; 6 tests (PC1 >98% var on a tilted line, orthonormal components, exact reconstruction at k=d, **matches sklearn's variance ratios**). Full suite: 33 passed, 2 skipped (mlp/naive_bayes scaffolds).
- Coding: stacks & queues pattern note written ([dsa/06](../dsa/06-stacks-queues.md)) — LIFO match stack, the **monotonic stack** (next-greater, O(n) amortized), BFS queue, **monotonic deque** for sliding-window max, RPN; templates, complexity, the `list.pop(0)`-is-O(n) gotcha, 9-problem list. To do on your side: 3–4 — suggested **Valid Parentheses + Implement Queue using Stacks** + the monotonic-stack rep **Daily Temperatures** + one of RPN/Number of Islands. Drill the monotonic stack — highest-yield idea here.
- Light slot (SVM read-through): note written ([ml-theory/10](../ml-theory/10-svm.md)) — max-margin hyperplane, **support vectors** (sparse solution), **soft margin = hinge loss + L2**, **C is *inverse* regularization**, the **kernel trick** (dual depends only on inner products; swap in K), RBF `gamma`, why kernel SVMs don't scale (~O(n²–n³)), SVM vs. logistic (hinge vs. log loss; no native probabilities). No flashcard deck (read-through).
- Revisit later: silhouette vs. elbow when they disagree; GMM as soft K-Means / when to reach for it; kernel-PCA vs. PCA for nonlinear manifolds; the SVM dual derivation only if a loop is heavy on classical ML theory.
