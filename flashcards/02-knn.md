# Flashcards: K-Nearest Neighbors

> One `Q:`/`A:` pair per card, atomic answers. Tags in `[brackets]` for later Anki export.
> 🟡 Seeded from the topic note's "Interview questions" section as content is filled.

---

**Q:** Why is KNN called a "lazy" learner?
**A:** It does no work at training time — `fit` just stores the data. All computation (distance + voting) is deferred to prediction time. `[knn]`

---

**Q:** Why is KNN "non-parametric"?
**A:** It assumes no fixed functional form for the decision boundary; the effective complexity grows with the data — the training set itself is the model. `[knn]`

---

**Q:** How does k affect the bias–variance tradeoff in KNN?
**A:** Small k → low bias, high variance (wiggly boundary, overfits noise). Large k → high bias, low variance (smooth, underfits). `[knn][bias-variance]`

---

**Q:** What happens with k=1?
**A:** Each training point is its own nearest neighbor → 0 training error but typically high variance / poor generalization. `[knn]`

---

**Q:** Why must features be standardized before KNN?
**A:** Distance sums over features, so a large-scale feature dominates the metric and drowns out smaller-scale (possibly more informative) features. `[knn][scaling]`

---

**Q:** What is the time complexity of KNN at training vs. prediction (brute force)?
**A:** Training O(1) (just store data); prediction O(n·p) per query for distances plus O(n) to select the top-k. `[knn][complexity]`

---

**Q:** Why does the curse of dimensionality hurt KNN specifically?
**A:** In high dimensions distances concentrate (nearest ≈ farthest), data becomes sparse, and noisy features inflate distance — so "nearest neighbor" stops being meaningful. `[knn][curse-of-dimensionality]`

---

**Q:** How do you choose k?
**A:** Cross-validation (sweep k, pick best validation score); use an odd k for binary classification to avoid ties; √n is a rough starting point. `[knn]`

---

**Q:** How can you speed up KNN prediction on large data?
**A:** Spatial indexes (KD-tree / ball-tree) for low dimensions, or approximate nearest neighbors (HNSW / FAISS) for large-scale or embedding data. `[knn][complexity]`

---

**Q:** KNN vs. K-Means — what's the difference?
**A:** KNN is supervised classification (k labeled neighbors vote on a query). K-Means is unsupervised clustering (k centroids partition unlabeled data). The shared "k" is the only similarity. `[knn][kmeans]`

---

**Q:** Can KNN do regression, and how?
**A:** Yes — predict the (optionally distance-weighted) average of the k nearest neighbors' target values. `[knn]`

---

**Q:** Why use an odd k for binary classification?
**A:** To avoid tie votes between the two classes. `[knn]`

