# Flashcards: K-Means & PCA

> One `Q:`/`A:` pair per card, atomic answers. Tags in `[brackets]` for later Anki export.
> ✅ Seeded from the topic note's "Interview questions" section.

---

**Q:** What objective does K-Means minimize?
**A:** Inertia — the within-cluster sum of squared distances, Σ_j Σ_{x∈C_j} ‖x − μ_j‖². `[kmeans][clustering]`

---

**Q:** Why is the K-Means update step the cluster *mean*?
**A:** The mean minimizes the summed squared Euclidean distance within a cluster, so it directly minimizes the objective for fixed assignments. `[kmeans]`

---

**Q:** Does K-Means converge? To the global optimum?
**A:** It always converges (each assign/update step can't increase inertia), but only to a **local** optimum — the objective is non-convex. Hence k-means++ and multiple restarts. `[kmeans]`

---

**Q:** What is k-means++ and why use it?
**A:** Smart initialization: first centroid uniform-random, each next chosen with probability ∝ D(x)² (squared distance to the nearest chosen centroid). Spreads seeds out → avoids bad local optima, converges faster. `[kmeans][init]`

---

**Q:** How do you choose k in K-Means?
**A:** Elbow method on inertia, silhouette score (more principled), or domain/downstream criteria — never just minimize inertia (it always drops as k grows). `[kmeans][model-selection]`

---

**Q:** K-Means vs. KNN?
**A:** K-Means = unsupervised clustering; learns k centroids by iterating assign/update. KNN = supervised classification; labels a point by majority vote of its nearest labeled neighbors. `[kmeans][knn]`

---

**Q:** Time complexity of one K-Means iteration?
**A:** O(n · k · d) — n points, k clusters, d dimensions. `[kmeans][complexity]`

---

**Q:** When does K-Means fail, and what do you use instead?
**A:** On non-spherical, varying-density, or nested clusters (it draws linear/Voronoi boundaries) — use DBSCAN, GMM (soft/elliptical), or spectral clustering. `[kmeans]`

---

**Q:** What does PCA optimize?
**A:** It finds orthogonal directions of maximum variance; equivalently, the linear projection that minimizes squared reconstruction error. `[pca][dimensionality-reduction]`

---

**Q:** How is PCA computed?
**A:** Eigendecomposition of the covariance matrix (eigenvectors = components, eigenvalues = variances) or, more numerically stably, SVD of the centered data. `[pca]`

---

**Q:** What do PCA's eigenvalues represent?
**A:** The variance captured along each principal component; their normalized values give explained_variance_ratio_. `[pca]`

---

**Q:** Why must you center (and usually scale) before PCA?
**A:** PCA measures variance about the mean — uncentered, PC1 points at the data's offset, not its spread; unscaled, large-unit features dominate the components. `[pca][preprocessing]`

---

**Q:** Is PCA supervised? What if you want class separation?
**A:** No — PCA ignores labels, so its top variance directions need not be discriminative. Use LDA for supervised class separation. `[pca]`

---

**Q:** How do you choose the number of PCA components?
**A:** Cumulative explained variance (e.g., keep 90–95%), scree-plot elbow, or cross-validate against the downstream task. `[pca][model-selection]`

---

**Q:** Why are both K-Means and PCA scale-sensitive?
**A:** Both are variance/distance-based — without standardizing, features with large units dominate distances (K-Means) or variance (PCA). `[kmeans][pca][preprocessing]`

---

**Q:** What's the empty-cluster edge case in K-Means and a fix?
**A:** A centroid can end up with no assigned points; re-seed it (e.g., on the point farthest from its centroid) so you don't lose a cluster. `[kmeans]`
