# Unsupervised: K-Means & PCA

> **Week:** 6 · **Status:** ✅ written · **From-scratch build:** yes — [`KMeans`](../implementations/src/kmeans.py) (canonical #4) + [`PCA`](../implementations/src/pca.py) (stretch)
> **One-line:** Two workhorses of unsupervised learning — **K-Means** partitions points into *k* clusters by minimizing within-cluster variance (inertia); **PCA** finds the orthogonal directions of *maximum variance* and projects onto the top few for linear dimensionality reduction. Both are fundamentally about **variance**, so both need **scaled** features.

## Table of contents
- [1. Intuition](#1-intuition)
- [2. K-Means](#2-k-means)
  - [2.1 The objective (inertia)](#21-the-objective-inertia)
  - [2.2 The algorithm (Lloyd's)](#22-the-algorithm-lloyds)
  - [2.3 Initialization: k-means++](#23-initialization-k-means)
  - [2.4 Choosing k](#24-choosing-k)
- [3. PCA](#3-pca)
  - [3.1 The objective (max variance = min reconstruction error)](#31-the-objective-max-variance--min-reconstruction-error)
  - [3.2 The algorithm (eigendecomposition / SVD)](#32-the-algorithm-eigendecomposition--svd)
  - [3.3 Choosing the number of components](#33-choosing-the-number-of-components)
- [4. From scratch](#4-from-scratch)
- [5. With scikit-learn](#5-with-scikit-learn)
- [6. Assumptions](#6-assumptions)
- [7. When to use / not use](#7-when-to-use--not-use)
- [8. Pitfalls & gotchas](#8-pitfalls--gotchas)
- [9. Interview questions](#9-interview-questions)
- [10. Connections](#10-connections)

---

## 1. Intuition

Unsupervised learning has **no labels** — the goal is to find structure in $X$ alone. Two of the most-asked classics:

- **K-Means (clustering):** "Group the points into $k$ blobs so each blob is as *tight* as possible." Pick $k$ centers, assign each point to its nearest center, move each center to the mean of its points, repeat. It's the canonical partition-clustering method.
- **PCA (dimensionality reduction):** "Find a few new axes — linear combinations of the original features — that capture most of the *spread* in the data, and throw the rest away." It compresses correlated, high-dimensional data into a small number of uncorrelated directions with minimal information loss.

> **The unifying theme: variance.** K-Means *minimizes* within-cluster variance; PCA *maximizes* the variance retained along each kept axis. That's also why **both are scale-sensitive** — a feature measured in large units will dominate distances/variance unless you standardize. ⭐

## 2. K-Means

### 2.1 The objective (inertia)

Given clusters $C_1, \dots, C_k$ with centroids $\mu_1, \dots, \mu_k$, K-Means minimizes the **inertia** = within-cluster sum of squared distances:
$$
J = \sum_{j=1}^{k} \sum_{x \in C_j} \lVert x - \mu_j \rVert^2
$$
This is non-convex (the assignment is combinatorial), so we can't solve it globally in general — K-Means finds a **local** optimum. Two facts fall straight out of the objective:
- For fixed assignments, the optimal centroid is the **mean** of the cluster (the point minimizing summed squared distance). That's *why* the update step uses the mean — and why K-Means implicitly assumes **Euclidean** geometry.
- Each step (assign, then update) can only **decrease or hold** $J$, so the algorithm is guaranteed to converge (to a local min). ⭐

### 2.2 The algorithm (Lloyd's)

```
LLOYD'S ALGORITHM (k, X):
    initialize k centroids   (see 2.3: k-means++)
    repeat until centroids stop moving (or max_iter):
        ASSIGN:  each point -> nearest centroid (squared Euclidean)
        UPDATE:  each centroid -> mean of the points assigned to it
    return centroids, assignments, inertia
```

This is **coordinate descent** on $J$: the ASSIGN step optimizes assignments with centroids fixed; the UPDATE step optimizes centroids with assignments fixed. Complexity per iteration is **O(n · k · d)** (n points, k clusters, d dims). Because the result depends on the start, run several **random restarts** (`n_init`) and keep the lowest-inertia one.

> **Empty-cluster edge case:** if a centroid ends up with no points, re-seed it (e.g., on the point farthest from its centroid) so you don't silently lose a cluster.

### 2.3 Initialization: k-means++

Naive random init can land two centroids in the same blob and converge to a bad local optimum. **k-means++** spreads the initial centroids out:
1. Pick the first centroid uniformly at random.
2. Pick each subsequent centroid with probability **proportional to $D(x)^2$**, the squared distance from $x$ to the nearest already-chosen centroid.

Far-away points are far more likely to be chosen, so the seeds start well-separated. This both improves the final clustering and speeds convergence — it's the default in sklearn. ⭐

### 2.4 Choosing k

$k$ is a hyperparameter; the objective always *decreases* as $k$ grows (more centroids → tighter clusters; at $k=n$, inertia is 0), so you can't just minimize inertia. Common heuristics:
- **Elbow method:** plot inertia vs. $k$; pick the "elbow" where the marginal gain flattens.
- **Silhouette score:** for each point, $(b - a)/\max(a,b)$ where $a$ = mean intra-cluster distance, $b$ = mean distance to the nearest *other* cluster. Ranges $[-1, 1]$; higher is better. More principled than the elbow.
- Domain knowledge / downstream task often beats both.

## 3. PCA

### 3.1 The objective (max variance = min reconstruction error)

PCA seeks an orthonormal set of directions (principal components) such that projecting the data onto them keeps as much variance as possible. The **first principal component** is the unit vector $w_1$ maximizing the variance of the projections:
$$
w_1 = \arg\max_{\lVert w \rVert = 1} \operatorname{Var}(X_c w) = \arg\max_{\lVert w \rVert = 1} w^\top \Sigma\, w
$$
where $X_c$ is the **centered** data and $\Sigma = \frac{1}{n-1} X_c^\top X_c$ is the covariance matrix. Each subsequent component maximizes remaining variance subject to being **orthogonal** to the previous ones.

> **Two equivalent views:** maximizing retained variance is *exactly* the same as **minimizing the squared reconstruction error** of the low-rank projection. PCA gives the best linear $k$-dimensional approximation of the data in least-squares sense. ⭐

### 3.2 The algorithm (eigendecomposition / SVD)

The variance-maximization problem is solved by the **eigenvectors of the covariance matrix**:
$$
\Sigma\, w_i = \lambda_i\, w_i
$$
The eigenvector with the largest eigenvalue is PC1; eigenvalues $\lambda_i$ are the variances along each component, so `explained_variance_ratio_` $= \lambda_i / \sum_j \lambda_j$.

```
PCA(X, k):
    center:   Xc = X - mean(X)          # (and usually scale to unit variance)
    either:   eigendecompose Σ = Xc^T Xc/(n-1),  sort eigvecs by eigval desc
    or (SVD): Xc = U S V^T  -> components = rows of V^T, variances ∝ S^2
    project:  Z = Xc · W_k              # W_k = top-k components
```

In practice use the **SVD of the centered data** rather than forming $\Sigma$ explicitly — it's more numerically stable (avoids squaring the condition number) and is what sklearn does. The right singular vectors are the components; the singular values give the variances ($\lambda_i = s_i^2/(n-1)$).

### 3.3 Choosing the number of components

- **Cumulative explained variance:** keep enough components to reach, say, 90–95% of total variance.
- **Scree plot:** plot eigenvalues; look for the elbow.
- For a **downstream model**, treat $k$ as a hyperparameter and cross-validate.

## 4. From scratch

Both builds live in `implementations/` and pass their test suites.

**K-Means** ([`kmeans.py`](../implementations/src/kmeans.py)) — the canonical build #4. The core loop is short once the squared-distance helper is vectorized:
```python
def _single_run(self, X, rng):
    centers = self._kmeanspp_init(X, rng)          # §2.3
    for n_iter in range(1, self.max_iter + 1):
        labels = np.argmin(self._sq_dists(X, centers), axis=1)   # ASSIGN
        new = np.array([X[labels == c].mean(0) if np.any(labels == c)
                        else X[far_point] for c in range(k)])    # UPDATE (+empty fix)
        if np.linalg.norm(new - centers) <= self.tol: break       # converged
        centers = new
    inertia = np.min(self._sq_dists(X, centers), axis=1).sum()
    return centers, labels, inertia, n_iter
```
Key pieces tested: recovers well-separated blobs, centroids land near the truth, **inertia strictly decreases as k grows**, reproducible under a seed, `k=n` gives zero inertia, and `n_clusters > n_samples` raises. The "code without notes" essentials: **k-means++ init**, the **assign/update alternation**, the **convergence check**, and **inertia tracking** across `n_init` restarts.

**PCA** ([`pca.py`](../implementations/src/pca.py), stretch) — center → SVD → take top-k rows of $V^\top$ → `explained_variance_ratio_` from $s^2$. Tests check that PC1 captures >98% variance on a tilted line, components are orthonormal, reconstruction is exact at $k=d$, and the variance ratios match sklearn.

## 5. With scikit-learn

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

Xs = StandardScaler().fit_transform(X)         # scale first — both are variance-based

km = KMeans(n_clusters=4, init="k-means++", n_init=10, random_state=0).fit(Xs)
km.inertia_, km.cluster_centers_, km.labels_
silhouette_score(Xs, km.labels_)               # pick k by this, not raw inertia

pca = PCA(n_components=0.95).fit(Xs)            # keep enough PCs for 95% variance
Z = pca.transform(Xs)
pca.explained_variance_ratio_                  # variance per component
```
`n_init="auto"`/10 guards against bad inits; `PCA(n_components=0.95)` is the handy "keep 95% of variance" form. Pipe PCA → a supervised model when features are many and correlated.

## 6. Assumptions

**K-Means:**
- Clusters are **roughly spherical, similar size, similar density** (it draws Voronoi/linear boundaries; it fails on elongated or nested shapes).
- **Euclidean distance is meaningful** → features should be **scaled**; the mean must be a sensible "center."
- You **know k** in advance.

**PCA:**
- Structure is **linear** and lives in directions of **high variance** (it can't capture nonlinear manifolds → use kernel PCA / t-SNE / UMAP for those).
- Features are **centered and (usually) scaled** — otherwise high-variance-by-units features hijack the components.
- High variance = high information (usually true, not always).

## 7. When to use / not use

**K-Means — use when:** you want fast, scalable partition clustering on roughly globular groups (customer/segment discovery, vector quantization, color compression, a featurization step). **Avoid when:** clusters are non-convex/varying density (→ **DBSCAN**, spectral, GMM for soft/elliptical clusters), or $k$ is genuinely unknown and unstable.

**PCA — use when:** you have many correlated numeric features and want to **compress, denoise, decorrelate, or visualize** (project to 2-D), or speed up a downstream model. **Avoid when:** you need **interpretable original features** (components are mixtures), the structure is **nonlinear** (→ kernel PCA, UMAP), or features are categorical.

## 8. Pitfalls & gotchas
- **Not scaling first** — the #1 mistake for *both*. Unscaled, a large-unit feature dominates distances (K-Means) or variance (PCA). ⭐
- **K-Means: bad local optima from poor init** — use **k-means++** and multiple `n_init` restarts; never trust a single random start.
- **K-Means: minimizing inertia to pick k** — inertia always drops with $k$; use the **elbow or silhouette**, not raw inertia. ⭐
- **Assuming K-Means finds the global optimum** — it doesn't; the objective is non-convex.
- **K-Means on non-spherical clusters** — it can't bend boundaries; switch to DBSCAN/GMM.
- **Confusing K-Means with KNN** — K-Means is *unsupervised clustering* (learns centroids); KNN is *supervised classification* (votes neighbors). Classic trap. ⭐
- **PCA: forgetting to center** — PCA is about variance around the mean; uncentered PCA points the first component at the data's offset, not its spread.
- **PCA components aren't features** — each PC is a linear mix of all originals; don't over-interpret loadings as "feature importance."
- **PCA for supervised separation** — PCA is unsupervised; its top variance directions need not be the discriminative ones (→ **LDA** if you want class separation).
- **Leakage:** fit the scaler **and** PCA inside the CV fold, not on the full data ([evaluation/03](../evaluation/03-cross-validation-leakage.md)).

## 9. Interview questions
Must answer cold (⭐):
- ⭐ **What objective does K-Means minimize?** Inertia — the within-cluster sum of squared distances $\sum_j \sum_{x\in C_j}\lVert x-\mu_j\rVert^2$. (§2.1)
- ⭐ **Why is the update step the mean?** The mean is the point that minimizes summed squared Euclidean distance within a cluster — directly minimizing the objective for fixed assignments. (§2.1)
- ⭐ **Does K-Means converge? To the global optimum?** It always converges (each step can't increase inertia) but only to a **local** optimum; the objective is non-convex. Hence k-means++ and restarts. (§2.1–2.3)
- ⭐ **What is k-means++ and why?** Seed centroids spread out — first uniform, each next with probability ∝ $D(x)^2$ — to avoid bad local optima and converge faster. (§2.3)
- ⭐ **How do you choose k?** Elbow on inertia, **silhouette score** (more principled), or downstream/domain criteria — never just minimize inertia. (§2.4, §8)
- ⭐ **K-Means vs. KNN?** K-Means = unsupervised clustering, learns $k$ centroids by iterating assign/update. KNN = supervised, classifies a point by majority vote of its nearest labeled neighbors. (§8)
- ⭐ **What does PCA do / optimize?** Finds orthogonal directions of maximum variance; equivalently, the linear projection minimizing reconstruction error. (§3.1)
- ⭐ **How is PCA computed?** Eigendecomposition of the covariance matrix (eigenvectors = components, eigenvalues = variances) or, more stably, **SVD of the centered data**. (§3.2)
- **Why scale before PCA/K-Means?** Both are variance/distance-based; without scaling, large-unit features dominate. (§1, §8)
- **What's the time complexity of K-Means?** O(n·k·d) per iteration. (§2.2)
- **How do you pick the number of PCA components?** Cumulative explained variance (e.g., 95%), scree-plot elbow, or CV against the downstream task. (§3.3)
- **Is PCA supervised?** No — it ignores labels; for class separation use LDA. (§8)
- **When does K-Means fail?** Non-spherical / varying-density / nested clusters, or unknown k → DBSCAN, GMM, spectral clustering. (§7)
- **What do PCA's eigenvalues mean?** The variance captured along each component; their ratios are `explained_variance_ratio_`. (§3.2)

## 10. Connections
- **K-Means vs. KNN** — both use nearest-distance, but one is unsupervised clustering and the other supervised classification ([Week 2 KNN](02-knn.md)). The single most common confusion to keep straight.
- **Scaling discipline** ties to KNN ([Week 2 §7](02-knn.md#7-assumptions)) and to leakage-safe pipelines ([evaluation/03](../evaluation/03-cross-validation-leakage.md)).
- **PCA as preprocessing** for any downstream supervised model — pairs with [regularization (Week 1)](01-regularization-and-optimization.md) for taming high-dimensional, correlated features.
- **Variance is the through-line** — see [bias–variance (statistics/02)](../statistics/02-bias-variance-mle.md).
- **GMM** is the soft/probabilistic generalization of K-Means (worth a one-liner if asked).
- Implementations: [`kmeans.py`](../implementations/src/kmeans.py), [`pca.py`](../implementations/src/pca.py). Flashcards: [flashcards/06-kmeans-pca.md](../flashcards/06-kmeans-pca.md).

---
### Sources
- *ISLR* — Ch. 12 (Unsupervised Learning: PCA, K-Means, hierarchical clustering).
- Géron, *Hands-On ML* — Dimensionality Reduction (PCA) and Unsupervised Learning (K-Means, DBSCAN, GMM) chapters.
- Arthur & Vassilvitskii (2007), *k-means++: The Advantages of Careful Seeding*.
