# From-scratch implementations

The "**code without notes by October**" set. Clean OOP, `fit`/`predict`, numpy only for the core algorithm (sklearn allowed for comparison/testing).

```
implementations/
├── src/      # the models
└── tests/    # pytest tests (run: pytest implementations/tests/ -q)
```

## Canonical checklist (AMLS classics)

| # | Model | File | Notes | Status |
|---|---|---|---|---|
| 1 | Linear Regression (GD, +L2) | [linear_regression.py](src/linear_regression.py) | [01](../ml-theory/01-regularization-and-optimization.md) | ✅ done |
| 2 | Logistic Regression (GD) | [logistic_regression.py](src/logistic_regression.py) | [00](../ml-theory/00-foundations.md) | ⬜ port existing |
| 3 | KNN Classifier | [knn.py](src/knn.py) | [02](../ml-theory/02-knn.md) | ✅ done |
| 4 | K-Means | [kmeans.py](src/kmeans.py) | [06](../ml-theory/06-kmeans-pca.md) | ✅ done |
| 5 | Decision Tree | [decision_tree.py](src/decision_tree.py) | [03](../ml-theory/03-decision-trees.md) | ✅ done |
| 6 | One-hidden-layer MLP | [mlp.py](src/mlp.py) | [07](../ml-theory/07-neural-networks.md) | 🟡 W7 |
| S | Naive Bayes (stretch) | [naive_bayes.py](src/naive_bayes.py) | [11](../ml-theory/11-naive-bayes.md) | 🟡 stretch |
| S | PCA (stretch) | [pca.py](src/pca.py) | [06](../ml-theory/06-kmeans-pca.md) | ✅ done |

## Running

```bash
# All tests
pytest implementations/tests/ -q

# A single module's smoke test
python implementations/src/linear_regression.py
```

## Conventions
- Every model: `__init__` (hyperparameters), `fit(X, y) -> self`, `predict(X)`.
- Standardize features where the algorithm needs it (GD models, KNN, K-Means, PCA).
- Store useful learned state with a trailing underscore (`w_`, `cluster_centers_`, `loss_history_`) — mirrors sklearn.
- Seeds via `random_state` for reproducible tests.

## Timed-rep tracker (Phase 3)
Re-code each from memory under 25 min. Log attempts:

| Model | Attempt 1 (date / time / pass?) | Attempt 2 | Attempt 3 |
|---|---|---|---|
| Linear Regression | | | |
| Logistic Regression | | | |
| KNN | | | |
| K-Means | | | |
| Decision Tree | | | |
| MLP | | | |
