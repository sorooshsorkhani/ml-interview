# Cross-Validation, Leakage & Split Discipline

> **Status:** ✅ written (W5 light)
> **One-line:** How to estimate generalization **honestly** — k-fold / stratified / time-series CV — and the **leakage** traps that quietly inflate every offline number until production disappoints.

This is the "did I actually measure what I think I measured?" topic. It rarely gets its own interview question, but a leakage mistake invalidates *everything else*, and "walk me through how you'd validate this model" is a staple of ML-case and system-design rounds.

## Why a single train/test split isn't enough
A single split gives **one noisy estimate** of generalization. With a small or unlucky split, the test score can swing several points purely by chance — you can't tell skill from luck. **Cross-validation** averages the estimate over multiple splits, giving both a **lower-variance mean** and a **spread (std)** that tells you how stable the model is.

It also lets you **use all the data** for both training and evaluation (every point is in a test fold exactly once), which matters when data is scarce — the main reason CV exists.

The discipline cost: you must keep a **final, untouched test set** for the one honest number at the end. CV is for *model selection and tuning*; the held-out test set is for the *final report*. Tuning against the test set turns it into a training set.

## k-fold & stratified k-fold
- **k-fold:** split into $k$ equal folds; train on $k-1$, validate on the held-out fold, rotate. Average the $k$ scores. Typical $k = 5$ or $10$. Larger $k$ → less bias (more training data per fold), more variance and compute; **LOOCV** ($k = n$) is the extreme — nearly unbiased but high-variance and expensive.
- **Stratified k-fold:** preserve the **class proportions** in each fold. **Always use this for classification**, especially with imbalance — plain k-fold can produce folds with almost none of the minority class, wrecking the estimate. ([Imbalance deep-dive: evaluation/04](04-calibration-imbalance.md).)
- **Repeated k-fold:** repeat the whole k-fold with different shuffles to further shrink the variance of the estimate.

> **Bias–variance of the estimator itself:** small $k$ → each model trains on less data → pessimistic (biased) estimate; large $k$ → near-unbiased but the $k$ scores are highly correlated → high-variance mean. $k=5$–$10$ is the standard sweet spot.

## Time-series / grouped CV
Plain k-fold **shuffles**, which is *wrong* whenever rows aren't exchangeable:

- **Time series → never shuffle.** Use **forward-chaining / expanding-window** CV (`TimeSeriesSplit`): always train on the past, validate on the future. Shuffling lets the model "see the future," a guaranteed leak. Add an embargo/gap if features use rolling windows.
- **Grouped data → keep groups together.** If you have multiple rows per user/patient/device, use **`GroupKFold`** so the *same group never appears in both train and validation*. Otherwise the model memorizes the group, not the pattern — leaks identity, inflates the score, fails on new groups.

## Data leakage: the catalog of ways it sneaks in
**Leakage = information from outside the training fold (or from the future/target) bleeds into training**, producing great offline metrics that collapse in production. The cardinal sin of applied ML. ⭐

- **Target leakage:** a feature that's a proxy for, or computed from, the label — or only available *after* the outcome is known (e.g., `account_closed_date` predicting churn). Ask of every feature: *would this value be available at prediction time, before the label exists?*
- **Train/test contamination (preprocessing leakage):** fitting any data-dependent transform on the **whole dataset** before splitting — scaler mean/std, imputation values, PCA, feature selection, target encoding. The test statistics leak into training. **Fit transforms on the training fold only**, then apply to validation/test. (See the Pipeline section.)
- **Temporal leakage:** training on rows that occur *after* the validation rows (the time-series case above).
- **Group/identity leakage:** the same entity in train and test (the grouped case above).
- **Duplicate / near-duplicate rows** split across train and test.
- **Leakage through tuning:** selecting features or hyperparameters using the test set; or doing feature selection once on all data, then CV-ing — the selection already saw the test folds. Selection must live *inside* the CV loop.

## Doing transforms inside the CV fold (Pipeline)
The fix for preprocessing leakage is to make every data-dependent step part of the model and let CV refit it per fold:

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

pipe = Pipeline([
    ("impute", SimpleImputer(strategy="median")),  # imputation values
    ("scale",  StandardScaler()),                  # mean/std...
    ("clf",    LogisticRegression()),
])  # ...are all refit on the TRAIN fold only, every fold

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
scores = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")
print(scores.mean(), scores.std())
```
Because the scaler/imputer live inside the `Pipeline`, `cross_val_score` **refits them on each training fold** — the validation fold's statistics never leak. For tuning, wrap this in `GridSearchCV`; to get an unbiased estimate *of the tuned model*, use **nested CV** (an inner loop tunes, an outer loop scores).

## Train/val/test discipline
The clean three-way split and its roles:

- **Train** — fit model parameters.
- **Validation** — tune hyperparameters, choose features, **early stopping** (e.g., for [boosting, Week 5](../ml-theory/05-ensembles-boosting.md)), pick the model.
- **Test** — touched **once**, at the very end, for the number you report. Every peek at it spends some of its honesty.

CV typically replaces the single train/val split (validation rotates through folds), but you **still hold out a separate test set** for the final estimate. Make the split **representative**: stratify by class, split by **time** for temporal data, split by **group** for grouped data — and lock the test set away before any exploration.

## Interview questions
Must answer cold (⭐):
- ⭐ **What is data leakage and why does it matter?** Information from outside the training fold (the future, the target, or the test set) leaks into training, so offline metrics look great but the model fails in production. It invalidates the whole evaluation. (Catalog above)
- ⭐ **Why fit your scaler/imputer inside CV instead of on the full data first?** Fitting on all data lets the validation/test statistics leak into training → optimistic scores. Use a Pipeline so transforms refit on each train fold only.
- ⭐ **How do you cross-validate time-series data?** Never shuffle — use forward-chaining / expanding-window (`TimeSeriesSplit`): always train on the past, validate on the future, optionally with an embargo gap.
- ⭐ **What is stratified k-fold and when do you need it?** k-fold that preserves class proportions in each fold; use it for classification, especially imbalanced — plain k-fold can leave a fold with no minority examples.
- **Why cross-validate instead of one split?** Lower-variance estimate (averaged over folds), a stability measure (std), and full use of scarce data.
- **How do you choose k?** 5–10 typically; bigger k → less bias, more variance and compute; LOOCV is the high-variance extreme.
- **You have multiple rows per user — how do you split?** GroupKFold so no user appears in both train and validation; otherwise the model leaks identity and won't generalize to new users.
- **Give an example of target leakage.** A feature available only after the label is known (e.g., a "closed date" predicting churn) or a near-copy of the target. Test: would it exist at prediction time?
- **Where should feature selection live in a CV pipeline?** Inside the CV loop (refit per fold); selecting once on all data leaks the test folds into the selection.
- **What is nested CV for?** An unbiased estimate of a *tuned* model: inner loop tunes hyperparameters, outer loop scores, so tuning never sees the outer test folds.

---
### Sources
- Géron, *Hands-On ML* — Ch. 2 (test-set discipline, pipelines, cross-validation) & model-evaluation sections.
- Huyen, *Designing ML Systems* — data leakage chapter (the production-facing catalog).
- *ISLR* — Ch. 5 (Cross-Validation).
