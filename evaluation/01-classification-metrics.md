# Classification Metrics

> **Status:** ✅ written · **Light slot:** Week 3
> **One-line:** Accuracy lies under class imbalance; precision/recall/F1 and ROC/PR-AUC tell you what the model actually does, and the right metric depends on the cost of false positives vs. false negatives.

## The confusion matrix (everything derives from this)

```
                 Predicted +     Predicted -
Actual +           TP              FN          ← recall = TP/(TP+FN)
Actual -           FP              TN
                   ↑
              precision = TP/(TP+FP)
```

- **TP/TN/FP/FN** — true/false positives/negatives.
- **FP = Type I error** (false alarm); **FN = Type II error** (miss).

## The core metrics

| Metric | Formula | Answers | Use when |
|---|---|---|---|
| Accuracy | (TP+TN)/all | overall correctness | balanced classes |
| **Precision** | TP/(TP+FP) | of predicted positives, how many are right | FP is costly (spam flag, ad spend) |
| **Recall** (sensitivity, TPR) | TP/(TP+FN) | of actual positives, how many we caught | FN is costly (cancer, fraud) |
| **F1** | 2·P·R/(P+R) | harmonic mean of P & R | imbalance + you care about both |
| Specificity (TNR) | TN/(TN+FP) | of actual negatives, how many we cleared | |
| FPR | FP/(FP+TN) | false-alarm rate | x-axis of ROC |

**Harmonic mean** for F1 (not arithmetic) so a model can't game it by being great at one and terrible at the other — F1 stays low unless *both* P and R are decent. Generalize to **Fβ** to weight recall (β>1) or precision (β<1).

## The precision–recall tradeoff
A classifier outputs a **score/probability**; you pick a **threshold**. Raising the threshold → fewer positives → higher precision, lower recall (and vice versa). There is no single "accuracy" — you choose the operating point from the cost structure.

## Threshold-independent curves
- **ROC curve:** TPR vs. FPR across all thresholds. **AUC** = P(model ranks a random positive above a random negative). 0.5 = random, 1.0 = perfect.
  - ➕ threshold-free, summarizes ranking quality.
  - ➖ **overly optimistic under heavy imbalance** (lots of TN inflate it).
- **PR curve:** Precision vs. Recall. **Prefer PR-AUC when positives are rare** (fraud, disease) — it focuses on the minority class and ignores the easy TN mass.
  - Baseline PR-AUC = the positive rate (not 0.5).

## Probabilistic / regression-style metric
- **Log loss (cross-entropy):** penalizes confident wrong predictions heavily; rewards **calibrated** probabilities, not just correct ranking.

## Calibration (often forgotten, often asked)
A model is **calibrated** if among samples predicted 0.7, ~70% are actually positive. AUC can be perfect while calibration is awful (ranking ≠ probabilities). Check with a **reliability diagram**; fix with **Platt scaling** (sigmoid) or **isotonic regression**. Matters when you use the probability itself (expected value, thresholds tied to cost).

## Class imbalance toolkit
- Metrics: use PR-AUC / F1 / recall, **not** accuracy.
- Data: resampling (SMOTE / undersampling), class weights.
- Threshold: tune it to the cost, don't default to 0.5.
- Report the **base rate** so numbers have context.

## Multiclass
- **Macro** average (mean over classes — treats classes equally) vs. **micro** (aggregate over all samples — dominated by frequent classes) vs. **weighted**.
- Confusion matrix generalizes to K×K.

## Interview questions
- ⭐ Precision vs. recall — define both and give a scenario where each dominates.
- ⭐ When is accuracy a bad metric? (imbalance) What do you use instead?
- ⭐ ROC-AUC vs. PR-AUC — when prefer PR? (rare positives)
- ⭐ What does AUC mean probabilistically? (ranking interpretation)
- Why harmonic mean for F1?
- What is calibration and how do you fix a miscalibrated model?
- A fraud model has 99% accuracy — why might that be useless? (1% fraud → predict-all-legit scores 99%)
- How do you choose a classification threshold?

## Connections
- Validation discipline & leakage: [03-cross-validation-leakage.md](03-cross-validation-leakage.md).
- Regression metrics: [02-regression-metrics.md](02-regression-metrics.md).
- Feeds [system design eval step](../system-design/00-framework.md#6-evaluation).

---
### Sources
- Géron, *Hands-On ML* — Ch. 3 (Classification: confusion matrix, P/R, ROC). · *ISLR* — Ch. 4.
