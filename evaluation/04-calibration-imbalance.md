# Calibration & Class Imbalance (deep dive)

> **Status:** ✅ written · **Light slot:** Week 4 (Evaluation part 2)
> **One-line:** Making predicted probabilities *trustworthy* (calibration), and handling rare-positive problems end to end (metrics, data, threshold) — the two failure modes accuracy/ROC hide.

This is the "part 2" companion to [classification metrics](01-classification-metrics.md). Part 1 covered the confusion matrix, precision/recall/F1, ROC vs. PR-AUC. Here we go deeper on **PR curves, log loss, calibration, and the imbalance toolkit** — the bits that separate "I know the metrics" from "I know what to do."

## Table of contents
- [1. PR curves & log loss (recap + depth)](#1-pr-curves--log-loss-recap--depth)
- [2. What calibration is](#2-what-calibration-is)
- [3. Reliability diagrams](#3-reliability-diagrams)
- [4. Platt scaling vs. isotonic regression](#4-platt-scaling-vs-isotonic-regression)
- [5. The class-imbalance toolkit](#5-the-class-imbalance-toolkit)
  - [5.1 Resampling (SMOTE, under/over)](#51-resampling-smote-underover)
  - [5.2 Class weights](#52-class-weights)
  - [5.3 Threshold tuning](#53-threshold-tuning)
- [6. Interview questions](#6-interview-questions)
- [7. Connections](#7-connections)

---

## 1. PR curves & log loss (recap + depth)

**PR curve** plots **precision (y) vs. recall (x)** as the threshold sweeps. Use it over ROC when **positives are rare**: ROC's x-axis is FPR = FP/(FP+TN), and with a huge negative pool TN swamps FP so FPR stays tiny → ROC looks great even for a weak model. PR uses precision = TP/(TP+FP), which *feels* every false positive, so it tracks minority-class performance honestly.
- **PR-AUC baseline = the positive rate** (e.g., 0.01 for 1% positives), **not 0.5**. Always state the baseline so a PR-AUC of 0.3 reads as "30× better than chance," not "bad."
- **Average Precision (AP)** is the standard summary of the PR curve (sklearn's `average_precision_score`).

**Log loss (cross-entropy)** scores **probabilities**, not labels:
$$
\text{LogLoss} = -\frac{1}{n}\sum_{i} \big[\, y_i \log \hat p_i + (1-y_i)\log(1-\hat p_i)\,\big]
$$
It punishes **confident and wrong** brutally (predicting 0.99 when the truth is 0) — a single such case can dominate. It is **minimized only by calibrated, correct probabilities**, which is why it's the natural training/eval loss when you care about the probability value, not just the ranking. (Brier score = mean squared error of probabilities is the other proper scoring rule; similar spirit, less aggressive on confident errors.)

> **Key distinction:** AUC/PR-AUC measure **ranking** (does the model order positives above negatives?). Log loss / Brier measure **calibration + correctness** (are the probabilities right?). A model can ace AUC and fail log loss. ⭐

## 2. What calibration is

A classifier is **calibrated** if its probabilities mean what they say: **among all samples it scores 0.7, about 70% are truly positive.** Formally $P(y=1 \mid \hat p = p) = p$ for all $p$.

Calibration is **orthogonal to ranking/accuracy**: a model can rank perfectly (AUC = 1.0) yet be wildly miscalibrated (e.g., all its scores squashed into [0.4, 0.6]). It only matters when you **use the probability itself**:
- Expected-value decisions (cost = p · loss).
- Thresholds tied to a business cost.
- Feeding the score downstream (bidding, triage, risk).

If you only need the top-K or a hard label, calibration may not matter — but interviewers love it because it's the subtle one.

**Which models are (mis)calibrated out of the box?**
- **Logistic regression** — naturally well-calibrated (it optimizes log loss).
- **Naive Bayes** — pushed toward 0/1 (independence assumption → overconfident).
- **SVM** — outputs un-probabilistic margins; needs calibration.
- **Random Forest / boosting** — often miscalibrated (RF averaging pulls probabilities toward the middle; boosting toward extremes).

## 3. Reliability diagrams

The diagnostic for calibration. Procedure:
1. Bin predictions by predicted probability (e.g., 10 bins: [0,0.1), …, [0.9,1]).
2. For each bin, plot **mean predicted probability (x)** vs. **observed positive fraction (y)**.
3. A perfectly calibrated model lies on the **diagonal** y = x.

Reading it: points **below** the diagonal = **overconfident** (predicts higher than reality); **above** = **underconfident**. Summarize the gap numerically with **Expected Calibration Error (ECE)** — the bin-size-weighted average of |predicted − observed|.

## 4. Platt scaling vs. isotonic regression

Both are **post-hoc** fixes: fit a calibration map on a **held-out set** (never the training data — that leaks) that transforms raw scores → calibrated probabilities.

| | Platt scaling | Isotonic regression |
|---|---|---|
| Model | fit a **sigmoid** `σ(a·s + b)` to scores `s` | fit a **monotonic step function** (non-parametric) |
| Flexibility | low (assumes sigmoidal distortion) | high (any monotonic shape) |
| Data needed | works on **small** held-out sets | needs **more** data or it overfits |
| Failure mode | underfits non-sigmoid distortion | overfits / steppy on little data |
| sklearn | `CalibratedClassifierCV(method="sigmoid")` | `CalibratedClassifierCV(method="isotonic")` |

Rule of thumb: **Platt for small calibration sets, isotonic when you have enough data** and the distortion isn't sigmoidal. Both only **monotonically remap** scores, so they **don't change the ranking** → AUC is unchanged; log loss / ECE improve.

## 5. The class-imbalance toolkit

Imbalance (say 1% positives) breaks naive training and evaluation. Attack it on **three fronts**: metrics, data, and threshold. (Metrics covered in [part 1](01-classification-metrics.md): use PR-AUC / F1 / recall, never accuracy — "99% accuracy" is just the base rate of predicting all-negative.)

### 5.1 Resampling (SMOTE, under/over)
- **Random oversampling** — duplicate minority examples. Simple; risks overfitting exact copies.
- **Random undersampling** — drop majority examples. Fast, but throws away data/signal.
- **SMOTE (Synthetic Minority Over-sampling)** — create *synthetic* minority points by interpolating between a minority sample and its nearest minority neighbors. Reduces the exact-duplicate overfitting of plain oversampling.
- ⚠️ **Resample only the training fold, inside CV** — resampling before the split **leaks** (synthetic/duplicated points bleed across folds) and the same logic applies to the test set: **never resample validation/test**; evaluate on the real distribution.

### 5.2 Class weights
Reweight the **loss** so minority errors cost more (e.g., weight ∝ inverse class frequency). `class_weight="balanced"` in sklearn (logistic regression, SVM, trees, RF). Often the **cleanest first move** — no data duplication, no leakage risk, one parameter. Equivalent in spirit to oversampling but acts on the objective rather than the data.

### 5.3 Threshold tuning
The default **0.5 threshold is arbitrary** under imbalance and under asymmetric costs. Choose the operating point deliberately:
- Pick the threshold that **maximizes F1** (or Fβ to favor recall), or
- Set it from the **cost matrix**: choose the threshold minimizing expected cost = `FP·cost_FP + FN·cost_FN`, or
- Hit a **target recall/precision** (e.g., "catch 95% of fraud," then read off the precision).

This is free (no retraining) and often the highest-impact lever. Threshold tuning needs **calibrated-enough scores or at least good ranking** — which loops back to §2.

> **The complete answer to "your data is imbalanced, what do you do?":** (1) right metric — PR-AUC/recall, report the base rate; (2) class weights or resampling (train fold only, inside CV); (3) tune the threshold to the cost; (4) check calibration if the probability is used. ⭐

## 6. Interview questions
- ⭐ **What is calibration and how is it different from accuracy/AUC?** Calibrated = predicted probabilities match observed frequencies (0.7 → 70% positive). Orthogonal to ranking: a model can have AUC 1.0 and terrible calibration. Matters when you use the probability value. (§2)
- ⭐ **How do you check and fix calibration?** Check with a reliability diagram / ECE; fix with Platt scaling (sigmoid, small data) or isotonic regression (monotonic, more data), fit on a held-out set. (§3, §4)
- ⭐ **Platt vs. isotonic — when each?** Platt for small calibration sets / sigmoidal distortion; isotonic when you have enough data and the distortion is arbitrary monotonic. Neither changes ranking/AUC. (§4)
- ⭐ **Your dataset is 1% positive — what do you do?** Right metric (PR-AUC/recall, report base rate), class weights or resampling on the train fold only, threshold tuning to the cost, check calibration. (§5)
- ⭐ **Why prefer PR-AUC over ROC-AUC under heavy imbalance?** ROC's FPR is diluted by the huge TN pool so it looks optimistic; PR's precision feels every FP and tracks the minority class. (§1)
- **What does log loss penalize that accuracy doesn't?** Confident wrong probabilities; it rewards calibrated, correct probabilities, not just correct labels. (§1)
- **Why is SMOTE before the train/test split a bug?** It leaks synthetic minority points across the split, inflating validation scores; resample only the training fold inside CV. (§5.1)
- **Which models are typically miscalibrated?** Naive Bayes and SVM (and often RF/boosting); logistic regression is naturally calibrated. (§2)
- **Class weights vs. resampling — difference?** Weights reweight the loss (no data change, no leakage risk); resampling changes the data distribution. Often try weights first. (§5.1–5.2)
- **Why is 0.5 the wrong default threshold under imbalance/asymmetric cost?** It's arbitrary; pick the threshold by max-F1, target recall, or minimum expected cost. (§5.3)

## 7. Connections
- Builds on [classification metrics part 1](01-classification-metrics.md) (confusion matrix, P/R, ROC vs PR).
- Resampling-leakage point ties to [cross-validation & leakage](03-cross-validation-leakage.md) — resample *inside* the CV loop.
- Calibration of tree ensembles connects to [Random Forests / boosting](../ml-theory/04-ensembles-bagging.md) (averaging pulls probs to the middle; boosting to the extremes).
- Logistic regression's natural calibration ties to [foundations](../ml-theory/00-foundations.md) (it minimizes log loss).
- Feeds the [system design eval step](../system-design/00-framework.md#6-evaluation) (pick metric + threshold from cost).
- Flashcards: [../flashcards/11-evaluation.md](../flashcards/11-evaluation.md).

---
### Sources
- Géron, *Hands-On ML* — Ch. 3 (classification metrics, threshold/precision–recall tradeoff).
- *ISLR* — Ch. 4 (classification, thresholds). · scikit-learn user guide — Probability calibration.
