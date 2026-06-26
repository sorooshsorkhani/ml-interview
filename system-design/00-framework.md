# ML System Design — The Framework (the spine to memorize)

> **Status:** ✅ written
> One reusable structure you can apply to *any* ML system design prompt. Say these section headers out loud and the interviewer will follow you. Depth > breadth — go deep where the interviewer pushes.

## The 8-step spine

```
1. Clarify   → problem + success metric (business AND ml)
2. Frame     → ML task: inputs, outputs, labels, online vs batch
3. Data      → sources, labeling, leakage, splits
4. Features  → engineering, encoding, scaling, real-time vs precomputed
5. Model     → baseline first, then candidates + tradeoffs
6. Eval      → offline metrics + online (A/B) + guardrails
7. Serving   → latency, batch vs real-time, infra
8. Monitor   → drift, retraining, failure modes, feedback loops
```

Memory hook: **C-F-D-F-M-E-S-M** ("Clarify Frames Data, Features Model Eval Serve Monitor").

---

## 1. Clarify the problem & success metric
**Spend real time here — interviewers grade this.** Don't jump to models.
- What's the **business goal**? (revenue, engagement, safety, cost) and how does ML serve it?
- What's the **ML objective** and how does it connect to the business metric? (often a proxy)
- **Constraints:** latency budget, QPS/scale, cost, privacy/regulatory, interpretability.
- **Online vs. batch?** Real-time scoring or precomputed?
- Scope: what's in/out? Cold-start? Who/what are the users?
- Ask 3–5 sharp clarifying questions before designing.

## 2. Frame as an ML task
- **Inputs → outputs:** what does one prediction look like?
- **Task type:** binary/multiclass/regression/ranking/retrieval/generation.
- **Labels:** where do they come from? Implicit (clicks) vs. explicit (human)? **Label delay**, noise, bias?
- Single model vs. a system (candidate generation → ranking → re-ranking)?

## 3. Data
- **Sources** and volume; can we log what we need?
- **Labeling strategy:** human annotation, weak/heuristic labels, implicit feedback. Cost & quality.
- **Class imbalance** strategy (resampling, class weights, threshold).
- **Leakage:** the silent killer. No future info, no target-derived features, fit transforms on train only.
- **Splits:** random vs. **time-based** (almost always time-based for production systems), grouped (avoid same-user leakage).

## 4. Features
- Engineering & encoding (categorical: one-hot/target/embeddings; numeric: scaling, bucketing).
- Text/image/sequence representations (embeddings).
- **Real-time vs. precomputed** features; a **feature store** for train/serve consistency.
- **Train/serve skew** — features must be computed identically online and offline.

## 5. Model
- **Baseline first** (heuristic or logistic regression / GBM) — sets the bar and ships fast.
- Candidate models + **tradeoffs**: accuracy vs. latency vs. interpretability vs. training cost.
- For most tabular problems: **gradient-boosted trees** are the strong default. Deep learning for text/image/large-scale.
- Loss function choice; how it ties to the metric.

## 6. Evaluation
- **Offline metrics** aligned to the task (see [evaluation/](../evaluation/)): AUC/PR for imbalanced classification, NDCG/MAP for ranking, RMSE/MAE for regression.
- **Pick the metric that matches the cost structure** (precision vs. recall depending on FP/FN cost).
- **Online:** A/B test the business metric; **guardrail metrics** (latency, revenue, fairness) so you don't ship a regression.
- Offline–online gap: offline wins don't always transfer.

## 7. Serving
- **Latency budget** and how the architecture meets it (model size, caching, approximate retrieval like ANN).
- **Batch** (precompute, store) vs. **real-time** (low-latency inference service).
- Candidate generation + ranking split to keep latency low at scale.
- Model deployment: shadow mode, canary, rollback.

## 8. Monitoring
- **Data drift** (input distribution shifts) and **concept drift** (relationship shifts).
- **Retraining cadence:** scheduled vs. triggered by drift/perf drop.
- **Failure modes:** stale features, training/serving skew, feedback loops (the model affects the data it later trains on).
- Alerting on prediction distribution, latency, and downstream business metrics.

---

## Worked cases
| Case | File | Status |
|---|---|---|
| Recommendation system | [01-recommendation.md](01-recommendation.md) | ✅ written (W9) |
| Fraud detection | [02-fraud-detection.md](02-fraud-detection.md) | 🟡 |
| Search / feed ranking | [03-search-ranking.md](03-search-ranking.md) | 🟡 |

## Cheat phrases that score points
- "Let me clarify the objective and constraints before I design."
- "I'd start with a simple baseline to establish a bar."
- "I'd use a **time-based split** to avoid leakage."
- "Offline I'd track X; online I'd A/B test the business metric with latency as a guardrail."
- "For monitoring I'd watch for data and concept drift and set a retraining trigger."

---
### Sources
- Chip Huyen, *Designing Machine Learning Systems* — the framework, data, and deployment chapters.
