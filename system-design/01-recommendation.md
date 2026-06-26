# System Design Case: Recommendation System

> **Status:** ✅ written (Week 9 light slot — the first end-to-end case)
> **Prompt:** "Design a recommendation system (e.g., recommend products / videos / posts to users)."

Worked end-to-end against the [framework spine](00-framework.md). Each section is what you'd say out loud; ⭐ marks the points interviewers reward. The headline architecture to state early: a **two-stage candidate-generation → ranking** funnel, because you can't score millions of items per request under latency. ⭐

## 1. Clarify the problem & success metric
**Ask first, design second** — spend real time here.
- **Business goal:** what are we optimizing — engagement (watch time, clicks), revenue (purchases/GMV), retention, or discovery? This drives everything downstream. Say: *"Let me pin the objective and constraints before designing."*
- **ML objective as a proxy:** e.g. predict P(click) or P(purchase | shown) — and acknowledge it's a *proxy* for the real goal (long-term engagement/revenue), which is why we also A/B test the business metric.
- **Constraints:** scale (how many users / items — millions?), latency budget (recommendations in <100–200 ms), QPS, freshness (do new items/users matter — **cold start**?), is it a feed (many items) or a single "next" item?
- **Online vs. batch:** typically **real-time ranking** with **precomputed candidates/embeddings**.
- **Clarifying questions to ask:** What surface (homepage feed, "related items", email)? How many items to return? Is implicit feedback (clicks) available or only explicit ratings? Any business rules (diversity, freshness, no-repeats, ads)?

## 2. Frame as an ML task
- **Inputs:** a (user, item, context) triple → **outputs:** a relevance score; rank items by score and return the top-N.
- **Task type:** **learning-to-rank** (or pointwise P(engagement) which we sort by). Two-stage:
  - **Candidate generation (retrieval):** millions → ~hundreds. Cheap, high-recall.
  - **Ranking:** hundreds → ordered top-N. Expensive, high-precision, many features. ⭐
- **Labels:** **implicit feedback** — clicks/watches/purchases as positives; impressions-without-action as (weak) negatives. Cheap and plentiful but **biased** (you only observe feedback on what you *showed* — *position bias*, *exposure bias*). Explicit ratings are cleaner but sparse. Note **label delay** (a purchase/return comes later).

## 3. Data
- **Sources:** interaction logs (clicks, views, watch time, purchases, add-to-cart), user profile/history, item metadata (category, price, text, images), context (time, device, location).
- **Labeling:** define a positive (click? watch > X seconds? purchase?) — clarify with the business. Sample negatives (impressed-not-clicked, plus **random/in-batch negatives** for retrieval).
- **Leakage:** the silent killer — no future interactions in features; **time-based split** (train on the past, validate on the future), never random (avoids leaking tomorrow's behavior into today). ⭐
- **Imbalance:** engagement is rare (most impressions aren't clicked) → negative downsampling, class weights, calibrate afterward ([evaluation/04](../evaluation/04-calibration-imbalance.md)).
- **Bias:** logs reflect what the *current* system showed (feedback loop / exposure bias) — mitigate with exploration (ε-random items), inverse-propensity weighting.

## 4. Features
- **User:** historical interactions, learned **user embedding**, demographics, recency/frequency aggregates.
- **Item:** **item embedding**, category, price, popularity, age, content features (text/image embeddings — ties to [Week 8](../ml-theory/08-nlp-transformers.md)).
- **User × item cross:** past affinity to this category/brand, similarity(user_emb, item_emb).
- **Context:** time of day, device, session, what they just viewed.
- **Real-time vs. precomputed:** item & user embeddings **precomputed** (batch, refreshed periodically); session/context features **computed online**. A **feature store** ensures **train/serve consistency** (no skew). ⭐

## 5. Model
- **Baseline first** (sets the bar, ships fast): **popularity** / most-recent, then **collaborative filtering** — user-item **matrix factorization** (learn user & item latent vectors; score = dot product). Explain the cold-start weakness. ⭐
- **Candidate generation:** **two-tower model** — a user tower and an item tower each produce an embedding; relevance = dot product. Precompute all item embeddings, index with **ANN** (approximate nearest neighbor) so retrieval is sublinear. (This is [KNN, Week 2](../ml-theory/02-knn.md) / the same machinery as [RAG retrieval](../ml-theory/09-llms-agentic.md).) Also content-based & co-visitation candidates → blend sources. ⭐
- **Ranking:** a richer model over the candidate set with all cross features — **gradient-boosted trees** ([Week 5](../ml-theory/05-ensembles-boosting.md)) as a strong default, or a **deep ranking net** (e.g. two-tower + MLP, DLRM, wide & deep) at scale. Optimize a ranking/logistic loss.
- **Re-ranking / business logic:** diversity (don't show 10 near-duplicates), freshness, de-dup, exploration, ads/sponsored, hard filters (out-of-stock, already-purchased).
- **Cold start:** new users → popularity/onboarding signals/context; new items → content features until interactions accrue.
- **Tradeoffs:** MF (simple, cold-start-weak) vs. two-tower (scalable retrieval) vs. deep ranker (accurate, costlier); accuracy vs. latency vs. interpretability.

## 6. Evaluation
- **Offline (ranking metrics):** **NDCG@k**, MAP, MRR, Recall@k / Hit-rate@k for retrieval, AUC for the pointwise scorer. Pick by surface (top-1 vs. a feed). ([evaluation/](../evaluation/))
- **Watch the offline–online gap** — offline wins don't always transfer (the logs are biased by the old model).
- **Online (the real test):** **A/B test** the business metric (CTR, watch time, GMV, retention). **Guardrail metrics**: latency, revenue, diversity, fairness — so a CTR win that tanks latency or diversity doesn't ship. ⭐
- **Counterfactual / off-policy** evaluation (IPS) to estimate online impact offline.

## 7. Serving
- **Two-stage pipeline at request time:** ANN retrieval (candidates) → feature fetch → ranker → re-rank → top-N. Meet the latency budget by keeping retrieval approximate and the ranker on a bounded candidate set. ⭐
- **Precompute** item embeddings + ANN index offline (refresh hourly/daily); compute user/session features online.
- **Caching** popular queries/users; **deployment** via shadow → canary → rollback.
- Scale: candidate generation keeps the expensive ranker off millions of items — the whole reason for the funnel.

## 8. Monitoring
- **Data drift** (input distributions shift — new items, seasonality) and **concept drift** (tastes change).
- **Feedback loop / popularity bias:** the model shapes the data it later trains on (rich-get-richer) — monitor catalog coverage/diversity; inject exploration. ⭐
- **Retraining cadence:** frequent (daily/streaming for embeddings) given fast-moving behavior; trigger on metric drops.
- **Failure modes:** stale features, train/serve skew, cold-start gaps, ANN index staleness.
- **Alerting:** on CTR/engagement, prediction distribution, latency, and coverage.

## Likely follow-ups
- **"How do you handle cold start?"** → content features + popularity + context for new users/items; onboarding signals; explore-exploit. (§5)
- **"How do you scale to millions of items under 100 ms?"** → two-stage funnel; ANN retrieval; precomputed embeddings; cache. (§1, §5, §7)
- **"Your offline AUC went up but the A/B test was flat — why?"** → offline–online gap from biased logs / position bias / proxy-vs-true objective mismatch. (§6)
- **"How do you avoid a filter bubble / over-recommending popular items?"** → diversity re-ranking, exploration, popularity-debiasing, coverage monitoring. (§5, §8)
- **"Collaborative vs. content-based filtering?"** → CF uses interaction patterns (great with data, cold-start-weak); content-based uses item/user features (handles cold start, can over-specialize); **hybrid** in practice. (§5)
- **"How do you pick positives/negatives from logs?"** → define engagement threshold; impressed-not-clicked + in-batch/random negatives; correct for exposure/position bias. (§2, §3)
- **"How would you incorporate an LLM?"** → content embeddings for items, semantic retrieval, cold-start via text, or generated explanations — but keep a GBM/two-tower core for latency/cost. (ties to [Week 8/9](../ml-theory/09-llms-agentic.md))
