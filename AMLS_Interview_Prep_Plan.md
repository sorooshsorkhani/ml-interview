# AMLS Interview Prep Plan

**Target:** Applied Machine Learning Scientist (weighted), open to ML Engineer / Data Scientist / AI Engineer
**Window:** late June to interviews in Oct (with September travel)
**Budget:** 5 to 8 hrs/week
**Starting point:** Linear Regression, Logistic Regression, Core ML Theory done; LeetCode Arrays Lessons 1 to 2 done

---

## The strategy in one paragraph

You have four weak areas and limited hours, so this plan does not try to cover everything evenly. It prioritizes what AMLS loops actually test: deep classical ML understanding, the ability to implement core models from scratch with clean classes, solid Easy-to-Medium coding across the high-frequency patterns, and a repeatable framework for ML problem-solving. ML theory and from-scratch implementation are merged wherever possible so one hour buys two outcomes. Frontier and agentic topics get a focused but lighter treatment near the end, which also matches your interests.

## Weekly rhythm (target ~6.5 hrs)

- **ML topic (~3 hrs):** taught in your usual format (intuition to math to algorithm to from-scratch to sklearn to assumptions to when-to-use to pitfalls)
- **Coding pattern (~2.5 hrs):** I teach the pattern, you solve 3 to 4 problems
- **Light rotating slot (~1 hr):** stats, evaluation, or system design

If a week is busy, protect the ML topic and one coding problem; let the light slot slide.

---

## Phase 1: Build (Weeks 1 to 9, late June to end of August)

### Week 1 (wk of Jun 29)
- **ML:** Regularization and optimization. Ridge, Lasso, ElasticNet, gradient descent vs SGD, learning rate, convergence. From-scratch: add L2 to your linear regression gradient descent.
- **Coding:** Finish Arrays (Lessons 3 to 4: sorting/searching, hash maps + recap), then 3 to 4 array problems.
- **Light:** Probability/stats refresh part 1 (distributions, expectation, Bayes rule).

### Week 2 (wk of Jul 6)
- **ML:** KNN. Distance metrics, k selection, curse of dimensionality. From-scratch OOP: `KNNClassifier`.
- **Coding:** Strings and hash maps (frequency counts, anagrams), 3 to 4 problems.
- **Light:** Probability/stats part 2 (bias vs variance formally, MLE intuition).

### Week 3 (wk of Jul 13)
- **ML:** Decision Trees. Entropy, Gini, information gain, overfitting, pruning. From-scratch OOP: the recursive splitting logic (a working `DecisionTree` is a stretch goal).
- **Coding:** Two pointers and sliding window consolidation, 3 to 4 problems.
- **Light:** Evaluation metrics part 1 (confusion matrix, precision, recall, F1, ROC-AUC).

### Week 4 (wk of Jul 20)
- **ML:** Ensembles 1: Bagging and Random Forests. Bootstrap, variance reduction, feature importance, OOB.
- **Coding:** Binary search and sorting, 3 to 4 problems.
- **Light:** Evaluation metrics part 2 (PR curves, log loss, calibration, class imbalance).

### Week 5 (wk of Jul 27)
- **ML:** Ensembles 2: Boosting. AdaBoost, Gradient Boosting, XGBoost intuition, bias reduction, bagging vs boosting tradeoffs.
- **Coding:** Linked lists, 3 to 4 problems.
- **Light:** Cross-validation, data leakage, train/val/test discipline.

### Week 6 (wk of Aug 3)
- **ML:** Unsupervised: K-Means and PCA. Objective, convergence, k selection, eigen-decomposition intuition. From-scratch OOP: `KMeans`.
- **Coding:** Stacks and queues, 3 to 4 problems.
- **Light:** SVM theory read-through (margin, hinge loss, kernel trick).

### Week 7 (wk of Aug 10)
- **ML:** Neural network fundamentals. Forward pass, backprop, activations, loss functions, optimizers. From-scratch: a one-hidden-layer MLP in numpy.
- **Coding:** Trees (BFS, DFS, BST operations), 3 to 4 problems.
- **Light:** Deep learning regularization (dropout, batch norm, early stopping).

### Week 8 (wk of Aug 17)
- **ML:** NLP and Transformers. Embeddings, the attention mechanism, transformer block, tokenization.
- **Coding:** Heaps and intervals, 2 to 3 problems.
- **Light:** ML system design framework intro (clarify, data, features, model, eval, serving, monitoring).

### Week 9 (wk of Aug 24)
- **ML:** LLMs and Agentic AI. Pretraining/finetuning/RLHF, RAG, agents and tool use, evaluation.
- **Coding:** Graphs basics + light intro to DP, 2 to 3 problems (or use as a catch-up buffer).
- **Light:** ML system design case 1 end to end (e.g., recommendation or fraud detection).

---

## Phase 2: Maintenance (September, travel)

Aim for 2 to 3 hrs/week, low effort, high retention:
- Spaced review of ML concepts (flashcards/Anki on the cards you build during Phase 1)
- 1 to 2 easy LeetCode problems weekly to stay warm
- Re-read the system design framework once

## Phase 3: Polish and mocks (October, into interviews)

- **Mock interviews** weekly: rotate coding, ML knowledge grilling, and an ML case/system design.
- **Timed from-scratch reps:** re-code the canonical models under 25-minute pressure.
- **Mixed Medium LeetCode** to simulate the real coding bar.
- **Behavioral:** 5 to 6 STAR stories + a deep-dive on one ML project you can defend.
- **Company-specific:** map the internal AMLS loop (format, who interviews, known focus areas) and tailor the final two weeks to it.

---

## Canonical from-scratch checklist (must code without notes by October)

These are the classics AMLS interviewers ask for. Each in clean OOP with `fit` / `predict`:

1. Linear Regression (gradient descent) [done]
2. Logistic Regression (gradient descent) [done]
3. KNN classifier
4. K-Means
5. Decision Tree (at least the split logic end to end)
6. One-hidden-layer MLP with backprop (numpy)

Stretch: Naive Bayes, PCA.

## DSA pattern priority (highest yield first)

Arrays/strings/hash maps, two pointers, sliding window, binary search, linked lists, stacks/queues, trees (BFS/DFS), heaps. Then if time: intervals, graphs, intro DP. The AMLS coding bar is usually Easy-to-Medium, so depth on these beats grinding Hard problems.

## ML system design skeleton (memorize this spine)

1. Clarify the problem and success metric (business + ML metric)
2. Frame as an ML task (inputs, outputs, labels, online vs batch)
3. Data: sources, labeling, leakage, splits
4. Features: engineering, encoding, scaling
5. Model: baseline first, then candidates, tradeoffs
6. Evaluation: offline metrics + online (A/B)
7. Serving: latency, batch vs real-time
8. Monitoring: drift, retraining, failure modes

## Resources (keep it minimal)

- **ML theory:** Hands-On Machine Learning (Geron) for applied depth; ISLR for statistical grounding.
- **DSA:** NeetCode pattern roadmap + your LeetCode Easy collection.
- **System design:** Designing Machine Learning Systems (Huyen) for the framework and cases.

## How we work each week

- ML topics run in your confirmed "conceptual with notation" style.
- Coding: I teach the pattern first, you attempt problems independently, then we review.
- I keep sessions fast and low-friction, building on prior weeks.

---

## Honest expectation-setting

At 5 to 8 hrs/week you will not master all of deep learning, transformers, and LLMs to research depth before October. That is fine. The realistic and sufficient target for an AMLS loop is: confident classical ML, the six from-scratch builds, solid Easy/Medium coding, a defensible project, and a working system-design framework. The frontier topics give you enough to discuss intelligently, which is what most loops actually probe at the scientist level.
