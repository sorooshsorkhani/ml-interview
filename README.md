# ml-interview

My personal, public, constantly-evolving prep for **Applied Machine Learning Scientist (AMLS)** interviews — open also to ML Engineer / Data Scientist / AI Engineer roles.

> Built for me. Public, but not (yet) intended as a course for others. Expect rough edges and personal shorthand.

- **Target window:** late June → interviews in October (September is a travel/maintenance month)
- **Budget:** ~5–8 hrs/week
- **Plan of record:** [AMLS_Interview_Prep_Plan.md](AMLS_Interview_Prep_Plan.md)

---

## How this repo is organized

| Folder | What lives here |
|---|---|
| [`ml-theory/`](ml-theory/) | The core study notes. One file per topic: intuition → math → algorithm → from-scratch → sklearn → assumptions → when-to-use → pitfalls. |
| [`implementations/`](implementations/) | From-scratch model code (`src/`) with `fit`/`predict` OOP, plus `tests/`. The "code without notes by October" set. |
| [`dsa/`](dsa/) | Data structures & algorithms. One file per pattern: the pattern, template code, and a curated problem list with notes. |
| [`system-design/`](system-design/) | The ML system design framework + worked case studies. |
| [`statistics/`](statistics/) | Probability & statistics refreshers (the "light slot" material). |
| [`evaluation/`](evaluation/) | Metrics, validation discipline, calibration, leakage. |
| [`behavioral/`](behavioral/) | STAR stories, project deep-dive, recruiter/role notes. |
| [`flashcards/`](flashcards/) | Markdown Q&A decks per topic for spaced repetition (Phase 2 maintenance). |
| [`statistics/`](statistics/), [`evaluation/`](evaluation/) | Rotating light-slot topics. |
| [`templates/`](templates/) | Reusable templates so every new note/topic looks the same. |
| [`progress/`](progress/) | Weekly log + the master checklist of what's done. |
| [`resources/`](resources/) | Book/course/link list, minimal and curated. |

---

## Master table of contents

### 1. ML Theory ([`ml-theory/`](ml-theory/))
| # | Topic | Week | Status |
|---|---|---|---|
| 00 | [Foundations recap (Linear & Logistic Regression, core theory)](ml-theory/00-foundations.md) | done | ✅ done (prior) |
| 01 | [Regularization & Optimization](ml-theory/01-regularization-and-optimization.md) | W1 | ✅ written |
| 02 | [K-Nearest Neighbors](ml-theory/02-knn.md) | W2 | ✅ written |
| 03 | [Decision Trees](ml-theory/03-decision-trees.md) | W3 | 🟡 scaffold |
| 04 | [Ensembles 1: Bagging & Random Forests](ml-theory/04-ensembles-bagging.md) | W4 | 🟡 scaffold |
| 05 | [Ensembles 2: Boosting (AdaBoost, GBM, XGBoost)](ml-theory/05-ensembles-boosting.md) | W5 | 🟡 scaffold |
| 06 | [Unsupervised: K-Means & PCA](ml-theory/06-kmeans-pca.md) | W6 | 🟡 scaffold |
| 07 | [Neural Network Fundamentals](ml-theory/07-neural-networks.md) | W7 | 🟡 scaffold |
| 08 | [NLP & Transformers](ml-theory/08-nlp-transformers.md) | W8 | 🟡 scaffold |
| 09 | [LLMs & Agentic AI](ml-theory/09-llms-agentic.md) | W9 | 🟡 scaffold |
| + | [SVM (theory read-through)](ml-theory/10-svm.md) | W6 light | 🟡 scaffold |
| + | [Naive Bayes (stretch)](ml-theory/11-naive-bayes.md) | stretch | 🟡 scaffold |

### 2. From-scratch implementations ([`implementations/`](implementations/))
Canonical builds — must code without notes by October. See [`implementations/README.md`](implementations/README.md).

| # | Model | Status |
|---|---|---|
| 1 | [Linear Regression (GD)](implementations/src/linear_regression.py) | ✅ done (prior) — to port |
| 2 | [Logistic Regression (GD)](implementations/src/logistic_regression.py) | ✅ done (prior) — to port |
| 3 | [KNN Classifier](implementations/src/knn.py) | ✅ written |
| 4 | [K-Means](implementations/src/kmeans.py) | 🟡 scaffold |
| 5 | [Decision Tree](implementations/src/decision_tree.py) | 🟡 scaffold |
| 6 | [One-hidden-layer MLP (numpy)](implementations/src/mlp.py) | 🟡 scaffold |
| S | [Naive Bayes (stretch)](implementations/src/naive_bayes.py) | 🟡 scaffold |
| S | [PCA (stretch)](implementations/src/pca.py) | 🟡 scaffold |

### 3. DSA ([`dsa/`](dsa/))
See [`dsa/README.md`](dsa/README.md) for the pattern priority order and progress.

Arrays/strings/hash maps → two pointers → sliding window → binary search → linked lists → stacks/queues → trees (BFS/DFS) → heaps → (then) intervals, graphs, intro DP.

### 4. System design ([`system-design/`](system-design/))
[Framework](system-design/00-framework.md) + cases (recommendation, fraud, search ranking, …).

### 5. Light-slot topics
- [Statistics & probability](statistics/)
- [Evaluation & validation](evaluation/)

### 6. Behavioral & logistics ([`behavioral/`](behavioral/))
STAR stories, project deep-dive, company/loop notes.

### 7. Flashcards ([`flashcards/`](flashcards/)) · Progress ([`progress/`](progress/)) · Resources ([`resources/`](resources/))

---

## Status legend
- ✅ written / done
- 🟡 scaffold (structure + outline present, content to be filled interactively)
- ⬜ not started

## Working agreement (how we evolve this repo)
- ML notes use my "conceptual with notation" style: intuition → math → algorithm → from-scratch → sklearn → assumptions → when-to-use → pitfalls.
- DSA: pattern taught first, then I attempt problems, then review.
- Everything is iterative — I'll ask for clarifications/expansions and we update files in place.
- New notes start from [`templates/`](templates/) so structure stays consistent.
