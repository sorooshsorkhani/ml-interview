# DSA — Data Structures & Algorithms

The AMLS coding bar is usually **Easy → Medium**. Depth on high-yield patterns beats grinding Hard problems. One file per pattern: the trigger, template code, complexity, and a curated problem list with my notes.

## Pattern priority (highest yield first)

| Order | Pattern | File | Week | Status |
|---|---|---|---|---|
| 1 | Arrays / Strings / Hash maps | [01-arrays-strings-hashmaps.md](01-arrays-strings-hashmaps.md) | W1–W2 | ✅ written |
| 2 | Two pointers | [02-two-pointers.md](02-two-pointers.md) | W3 | 🟡 |
| 3 | Sliding window | [03-sliding-window.md](03-sliding-window.md) | W3 | 🟡 |
| 4 | Binary search | [04-binary-search.md](04-binary-search.md) | W4 | 🟡 |
| 5 | Linked lists | [05-linked-lists.md](05-linked-lists.md) | W5 | 🟡 |
| 6 | Stacks & queues | [06-stacks-queues.md](06-stacks-queues.md) | W6 | 🟡 |
| 7 | Trees (BFS / DFS / BST) | [07-trees.md](07-trees.md) | W7 | 🟡 |
| 8 | Heaps | [08-heaps.md](08-heaps.md) | W8 | 🟡 |
| 9 | Intervals | [09-intervals.md](09-intervals.md) | W8 | 🟡 |
| 10 | Graphs | [10-graphs.md](10-graphs.md) | W9 | 🟡 |
| 11 | Intro DP | [11-dynamic-programming.md](11-dynamic-programming.md) | W9 | 🟡 |

## How I work coding
1. Read the pattern note (trigger + template).
2. Attempt 3–4 problems myself, timed.
3. Review: where did I reach for the wrong pattern? what edge case bit me? → log in the note's "Notes after solving".

## Big-O quick reference

| Structure | Access | Search | Insert | Delete |
|---|---|---|---|---|
| Array | O(1) | O(n) | O(n) | O(n) |
| Hash map | — | O(1) avg | O(1) avg | O(1) avg |
| Linked list | O(n) | O(n) | O(1)* | O(1)* |
| Balanced BST | O(log n) | O(log n) | O(log n) | O(log n) |
| Heap | — | O(n) | O(log n) | O(log n) (top) |

\* given the node reference.

## Resource
NeetCode pattern roadmap + my LeetCode Easy collection. Track overall progress in [../progress/checklist.md](../progress/checklist.md).
