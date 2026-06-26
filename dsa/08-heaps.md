# DSA Pattern: Heaps / Priority Queues

> **Priority:** see [README](README.md) · **Status:** ✅ written
> **One-line:** A heap gives you the **min (or max) in O(1)** and push/pop in **O(log n)** — the tool whenever you need a **running extreme** or the **top-k** without fully sorting. The tells: *"k largest/smallest/closest", "merge k sorted things", "median of a stream"* (two heaps), *"schedule by priority."*

The unifying idea: when you repeatedly need "the smallest (or largest) thing remaining" and the set keeps changing, a heap beats re-sorting (O(n log n) each time) or scanning (O(n) each time). It maintains a *partial* order — just enough to know the extreme — at O(log n) per update. Python's `heapq` is a **min-heap**; for a max-heap, push **negated** values.

## When to use it (the trigger)
- **Top-k / k largest / k smallest / k closest** → a heap of size **k** (counter-intuitively, use a **min**-heap to keep the *k largest*). ⭐
- **Repeatedly grab the min/max while inserting** → priority queue (Dijkstra, task scheduling, Huffman).
- **Merge k sorted lists/streams** → heap of the k current heads.
- **Median of a data stream** → **two heaps** (a max-heap of the low half, a min-heap of the high half).
- **"Process events/tasks by priority or deadline"**, "least/most frequent", "minimum cost to connect" → heap.
- Keyword giveaways: *k-th*, *top/closest/most-least frequent*, *merge k*, *median*, *schedule*, *priority*.

## The core idea
A **binary heap** is a complete binary tree stored in an array where each parent ≤ its children (min-heap). The root is the min. Push/pop "bubble" an element up/down in O(log n) to restore the property. You don't implement it — you call `heapq` — but know the **size-k trick**: to track the *k largest*, keep a **min-heap of size k**; the root is the *k*-th largest, and any new element bigger than the root replaces it. (Symmetrically: *k smallest* → max-heap of size k.) This is O(n log k), better than sorting's O(n log n) when k ≪ n.

## Template code

```python
import heapq

# 1) k LARGEST elements -> min-heap of size k (root = the smallest of the top-k)
def k_largest(nums, k):
    heap = []                                   # min-heap
    for x in nums:
        heapq.heappush(heap, x)
        if len(heap) > k:
            heapq.heappop(heap)                 # evict the smallest -> keep top-k
    return heap                                 # the k largest (unordered)
# Shortcut: heapq.nlargest(k, nums) / heapq.nsmallest(k, nums)

# 2) Kth largest element (same trick; answer is the root)
def kth_largest(nums, k):
    heap = nums[:k]
    heapq.heapify(heap)                         # O(n) build
    for x in nums[k:]:
        if x > heap[0]:
            heapq.heapreplace(heap, x)          # pop+push in one op
    return heap[0]

# 3) Merge k sorted lists -> heap of (value, list_idx, elem_idx)
def merge_k(lists):
    heap = [(lst[0], i, 0) for i, lst in enumerate(lists) if lst]
    heapq.heapify(heap)
    out = []
    while heap:
        val, i, j = heapq.heappop(heap)
        out.append(val)
        if j + 1 < len(lists[i]):
            heapq.heappush(heap, (lists[i][j + 1], i, j + 1))
    return out

# 4) Median of a stream -> two heaps, kept balanced
class MedianFinder:
    def __init__(self):
        self.lo = []        # MAX-heap (store negatives) — smaller half
        self.hi = []        # MIN-heap                    — larger half
    def add(self, x):
        heapq.heappush(self.lo, -x)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))   # balance: lo's max -> hi
        if len(self.hi) > len(self.lo):                    # keep lo >= hi in size
            heapq.heappush(self.lo, -heapq.heappop(self.hi))
    def median(self):
        if len(self.lo) > len(self.hi):
            return -self.lo[0]
        return (-self.lo[0] + self.hi[0]) / 2

# 5) Max-heap via negation (heapq is min-only)
def max_heap_demo(nums):
    h = [-x for x in nums]
    heapq.heapify(h)
    return -heapq.heappop(h)        # the maximum
```

## Complexity
- **push / pop:** **O(log n)**. **peek (min/max):** **O(1)** (`heap[0]`).
- **heapify** an existing list: **O(n)** (better than n pushes = O(n log n)).
- **Top-k via size-k heap:** **O(n log k)** time, **O(k)** space — beats sorting (O(n log n)) when k ≪ n.
- **Merge k lists** (total N elements): **O(N log k)**.
- **`heapq.nlargest/nsmallest(k, …)`:** O(n log k) — use them directly unless you need the streaming/online version.

## Variations / sub-patterns
- **Size-k heap** — top-k / kth largest / k closest points to origin / k most frequent (heap over counts).
- **Two heaps** — median of a stream; also "balance two halves" problems (e.g. IPO, scheduling around a pivot).
- **Heap as a priority queue** — Dijkstra's shortest path ([Graphs](10-graphs.md)), Prim's MST, task scheduler, meeting rooms II ([Intervals](09-intervals.md)).
- **Merge k sorted** — lists, arrays, or streams; "smallest range covering k lists".
- **Lazy deletion** — when you can't remove an arbitrary element, push updated entries and skip stale ones on pop.
- **Greedy + heap** — "reorganize string", "task scheduler" (always take the most frequent remaining).

## Curated problems
| # | Problem | Difficulty | Pattern note | Done |
|---|---|---|---|---|
| 1 | Kth Largest Element in an Array | Medium | size-k min-heap (or quickselect) | ⬜ |
| 2 | K Closest Points to Origin | Medium | size-k heap on squared distance | ⬜ |
| 3 | Top K Frequent Elements | Medium | count, then heap/bucket over counts | ⬜ |
| 4 | Merge k Sorted Lists | Hard | heap of the k current heads | ⬜ |
| 5 | Find Median from Data Stream | Hard | two heaps, balanced | ⬜ |
| 6 | Task Scheduler | Medium | greedy with a max-heap of counts | ⬜ |
| 7 | Last Stone Weight | Easy | max-heap, repeatedly smash top two | ⬜ |

(Week 8 plan: heaps — aim for 2–3 (shared with intervals). Suggested: **#1 + #3** (lock the size-k trick) and one of the two-heaps reps **#5**. The size-k heap and the two-heaps median are the two ideas interviewers reach for most.)

## Common mistakes
- **Wrong heap polarity for top-k** — to keep the *k largest* you use a **min**-heap (so you can cheaply evict the smallest); the instinct to use a max-heap is the classic trap. ⭐
- **Forgetting `heapq` is min-only** — negate values (and remember to negate back) for a max-heap.
- **Comparing un-orderable tuples** — when pushing `(priority, item)`, ties try to compare `item`; add a unique counter as a tiebreaker: `(priority, count, item)`.
- **Using sorted() when k ≪ n** — sorting is O(n log n); a size-k heap is O(n log k). For a single pass, `heapq.nlargest(k, …)`.
- **Two-heaps imbalance** — re-balance sizes after every insert, or the median read is wrong.
- **Mutating the heap list directly** — always go through `heappush/heappop/heapreplace`; manual edits break the invariant.
- **Reaching for a heap when a full sort is needed anyway** — if you need *all* elements ordered, just sort.

## Notes after solving
_Filled in as I work the problems — my own gotchas and which ones I want to re-do._
