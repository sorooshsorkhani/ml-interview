# DSA Pattern: Binary Search

> **Priority:** see [README](README.md) · **Status:** ✅ written
> **One-line:** Halve the search space each step on anything **monotonic** — sorted arrays, "first/last position," rotated arrays, and "search on the answer."

The highest-leverage logarithmic pattern. The trick most people miss: binary search isn't only for "find x in a sorted array" — it's for *any* predicate that flips from False to True exactly once. Spotting that monotonic boundary is the whole skill.

## When to use it (the trigger)
- **Sorted (or rotated-sorted) array** + "find / find the boundary / find the insert position."
- **"First/last element that satisfies P"** where `P` is monotonic (all False, then all True).
- **"Minimize/maximize X subject to feasible(X)"** where feasibility is monotonic → **binary search on the answer** (e.g., min capacity, min eating speed, smallest divisor). This is the Medium-interview workhorse.
- Anything where an O(n) or O(n²) scan can be replaced because the data/answer space is ordered.
- Keyword giveaways: *sorted*, *rotated*, *minimum/maximum such that*, *O(log n)*.

## The core idea
Maintain a candidate interval `[lo, hi]`. Each step, evaluate the midpoint and discard the half that can't contain the answer → the space shrinks geometrically, **O(log n)** steps. For "find the boundary" problems, reframe as: *find the first index where predicate(i) is True.* Get the **invariant** right (what's always true about `lo`/`hi`) and the loop bounds fall out.

The single biggest correctness lever is the loop form. Two reliable templates below — the **`while lo < hi`** boundary template generalizes to almost everything; learn it well.

## Template code

```python
import bisect

# 1) Classic: find target's index in a sorted array (or -1)
def binary_search(nums, target):
    lo, hi = 0, len(nums) - 1          # inclusive interval
    while lo <= hi:
        mid = (lo + hi) // 2           # in Python no overflow; elsewhere lo + (hi-lo)//2
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            lo = mid + 1
        else:
            hi = mid - 1
    return -1

# 2) Boundary template (PREFERRED): first index where predicate is True.
#    predicate must be monotonic: F F F T T T  -> returns index of first T (or len if none)
def first_true(lo, hi, predicate):     # search the half-open space [lo, hi)
    while lo < hi:
        mid = (lo + hi) // 2
        if predicate(mid):
            hi = mid                   # mid might be the answer; keep it in range
        else:
            lo = mid + 1               # mid is not, discard it
    return lo                          # lo == hi == first True

# e.g. leftmost insertion point == bisect_left:
def search_insert(nums, target):
    return first_true(0, len(nums), lambda i: nums[i] >= target)

# 3) Binary search on the ANSWER (e.g., min eating speed / Koko, ship capacity)
def min_feasible(low, high, feasible):  # feasible: monotonic F...F T...T over the answer range
    while low < high:
        mid = (low + high) // 2
        if feasible(mid):
            high = mid
        else:
            low = mid + 1
    return low

# 4) Search in a rotated sorted array (unique values)
def search_rotated(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:           # left half is sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                               # right half is sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1

# 5) Standard-library shortcuts — say these in interviews
#    bisect_left(a, x)  -> first index where a[i] >= x
#    bisect_right(a, x) -> first index where a[i] >  x
```

## Complexity
- Time: **O(log n)** for array search; for "binary search on the answer," **O(log(range) · cost_of_feasible)**.
- Space: **O(1)** iterative.
- **Sorting** (the partner topic): comparison sorts are **O(n log n)** time; Python's `sorted`/`.sort()` is Timsort (stable, O(n) on nearly-sorted data). Often the unlock is "sort first (O(n log n)), then binary search / two-pointer."

## Variations / sub-patterns
- **Leftmost / rightmost** occurrence (lower/upper bound) → the boundary template with `>=` vs `>`.
- **Insertion point** → `bisect_left` / `bisect_right`.
- **Rotated sorted array** (find target, or find the minimum/pivot).
- **Search on the answer** → "min/max value such that a feasibility check passes" (capacity, speed, threshold). The check is usually an O(n) greedy.
- **2D / matrix** binary search (treat as a flattened sorted array, or staircase search).
- **Peak finding** / "find any local max" via slope direction.
- **Float binary search** (loop a fixed number of iterations or until `hi - lo < eps`).

## Curated problems
| # | Problem | Difficulty | Pattern note | Done |
|---|---|---|---|---|
| 1 | Binary Search | Easy | the canonical `lo <= hi` template | ⬜ |
| 2 | Search Insert Position | Easy | boundary template = bisect_left | ⬜ |
| 3 | First Bad Version | Easy | first-True predicate, minimize API calls | ⬜ |
| 4 | Find First and Last Position | Medium | lower bound + upper bound | ⬜ |
| 5 | Search in Rotated Sorted Array | Medium | which half is sorted? | ⬜ |
| 6 | Find Minimum in Rotated Sorted Array | Medium | compare mid to hi to locate pivot | ⬜ |
| 7 | Koko Eating Bananas | Medium | binary search on the answer (speed) | ⬜ |
| 8 | Median of Two Sorted Arrays | Hard | partition-based binary search | ⬜ |

(Week 4 plan: binary search + sorting — aim for 3–4. Suggested: #1, #2, #5 + one "search on the answer," #7.)

## Common mistakes
- **Infinite loop from a bad mid update** — with `while lo < hi`, the False branch must be `lo = mid + 1` (otherwise `mid` can equal `lo` forever). The True branch keeps `hi = mid`.
- **Mixing interval conventions** — pick one: inclusive `[lo, hi]` with `lo <= hi`, *or* half-open `[lo, hi)` with `lo < hi`. Don't blend them.
- **Off-by-one on lower vs. upper bound** — leftmost uses `nums[mid] >= target`; rightmost/upper uses `> target`.
- **Forgetting the array must be sorted** (or that the *predicate* is monotonic) — binary search is meaningless otherwise.
- **`(lo + hi)` overflow** in languages with fixed-width ints — use `lo + (hi - lo) // 2` (a non-issue in Python, but say it).
- **Rotated array: wrong half** — decide which half is sorted by comparing `nums[lo]` to `nums[mid]`, then check if target lies within that sorted half.
- **Returning the wrong variable** — in the boundary template the answer is `lo` (== `hi`) after the loop, not `mid`.

## Notes after solving
_Filled in as I work the problems — my own gotchas and which ones I want to re-do._
