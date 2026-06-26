# DSA Pattern: Two Pointers

> **Priority:** see [README](README.md) · **Status:** ✅ written
> **One-line:** Use two indices that move with intent — toward each other, or one chasing the other — to collapse an O(n²) nested scan into a single O(n) pass.

The second-highest-yield pattern after hash maps. Whenever a brute force checks **pairs** of positions, ask whether two coordinated pointers can do it in one pass — usually after sorting, or on already-sorted / two-sequence data.

## When to use it (the trigger)
- **Sorted array + find a pair/triple** matching a sum/condition → opposite-ends pointers.
- **Palindrome / mirror checks** → pointers from both ends moving inward.
- **In-place array surgery** — remove/move/dedupe elements, partition → a slow "write" pointer and a fast "read" pointer.
- **Merging two sorted sequences** → one pointer per sequence.
- **Reverse in place** → swap ends, move inward.
- Heuristic: *"the array is sorted"* or *"do it in O(1) extra space"* is a strong two-pointers signal.

## The core idea
Two pointers exploit **monotonic structure** so you never have to back up. The classic is the **converging** variant on a sorted array: `lo` at the start, `hi` at the end. If `nums[lo] + nums[hi]` is too small, the only way to grow the sum is `lo += 1`; if too big, `hi -= 1`. Each step eliminates a whole row/column of the implicit pair matrix → O(n) instead of O(n²).

The **fast/slow** (same-direction) variant uses a `write` pointer that lags behind a `read` pointer to compact an array in place with O(1) extra space.

## Template code

```python
# 1) Converging pointers on a SORTED array (e.g., Two Sum II, 3Sum inner loop)
def two_sum_sorted(nums, target):
    lo, hi = 0, len(nums) - 1
    while lo < hi:
        s = nums[lo] + nums[hi]
        if s == target:
            return [lo, hi]
        elif s < target:
            lo += 1          # need a bigger sum
        else:
            hi -= 1          # need a smaller sum
    return []

# 2) Palindrome check (skip non-alphanumerics, case-insensitive)
def is_palindrome(s):
    lo, hi = 0, len(s) - 1
    while lo < hi:
        while lo < hi and not s[lo].isalnum(): lo += 1
        while lo < hi and not s[hi].isalnum(): hi -= 1
        if s[lo].lower() != s[hi].lower():
            return False
        lo += 1; hi -= 1
    return True

# 3) Fast/slow write pointer — in-place dedupe of a SORTED array
def remove_duplicates(nums):
    if not nums:
        return 0
    write = 1                      # next slot to write a new unique value
    for read in range(1, len(nums)):
        if nums[read] != nums[write - 1]:
            nums[write] = nums[read]
            write += 1
    return write                   # new logical length

# 4) 3Sum = sort + fix one + two-pointer the rest (O(n^2))
def three_sum(nums):
    nums.sort()
    res = []
    for i in range(len(nums) - 2):
        if i > 0 and nums[i] == nums[i - 1]:
            continue               # skip duplicate anchors
        lo, hi = i + 1, len(nums) - 1
        while lo < hi:
            s = nums[i] + nums[lo] + nums[hi]
            if s < 0:   lo += 1
            elif s > 0: hi -= 1
            else:
                res.append([nums[i], nums[lo], nums[hi]])
                lo += 1; hi -= 1
                while lo < hi and nums[lo] == nums[lo - 1]: lo += 1   # skip dup
                while lo < hi and nums[hi] == nums[hi + 1]: hi -= 1
    return res
```

## Complexity
- Converging / fast-slow scan: **O(n) time, O(1) extra space.**
- If you must sort first (unsorted input): **O(n log n)** dominates.
- 3Sum and friends: **O(n²)** (outer fix + inner two-pointer), O(1)–O(n) space depending on output.

## Variations / sub-patterns
- **Opposite ends (converging):** sorted pair/triple sums, container-with-most-water, palindrome, reverse.
- **Same direction (fast/slow):** in-place removal/dedupe/partition, "move zeroes," cycle detection (Floyd's — see [linked lists](05-linked-lists.md)).
- **Two sequences:** merge two sorted arrays, intersection of sorted arrays, `is subsequence`.
- **Fix-one-then-two-pointer:** kSum problems reduce to a loop around the two-pointer core.
- Closely related to (and often confused with) [sliding window](03-sliding-window.md) — that's the *variable-gap* same-direction special case.

## Curated problems
| # | Problem | Difficulty | Pattern note | Done |
|---|---|---|---|---|
| 1 | Valid Palindrome | Easy | converge from both ends, skip non-alnum | ⬜ |
| 2 | Two Sum II (sorted input) | Easy | converging pointers, the canonical | ⬜ |
| 3 | Remove Duplicates from Sorted Array | Easy | fast/slow write pointer, in place | ⬜ |
| 4 | Move Zeroes | Easy | write pointer for non-zeros | ⬜ |
| 5 | Container With Most Water | Medium | move the shorter wall inward | ⬜ |
| 6 | 3Sum | Medium | sort + fix one + two-pointer, skip dups | ⬜ |
| 7 | Sort Colors (Dutch flag) | Medium | three pointers / in-place partition | ⬜ |

(Week 3 plan: consolidate two pointers + [sliding window](03-sliding-window.md) — aim for 3–4 across both, e.g. #1, #2, #5 here + one window problem.)

## Common mistakes
- **Forgetting to sort** when the converging logic relies on order.
- **Loop bound `lo < hi` vs `lo <= hi`** — `<` for pairs (don't reuse one element), `<=` when a single middle element is valid.
- **Not skipping duplicates** in 3Sum → duplicate triples in the output.
- **Moving the wrong pointer** in container-with-most-water (always move the *shorter* wall; moving the taller can't improve area).
- **Off-by-one on the write pointer** — it marks the *next slot to fill*, return it as the new length.
- Mutating `len(nums)` assumptions while writing in place.

## Notes after solving
_Filled in as I work the problems — my own gotchas and which ones I want to re-do._
