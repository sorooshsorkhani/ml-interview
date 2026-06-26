# DSA Pattern: Sliding Window

> **Priority:** see [README](README.md) · **Status:** ✅ written
> **One-line:** Maintain a contiguous window `[l, r]` and slide it — expand `r` to include, shrink `l` to satisfy a constraint — so each element enters and leaves at most once → O(n).

The same-direction special case of [two pointers](02-two-pointers.md): both pointers move forward, the *gap between them* is the answer. The go-to for **contiguous subarray / substring** questions.

## When to use it (the trigger)
- "**Longest / shortest / max / min** contiguous subarray or substring such that ⟨condition⟩."
- "Subarray/substring **containing / avoiding** certain characters" (no repeats, at most k distinct, all of T).
- A **fixed-size window** ("of size k") → fixed window; a **variable condition** ("sum ≥ target", "≤ k distinct") → variable window.
- Brute force enumerates all subarrays (O(n²)/O(n³)) and the validity is **monotonic** as the window grows/shrinks → window collapses it to O(n).
- Keyword giveaways: *contiguous*, *substring*, *consecutive*, *window of size k*.

## The core idea
A window holds a running summary (a sum, a count, a frequency map) of the elements between `l` and `r`. You **grow `r`** one step at a time, updating the summary in O(1). When the window **violates** the constraint, you **shrink from `l`**, undoing the summary, until it's valid again. Because each index is added once and removed once, total work is **O(n)** — not O(n) per window.

The art is choosing the loop shape: **fixed-size** (advance `r`, drop `l = r-k`), or **variable** (advance `r` always; `while invalid: shrink l`).

## Template code

```python
from collections import Counter, defaultdict

# 1) FIXED window of size k (e.g., max sum of k consecutive)
def max_sum_k(nums, k):
    window = sum(nums[:k])
    best = window
    for r in range(k, len(nums)):
        window += nums[r] - nums[r - k]   # add new, drop the one falling off
        best = max(best, window)
    return best

# 2) VARIABLE window — longest substring without repeating characters
def longest_unique_substring(s):
    last = {}                # char -> last index seen
    l = 0
    best = 0
    for r, ch in enumerate(s):
        if ch in last and last[ch] >= l:
            l = last[ch] + 1          # jump l past the previous occurrence
        last[ch] = r
        best = max(best, r - l + 1)
    return best

# 3) VARIABLE window — shrink-to-valid (smallest subarray with sum >= target)
def min_subarray_len(target, nums):
    l = 0
    total = 0
    best = float("inf")
    for r in range(len(nums)):
        total += nums[r]
        while total >= target:         # shrink while still valid
            best = min(best, r - l + 1)
            total -= nums[l]
            l += 1
    return best if best != float("inf") else 0

# 4) VARIABLE window with a frequency map — at most k distinct characters
def longest_at_most_k_distinct(s, k):
    freq = defaultdict(int)
    l = 0
    best = 0
    for r, ch in enumerate(s):
        freq[ch] += 1
        while len(freq) > k:           # too many distinct -> shrink
            freq[s[l]] -= 1
            if freq[s[l]] == 0:
                del freq[s[l]]
            l += 1
        best = max(best, r - l + 1)
    return best
```

## Complexity
- **O(n) time** — each index enters and exits the window once (`l` and `r` only move forward; the inner `while` is amortized O(1) per step).
- **Space:** O(1) for sum/count windows; **O(k)** or O(alphabet) when a frequency map of the window contents is needed.

## Variations / sub-patterns
- **Fixed-size window:** running sum/average, max in each window (with a deque → monotonic queue).
- **Variable "longest valid":** grow `r`, shrink `l` only when invalid; answer is the max window seen.
- **Variable "shortest valid":** grow `r`, shrink `l` while *still valid*, recording the min.
- **Counting windows** (number of substrings with ≤ k distinct = f(≤k) − f(≤k−1)).
- **Window + frequency map:** anagrams in a string, permutation-in-string, minimum window substring.
- **Monotonic deque** for sliding-window maximum/minimum (links to [stacks/queues](06-stacks-queues.md)).

## Curated problems
| # | Problem | Difficulty | Pattern note | Done |
|---|---|---|---|---|
| 1 | Best Time to Buy and Sell Stock | Easy | track min-so-far = a degenerate window | ⬜ |
| 2 | Maximum Average Subarray I | Easy | fixed window of size k | ⬜ |
| 3 | Longest Substring Without Repeating Characters | Medium | variable window + last-seen map | ⬜ |
| 4 | Minimum Size Subarray Sum | Medium | shrink-to-valid, smallest window | ⬜ |
| 5 | Longest Repeating Character Replacement | Medium | window valid while (len − maxFreq) ≤ k | ⬜ |
| 6 | Permutation in String | Medium | fixed window + frequency match | ⬜ |
| 7 | Minimum Window Substring | Hard | variable window + need/have counts | ⬜ |

(Week 3 plan: consolidate [two pointers](02-two-pointers.md) + sliding window — pick 3–4 total across both. Good window starters: #2, #3, #4.)

## Common mistakes
- **Shrinking with `if` instead of `while`** in variable windows — one shrink may not restore validity.
- **Forgetting to undo the summary** when `l` advances (subtract from sum, decrement/delete from the freq map).
- **Window length is `r - l + 1`**, not `r - l` (classic off-by-one).
- **Letting `l` jump backward** — in the "no repeats" template, guard with `last[ch] >= l` so a stale index doesn't drag `l` back.
- **Recomputing the window from scratch** each step → silently O(n·k); the whole point is incremental O(1) updates.
- Mixing up "longest valid" (shrink only when invalid) with "shortest valid" (shrink while valid) loop logic.

## Notes after solving
_Filled in as I work the problems — my own gotchas and which ones I want to re-do._
