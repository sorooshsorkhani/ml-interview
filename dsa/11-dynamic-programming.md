# DSA Pattern: Dynamic Programming (intro)

> **Priority:** see [README](README.md) · **Status:** ✅ written (intro depth)
> **One-line:** DP applies when a problem has **overlapping subproblems** + **optimal substructure**. The recipe never changes: **define the state**, write the **recurrence**, set the **base cases**, then either **memoize** (top-down recursion + cache) or **tabulate** (bottom-up array). The whole skill is *defining the state* — once the state and recurrence are right, the code is mechanical.

The AMLS coding bar is Easy→Medium, so the goal here is **fluency on the canonical 1-D patterns** (climbing stairs, house robber, coin change), not hard 2-D/bitmask DP. The mental shift from brute-force recursion to DP: a naive recursion recomputes the same subproblem exponentially often (e.g. Fibonacci's tree); **caching each subproblem's answer once** collapses it to polynomial time. That cache *is* DP. ⭐

## When to use it (the trigger)
- **"Count the number of ways"** to do something (climb stairs, make change, decode) → DP.
- **"Min/max cost/length/value"** over a sequence of choices ("min coins", "longest increasing subsequence", "max profit") → DP.
- **"Can you reach / partition / make"** a target (word break, subset sum, partition equal subset) → DP (often boolean).
- The brute-force is **exponential recursion that revisits the same arguments** → add memoization.
- Optimization over **sequences, grids, or choices with a clean "take it / leave it" structure**.
- Keyword giveaways: *number of ways, min/max, longest/shortest, can you, optimal, count*; and "I wrote a recursion and it's re-solving the same thing."

## The core idea — the 4-step recipe
1. **State:** what does a subproblem look like? Define `dp[i]` (or `dp[i][j]`) in *words* — e.g. "the max money robbable from the first `i` houses." Getting this right is 90% of the work. ⭐
2. **Recurrence:** how does a state combine smaller states? e.g. `dp[i] = max(dp[i-1], dp[i-2] + nums[i])` (rob this house or skip it).
3. **Base cases:** the smallest states you can answer directly (`dp[0]`, empty input).
4. **Order / implementation:** **top-down** (recurse + `@lru_cache`) follows the recurrence literally and only computes needed states; **bottom-up** (fill a table from base cases up) avoids recursion overhead and enables **space optimization** (often you only need the last one or two rows/cells).

## Template code

```python
from functools import lru_cache

# 1) TOP-DOWN (memoization): coin change — fewest coins to make `amount`
def coin_change(coins, amount):
    @lru_cache(None)
    def dp(rem):                         # STATE: min coins to make `rem`
        if rem == 0: return 0            # BASE
        if rem < 0: return float("inf")
        return min(dp(rem - c) + 1 for c in coins)   # RECURRENCE
    ans = dp(amount)
    return ans if ans != float("inf") else -1

# 2) BOTTOM-UP (tabulation): climbing stairs — # ways to reach step n (1 or 2 at a time)
def climb_stairs(n):
    if n <= 2: return n
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2                  # BASE
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]    # RECURRENCE (it's Fibonacci)
    return dp[n]

# 3) SPACE-OPTIMIZED: house robber — only the last two states matter
def rob(nums):
    prev, curr = 0, 0                    # dp[i-2], dp[i-1]
    for x in nums:
        prev, curr = curr, max(curr, prev + x)   # rob x (+prev) or skip (curr)
    return curr

# 4) BOOLEAN / reachability DP: word break
def word_break(s, words):
    words = set(words)
    dp = [False] * (len(s) + 1)
    dp[0] = True                         # empty prefix is reachable
    for i in range(1, len(s) + 1):
        for j in range(i):
            if dp[j] and s[j:i] in words:
                dp[i] = True
                break
    return dp[-1]

# 5) 2-D DP: longest common subsequence (the canonical grid recurrence)
def lcs(a, b):
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if a[i-1] == b[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1            # match: extend diagonal
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1]) # skip one char
    return dp[m][n]
```

## Complexity
- **Time:** (number of distinct states) × (work per state/transition). 1-D over n with O(1) transition → **O(n)**; coin change → O(amount · #coins); LCS / edit distance → **O(m·n)**.
- **Space:** O(number of states) for the table; often reducible to **O(1)** or O(n) when each state depends only on the last few (house robber → O(1); many grid DPs → one row).
- **Memoization vs. tabulation:** same asymptotics. Top-down computes only reachable states and is easier to write from the recurrence; bottom-up has no recursion overhead and enables space tricks.

## Variations / sub-patterns
- **1-D sequence DP** — climbing stairs, house robber (I/II), max subarray (Kadane's), decode ways, min cost climbing stairs.
- **Unbounded / bounded "knapsack-ish"** — coin change (count ways / min coins), combination sum, partition equal subset sum, 0/1 knapsack.
- **Subsequence DP** — longest increasing subsequence (O(n²) DP or O(n log n) patience), longest common subsequence, edit distance.
- **Grid DP** — unique paths, min path sum, maximal square (each cell from neighbors).
- **Interval DP** (harder, usually beyond the bar) — matrix chain, burst balloons.
- **Kadane's algorithm** (max subarray) — the "running best ending here" 1-D DP worth knowing cold.

## Curated problems
| # | Problem | Difficulty | Pattern note | Done |
|---|---|---|---|---|
| 1 | Climbing Stairs | Easy | the "hello world" of DP (Fibonacci) | ⬜ |
| 2 | House Robber | Medium | take/skip; space-optimize to O(1) | ⬜ |
| 3 | Coin Change | Medium | min-over-choices, top-down or bottom-up | ⬜ |
| 4 | Longest Increasing Subsequence | Medium | dp[i] = best ending at i | ⬜ |
| 5 | Maximum Subarray (Kadane's) | Medium | running best-ending-here | ⬜ |
| 6 | Word Break | Medium | boolean reachability DP | ⬜ |
| 7 | Unique Paths | Medium | 2-D grid, dp[i][j] from top + left | ⬜ |

(Week 9 plan: light DP intro — aim for 2–3, or treat as catch-up buffer. Suggested: **#1 → #2 → #3** in order; they build the "define state → recurrence → memoize/tabulate" muscle. Don't grind hard DP — fluency on these 1-D patterns is the right bar for an AMLS loop.)

## Common mistakes
- **Jumping to code before defining the state in words** — an unclear state means a wrong recurrence. Write "`dp[i]` = …" first. ⭐
- **Wrong or missing base cases** — `dp[0]`, empty input, or the `rem < 0` guard; off-by-one in array sizing (`n+1`).
- **Recomputation without memoization** — a clean recursion that TLEs is usually just missing `@lru_cache`. ⭐
- **Unhashable arguments to `lru_cache`** — pass tuples/ints, not lists.
- **Confusing "count ways" vs. "min/max"** — the recurrence aggregates differently (sum vs. min/max).
- **Iterating in the wrong order** — bottom-up must fill states *before* they're needed (e.g. 0/1 knapsack iterates capacity in reverse to avoid reusing an item).
- **Over-optimizing space too early** — get the full table correct first, then collapse dimensions.
- **Forcing DP where greedy/BFS is right** — not every "min steps" is DP; if there's no overlapping-subproblem reuse, DP buys nothing.

## Notes after solving
_Filled in as I work the problems — my own gotchas and which ones I want to re-do._
