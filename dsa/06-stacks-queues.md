# DSA Pattern: Stacks & Queues

> **Priority:** see [README](README.md) · **Status:** ✅ written
> **One-line:** **LIFO** stack for matching/nesting and **monotonic stacks** (next-greater-element); **FIFO** queue for BFS; **deque** for sliding-window max. The skill is recognizing "the thing I need is the *most recent* unresolved item" → stack.

The unifying tell: when correctness depends on the **most recently seen** unresolved element (a matching bracket, the previous taller bar, the last operator), reach for a **stack**. When you process in arrival order (level-by-level traversal), reach for a **queue**. The high-value sub-pattern for Mediums is the **monotonic stack** — keep the stack sorted so each element is pushed and popped at most once → O(n).

## When to use it (the trigger)
- **Matching / nesting / balancing** — parentheses, tags, "valid" structures → **stack**.
- **"Next/previous greater/smaller element"**, "days until warmer", "largest rectangle" → **monotonic stack**.
- **Evaluate expressions** (RPN, basic calculators), **undo**, **backtracking-ish** "pop the last" logic → stack.
- **Process level by level / shortest path in unweighted graph / first-in-first-out** → **queue** (BFS, see [Trees](07-trees.md), [Graphs](10-graphs.md)).
- **Sliding-window max/min** → **monotonic deque**.
- **Implement one with the other** (stack-from-queues, queue-from-stacks) — classic warm-up.
- Keyword giveaways: *valid*, *balanced*, *next greater*, *nested*, *innermost*, *level order*, *most recent*.

## The core idea
A **stack** answers "what's the most recent unresolved thing?" in O(1) — push when you see something open/pending, pop when it resolves. A **monotonic stack** maintains elements in increasing (or decreasing) order: before pushing `x`, pop everything that `x` "resolves" (e.g., every smaller bar that `x` is the next-greater for). Each element enters and leaves once → **O(n)** total despite the inner while-loop.

A **queue** answers "what's next in arrival order?" — the backbone of **BFS**, where you process all nodes at distance d before distance d+1. A **deque** (double-ended queue) generalizes both and is the tool for windowed extrema.

## Template code

```python
from collections import deque

# 1) Balanced parentheses (the canonical stack match)
def is_valid(s):
    pairs = {")": "(", "]": "[", "}": "{"}
    stack = []
    for ch in s:
        if ch in pairs.values():          # opening -> push
            stack.append(ch)
        elif ch in pairs:                 # closing -> must match top
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack                      # all opens resolved

# 2) Monotonic stack: next greater element to the right (indices)
#    answer[i] = index of the next element greater than nums[i], else -1
def next_greater(nums):
    n = len(nums)
    ans = [-1] * n
    stack = []                            # holds indices, values DECREASING bottom->top
    for i, x in enumerate(nums):
        while stack and nums[stack[-1]] < x:   # x resolves these waiting elements
            ans[stack.pop()] = i
        stack.append(i)
    return ans

# 3) BFS with a queue (level-order; also shortest path in an unweighted graph)
def bfs(start, neighbors):
    q = deque([start])
    seen = {start}
    while q:
        node = q.popleft()                # FIFO
        for nxt in neighbors(node):
            if nxt not in seen:
                seen.add(nxt)
                q.append(nxt)

# 4) Sliding-window maximum (monotonic deque, O(n))
def max_sliding_window(nums, k):
    dq = deque()                          # indices, values DECREASING; front = current max
    out = []
    for i, x in enumerate(nums):
        while dq and nums[dq[-1]] <= x:   # pop smaller-or-equal from the back
            dq.pop()
        dq.append(i)
        if dq[0] <= i - k:                # drop indices that fell out of the window
            dq.popleft()
        if i >= k - 1:
            out.append(nums[dq[0]])
    return out

# 5) Evaluate Reverse Polish Notation (stack of operands)
def eval_rpn(tokens):
    stack = []
    ops = {"+": lambda a, b: a + b, "-": lambda a, b: a - b,
           "*": lambda a, b: a * b, "/": lambda a, b: int(a / b)}  # trunc toward zero
    for t in tokens:
        if t in ops:
            b, a = stack.pop(), stack.pop()
            stack.append(ops[t](a, b))
        else:
            stack.append(int(t))
    return stack[0]
```

## Complexity
- **Stack / queue ops:** push, pop, peek all **O(1)**.
- **Monotonic stack / deque scans:** **O(n)** total — each element is pushed and popped at most once (amortized), even with the inner `while`.
- **Space:** O(n) worst case (e.g., a strictly increasing array fills a "next greater" stack).
- Use **`collections.deque`** for a queue/deque (O(1) both ends). A plain Python `list` is a fine stack (`append`/`pop`) but a **bad queue** — `pop(0)` is O(n).

## Variations / sub-patterns
- **Plain matching stack** — brackets, valid string, remove adjacent duplicates, simplify path.
- **Monotonic stack** — next/previous greater/smaller, daily temperatures, largest rectangle in histogram, trapping rain water, stock span, remove-k-digits.
- **Two stacks / stack + state** — min-stack (track min alongside), basic calculator (numbers + signs), backspace string compare.
- **Queue / BFS** — level-order tree traversal, shortest path in unweighted grid/graph, multi-source BFS (rotting oranges), topological sort (Kahn's).
- **Monotonic deque** — sliding-window maximum/minimum.
- **Implement-the-other** — queue using two stacks (amortized O(1)), stack using two queues.

## Curated problems
| # | Problem | Difficulty | Pattern note | Done |
|---|---|---|---|---|
| 1 | Valid Parentheses | Easy | the canonical match stack | ⬜ |
| 2 | Min Stack | Easy | track current min alongside values | ⬜ |
| 3 | Implement Queue using Stacks | Easy | two stacks, amortized O(1) | ⬜ |
| 4 | Evaluate Reverse Polish Notation | Medium | operand stack | ⬜ |
| 5 | Daily Temperatures | Medium | monotonic (decreasing) stack of indices | ⬜ |
| 6 | Next Greater Element II (circular) | Medium | monotonic stack, loop twice | ⬜ |
| 7 | Number of Islands (BFS variant) | Medium | queue BFS / flood fill ([Graphs](10-graphs.md)) | ⬜ |
| 8 | Sliding Window Maximum | Hard | monotonic deque | ⬜ |
| 9 | Largest Rectangle in Histogram | Hard | monotonic stack, the boss fight | ⬜ |

(Week 6 plan: stacks & queues — aim for 3–4. Suggested: **#1, #3** + the monotonic-stack rep **#5**, and one of **#4/#7**. The monotonic stack is the highest-yield idea here — drill #5 until it's automatic.)

## Common mistakes
- **Using `list.pop(0)` as a queue** — it's O(n); use `deque.popleft()`.
- **Forgetting the final emptiness check** — in bracket matching, leftover opens on the stack mean *invalid*; returning `True` too early misses `"((("`.
- **Wrong monotonic direction** — "next greater" needs a stack whose values *decrease* bottom→top; pick the direction by what you pop (you pop the elements the current one *resolves*).
- **Storing values instead of indices** in monotonic problems — you usually need the index to compute distances/widths.
- **Deque window: stale front** — must evict indices that slid out (`dq[0] <= i - k`) before reading the max.
- **Empty stack pop** — guard `if stack` before `pop()`/`peek()` (closing bracket with nothing open, operator with too few operands).
- **RPN operand order** — pop `b` then `a`; compute `a op b` (order matters for `-` and `/`).

## Notes after solving
_Filled in as I work the problems — my own gotchas and which ones I want to re-do._
