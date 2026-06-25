# DSA Pattern: Arrays, Strings & Hash Maps

> **Priority:** highest · **Status:** ✅ written
> **One-line:** Use a hash map to trade space for time — turn an O(n²) "have I seen something compatible?" scan into an O(n) single pass.

This is the foundational, highest-frequency pattern. Most Easy/Medium problems reduce to "remember what you've seen" (hash map / hash set) or "count things" (frequency map).

## When to use it (the trigger)
- "Find two/three elements that satisfy a relation" (sum, difference, product).
- "Has this value appeared before?" / "find duplicates" → **hash set**.
- "How many times does each X occur?" → **frequency map** (`collections.Counter`).
- "Group things by a signature" (anagrams) → map signature → list.
- "First/only non-repeating element."
- Anything where a brute-force double loop checks pairs — ask: *can a hash map remember the complement?*

## The core idea
A hash map gives **O(1) average** lookup/insert. Instead of re-scanning the array to check a condition (O(n) per element → O(n²) total), you store what you've seen so the check becomes O(1). Single pass → **O(n) time, O(n) space**.

The recurring trick — **the complement**: while scanning for a pair summing to `target`, at element `x` you don't search for the partner, you check if `target - x` is already in the map.

## Template code

```python
from collections import Counter, defaultdict

# 1) "Seen" map for pair/complement problems (e.g., Two Sum)
def two_sum(nums, target):
    seen = {}                       # value -> index
    for i, x in enumerate(nums):
        if target - x in seen:      # complement already seen?
            return [seen[target - x], i]
        seen[x] = i
    return []

# 2) Frequency map (counts, anagrams, majority, etc.)
def char_counts(s):
    return Counter(s)               # {char: count}

# 3) Group by signature (e.g., group anagrams)
def group_anagrams(words):
    groups = defaultdict(list)
    for w in words:
        key = tuple(sorted(w))      # or a 26-length count tuple for O(n)
        groups[key].append(w)
    return list(groups.values())

# 4) Hash set for dedupe / membership
def has_duplicate(nums):
    seen = set()
    for x in nums:
        if x in seen:
            return True
        seen.add(x)
    return False
```

## Complexity
- Hash map/set ops: **O(1) average**, O(n) worst case (pathological collisions — rare).
- Single-pass solutions: **O(n) time, O(n) space**.
- Sorting-based alternatives (when no extra space allowed): O(n log n) time, O(1) extra.

## Variations / sub-patterns
- **Prefix sums + hash map:** "subarray summing to k" → store running sum frequencies (turns O(n²) into O(n)). Bridge to harder problems.
- **Frequency-then-decide:** count first, then a second pass uses the counts.
- **Index map vs. value set:** store indices when you must return positions; store just values when membership suffices.
- **Fixed-alphabet counting:** for lowercase letters, a length-26 array beats a dict (constant factor + O(n) anagram keys).

## Curated problems
| # | Problem | Difficulty | Pattern note | Done |
|---|---|---|---|---|
| 1 | Two Sum | Easy | complement map (the canonical) | ⬜ |
| 2 | Contains Duplicate | Easy | hash set membership | ⬜ |
| 3 | Valid Anagram | Easy | frequency map equality | ⬜ |
| 4 | Group Anagrams | Medium | map signature → list | ⬜ |
| 5 | Top K Frequent Elements | Medium | Counter + bucket/heap | ⬜ |
| 6 | Valid Sudoku | Medium | sets per row/col/box | ⬜ |
| 7 | Longest Consecutive Sequence | Medium | set + start-of-run check (O(n)) | ⬜ |
| 8 | Subarray Sum Equals K | Medium | prefix sum + count map | ⬜ |
| 9 | First Unique Character | Easy | frequency map, two passes | ⬜ |
| 10 | Product of Array Except Self | Medium | prefix/suffix products (no division) | ⬜ |

(Start with 1–3 + one Medium for Week 1; the plan says finish Arrays lessons 3–4 first.)

## Common mistakes
- Storing the **value** when you need the **index** (or vice versa).
- Adding the current element to the "seen" map **before** checking the complement → matches an element with itself.
- For anagram grouping, `sorted(w)` returns a list (unhashable) — use `tuple(sorted(w))` or a count tuple as the key.
- Assuming hash ops are O(1) worst-case in interviews — say "O(1) average."
- Mutating a dict while iterating it.
- Off-by-one / empty-input edge cases (empty array, single element).

## Notes after solving
_Filled in as I work the problems — my own gotchas and which ones I want to re-do._
