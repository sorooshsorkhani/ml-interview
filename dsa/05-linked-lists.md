# DSA Pattern: Linked Lists

> **Priority:** see [README](README.md) · **Status:** ✅ written
> **One-line:** Pointer manipulation on node chains — **reverse**, **fast/slow two pointers** (cycle, middle), **merge**, and **dummy-head** to kill edge cases. Almost everything is "re-wire `.next` carefully without losing a reference."

The whole pattern is about not losing track of nodes while you rewire pointers. Two habits make every problem easier: **a dummy head node** (so you never special-case the head), and **drawing the three pointers** (`prev`, `cur`, `nxt`) before you write the swap. There's very little algorithmic depth here — it's pointer hygiene under time pressure.

## When to use it (the trigger)
- The input **is** a linked list (`ListNode`), or the problem forbids using the array's index access.
- **Reverse** a list or a sub-section of it.
- **Detect a cycle**, or find where the cycle begins → **Floyd's fast/slow pointers**.
- **Find the middle** / the k-th-from-end node → fast/slow or two-pointer gap.
- **Merge** two sorted lists, or merge/reorder in place.
- "Do it in **O(1) extra space**" on a list → almost always pointer rewiring, not a copy.
- Keyword giveaways: *linked list*, *node*, *next pointer*, *cycle*, *reverse*, *in place*.

## The core idea
You can only move **forward** (in a singly linked list) and you only hold the references you save. So the recurring micro-skill is the **three-pointer shuffle**: cache `nxt = cur.next` *before* you overwrite `cur.next`, move `prev` and `cur` forward. For traversal-with-two-speeds problems, run a **slow** pointer (1 step) and a **fast** pointer (2 steps): when fast hits the end, slow is at the middle; if they ever meet, there's a cycle.

The **dummy head** trick: allocate a throwaway node pointing at the real head, build/operate from there, and `return dummy.next`. This removes the "what if I'm modifying the first node?" special case in merges, deletions, and insertions.

## Template code

```python
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

# 1) Reverse a singly linked list (the three-pointer shuffle — memorize cold)
def reverse_list(head):
    prev, cur = None, head
    while cur:
        nxt = cur.next      # cache before we clobber it
        cur.next = prev     # rewire
        prev = cur          # advance both
        cur = nxt
    return prev             # new head

# 2) Fast/slow: find the middle (slow lands on middle / 2nd-of-two-middles)
def middle(head):
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
    return slow

# 3) Floyd's cycle detection + find cycle start
def detect_cycle(head):
    slow = fast = head
    while fast and fast.next:
        slow, fast = slow.next, fast.next.next
        if slow is fast:                 # they met -> cycle exists
            p = head
            while p is not slow:         # reset one to head; equal speed now
                p, slow = p.next, slow.next
            return p                     # cycle start
    return None                          # no cycle

# 4) Merge two sorted lists (dummy head kills the edge cases)
def merge_two(l1, l2):
    dummy = tail = ListNode()
    while l1 and l2:
        if l1.val <= l2.val:
            tail.next, l1 = l1, l1.next
        else:
            tail.next, l2 = l2, l2.next
        tail = tail.next
    tail.next = l1 or l2                 # attach the remaining run
    return dummy.next

# 5) Remove n-th node from the end (two pointers, fixed gap, dummy head)
def remove_nth_from_end(head, n):
    dummy = ListNode(0, head)
    fast = slow = dummy
    for _ in range(n):                   # open a gap of n
        fast = fast.next
    while fast.next:                     # walk both until fast is last
        fast, slow = fast.next, slow.next
    slow.next = slow.next.next           # slow is just before the target
    return dummy.next
```

## Complexity
- Time: **O(n)** for a single pass (reverse, middle, cycle, merge of length-n inputs).
- Space: **O(1)** — the point of the pattern is in-place pointer rewiring, no extra structures.
- Compare to arrays: linked lists give **O(1) insert/delete given the node** but **O(n) access** (no indexing) — see the [README big-O table](README.md#big-o-quick-reference).

## Variations / sub-patterns
- **Reverse a sub-list** `[left, right]` (reverse-between) — reverse template applied to a window, stitched back with a dummy head.
- **Reverse in k-groups** — reverse template in chunks; the Hard escalation of #1.
- **Fast/slow** family: middle, cycle detect, cycle start, k-th from end, palindrome check (find middle → reverse second half → compare).
- **Merge** family: merge two sorted, merge k sorted (with a heap — see [Heaps](08-heaps.md)), sort a list (merge sort on a list).
- **Reorder / partition** — split, reverse, interleave; partition by value with two builder lists + dummy heads.
- **Dummy-head insert/delete** — any modification near the head.

## Curated problems
| # | Problem | Difficulty | Pattern note | Done |
|---|---|---|---|---|
| 1 | Reverse Linked List | Easy | the three-pointer shuffle — the rep to overlearn | ⬜ |
| 2 | Merge Two Sorted Lists | Easy | dummy head + tail pointer | ⬜ |
| 3 | Linked List Cycle | Easy | Floyd's fast/slow, "do they meet?" | ⬜ |
| 4 | Middle of the Linked List | Easy | fast/slow | ⬜ |
| 5 | Remove Nth Node From End | Medium | two pointers, fixed gap + dummy head | ⬜ |
| 6 | Reorder List | Medium | middle → reverse 2nd half → interleave | ⬜ |
| 7 | Linked List Cycle II (cycle start) | Medium | Floyd's + reset-to-head | ⬜ |
| 8 | Reverse Linked List II (between) | Medium | reverse a sub-list, stitch with dummy | ⬜ |
| 9 | Merge k Sorted Lists | Hard | min-heap of heads ([Heaps](08-heaps.md)) | ⬜ |

(Week 5 plan: linked lists — aim for 3–4. Suggested: **#1, #2, #3** + one Medium, **#5**. Overlearn #1 — reverse shows up *inside* half the others.)

## Common mistakes
- **Losing the rest of the list** — overwriting `cur.next` before caching `nxt`. Always `nxt = cur.next` first.
- **Null-pointer in fast/slow** — the loop guard must be `while fast and fast.next` (you dereference `fast.next.next`); getting this wrong crashes on even-length lists or empty input.
- **Not using a dummy head** — then every head-modifying case (delete head, merge into empty, insert at front) needs a special branch. The dummy removes all of them.
- **Returning the wrong node after reverse** — the new head is `prev` (where you ended), not `head` (now the tail).
- **Off-by-one on "n-th from end"** — open the gap *from the dummy*, not from head, so deleting the head also works.
- **Cycle-start logic** — after the meet, you must reset *one* pointer to `head` and advance **both one step at a time**; keeping fast at double speed gives the wrong node.
- **Comparing values vs. identity** — use `is` for node identity (`slow is fast`), not `==`.

## Notes after solving
_Filled in as I work the problems — my own gotchas and which ones I want to re-do._
