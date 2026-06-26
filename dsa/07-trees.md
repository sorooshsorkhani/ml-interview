# DSA Pattern: Trees (BFS / DFS / BST)

> **Priority:** see [README](README.md) · **Status:** ✅ written
> **One-line:** **Recursion is the default tool.** Use **DFS** (pre/in/post-order) for path/subtree/depth questions; **BFS** (a queue) for level-order and shortest-depth questions; exploit the **BST invariant** (left < node < right) for ordered operations in O(h). The skill is matching the question to a traversal and trusting the recursion to handle the subtrees.

The unifying tell: a binary tree is a **self-similar recursive structure** — every node is the root of its own subtree. So most tree problems reduce to "solve it for the left subtree, solve it for the right subtree, combine." Pick the traversal by *when* you need to combine: **before** recursing (pre-order, top-down), **between** (in-order, gives sorted order in a BST), or **after** (post-order, bottom-up — when a node's answer needs its children's answers).

## When to use it (the trigger)
- **Anything shaped like a tree** — binary trees, n-ary trees, tries, parse trees, the recursion tree of another problem.
- **Depth / height / path-from-root / "does a path exist"** → **DFS** (usually recursion).
- **"Level order", "by depth", "minimum depth", "closest node", "level averages"** → **BFS** with a queue.
- **A node's answer depends on its children** (height, diameter, "is balanced", subtree sums) → **post-order DFS** (compute children first, return up).
- **Sorted output, kth smallest, validate ordering, range queries** in a **BST** → **in-order DFS** (in-order of a BST is sorted) or the BST search invariant.
- Keyword giveaways: *root*, *leaf*, *subtree*, *ancestor*, *level*, *depth/height*, *path*, *balanced*, *BST*, *in-order*.

## The core idea
Define a node as `class TreeNode: val, left, right`. **DFS recursion** is three lines plus a base case; the only decision is where you do the "work" relative to the two recursive calls (pre/in/post-order). **BFS** uses a queue and processes the tree level by level — snapshot the queue length at the start of each level to group nodes by depth. The **BST** invariant (everything left < node < everything right) turns search/insert/kth-smallest into O(h) operations (O(log n) if balanced, O(n) if degenerate).

## Template code

```python
from collections import deque

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val, self.left, self.right = val, left, right

# 1) DFS recursion — the universal skeleton (here: post-order to compute height)
def height(node):
    if not node:                      # base case
        return 0
    lh = height(node.left)            # solve left subtree
    rh = height(node.right)           # solve right subtree
    return 1 + max(lh, rh)            # combine (work happens AFTER -> post-order)

# 2) In-order DFS of a BST yields values in SORTED order
def inorder(node, out):
    if not node:
        return
    inorder(node.left, out)
    out.append(node.val)              # work happens BETWEEN -> in-order
    inorder(node.right, out)

# 3) BFS level-order (group nodes by depth)
def level_order(root):
    if not root:
        return []
    levels, q = [], deque([root])
    while q:
        level = []
        for _ in range(len(q)):       # snapshot: process exactly this level
            node = q.popleft()
            level.append(node.val)
            if node.left:  q.append(node.left)
            if node.right: q.append(node.right)
        levels.append(level)
    return levels

# 4) BST search — exploit the invariant, O(h)
def bst_search(node, target):
    while node:
        if target == node.val: return node
        node = node.left if target < node.val else node.right
    return None

# 5) "Bubble up extra info" pattern: diameter via a post-order that returns height
def diameter(root):
    best = 0
    def depth(node):
        nonlocal best
        if not node: return 0
        l, r = depth(node.left), depth(node.right)
        best = max(best, l + r)       # longest path THROUGH this node
        return 1 + max(l, r)          # height returned to the parent
    depth(root)
    return best

# 6) Lowest Common Ancestor (general binary tree), post-order
def lca(root, p, q):
    if not root or root is p or root is q:
        return root
    L, R = lca(root.left, p, q), lca(root.right, p, q)
    if L and R: return root           # p and q split here -> this is the LCA
    return L or R                     # both on one side -> bubble it up
```

## Complexity
- **DFS / BFS traversal:** **O(n)** time (visit each node once), where n = number of nodes.
- **Space:** O(h) for DFS (recursion stack, h = height); O(w) for BFS (queue holds a level, w = max width, up to n/2 for the bottom level). Balanced tree h = O(log n); degenerate (linked-list-shaped) h = O(n).
- **BST operations** (search/insert/delete): O(h) — **O(log n)** if balanced, **O(n)** worst case (unbalanced). Self-balancing trees (AVL, red-black) guarantee O(log n).

## Variations / sub-patterns
- **Plain DFS recursion** — max depth, invert/mirror, same-tree, symmetric tree, path sum.
- **Post-order "return info up"** — height, is-balanced, diameter, max path sum, count good nodes, subtree sums. The signature move: the helper returns one value to the parent while updating a `nonlocal` best.
- **In-order on a BST** — validate BST, kth smallest, two-sum in BST, range sum — all rely on in-order = sorted.
- **BFS / level-order** — level-order traversal, right-side view, level averages, min depth, zigzag, connect-next-pointers.
- **LCA** — general binary tree (post-order) vs. BST (walk down using the invariant).
- **Construction** — build a tree from preorder+inorder, serialize/deserialize.
- **Path problems** — root-to-leaf paths, path sum I/II/III, "count paths summing to target" (DFS + prefix-sum hashmap).

## Curated problems
| # | Problem | Difficulty | Pattern note | Done |
|---|---|---|---|---|
| 1 | Maximum Depth of Binary Tree | Easy | the DFS recursion skeleton | ⬜ |
| 2 | Invert Binary Tree | Easy | swap children, recurse | ⬜ |
| 3 | Same Tree / Symmetric Tree | Easy | parallel recursion on two nodes | ⬜ |
| 4 | Binary Tree Level Order Traversal | Medium | BFS with level snapshot | ⬜ |
| 5 | Validate Binary Search Tree | Medium | in-order is sorted (or min/max bounds) | ⬜ |
| 6 | Lowest Common Ancestor of a BST | Medium | walk down using the BST invariant | ⬜ |
| 7 | Kth Smallest Element in a BST | Medium | in-order, stop at k | ⬜ |
| 8 | Diameter of Binary Tree | Easy/Med | post-order returning height, track best | ⬜ |
| 9 | Binary Tree Right Side View | Medium | BFS, take last of each level | ⬜ |
| 10 | Construct Tree from Preorder & Inorder | Medium | recursion + index map | ⬜ |

(Week 7 plan: trees — aim for 3–4. Suggested: **#1 + #4** (lock DFS and BFS skeletons cold), the BST rep **#5**, and one post-order "return info up" rep **#8**. Overlearn the post-order pattern (#8) — it's the trick behind diameter, is-balanced, and max-path-sum.)

## Common mistakes
- **Forgetting the base case** (`if not node: return ...`) → `AttributeError` on `None.left`. Every recursion needs it.
- **Wrong traversal order** — using pre-order when the node needs its children's results first (use **post-order**); forgetting that **in-order of a BST is sorted**.
- **Validating a BST by only checking immediate children** — a node can be > its parent yet violate an ancestor's bound. Pass down (min, max) bounds or use a clean in-order check. ⭐
- **BFS without the level snapshot** — read `len(q)` *once* per level before the inner loop; appending children mid-loop corrupts the level grouping.
- **Using a `list` as a BFS queue** — `pop(0)` is O(n); use `collections.deque` and `popleft()`.
- **Mutable default / shared path lists** in path problems — append then **pop on the way back up** (backtracking) or pass copies.
- **Recursion depth on a degenerate tree** — a skewed tree is O(n) deep; deep recursion can blow Python's stack (convert to iterative DFS with an explicit stack if needed).
- **LCA: returning too early** — only return the current node when *both* sides find a target; otherwise bubble up whichever side is non-null.

## Notes after solving
_Filled in as I work the problems — my own gotchas and which ones I want to re-do._
