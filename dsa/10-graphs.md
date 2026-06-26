# DSA Pattern: Graphs

> **Priority:** see [README](README.md) · **Status:** ✅ written
> **One-line:** Model the problem as **nodes + edges**, build an **adjacency list**, then **BFS or DFS** with a **visited set**. BFS for shortest path in an *unweighted* graph; DFS for connectivity/cycles/components; **topological sort** (Kahn's) for ordering with dependencies; **union-find** for "are these connected?"; **Dijkstra** for weighted shortest path. Grids are graphs in disguise.

The unifying skill is **recognizing a graph**: any problem about *relationships, connectivity, reachability, dependencies, or paths* is a graph problem, even when the word "graph" never appears (a grid of cells, course prerequisites, friend networks, word ladders). Once you see it, the machinery is the same small toolkit — and the single most common bug is **forgetting the `visited` set** (→ infinite loops / re-processing). ⭐

## When to use it (the trigger)
- **"Connected components", "number of islands", "is it connected", "flood fill"** → BFS/DFS or **union-find**.
- **Shortest path in an *unweighted* graph/grid** ("fewest steps", "min moves") → **BFS** (first time you reach a node = shortest). ⭐
- **Shortest path with *weights*** → **Dijkstra** (heap) / Bellman-Ford (negative edges).
- **"Order tasks with prerequisites", "course schedule", "build order"**, detect a cycle in a directed graph → **topological sort** (Kahn's BFS on in-degrees, or DFS).
- **Dynamic connectivity / "do these merge?"** ("accounts merge", "redundant connection", "number of provinces") → **union-find (DSU)**.
- Keyword giveaways: *connected, component, island, grid, path, neighbors, prerequisite/dependency, cycle, network, reach, shortest*.

## The core idea
Represent the graph as an **adjacency list** `{node: [neighbors]}` (or a grid where neighbors are the 4/8 adjacent cells). Then traverse:
- **BFS** (queue) explores in **layers** — all nodes at distance 1, then 2, … so the *first* time it reaches a node is via a **shortest path** (unweighted). Use it for shortest-path/level questions.
- **DFS** (recursion or stack) goes deep first — natural for **connectivity, components, cycle detection, path existence**.
Both need a **`visited` set** so you never revisit a node. **Topological sort** orders a DAG so every edge points "forward" (Kahn's: repeatedly remove a node with in-degree 0). **Union-find** answers connectivity queries in near-O(1) amortized via union-by-rank + path compression.

## Template code

```python
from collections import deque, defaultdict

# 0) Build an adjacency list from an edge list
def build_graph(n, edges, directed=False):
    g = defaultdict(list)
    for u, v in edges:
        g[u].append(v)
        if not directed:
            g[v].append(u)
    return g

# 1) BFS — shortest path / levels in an UNWEIGHTED graph
def bfs_shortest(g, start, target):
    q = deque([(start, 0)])
    seen = {start}
    while q:
        node, dist = q.popleft()
        if node == target:
            return dist
        for nxt in g[node]:
            if nxt not in seen:        # mark on ENQUEUE, not dequeue (avoids dup work)
                seen.add(nxt)
                q.append((nxt, dist + 1))
    return -1

# 2) DFS — connected components / flood fill (here: count islands in a grid)
def num_islands(grid):
    if not grid: return 0
    R, C = len(grid), len(grid[0])
    seen = set()
    def dfs(r, c):
        if r < 0 or r >= R or c < 0 or c >= C: return
        if grid[r][c] != "1" or (r, c) in seen: return
        seen.add((r, c))
        for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
            dfs(r + dr, c + dc)
    count = 0
    for r in range(R):
        for c in range(C):
            if grid[r][c] == "1" and (r, c) not in seen:
                count += 1
                dfs(r, c)
    return count

# 3) Topological sort (Kahn's) — order a DAG / detect a cycle
def topo_sort(n, edges):                # edges: u -> v means u before v
    g = defaultdict(list); indeg = [0] * n
    for u, v in edges:
        g[u].append(v); indeg[v] += 1
    q = deque([i for i in range(n) if indeg[i] == 0])
    order = []
    while q:
        node = q.popleft(); order.append(node)
        for nxt in g[node]:
            indeg[nxt] -= 1
            if indeg[nxt] == 0:
                q.append(nxt)
    return order if len(order) == n else []   # [] => a cycle exists

# 4) Union-Find (DSU) — connectivity, with path compression + union by rank
class DSU:
    def __init__(self, n):
        self.parent = list(range(n)); self.rank = [0] * n
    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]   # path compression
            x = self.parent[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra == rb: return False                          # already connected
        if self.rank[ra] < self.rank[rb]: ra, rb = rb, ra
        self.parent[rb] = ra
        self.rank[ra] += self.rank[ra] == self.rank[rb]
        return True

# 5) Dijkstra — shortest path with non-negative WEIGHTS (heap)
import heapq
def dijkstra(g, start, n):              # g[u] = list of (v, weight)
    dist = [float("inf")] * n; dist[start] = 0
    pq = [(0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d > dist[u]: continue        # stale entry
        for v, w in g[u]:
            if d + w < dist[v]:
                dist[v] = d + w
                heapq.heappush(pq, (dist[v], v))
    return dist
```

## Complexity
- **BFS / DFS:** **O(V + E)** time (visit each vertex and edge once), O(V) space for `visited` + frontier.
- **Topological sort (Kahn's):** O(V + E).
- **Union-find:** ~**O(α(n)) ≈ O(1)** amortized per op with path compression + union by rank (α = inverse Ackermann).
- **Dijkstra (binary heap):** **O((V + E) log V)**.
- **Grid** (R×C): V = R·C, E ≈ 4·R·C → O(R·C).

## Variations / sub-patterns
- **Connectivity / components** — number of islands/provinces, friend circles, flood fill (DFS/BFS or DSU).
- **Shortest path (unweighted)** — BFS: word ladder, shortest path in a binary matrix, rotting oranges (**multi-source BFS** — seed the queue with all sources), knight moves.
- **Topological sort** — course schedule I/II, alien dictionary, build order; also cycle detection in a digraph.
- **Cycle detection** — undirected (DSU or DFS with parent), directed (DFS 3-color / Kahn's leftover nodes).
- **Union-find** — redundant connection, accounts merge, number of connected components, Kruskal's MST.
- **Weighted shortest path** — Dijkstra (non-negative), Bellman-Ford (negatives), 0-1 BFS (deque).
- **Bipartite check / graph coloring** — BFS/DFS 2-coloring.

## Curated problems
| # | Problem | Difficulty | Pattern note | Done |
|---|---|---|---|---|
| 1 | Number of Islands | Medium | grid DFS/BFS flood fill | ⬜ |
| 2 | Clone Graph | Medium | DFS/BFS + hashmap old→new | ⬜ |
| 3 | Course Schedule (I/II) | Medium | topological sort / cycle detect | ⬜ |
| 4 | Rotting Oranges | Medium | multi-source BFS (levels = minutes) | ⬜ |
| 5 | Number of Connected Components | Medium | union-find (or DFS) | ⬜ |
| 6 | Word Ladder | Hard | BFS shortest transformation | ⬜ |
| 7 | Network Delay Time | Medium | Dijkstra | ⬜ |

(Week 9 plan: graphs + light DP — aim for 2–3 total, or use as a catch-up buffer. Suggested graphs: **#1** (lock grid DFS/BFS) + **#3** (topological sort) and optionally **#4** (multi-source BFS). Islands and Course Schedule are the two highest-frequency graph reps.)

## Common mistakes
- **No `visited` set** → infinite loops or exponential re-processing. The #1 graph bug. ⭐
- **Marking visited on *dequeue* instead of *enqueue* in BFS** → the same node gets queued many times before being processed (still correct but slow, and can blow memory). Mark when you enqueue. ⭐
- **Using a `list` as the BFS queue** — `pop(0)` is O(n); use `deque.popleft()`.
- **Grid bounds / out-of-range** — check `0 <= r < R and 0 <= c < C` before indexing.
- **Topological sort without a cycle check** — if `len(order) != n`, there's a cycle; don't return a partial order as valid.
- **Dijkstra with negative weights** — it's wrong; use Bellman-Ford.
- **DFS recursion depth** on a big graph → Python stack overflow; switch to an explicit stack / BFS.
- **Directed vs. undirected confusion** — add both directions only for undirected graphs; for cycle detection in undirected graphs, track the parent to avoid the trivial back-edge.
- **Forgetting the stale-entry skip in Dijkstra** (`if d > dist[u]: continue`) → redundant work.

## Notes after solving
_Filled in as I work the problems — my own gotchas and which ones I want to re-do._
