# DSA Pattern: Intervals

> **Priority:** see [README](README.md) · **Status:** ✅ written
> **One-line:** Almost every interval problem starts with **sort by start time**, then a single linear sweep. The one fact that drives everything: two sorted intervals **overlap iff `curr.start <= prev.end`**. Tells: *"merge", "insert", "meeting rooms", "non-overlapping", "free time."*

The unifying move: once intervals are **sorted by start**, you only ever compare each interval to the one "frontier" you're tracking (the last merged interval, or the earliest end-time across active intervals). That turns an O(n²) "compare all pairs" instinct into an O(n log n) sort + O(n) sweep. When you need to count **concurrent** intervals (max rooms), switch to a **min-heap of end times** or a **sweep-line** of +1/−1 events.

## When to use it (the trigger)
- **Merge overlapping intervals** → sort by start, fold each into the previous if it overlaps.
- **Insert a new interval** into a sorted, non-overlapping list → three phases (before / overlapping / after).
- **"How many rooms/resources at once?" / max concurrent** → min-heap of end times, or sweep-line events.
- **"Can a person attend all meetings?" / any overlap?** → sort, check adjacent pairs.
- **Non-overlapping: remove the fewest / pick the most** → greedy by **earliest end time**.
- Keyword giveaways: *interval, range, [start, end], overlap, merge, meeting/room, schedule, conflict, calendar*.

## The core idea
**Sort by start.** Then walk left to right maintaining a frontier:
- **Merging:** keep the last interval in the output; if the next one's `start <= last.end`, extend `last.end = max(last.end, next.end)`; else append a new interval.
- **Counting concurrency:** a **min-heap of end times** holds currently-active intervals. For each new interval (in start order), pop every end `<= current.start` (those meetings have finished), then push the current end. The **heap size = rooms in use**; its max over the sweep is the answer.
- **Greedy selection:** to keep the most non-overlapping intervals, sort by **end** and greedily take each interval that starts after the last taken one ends.

## Template code

```python
# 1) Merge overlapping intervals
def merge(intervals):
    intervals.sort(key=lambda iv: iv[0])           # sort by start
    out = [intervals[0]]
    for s, e in intervals[1:]:
        if s <= out[-1][1]:                        # overlap with last merged
            out[-1][1] = max(out[-1][1], e)        # extend the end
        else:
            out.append([s, e])
    return out

# 2) Insert an interval into a sorted, non-overlapping list
def insert(intervals, new):
    out, i, n = [], 0, len(intervals)
    while i < n and intervals[i][1] < new[0]:      # ends entirely before new
        out.append(intervals[i]); i += 1
    while i < n and intervals[i][0] <= new[1]:     # overlaps new -> absorb
        new = [min(new[0], intervals[i][0]), max(new[1], intervals[i][1])]
        i += 1
    out.append(new)
    out.extend(intervals[i:])                      # the rest, entirely after
    return out

# 3) Minimum meeting rooms (max concurrent intervals) -> min-heap of end times
import heapq
def min_meeting_rooms(intervals):
    intervals.sort(key=lambda iv: iv[0])
    ends = []                                      # min-heap of active end times
    rooms = 0
    for s, e in intervals:
        while ends and ends[0] <= s:               # a meeting freed up before s
            heapq.heappop(ends)
        heapq.heappush(ends, e)
        rooms = max(rooms, len(ends))              # peak concurrency
    return rooms

# 4) Can attend all meetings? (any overlap?)
def can_attend_all(intervals):
    intervals.sort(key=lambda iv: iv[0])
    return all(intervals[i][0] >= intervals[i-1][1] for i in range(1, len(intervals)))

# 5) Max non-overlapping intervals -> greedy by EARLIEST END
def max_non_overlapping(intervals):
    intervals.sort(key=lambda iv: iv[1])           # sort by END
    count, last_end = 0, float("-inf")
    for s, e in intervals:
        if s >= last_end:                          # doesn't conflict with last taken
            count += 1
            last_end = e
    return count
```

## Complexity
- **Sort:** O(n log n) — dominates almost every interval solution.
- **Sweep / merge:** O(n) after sorting.
- **Meeting-rooms heap:** O(n log n) (each interval pushed/popped once).
- **Space:** O(n) for output; O(n) heap worst case (all concurrent).

## Variations / sub-patterns
- **Merge** — merge intervals, merge two sorted interval lists, interval list intersections.
- **Insert** — insert interval, "summary ranges."
- **Concurrency / sweep-line** — meeting rooms II, car pooling, "my calendar I/II/III", max events. The general tool: emit `(start, +1)` and `(end, −1)` events, sort, accumulate, track the running max.
- **Greedy selection** — non-overlapping intervals (min removals), minimum arrows to burst balloons, activity selection — all "sort by end, take greedily."
- **Employee free time** — merge everyone's intervals, then report the gaps.

## Curated problems
| # | Problem | Difficulty | Pattern note | Done |
|---|---|---|---|---|
| 1 | Merge Intervals | Medium | the canonical sort-by-start fold | ⬜ |
| 2 | Insert Interval | Medium | three-phase (before/overlap/after) | ⬜ |
| 3 | Meeting Rooms | Easy | sort, check adjacent overlap | ⬜ |
| 4 | Meeting Rooms II | Medium | min-heap of end times (max concurrency) | ⬜ |
| 5 | Non-overlapping Intervals | Medium | greedy by earliest end | ⬜ |
| 6 | Minimum Number of Arrows to Burst Balloons | Medium | greedy by end (same idea) | ⬜ |

(Week 8 plan: intervals — aim for 2–3 (shared with heaps). Suggested: **#1 + #2** (merge and insert are the bread-and-butter) and the concurrency rep **#4**. Lock the `curr.start <= prev.end` overlap test and the "sort by end for greedy" rule.)

## Common mistakes
- **Forgetting to sort first** — every template assumes start-sorted (or end-sorted for greedy) input. ⭐
- **Wrong overlap condition** — overlap is `curr.start <= prev.end` (use `<` vs `<=` consistently depending on whether touching endpoints count as overlapping; *clarify with the interviewer*).
- **Extending with `next.end` instead of `max(prev.end, next.end)`** — a fully-contained interval would wrongly shrink the merged range.
- **Sorting by the wrong key** — merge/sweep sort by **start**; greedy non-overlapping selection sorts by **end**. Mixing them breaks the logic. ⭐
- **Meeting rooms: not popping finished meetings** — pop all ends `<= current.start` before pushing, or you over-count rooms.
- **Mutating intervals while iterating** — build a fresh output list.
- **Off-by-one on touching intervals** — `[1,2]` and `[2,3]`: do they overlap? Depends on whether the endpoint is inclusive — ask.

## Notes after solving
_Filled in as I work the problems — my own gotchas and which ones I want to re-do._
