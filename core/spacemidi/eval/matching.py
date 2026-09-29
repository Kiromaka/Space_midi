"""Maximum bipartite matching between reference and estimated events.

Scores in music transcription count how many estimated events can be paired
one-to-one with reference events under a tolerance. Greedy pairing (closest
first) can miss pairs, so we compute a *maximum* matching with the
Hopcroft-Karp algorithm, as mir_eval does. The results are cross-checked
against mir_eval in the tests.
"""

from __future__ import annotations

import bisect
from collections import deque
from collections.abc import Callable, Sequence

# mir_eval rounds note-timing distances to 4 decimals before comparing them with
# the tolerance (so 0.05004 s still counts as within 50 ms), but compares event
# times (onsets, beats) exactly. We copy both behaviours so boundary cases agree.
_NOTE_DECIMALS = 4


def _round(x: float, decimals: int) -> float:
    """Round like numpy.around: scale, round half to even, scale back."""
    scale = 10.0**decimals
    return round(x * scale) / scale


def maximum_matching(adjacency: Sequence[Sequence[int]], n_right: int) -> list[tuple[int, int]]:
    """Hopcroft-Karp. ``adjacency[i]`` lists the right nodes left node ``i`` may pair with.

    Returns the matched ``(left, right)`` pairs, sorted by left index.
    """
    n_left = len(adjacency)
    match_left = [-1] * n_left
    match_right = [-1] * n_right
    inf = n_left + n_right + 1
    dist = [inf] * n_left

    def bfs() -> bool:
        queue = deque()
        for u in range(n_left):
            if match_left[u] == -1:
                dist[u] = 0
                queue.append(u)
            else:
                dist[u] = inf
        found = False
        while queue:
            u = queue.popleft()
            for v in adjacency[u]:
                w = match_right[v]
                if w == -1:
                    found = True
                elif dist[w] == inf:
                    dist[w] = dist[u] + 1
                    queue.append(w)
        return found

    def augment(root: int) -> bool:
        # Iterative DFS along the BFS layers (no recursion limit on long paths).
        stack = [root]
        via: dict[int, int] = {}
        while stack:
            u = stack[-1]
            if pos[u] < len(adjacency[u]):
                v = adjacency[u][pos[u]]
                pos[u] += 1
                w = match_right[v]
                if w == -1:
                    for k, node in enumerate(stack):
                        right = via[node] if k < len(stack) - 1 else v
                        match_left[node] = right
                        match_right[right] = node
                    return True
                if dist[w] == dist[u] + 1:
                    via[u] = v
                    stack.append(w)
            else:
                dist[u] = inf
                stack.pop()
        return False

    while bfs():
        pos = [0] * n_left
        for u in range(n_left):
            if match_left[u] == -1:
                augment(u)
    return [(u, v) for u, v in enumerate(match_left) if v != -1]


def match_events(reference: Sequence[float], estimate: Sequence[float], window: float) -> list[tuple[int, int]]:
    """Pair event times one-to-one when ``|ref - est| <= window`` (compared exactly)."""
    return _match(reference, estimate, window, lambda i, j: True, decimals=None)


def match_notes(
    ref: Sequence[tuple[float, float, int]],
    est: Sequence[tuple[float, float, int]],
    onset_tolerance: float = 0.05,
    offset_ratio: float | None = 0.2,
    offset_min_tolerance: float = 0.05,
) -> list[tuple[int, int]]:
    """Pair notes given as ``(onset, offset, pitch)``.

    A pair needs the same pitch and onsets within ``onset_tolerance``. With an
    ``offset_ratio``, offsets must also agree within
    ``max(offset_ratio * reference duration, offset_min_tolerance)``; with
    ``None`` offsets are ignored (the "onset only" score).
    """

    def compatible(i: int, j: int) -> bool:
        r_on, r_off, r_pitch = ref[i]
        e_on, e_off, e_pitch = est[j]
        if r_pitch != e_pitch:
            return False
        if offset_ratio is None:
            return True
        tolerance = max(offset_ratio * (r_off - r_on), offset_min_tolerance)
        return _round(abs(r_off - e_off), _NOTE_DECIMALS) <= tolerance

    onsets_ref = [n[0] for n in ref]
    onsets_est = [n[0] for n in est]
    return _match(onsets_ref, onsets_est, onset_tolerance, compatible, decimals=_NOTE_DECIMALS)


def _match(ref_times, est_times, window: float, compatible: Callable[[int, int], bool], decimals: int | None):
    """Candidate pairs within the window (rounded to ``decimals`` if given), then a maximum matching."""
    order = sorted(range(len(est_times)), key=lambda j: est_times[j])
    sorted_times = [est_times[j] for j in order]
    # The pre-filter must be at least as wide as what rounding can still accept.
    margin = window + (0.5 * 10**-decimals if decimals is not None else 0.0) + 1e-9

    def hit(r: float, e: float) -> bool:
        if decimals is None:
            return e - window <= r <= e + window  # the same float operations as mir_eval
        return _round(abs(r - e), decimals) <= window

    adjacency: list[list[int]] = []
    for i, t in enumerate(ref_times):
        lo = bisect.bisect_left(sorted_times, t - margin)
        hi = bisect.bisect_right(sorted_times, t + margin)
        adjacency.append([order[k] for k in range(lo, hi) if hit(t, sorted_times[k]) and compatible(i, order[k])])
    return maximum_matching(adjacency, len(est_times))
