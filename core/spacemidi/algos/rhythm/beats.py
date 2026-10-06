"""Beat tracking by dynamic programming (Ellis, "Beat tracking by dynamic programming", JNMR 2007).

Given the onset envelope and a target beat period P, find the sequence of
beat times that (a) lands on strong onsets and (b) keeps consecutive beats
about P apart. The score of a beat at frame t is

    C(t) = O(t) + max over t' in [t - 2P, t - P/2] of ( C(t') - tightness * log((t - t') / P)^2 )

where O is the onset envelope smoothed around each frame. The log-squared
penalty is zero for a gap of exactly P and grows symmetrically in tempo
ratio. Keeping the best predecessor of every frame and backtracking from the
best final beat gives the globally best beat sequence in one pass.
"""

from __future__ import annotations

import numpy as np


def local_score(env: np.ndarray, period: float) -> np.ndarray:
    """Onset envelope (scaled to unit standard deviation) smoothed by a Gaussian of width period / 32."""
    env = np.asarray(env, dtype=np.float64)
    std = env.std(ddof=1) if len(env) > 1 else 0.0
    if std <= 0:
        return np.zeros_like(env)
    half = max(1, int(round(period)))
    kernel = np.exp(-0.5 * (np.arange(-half, half + 1) * 32.0 / period) ** 2)
    return np.convolve(env / std, kernel, mode="same")


def track_beats(env: np.ndarray, fps: float, bpm: float, *, tightness: float = 100.0, trim: bool = True) -> np.ndarray:
    """Beat positions as frame indices of ``env``, in time order."""
    if bpm <= 0:
        raise ValueError("bpm must be positive")
    period = 60.0 * fps / bpm
    local = local_score(env, period)
    n = len(local)
    if n == 0 or not np.any(local > 0):
        return np.zeros(0, dtype=int)

    offsets = np.arange(-int(round(2 * period)), -int(round(period / 2)) + 1)
    penalty = -tightness * np.log(-offsets / period) ** 2
    cumulative = np.zeros(n)
    backlink = np.full(n, -1, dtype=int)
    for t in range(n):
        idx = t + offsets
        valid = idx >= 0
        best = 0.0
        if valid.any():
            cand = penalty[valid] + cumulative[idx[valid]]
            k = int(np.argmax(cand))
            if cand[k] > 0:  # extending a chain only pays when it adds score
                best = cand[k]
                backlink[t] = idx[valid][k]
        cumulative[t] = local[t] + best

    last = _last_beat(cumulative)
    beats = [last]
    while backlink[beats[-1]] >= 0:
        beats.append(int(backlink[beats[-1]]))
    beats = np.asarray(beats[::-1], dtype=int)
    return _trim_weak(beats, local) if trim else beats


def _last_beat(cumulative: np.ndarray) -> int:
    """The last local maximum of the cumulative score that is at least half the median maximum."""
    c = cumulative
    is_max = np.zeros(len(c), dtype=bool)
    if len(c) > 2:
        is_max[1:-1] = (c[1:-1] > c[:-2]) & (c[1:-1] >= c[2:])
    is_max[-1] = len(c) == 1 or c[-1] > c[-2]
    if not is_max.any():
        return int(np.argmax(c))
    threshold = 0.5 * np.median(c[is_max])
    return int(np.nonzero(is_max & (c >= threshold))[0][-1])


def _trim_weak(beats: np.ndarray, local: np.ndarray) -> np.ndarray:
    """Drop leading and trailing beats much weaker than the rest (silence before and after the music)."""
    if len(beats) < 3:
        return beats
    w = local[beats]
    smooth = np.convolve(w, np.hanning(5)[1:-1] / np.hanning(5)[1:-1].sum(), mode="same")
    threshold = 0.5 * np.sqrt(np.mean(smooth**2))
    keep = np.nonzero(smooth >= threshold)[0]
    if len(keep) == 0:
        return beats
    return beats[keep[0] : keep[-1] + 1]
