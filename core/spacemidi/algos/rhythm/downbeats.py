"""Downbeats: which beat of the bar is "one".

Three cues tend to land on the first beat of a bar in popular music:

* accent - the onset envelope is strongest there;
* low end - kick drum and bass notes start there;
* harmony - chords change there.

Each cue is measured once per beat and turned into z-scores. For every
candidate bar length (4, optionally 3) and every phase, the score is how much
the beats at that phase stand out from the others. The best phase gives the
downbeats; 3/4 has to beat 4/4 by ``meter_bias`` because 4/4 is far more
common. This assumes the bar phase never changes within a song (no odd bars),
which is the main simplification of this L1 version.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class DownbeatChoice:
    meter: int
    phase: int
    scores: dict[str, float]  # "meter/phase" -> score, for inspection


def beat_features(
    beats: np.ndarray, env: np.ndarray, low_env: np.ndarray, chroma: np.ndarray, radius: int = 1
) -> dict[str, np.ndarray]:
    """Per-beat accent, low-end onset strength and harmonic change (all frame-indexed inputs)."""
    beats = np.asarray(beats, dtype=int)
    n = len(env)

    def peak(signal: np.ndarray) -> np.ndarray:
        return np.array([signal[max(0, b - radius) : min(n, b + radius + 1)].max() if n else 0.0 for b in beats])

    edges = np.concatenate([beats, [min(n, beats[-1] + (beats[-1] - beats[-2] if len(beats) > 1 else 1))]])
    segments = [chroma[:, a:max(a + 1, b)].mean(axis=1) for a, b in zip(edges[:-1], edges[1:])]
    harmony = np.zeros(len(beats))
    for i in range(1, len(beats)):
        u, v = segments[i - 1], segments[i]
        norm = np.linalg.norm(u) * np.linalg.norm(v)
        harmony[i] = 1.0 - (float(u @ v) / norm if norm > 0 else 1.0)
    return {"accent": peak(env), "low": peak(low_env), "harmony": harmony}


def _zscore(x: np.ndarray) -> np.ndarray:
    std = x.std()
    return (x - x.mean()) / std if std > 0 else np.zeros_like(x)


def choose_downbeats(
    features: dict[str, np.ndarray],
    weights: dict[str, float],
    meters: tuple[int, ...] = (4, 3),
    meter_bias: float = 0.1,
) -> DownbeatChoice:
    """The bar length and phase whose beats stand out most in the weighted features."""
    n = len(next(iter(features.values())))
    combined = sum(weights.get(name, 0.0) * _zscore(values) for name, values in features.items())
    combined = np.asarray(combined, dtype=np.float64)
    scores: dict[str, float] = {}
    best: dict[int, tuple[float, int]] = {}
    for m in meters:
        for p in range(m):
            on = combined[p::m]
            off = np.delete(combined, np.arange(p, n, m))
            if len(on) == 0 or len(off) == 0:
                continue
            s = float(on.mean() - off.mean())
            scores[f"{m}/{p}"] = s
            if m not in best or s > best[m][0]:
                best[m] = (s, p)
    if not best:
        return DownbeatChoice(meter=4, phase=0, scores=scores)
    meter = 4 if 4 in best else min(best)
    for m, (s, _) in best.items():
        if m != 4 and s > best.get(4, (-np.inf, 0))[0] + meter_bias and s > best[meter][0]:
            meter = m
    return DownbeatChoice(meter=meter, phase=best[meter][1], scores=scores)
