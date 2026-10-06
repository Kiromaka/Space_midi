"""Global tempo from the autocorrelation of an onset envelope.

A steady pulse makes the onset envelope correlate with itself shifted by one
beat (and by two, three... beats). The lag with the strongest correlation is
the beat period. Every multiple and fraction of the true period also
correlates, so a log-normal prior over tempo (centred on ``start_bpm``,
``std_octaves`` wide; Ellis 2007, as in librosa) picks the most likely
metrical level. ``harmonics`` > 1 also adds the correlation at 2, 3, ...
times each lag, which favours lags whose multiples line up too (the true
beat) over lags that only match by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def autocorrelation(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Unnormalized autocorrelation of ``x - mean(x)`` for lags ``0..max_lag`` (via FFT)."""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    n = len(x)
    size = 1 << int(np.ceil(np.log2(max(2 * n, 2))))
    spec = np.fft.rfft(x, size)
    acf = np.fft.irfft(spec * np.conj(spec), size)[: max_lag + 1]
    if len(acf) < max_lag + 1:
        acf = np.pad(acf, (0, max_lag + 1 - len(acf)))
    return acf


@dataclass
class TempoEstimate:
    bpm: float
    candidates: list[tuple[float, float]] = field(default_factory=list)  # (bpm, relative strength), best first


def tempo_prior(bpm: np.ndarray, start_bpm: float = 120.0, std_octaves: float = 1.0) -> np.ndarray:
    """Log-normal weighting over tempo, 1 at ``start_bpm``."""
    return np.exp(-0.5 * (np.log2(np.asarray(bpm, dtype=np.float64) / start_bpm) / std_octaves) ** 2)


def estimate_tempo(
    env: np.ndarray,
    fps: float,
    *,
    start_bpm: float = 120.0,
    std_octaves: float = 1.0,
    min_bpm: float = 40.0,
    max_bpm: float = 240.0,
    harmonics: int = 1,
    smooth: float = 1.0,
) -> TempoEstimate:
    """The most likely tempo of an onset envelope sampled at ``fps`` frames per second.

    ``smooth`` is the width (standard deviation, in frames) of a Gaussian
    applied to the envelope first. Without it, a beat period that is not a
    whole number of frames (21.5 frames at 120 BPM and 43 fps) spreads its
    correlation over two lags, and the 2-beat lag, which is closer to a whole
    number, wins: the classic half-tempo error.
    """
    if min_bpm <= 0 or max_bpm <= min_bpm:
        raise ValueError("need 0 < min_bpm < max_bpm")
    lo = max(1, int(np.floor(60.0 * fps / max_bpm)))
    hi = int(np.ceil(60.0 * fps / min_bpm))
    harmonics = max(1, int(harmonics))
    env = np.asarray(env, dtype=np.float64)
    if smooth > 0:
        half = int(np.ceil(3 * smooth))
        kernel = np.exp(-0.5 * (np.arange(-half, half + 1) / smooth) ** 2)
        env = np.convolve(env, kernel / kernel.sum(), mode="same")
    acf = autocorrelation(env, hi * harmonics + 1)
    if acf[0] <= 0:
        return TempoEstimate(bpm=float(start_bpm), candidates=[])
    acf = acf / acf[0]
    lags = np.arange(lo, hi + 1)
    strength = np.zeros(len(lags))
    for k in range(1, harmonics + 1):
        strength += acf[np.minimum(k * lags, len(acf) - 1)] / k
    score = np.maximum(strength, 0.0) * tempo_prior(60.0 * fps / lags, start_bpm, std_octaves)

    peaks = [i for i in range(len(score)) if score[i] > 0
             and (i == 0 or score[i] >= score[i - 1]) and (i == len(score) - 1 or score[i] > score[i + 1])]
    if not peaks:
        return TempoEstimate(bpm=float(start_bpm), candidates=[])
    peaks.sort(key=lambda i: score[i], reverse=True)
    top = score[peaks[0]]
    candidates = []
    for i in peaks[:5]:
        lag = float(lags[i])
        if 0 < i < len(score) - 1:  # parabolic interpolation for a sub-frame lag
            a, b, c = score[i - 1], score[i], score[i + 1]
            denom = a - 2 * b + c
            if denom < 0:
                lag += 0.5 * (a - c) / denom
        candidates.append((60.0 * fps / lag, float(score[i] / top)))
    return TempoEstimate(bpm=candidates[0][0], candidates=candidates)
