"""Filterbanks over the STFT: mel (with mel spectrogram) and log-frequency.

The default mel scale is Slaney's (linear below 1 kHz, logarithmic above),
with each triangular filter normalized to unit area - the same defaults as
librosa, so mel features are directly comparable. The log-frequency bank
(fixed bands per octave) is the one SuperFlux uses for onset detection.
"""

from __future__ import annotations

import numpy as np

from .spectral import fft_frequencies, stft

_F_SP = 200.0 / 3  # Hz per mel in the linear part
_MIN_LOG_HZ = 1000.0
_MIN_LOG_MEL = _MIN_LOG_HZ / _F_SP
_LOGSTEP = np.log(6.4) / 27.0


def hz_to_mel(hz, htk: bool = False) -> np.ndarray:
    hz = np.asarray(hz, dtype=np.float64)
    if htk:
        return 2595.0 * np.log10(1.0 + hz / 700.0)
    mel = hz / _F_SP
    log_part = hz >= _MIN_LOG_HZ
    return np.where(log_part, _MIN_LOG_MEL + np.log(np.maximum(hz, _MIN_LOG_HZ) / _MIN_LOG_HZ) / _LOGSTEP, mel)


def mel_to_hz(mel, htk: bool = False) -> np.ndarray:
    mel = np.asarray(mel, dtype=np.float64)
    if htk:
        return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    hz = mel * _F_SP
    return np.where(mel >= _MIN_LOG_MEL, _MIN_LOG_HZ * np.exp(_LOGSTEP * (mel - _MIN_LOG_MEL)), hz)


def mel_frequencies(n_mels: int, fmin: float, fmax: float, htk: bool = False) -> np.ndarray:
    """``n_mels`` frequencies in Hz, evenly spaced on the mel scale."""
    return mel_to_hz(np.linspace(hz_to_mel(fmin, htk), hz_to_mel(fmax, htk), n_mels), htk)


def mel_filterbank(
    sr: float,
    n_fft: int,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    *,
    htk: bool = False,
    norm: str | None = "slaney",
) -> np.ndarray:
    """Triangular filters, shape ``(n_mels, n_fft // 2 + 1)``.

    ``norm="slaney"`` scales each filter to unit area, so wide high-frequency
    bands do not dominate; ``None`` leaves peaks at 1.
    """
    fmax = sr / 2.0 if fmax is None else fmax
    freqs = fft_frequencies(sr, n_fft)
    edges = mel_frequencies(n_mels + 2, fmin, fmax, htk)
    widths = np.diff(edges)
    ramps = edges[:, None] - freqs[None, :]
    lower = -ramps[:-2] / widths[:-1, None]
    upper = ramps[2:] / widths[1:, None]
    weights = np.maximum(0.0, np.minimum(lower, upper))
    if norm == "slaney":
        weights *= (2.0 / (edges[2:] - edges[:-2]))[:, None]
    elif norm is not None:
        raise ValueError(f"unknown mel norm '{norm}'")
    return weights


def mel_spectrogram(
    x: np.ndarray | None = None,
    sr: float = 22050,
    *,
    spec: np.ndarray | None = None,
    n_fft: int = 2048,
    hop: int = 512,
    n_mels: int = 128,
    fmin: float = 0.0,
    fmax: float | None = None,
    power: float = 2.0,
) -> np.ndarray:
    """Mel power spectrogram, shape ``(n_mels, n_frames)``.

    Pass either the signal ``x`` or a magnitude spectrogram ``spec`` (already
    raised to ``power``).
    """
    if spec is None:
        if x is None:
            raise ValueError("give a signal or a spectrogram")
        spec = np.abs(stft(x, n_fft, hop)) ** power
    else:
        n_fft = 2 * (spec.shape[0] - 1)
    return mel_filterbank(sr, n_fft, n_mels, fmin, fmax) @ spec


def log_filterbank(
    sr: float,
    n_fft: int,
    bands_per_octave: int = 24,
    fmin: float = 27.5,
    fmax: float = 16000.0,
    *,
    norm: bool = True,
) -> np.ndarray:
    """Triangular filters with log-spaced centres, shape ``(n_bands, n_fft // 2 + 1)``.

    Centres are rounded to STFT bins and duplicates dropped, so at low
    frequencies (where bins are wider than the bands) each band is one bin.
    Each filter spans from the previous centre bin to the next; ``norm``
    scales filters to unit sum.
    """
    fmax = min(fmax, sr / 2.0)
    n_oct = np.log2(fmax / fmin)
    centres = fmin * 2.0 ** (np.arange(int(np.floor(n_oct * bands_per_octave)) + 1) / bands_per_octave)
    bins = np.unique(np.round(centres * n_fft / sr).astype(int))
    bins = bins[(bins > 0) & (bins < n_fft // 2)]
    if len(bins) < 3:
        raise ValueError("log filterbank needs at least 3 distinct bins; raise n_fft or widen fmin..fmax")
    weights = np.zeros((len(bins) - 2, n_fft // 2 + 1))
    for i, (lo, mid, hi) in enumerate(zip(bins[:-2], bins[1:-1], bins[2:])):
        rise = np.arange(lo, mid + 1)
        fall = np.arange(mid, hi + 1)
        weights[i, rise] = (rise - lo + 1) / (mid - lo + 1)
        weights[i, fall] = np.maximum(weights[i, fall], (hi - fall + 1) / (hi - mid + 1))
        if norm:
            weights[i] /= weights[i].sum()
    return weights
