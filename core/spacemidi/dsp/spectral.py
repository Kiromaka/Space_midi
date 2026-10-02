"""Short-time Fourier transform and the helpers around it.

Conventions match librosa (so results can be cross-checked and compared
with published numbers):

* periodic Hann window;
* ``center=True`` pads ``n_fft // 2`` zeros on both sides, so frame ``t`` is
  centred on sample ``t * hop``;
* spectra have shape ``(n_fft // 2 + 1, n_frames)``, no normalization.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def window(name: str, length: int) -> np.ndarray:
    """A periodic window (the right kind for spectral analysis)."""
    n = np.arange(length)
    if name == "hann":
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / length)
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * n / length)
    if name in ("rect", "boxcar", "ones"):
        return np.ones(length)
    raise ValueError(f"unknown window '{name}'")


def _window_for(name: str | np.ndarray, n_fft: int, win_length: int | None) -> np.ndarray:
    if isinstance(name, np.ndarray):
        w = np.asarray(name, dtype=np.float64)
    else:
        w = window(name, win_length or n_fft)
    if len(w) > n_fft:
        raise ValueError(f"window length {len(w)} is longer than n_fft={n_fft}")
    left = (n_fft - len(w)) // 2
    return np.pad(w, (left, n_fft - len(w) - left))


def n_frames(n_samples: int, n_fft: int, hop: int, center: bool = True) -> int:
    padded = n_samples + (2 * (n_fft // 2) if center else 0)
    return 1 + (padded - n_fft) // hop if padded >= n_fft else 0


def stft(
    x: np.ndarray,
    n_fft: int = 2048,
    hop: int = 512,
    *,
    win_length: int | None = None,
    window: str | np.ndarray = "hann",
    center: bool = True,
    pad_mode: str = "constant",
) -> np.ndarray:
    """Complex spectrogram of a mono signal, shape ``(n_fft // 2 + 1, n_frames)``."""
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("stft expects a mono signal (1-D array)")
    w = _window_for(window, n_fft, win_length)
    if center:
        x = np.pad(x, n_fft // 2, mode=pad_mode)
    if len(x) < n_fft:
        raise ValueError(f"signal of {len(x)} samples is shorter than n_fft={n_fft}")
    frames = sliding_window_view(x, n_fft)[::hop]
    out = np.empty((n_fft // 2 + 1, len(frames)), dtype=np.complex128)
    block = max(1, 2**22 // n_fft)  # about 32 MB of frames at a time
    for start in range(0, len(frames), block):
        out[:, start : start + block] = np.fft.rfft(frames[start : start + block] * w, axis=-1).T
    return out


def istft(
    spec: np.ndarray,
    hop: int = 512,
    *,
    win_length: int | None = None,
    window: str | np.ndarray = "hann",
    center: bool = True,
    length: int | None = None,
) -> np.ndarray:
    """Inverse of :func:`stft` by weighted overlap-add (least-squares estimate)."""
    n_fft = 2 * (spec.shape[0] - 1)
    w = _window_for(window, n_fft, win_length)
    frames = np.fft.irfft(spec.T, n=n_fft, axis=-1) * w
    total = n_fft + hop * (len(frames) - 1)
    y = np.zeros(total)
    wsum = np.zeros(total)
    w2 = w**2
    for t, frame in enumerate(frames):
        y[t * hop : t * hop + n_fft] += frame
        wsum[t * hop : t * hop + n_fft] += w2
    nonzero = wsum > np.finfo(np.float64).tiny
    y[nonzero] /= wsum[nonzero]
    start = n_fft // 2 if center else 0
    if length is None:
        return y[start : total - start] if center else y
    y = y[start:]
    return y[:length] if len(y) >= length else np.pad(y, (0, length - len(y)))


def fft_frequencies(sr: float, n_fft: int) -> np.ndarray:
    """Centre frequency in Hz of each STFT bin."""
    return np.fft.rfftfreq(n_fft, 1.0 / sr)


def frames_to_time(frames, sr: float, hop: int) -> np.ndarray:
    return np.asarray(frames, dtype=np.float64) * hop / sr


def time_to_frames(times, sr: float, hop: int) -> np.ndarray:
    return np.floor(np.asarray(times, dtype=np.float64) * sr / hop).astype(int)


def power_to_db(
    s: np.ndarray, ref: float | Callable[[np.ndarray], float] = 1.0, amin: float = 1e-10, top_db: float | None = 80.0
) -> np.ndarray:
    """``10 * log10(s / ref)``, floored at ``amin`` and at ``max - top_db``."""
    s = np.asarray(s, dtype=np.float64)
    ref_value = ref(s) if callable(ref) else ref
    out = 10.0 * np.log10(np.maximum(amin, s)) - 10.0 * np.log10(np.maximum(amin, abs(ref_value)))
    if top_db is not None:
        if top_db < 0:
            raise ValueError("top_db must be non-negative")
        out = np.maximum(out, out.max() - top_db)
    return out


def amplitude_to_db(
    s: np.ndarray, ref: float | Callable[[np.ndarray], float] = 1.0, amin: float = 1e-5, top_db: float | None = 80.0
) -> np.ndarray:
    """``20 * log10(s / ref)`` with the same floors as :func:`power_to_db`."""
    s = np.abs(np.asarray(s))
    ref_value = ref(s) if callable(ref) else ref
    return power_to_db(s**2, ref=ref_value**2, amin=amin**2, top_db=top_db)


def hz_to_midi(hz) -> np.ndarray:
    return 69.0 + 12.0 * np.log2(np.asarray(hz, dtype=np.float64) / 440.0)


def midi_to_hz(midi, tuning_hz: float = 440.0) -> np.ndarray:
    return tuning_hz * 2.0 ** ((np.asarray(midi, dtype=np.float64) - 69.0) / 12.0)
