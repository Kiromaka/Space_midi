"""Constant-Q transform and chroma.

The CQT has logarithmically spaced bins (``bins_per_octave`` per octave), so
every semitone gets the same number of bins and low notes get long analysis
windows. This is the direct "spectral kernel" method (Brown & Puckette 1992):
each bin's windowed complex exponential is transformed once with the FFT,
and every frame's spectrum is multiplied by that sparse kernel matrix.

Scaling: a sinusoid of amplitude ``A`` at a bin's centre frequency gives a
magnitude of about ``A / 2`` in that bin, at any frequency.

A recursive (octave-by-octave downsampling) variant would be faster for long
low-frequency windows; whether it is worth it is one of the stage 1 research
questions (docs/research/stage1-spectral.md).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy import sparse

from .spectral import midi_to_hz, window as make_window

#: C1, the default lowest CQT bin (MIDI 24).
C1_HZ = float(midi_to_hz(24))


def cqt_frequencies(n_bins: int, fmin: float, bins_per_octave: int = 12) -> np.ndarray:
    return fmin * 2.0 ** (np.arange(n_bins) / bins_per_octave)


@dataclass(frozen=True)
class CQTKernel:
    """Precomputed spectral kernels for one CQT configuration."""

    freqs: np.ndarray
    lengths: np.ndarray  # window length in samples per bin
    n_fft: int
    matrix: sparse.csr_matrix  # (n_bins, n_fft // 2 + 1), conjugated and scaled


@lru_cache(maxsize=16)
def cqt_kernel(
    sr: float,
    fmin: float = C1_HZ,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    filter_scale: float = 1.0,
    window: str = "hann",
    sparsity: float = 1e-3,
) -> CQTKernel:
    """Kernels for each bin; entries below ``sparsity`` times a row's peak are dropped."""
    freqs = cqt_frequencies(n_bins, fmin, bins_per_octave)
    q = filter_scale / (2.0 ** (1.0 / bins_per_octave) - 1.0)
    lengths = q * sr / freqs
    top = freqs[-1] + 2.0 * sr / lengths[-1]  # centre + half the Hann main lobe
    if top > sr / 2.0:
        raise ValueError(
            f"highest CQT bin ({freqs[-1]:.0f} Hz) reaches above the Nyquist frequency ({sr / 2:.0f} Hz)"
        )
    n_fft = int(2 ** np.ceil(np.log2(np.ceil(lengths[0]))))
    rows, cols, values = [], [], []
    for k, (f, length) in enumerate(zip(freqs, lengths)):
        n = int(np.ceil(length))
        w = make_window(window, n)
        t = np.arange(n) - (n - 1) / 2.0  # phase reference at the window centre
        kernel = w * np.exp(2j * np.pi * f * t / sr) / w.sum()
        buf = np.zeros(n_fft, dtype=np.complex128)
        start = n_fft // 2 - n // 2
        buf[start : start + n] = kernel
        spec = np.conj(np.fft.fft(buf)[: n_fft // 2 + 1]) / n_fft
        mag = np.abs(spec)
        keep = np.nonzero(mag >= sparsity * mag.max())[0]
        rows.append(np.full(len(keep), k))
        cols.append(keep)
        values.append(spec[keep])
    matrix = sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))), shape=(n_bins, n_fft // 2 + 1)
    )
    return CQTKernel(freqs, lengths, n_fft, matrix)


def cqt(
    x: np.ndarray,
    sr: float = 22050,
    hop: int = 512,
    *,
    fmin: float = C1_HZ,
    n_bins: int = 84,
    bins_per_octave: int = 12,
    tuning: float = 0.0,
    filter_scale: float = 1.0,
) -> np.ndarray:
    """Complex CQT, shape ``(n_bins, n_frames)``; frame ``t`` is centred on sample ``t * hop``.

    ``tuning`` shifts all bins by that many cents (for recordings not tuned
    to A440). The number of frames equals that of :func:`stft` with the same hop.
    """
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("cqt expects a mono signal (1-D array)")
    fmin = fmin * 2.0 ** (tuning / 1200.0)
    kernel = cqt_kernel(float(sr), float(fmin), int(n_bins), int(bins_per_octave), float(filter_scale))
    n_fft = kernel.n_fft
    padded = np.pad(x, n_fft // 2)
    n_out = 1 + len(x) // hop
    if len(padded) < n_fft + (n_out - 1) * hop:
        padded = np.pad(padded, (0, n_fft + (n_out - 1) * hop - len(padded)))
    frames = sliding_window_view(padded, n_fft)[::hop][:n_out]
    out = np.empty((n_bins, n_out), dtype=np.complex128)
    block = max(1, 2**22 // n_fft)
    for start in range(0, n_out, block):
        spectra = np.fft.rfft(frames[start : start + block], axis=-1)
        out[:, start : start + block] = kernel.matrix @ spectra.T
    return out


def chroma_matrix(n_bins: int, bins_per_octave: int, n_chroma: int = 12) -> np.ndarray:
    """Maps CQT bins (lowest bin = pitch class 0) to ``n_chroma`` pitch classes.

    With 3 bins per semitone, bins ``3s - 1, 3s, 3s + 1`` all go to semitone ``s``.
    """
    if bins_per_octave % n_chroma:
        raise ValueError("bins_per_octave must be a multiple of n_chroma")
    per = bins_per_octave // n_chroma
    k = np.arange(n_bins)
    pitch_class = (np.floor((k + per // 2) / per).astype(int)) % n_chroma
    m = np.zeros((n_chroma, n_bins))
    m[pitch_class, k] = 1.0
    return m


def chroma_cqt(
    x: np.ndarray,
    sr: float = 22050,
    hop: int = 512,
    *,
    fmin: float = C1_HZ,
    n_octaves: int = 7,
    bins_per_octave: int = 36,
    tuning: float = 0.0,
    norm: float | None = np.inf,
) -> np.ndarray:
    """Pitch-class energy, shape ``(12, n_frames)``; row 0 is the pitch class of ``fmin`` (C by default).

    Each frame is scaled so its largest value is 1 (``norm=np.inf``); silent
    frames stay zero.
    """
    c = np.abs(cqt(x, sr, hop, fmin=fmin, n_bins=n_octaves * bins_per_octave,
                   bins_per_octave=bins_per_octave, tuning=tuning))
    chroma = chroma_matrix(c.shape[0], bins_per_octave) @ c
    if norm is None:
        return chroma
    scale = np.linalg.norm(chroma, ord=norm, axis=0) if norm != np.inf else chroma.max(axis=0)
    nonzero = scale > np.finfo(np.float64).tiny
    chroma[:, nonzero] /= scale[nonzero]
    return chroma


def chroma_stft(
    mag: np.ndarray, sr: float, n_fft: int, *, fmin: float = 100.0, fmax: float = 5000.0, tuning_hz: float = 440.0
) -> np.ndarray:
    """Cheap chroma from an existing magnitude STFT, shape ``(12, frames)``; row 0 is C.

    Every bin between ``fmin`` and ``fmax`` adds its energy to the nearest
    pitch class. Much coarser than :func:`chroma_cqt` at low frequencies, but
    free when the STFT is already there; good enough to see chord changes.
    Frames are scaled so their largest value is 1.
    """
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    use = np.nonzero((freqs >= fmin) & (freqs <= fmax))[0]
    pitch_class = np.round(69.0 + 12.0 * np.log2(freqs[use] / tuning_hz)).astype(int) % 12
    m = np.zeros((12, mag.shape[0]))
    m[pitch_class, use] = 1.0
    chroma = m @ (mag**2)
    peak = chroma.max(axis=0)
    nonzero = peak > np.finfo(np.float64).tiny
    chroma[:, nonzero] /= peak[nonzero]
    return chroma
