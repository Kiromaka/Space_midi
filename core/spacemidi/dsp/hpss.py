"""Harmonic-percussive source separation by median filtering.

Fitzgerald, "Harmonic/percussive separation using median filtering" (DAFx
2010): in a spectrogram, sustained (harmonic) sounds are horizontal lines and
short (percussive) sounds are vertical lines. A median filter along time
keeps the horizontal lines and removes the vertical ones; a median filter
along frequency does the opposite. Comparing the two gives, for every
time-frequency bin, how harmonic or how percussive it is.

The two filtered spectrograms become soft masks (Wiener-like):
``M_h = H^p / (H^p + P^p)``. With ``margin`` > 1 (Driedger et al., ISMIR
2014) a bin only counts as harmonic when it is ``margin`` times more
harmonic than percussive (and vice versa); what is neither goes to a third,
residual part. With ``margin = 1`` the masks sum to 1 and harmonic +
percussive reconstructs the input exactly.

Conventions follow ``librosa.decompose.hpss`` (kernel sizes in frames and
bins, reflected edges, zero bins split evenly), which the tests check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import median_filter

from .spectral import istft, stft


def softmask(x: np.ndarray, x_ref: np.ndarray, power: float = 2.0, split_zeros: bool = False) -> np.ndarray:
    """``x^p / (x^p + x_ref^p)``, computed stably; bins where both are zero get 0.5 (``split_zeros``) or 0."""
    x = np.asarray(x, dtype=np.float64)
    x_ref = np.asarray(x_ref, dtype=np.float64)
    if np.any(x < 0) or np.any(x_ref < 0):
        raise ValueError("softmask inputs must be non-negative")
    z = np.maximum(x, x_ref)
    bad = z < np.finfo(np.float64).tiny
    z = np.where(bad, 1.0, z)
    if np.isinf(power):
        mask = (x > x_ref).astype(np.float64)
    else:
        a = (x / z) ** power
        b = (x_ref / z) ** power
        mask = np.where(bad, 0.0, a / np.where(bad, 1.0, a + b))
    mask[bad] = 0.5 if split_zeros else 0.0
    return mask


def median_parts(mag: np.ndarray, harm_frames: int = 31, perc_bins: int = 31) -> tuple[np.ndarray, np.ndarray]:
    """The two median-filtered spectrograms: (along time = harmonic, along frequency = percussive)."""
    mag = np.abs(np.asarray(mag, dtype=np.float64))
    if harm_frames < 1 or perc_bins < 1:
        raise ValueError("kernel sizes must be at least 1")
    harm = median_filter(mag, size=(1, harm_frames), mode="reflect")
    perc = median_filter(mag, size=(perc_bins, 1), mode="reflect")
    return harm, perc


def masks_from_parts(
    harm: np.ndarray, perc: np.ndarray, *, power: float = 2.0, margin: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Soft masks (harmonic, percussive) from :func:`median_parts`."""
    if margin < 1:
        raise ValueError("margin must be at least 1")
    split = margin == 1.0
    return (
        softmask(harm, perc * margin, power, split_zeros=split),
        softmask(perc, harm * margin, power, split_zeros=split),
    )


def hpss_masks(
    mag: np.ndarray,
    harm_frames: int = 31,
    perc_bins: int = 31,
    *,
    power: float = 2.0,
    margin: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Soft masks (harmonic, percussive) for a magnitude spectrogram ``(bins, frames)``.

    ``harm_frames``: median filter length along time (longer = only longer
    sounds count as harmonic). ``perc_bins``: length along frequency.
    """
    if margin < 1:
        raise ValueError("margin must be at least 1")
    harm, perc = median_parts(mag, harm_frames, perc_bins)
    return masks_from_parts(harm, perc, power=power, margin=margin)


@dataclass
class HPSS:
    """Separated signals (same length as the input)."""

    harmonic: np.ndarray
    percussive: np.ndarray
    residual: np.ndarray


def hpss(
    x: np.ndarray,
    n_fft: int = 2048,
    hop: int = 512,
    *,
    harm_frames: int = 31,
    perc_bins: int = 31,
    power: float = 2.0,
    margin: float = 1.0,
) -> HPSS:
    """Split a mono signal into harmonic, percussive and (with ``margin`` > 1) residual parts."""
    x = np.asarray(x, dtype=np.float64)
    spec = stft(x, n_fft, hop)
    mask_h, mask_p = hpss_masks(np.abs(spec), harm_frames, perc_bins, power=power, margin=margin)
    harmonic = istft(spec * mask_h, hop, length=len(x))
    percussive = istft(spec * mask_p, hop, length=len(x))
    return HPSS(harmonic, percussive, x - harmonic - percussive)
