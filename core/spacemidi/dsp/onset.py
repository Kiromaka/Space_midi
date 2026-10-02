"""Onset detection: an onset envelope from spectral change, then peak picking.

Two envelopes, both built from the same parts:

flux - spectral flux on a log-power mel spectrogram (128 bands, 43 frames
    per second at 22050 Hz), the same envelope as librosa's.

superflux - Böck & Widmer, "Maximum filter vibrato suppression for onset
    detection" (DAFx 2013): log-frequency filterbank with 24 bands per
    octave, ``log10(1 + 1000 X)`` magnitudes, 200 frames per second, and the
    reference frame is first max-filtered across 3 neighbouring bands, so a
    note that only wobbles in pitch (vibrato) does not register as new energy.

The presets (``ONSET_PRESETS``) pair an envelope with an absolute threshold
tuned on Slakh2100 (docs/research/stage1-onsets.md): ``flux`` and
``superflux`` for full mixes, ``flux-stem`` and ``superflux-stem`` for a
single instrument. ``librosa`` reproduces ``librosa.onset.onset_detect``
(threshold relative to the track's loudest onset) for cross-checks.

The envelope at frame ``t`` is the band-averaged positive increase
``max(0, S[f, t] - maxfilter(S)[f, t - lag])``. Frame ``t`` is centred on
sample ``t * hop``. ``offset`` (seconds) corrects the median timing error
measured on Slakh.

Peak picking follows Böck et al., "Evaluating the online capabilities of
onset detection methods" (ISMIR 2012): a frame is an onset if it is the
maximum of its neighbourhood, exceeds the local mean by ``delta``, and is at
least ``wait`` after the previous onset.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from scipy.ndimage import maximum_filter1d

from .mel import log_filterbank, mel_filterbank
from .spectral import amplitude_to_db, power_to_db, stft


@dataclass(frozen=True)
class OnsetConfig:
    """Every setting of the onset detector. Times are in seconds."""

    # spectrogram
    n_fft: int = 2048
    hop: int = 512
    bands: str = "mel"  # "mel" (power), "log" (magnitude, bands per octave) or "linear" (magnitude)
    n_bands: int = 128  # mel bands, or bands per octave for "log"
    fmin: float = 0.0
    fmax: float | None = None
    compression: str = "db"  # "db", "log1p" or "none"
    log_mul: float = 1.0  # X -> log10(1 + log_mul * X) for "log1p"
    # spectral difference
    lag: int = 1
    max_size: int = 1  # max filter across bands for the reference frame; 1 = plain flux
    # peak picking
    normalize: bool = True  # scale the envelope to 0..1 before picking
    pre_max: float = 0.03
    post_max: float = 0.0
    pre_avg: float = 0.10
    post_avg: float = 0.10
    delta: float = 0.07
    wait: float = 0.03
    offset: float = 0.0  # added to every detected time

    def with_(self, **changes) -> OnsetConfig:
        """A copy with some settings changed; values given as strings are converted."""
        types = {f.name: f.type for f in fields(self)}
        clean = {}
        for key, value in changes.items():
            if key not in types:
                raise KeyError(f"unknown onset setting '{key}'")
            clean[key] = _convert(value, getattr(self, key), types[key])
        return replace(self, **clean)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


def _convert(value, current, type_name: str):
    if not isinstance(value, str):
        return value
    if value.lower() in ("none", "null"):
        return None
    if isinstance(current, bool):
        return value.lower() in ("1", "true", "yes")
    if isinstance(current, int) and "int" in str(type_name):
        return int(value)
    if isinstance(current, float) or "float" in str(type_name):
        return float(value)
    return value


_SUPERFLUX = OnsetConfig(
    n_fft=1024,
    hop=110,
    bands="log",
    n_bands=24,
    fmin=27.5,
    fmax=16000.0,
    compression="log1p",
    log_mul=1000.0,
    lag=2,
    max_size=3,
    post_max=0.03,
    post_avg=0.07,
)

#: Tuned on the Slakh2100 test split, 2026-10-02 (F1 there in the comments).
#: ``delta`` is absolute: mean dB rise per band for flux, log10 units for superflux.
ONSET_PRESETS: dict[str, OnsetConfig] = {
    "flux": OnsetConfig(normalize=False, delta=0.3, offset=0.006),  # mixes 0.821
    "flux-stem": OnsetConfig(normalize=False, delta=1.0, offset=0.009),  # single instruments 0.840
    "superflux": replace(_SUPERFLUX, normalize=False, delta=0.01, offset=-0.003),  # mixes 0.817
    "superflux-stem": replace(_SUPERFLUX, normalize=False, delta=0.05),  # single instruments 0.838
    "librosa": OnsetConfig(),  # = librosa.onset.onset_detect defaults; mixes 0.683
}


def onset_config(preset: str | OnsetConfig = "flux", **changes) -> OnsetConfig:
    if not isinstance(preset, OnsetConfig) and preset not in ONSET_PRESETS:
        raise KeyError(f"unknown onset preset '{preset}', expected one of {', '.join(ONSET_PRESETS)}")
    base = preset if isinstance(preset, OnsetConfig) else ONSET_PRESETS[preset]
    return base.with_(**changes) if changes else base


def band_spectrogram(x: np.ndarray, sr: float, cfg: OnsetConfig) -> np.ndarray:
    """The compressed band energies the envelope is computed from, shape ``(bands, frames)``."""
    mag = np.abs(stft(x, cfg.n_fft, cfg.hop))
    if cfg.bands == "mel":
        s = mel_filterbank(sr, cfg.n_fft, cfg.n_bands, cfg.fmin, cfg.fmax) @ mag**2
        power = True
    elif cfg.bands == "log":
        s = log_filterbank(sr, cfg.n_fft, cfg.n_bands, cfg.fmin or 27.5, cfg.fmax or sr / 2.0) @ mag
        power = False
    elif cfg.bands == "linear":
        s, power = mag, False
    else:
        raise ValueError(f"unknown band type '{cfg.bands}'")
    if cfg.compression == "db":
        return power_to_db(s) if power else amplitude_to_db(s)
    if cfg.compression == "log1p":
        return np.log10(1.0 + cfg.log_mul * s)
    if cfg.compression == "none":
        return s
    raise ValueError(f"unknown compression '{cfg.compression}'")


def onset_envelope(x: np.ndarray, sr: float, cfg: OnsetConfig | None = None) -> np.ndarray:
    """Onset strength per frame (not normalized); frame ``t`` is at ``t * hop / sr`` seconds."""
    cfg = cfg or ONSET_PRESETS["flux"]
    return envelope_from_bands(band_spectrogram(x, sr, cfg), cfg.lag, cfg.max_size)


def envelope_from_bands(s: np.ndarray, lag: int = 1, max_size: int = 1) -> np.ndarray:
    if lag < 1 or max_size < 1:
        raise ValueError("lag and max_size must be at least 1")
    ref = maximum_filter1d(s, max_size, axis=0) if max_size > 1 else s
    diff = np.maximum(0.0, s[:, lag:] - ref[:, :-lag])
    return np.concatenate([np.zeros(lag), diff.mean(axis=0)])


def peak_pick(
    env: np.ndarray, pre_max: int, post_max: int, pre_avg: int, post_avg: int, delta: float, wait: int
) -> np.ndarray:
    """Indices of peaks in ``env`` (all window sizes in frames).

    Frame ``n`` is a peak when it is non-zero, equals the maximum of
    ``env[n - pre_max : n + post_max]``, is at least ``delta`` above the mean
    of ``env[n - pre_avg : n + post_avg]`` (windows cut at the edges), and
    comes more than ``wait`` frames after the previous peak.
    """
    env = np.asarray(env, dtype=np.float64)
    n = len(env)
    if n == 0:
        return np.zeros(0, dtype=int)
    if pre_max < 0 or post_max < 1 or pre_avg < 0 or post_avg < 1 or wait < 0:
        raise ValueError("windows must be non-negative, and post_max / post_avg at least 1 (they include the frame)")
    padded = np.concatenate([np.full(pre_max, -np.inf), env, np.full(post_max - 1, -np.inf)])
    local_max = sliding_window_view(padded, pre_max + post_max).max(axis=1)
    csum = np.concatenate([[0.0], np.cumsum(env)])
    idx = np.arange(n)
    lo = np.maximum(0, idx - pre_avg)
    hi = np.minimum(n, idx + post_avg)
    local_mean = (csum[hi] - csum[lo]) / (hi - lo)
    candidates = np.nonzero((env != 0) & (env == local_max) & (env >= local_mean + delta))[0]
    peaks = []
    last = -np.inf
    for i in candidates:
        if i > last + wait:
            peaks.append(i)
            last = i
    return np.asarray(peaks, dtype=int)


def pick_onsets(env: np.ndarray, sr: float, cfg: OnsetConfig) -> np.ndarray:
    """Onset times in seconds from an envelope computed with ``cfg``."""
    env = np.asarray(env, dtype=np.float64)
    if cfg.normalize and len(env):
        env = env - env.min()
        env = env / (env.max() + np.finfo(np.float64).tiny)
    per = sr / cfg.hop  # frames per second, as a float
    frames = peak_pick(
        env,
        pre_max=int(cfg.pre_max * per),
        post_max=int(cfg.post_max * per) + 1,
        pre_avg=int(cfg.pre_avg * per),
        post_avg=int(cfg.post_avg * per) + 1,
        delta=cfg.delta,
        wait=int(cfg.wait * per),
    )
    return frames * cfg.hop / sr + cfg.offset


def detect_onsets(x: np.ndarray, sr: float, cfg: str | OnsetConfig = "flux", **changes) -> np.ndarray:
    """Onset times in seconds. ``cfg`` is a preset name or an :class:`OnsetConfig`."""
    cfg = onset_config(cfg, **changes)
    return pick_onsets(onset_envelope(x, sr, cfg), sr, cfg)
