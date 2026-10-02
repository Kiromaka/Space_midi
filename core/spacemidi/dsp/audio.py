"""Reading audio files and changing the sample rate.

Everything downstream works on mono float64 arrays in [-1, 1] at one sample
rate (22050 Hz unless a caller asks otherwise), so loading also mixes down
and resamples.
"""

from __future__ import annotations

from math import gcd
from pathlib import Path

import numpy as np
from scipy.signal import resample_poly

#: Default analysis rate: enough for onsets, beats and pitch up to ~11 kHz,
#: and the rate most published MIR results use.
DEFAULT_SR = 22050


def load_audio(path: str | Path, sr: int | None = DEFAULT_SR, *, mono: bool = True) -> tuple[np.ndarray, int]:
    """Read an audio file. Returns ``(samples, sample_rate)``.

    Mono output has shape ``(n,)``; with ``mono=False`` the shape is
    ``(channels, n)``. ``sr=None`` keeps the file's own rate. WAV, FLAC and
    OGG are read with soundfile; without soundfile only WAV works.
    """
    path = Path(path)
    try:
        import soundfile
    except ImportError:  # pragma: no cover - soundfile is a core dependency
        soundfile = None
    if soundfile is not None:
        data, rate = soundfile.read(str(path), dtype="float64", always_2d=True)
        data = data.T
    else:
        data, rate = _read_wav_stdlib(path)
    if mono:
        data = data.mean(axis=0)
    if sr is not None and sr != rate:
        data = resample(data, rate, sr)
        rate = sr
    return np.ascontiguousarray(data), int(rate)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    """Polyphase resampling along the last axis (anti-aliasing filter included)."""
    if sr_from == sr_to:
        return x
    g = gcd(int(sr_from), int(sr_to))
    return resample_poly(x, int(sr_to) // g, int(sr_from) // g, axis=-1)


def to_mono(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x if x.ndim == 1 else x.mean(axis=0)


def _read_wav_stdlib(path: Path) -> tuple[np.ndarray, int]:
    import wave

    with wave.open(str(path), "rb") as f:
        channels, width, rate, n = f.getnchannels(), f.getsampwidth(), f.getframerate(), f.getnframes()
        raw = f.readframes(n)
    if width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    elif width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    else:
        raise ValueError(f"{path}: {8 * width}-bit WAV needs soundfile")
    return data.reshape(-1, channels).T, rate
