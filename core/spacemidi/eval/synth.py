"""Render a NoteDocument to audio with simple built-in sounds.

The point is test audio whose correct answer is known exactly: every onset,
pitch and drum hit comes from the document. The sounds are deliberately
simple (additive harmonics, decaying noise), but each group gets its own
timbre and envelope so that onsets, sustains and mixtures behave roughly like
real instruments.
"""

from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from spacemidi.notes import Note, NoteDocument

from .metrics import drum_class


@dataclass(frozen=True)
class Voice:
    """Timbre and envelope of a pitched instrument group."""

    harmonics: int
    rolloff: float  # amplitude of harmonic k is 1 / k**rolloff
    attack: float  # seconds
    decay: float | None  # time constant of the exponential decay while held; None = sustained
    release: float  # seconds after the note ends
    vibrato_cents: float = 0.0


VOICES = {
    "piano": Voice(harmonics=10, rolloff=1.0, attack=0.005, decay=1.0, release=0.08),
    "guitar": Voice(harmonics=12, rolloff=1.2, attack=0.003, decay=0.6, release=0.06),
    "bass": Voice(harmonics=6, rolloff=1.5, attack=0.01, decay=None, release=0.06),
    "vocals": Voice(harmonics=8, rolloff=1.0, attack=0.04, decay=None, release=0.1, vibrato_cents=20.0),
    "other": Voice(harmonics=12, rolloff=1.0, attack=0.03, decay=None, release=0.1),
}


def render(
    doc: NoteDocument,
    sample_rate: int = 44100,
    *,
    noise_snr_db: float | None = None,
    seed: int = 0,
    tail: float = 0.5,
) -> np.ndarray:
    """Mono float32 audio of the whole document, peak-normalized to -1 dBFS.

    ``noise_snr_db`` adds white noise at that signal-to-noise ratio, to test
    robustness. The same document and seed always give the same audio.
    """
    rng = np.random.default_rng(seed)
    end = max((n.offset for t in doc.tracks for n in t.notes), default=0.0) + tail
    if doc.source and doc.source.duration:
        end = max(end, doc.source.duration)
    out = np.zeros(int(np.ceil(end * sample_rate)) + 1, dtype=np.float64)

    for track in doc.tracks:
        drum = track.is_drum or track.group == "drums"
        voice = VOICES.get(track.group, VOICES["other"])
        for note in track.notes:
            if drum:
                sound = _drum(note, sample_rate, rng)
            else:
                sound = _pitched(note, voice, doc.tuning_hz, sample_rate)
            start = int(round(note.onset * sample_rate))
            stop = min(len(out), start + len(sound))
            out[start:stop] += sound[: stop - start]

    signal_power = float(np.mean(out**2))
    if noise_snr_db is not None and signal_power > 0:
        noise_power = signal_power / 10 ** (noise_snr_db / 10)
        out += rng.normal(0.0, np.sqrt(noise_power), len(out))
    peak = float(np.max(np.abs(out))) if len(out) else 0.0
    if peak > 0:
        out *= 10 ** (-1 / 20) / peak
    return out.astype(np.float32)


def _envelope(n: int, held: int, voice: Voice, sr: int) -> np.ndarray:
    t = np.arange(n) / sr
    env = np.minimum(1.0, t / voice.attack) if voice.attack > 0 else np.ones(n)
    if voice.decay is not None:
        env = env * np.exp(-t / voice.decay)
    if n > held:  # linear release from the level reached at the note end
        level = env[held - 1] if held > 0 else 0.0
        env[held:] = level * np.linspace(1.0, 0.0, n - held)
    return env


def _pitched(note: Note, voice: Voice, tuning_hz: float, sr: int) -> np.ndarray:
    held = max(1, int(round(note.duration * sr)))
    n = held + int(round(voice.release * sr))
    t = np.arange(n) / sr
    cents = np.zeros(n)
    if note.bend:
        times = [p[0] for p in note.bend]
        values = [p[1] for p in note.bend]
        cents += np.interp(t, times, values)  # held at the end values outside the curve
    if voice.vibrato_cents:
        depth = voice.vibrato_cents * np.clip((t - 0.15) / 0.2, 0.0, 1.0)  # vibrato fades in
        cents += depth * np.sin(2 * np.pi * 5.5 * t)
    f0 = tuning_hz * 2 ** ((note.pitch - 69) / 12) * 2 ** (cents / 1200)
    phase = 2 * np.pi * np.cumsum(f0) / sr
    wave_ = np.zeros(n)
    nyquist = sr / 2
    for k in range(1, voice.harmonics + 1):
        if k * f0.max() >= nyquist * 0.95:
            break
        wave_ += np.sin(k * phase) / k**voice.rolloff
    amplitude = 0.3 * (note.velocity / 127) ** 1.5
    return amplitude * wave_ * _envelope(n, held, voice, sr)


def _drum(note: Note, sr: int, rng: np.random.Generator) -> np.ndarray:
    cls = drum_class(note.pitch)
    amplitude = 0.5 * (note.velocity / 127) ** 1.5
    if cls == "kick":
        n = int(0.4 * sr)
        t = np.arange(n) / sr
        freq = 45 + 105 * np.exp(-t / 0.03)  # 150 Hz sweeping down to 45 Hz
        body = np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t / 0.15)
        click = rng.normal(0, 1, n) * np.exp(-t / 0.003) * 0.3
        return amplitude * (body + click)
    if cls == "snare":
        n = int(0.3 * sr)
        t = np.arange(n) / sr
        noise = np.diff(rng.normal(0, 1, n + 1)) * np.exp(-t / 0.12) * 0.5
        tone = np.sin(2 * np.pi * 185 * t) * np.exp(-t / 0.08)
        return amplitude * (noise + tone)
    if cls == "hihat":
        length = 0.25 if note.pitch == 46 else 0.06  # open vs closed
        n = int(4 * length * sr)
        t = np.arange(n) / sr
        noise = np.diff(rng.normal(0, 1, n + 2), 2)  # second difference: mostly high frequencies
        return amplitude * 0.3 * noise * np.exp(-t / length)
    if cls == "tom":
        n = int(0.5 * sr)
        t = np.arange(n) / sr
        base = 80 + 12 * (note.pitch - 41)
        freq = base * (1 + 0.5 * np.exp(-t / 0.05))
        return amplitude * np.sin(2 * np.pi * np.cumsum(freq) / sr) * np.exp(-t / 0.2)
    # cymbals and anything else: long bright noise
    n = int(1.5 * sr)
    t = np.arange(n) / sr
    noise = np.diff(rng.normal(0, 1, n + 1))
    return amplitude * 0.25 * noise * np.exp(-t / 0.5)


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> None:
    """16-bit PCM mono WAV."""
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(sample_rate)
        f.writeframes(pcm.tobytes())


def read_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a 16-bit PCM WAV (mono or the mean of its channels) as float32."""
    with wave.open(str(path), "rb") as f:
        if f.getsampwidth() != 2:
            raise ValueError("only 16-bit PCM WAV is supported here")
        channels, sr = f.getnchannels(), f.getframerate()
        data = np.frombuffer(f.readframes(f.getnframes()), dtype="<i2").astype(np.float32) / 32768
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data, sr
