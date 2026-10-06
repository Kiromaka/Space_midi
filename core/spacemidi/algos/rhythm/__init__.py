"""Tempo, beats, downbeats: autocorrelation + dynamic programming (L1).

Pipeline (:func:`estimate_rhythm`):

1. onset envelope from the ``flux`` onset preset (spacemidi.dsp.onset);
2. global tempo from its autocorrelation with a log-normal tempo prior (:mod:`.tempo`);
3. beats by Ellis's dynamic programming at that tempo (:mod:`.beats`);
4. downbeats and 3/4 vs 4/4 from accent, low-end onsets and chord changes (:mod:`.downbeats`).

Notes: docs/research/stage1-beats.md.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields, replace

import numpy as np

from spacemidi.notes import TempoMap

from .beats import local_score, track_beats
from .downbeats import DownbeatChoice, beat_features, choose_downbeats
from .tempo import TempoEstimate, autocorrelation, estimate_tempo, tempo_prior


@dataclass(frozen=True)
class RhythmConfig:
    """Every setting of the rhythm estimator."""

    onset_preset: str = "flux"  # envelope source (its threshold settings are not used)
    # tempo
    start_bpm: float = 120.0
    std_octaves: float = 1.0
    min_bpm: float = 40.0
    max_bpm: float = 240.0
    harmonics: int = 1
    tempo_smooth: float = 1.0  # Gaussian smoothing of the envelope before autocorrelation, in frames
    # beats
    tightness: float = 400.0  # tuned on Slakh (100 is Ellis's and librosa's default)
    trim: bool = True
    offset: float = 0.0  # seconds added to every beat
    # downbeats
    triple: bool = True  # also consider 3 beats per bar
    meter_bias: float = 0.1  # how much better 3/4 must score than 4/4
    w_accent: float = 0.0  # tuned on Slakh: the loudest beat is usually the backbeat, not "one"
    w_low: float = 1.0
    w_harmony: float = 2.0
    low_hz: float = 150.0  # "low end" = energy below this

    def with_(self, **changes) -> RhythmConfig:
        from spacemidi.dsp.onset import _convert

        types = {f.name: f.type for f in fields(self)}
        clean = {}
        for key, value in changes.items():
            if key not in types:
                raise KeyError(f"unknown rhythm setting '{key}'")
            clean[key] = _convert(value, getattr(self, key), types[key])
        return replace(self, **clean)

    def to_dict(self) -> dict:
        return {f.name: getattr(self, f.name) for f in fields(self)}


@dataclass
class Rhythm:
    """Estimated tempo, beats and downbeats; times in seconds."""

    bpm: float
    beats: list[float]
    downbeats: list[float]
    time_signature: tuple[int, int] = (4, 4)
    tempo_candidates: list[tuple[float, float]] = field(default_factory=list)
    downbeat_scores: dict[str, float] = field(default_factory=dict)

    def tempo_map(self) -> TempoMap:
        return TempoMap(beats=list(self.beats), downbeats=list(self.downbeats), time_signature=self.time_signature)


@dataclass
class RhythmFeatures:
    """Everything computed from the audio once; the settings that follow only read it."""

    fps: float
    env: np.ndarray
    low_env: np.ndarray
    chroma: np.ndarray


def rhythm_features(x: np.ndarray, sr: float, onset_preset: str = "flux", low_hz: float = 150.0) -> RhythmFeatures:
    from spacemidi.dsp import band_spectrogram, chroma_stft, envelope_from_bands, onset_config, power_to_db, stft

    ocfg = onset_config(onset_preset)
    mag = np.abs(stft(np.asarray(x, dtype=np.float64), ocfg.n_fft, ocfg.hop))
    env = envelope_from_bands(band_spectrogram(x, sr, ocfg, mag=mag), ocfg.lag, ocfg.max_size)
    freqs = np.fft.rfftfreq(ocfg.n_fft, 1.0 / sr)
    low = power_to_db((mag[(freqs > 20.0) & (freqs < low_hz)] ** 2).sum(axis=0, keepdims=True))
    low_env = envelope_from_bands(low, 1, 1)
    return RhythmFeatures(sr / ocfg.hop, env, low_env, chroma_stft(mag, sr, ocfg.n_fft))


def rhythm_from_features(feat: RhythmFeatures, cfg: RhythmConfig) -> Rhythm:
    tempo = estimate_tempo(
        feat.env, feat.fps, start_bpm=cfg.start_bpm, std_octaves=cfg.std_octaves,
        min_bpm=cfg.min_bpm, max_bpm=cfg.max_bpm, harmonics=cfg.harmonics, smooth=cfg.tempo_smooth,
    )
    frames = track_beats(feat.env, feat.fps, tempo.bpm, tightness=cfg.tightness, trim=cfg.trim)
    beats = (frames / feat.fps + cfg.offset).tolist()
    if len(frames) < 2:
        return Rhythm(tempo.bpm, beats, beats[:1], (4, 4), tempo.candidates)
    choice = choose_downbeats(
        beat_features(frames, feat.env, feat.low_env, feat.chroma),
        {"accent": cfg.w_accent, "low": cfg.w_low, "harmony": cfg.w_harmony},
        meters=(4, 3) if cfg.triple else (4,),
        meter_bias=cfg.meter_bias,
    )
    return Rhythm(
        bpm=tempo.bpm,
        beats=beats,
        downbeats=beats[choice.phase :: choice.meter],
        time_signature=(choice.meter, 4),
        tempo_candidates=tempo.candidates,
        downbeat_scores=choice.scores,
    )


def estimate_rhythm(x: np.ndarray, sr: float, cfg: RhythmConfig | None = None, **changes) -> Rhythm:
    """Tempo, beats and downbeats of a mono signal."""
    cfg = (cfg or RhythmConfig()).with_(**changes) if changes else (cfg or RhythmConfig())
    return rhythm_from_features(rhythm_features(x, sr, cfg.onset_preset, cfg.low_hz), cfg)


__all__ = [
    "DownbeatChoice",
    "Rhythm",
    "RhythmConfig",
    "RhythmFeatures",
    "TempoEstimate",
    "autocorrelation",
    "beat_features",
    "choose_downbeats",
    "estimate_rhythm",
    "estimate_tempo",
    "local_score",
    "rhythm_features",
    "rhythm_from_features",
    "tempo_prior",
    "track_beats",
]
