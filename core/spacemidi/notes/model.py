"""Data model of the intermediate note format (``spacemidi-notes``, version 1).

Every pipeline stage reads and writes this model; MIDI and SS13 exporters
consume it. Times are always seconds from the start of the source audio, so
the model stays aligned with the audio. Beats, bars and quantized positions
are derived from ``TempoMap`` at export time, never stored on notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

FORMAT_NAME = "spacemidi-notes"
FORMAT_VERSION = 1

#: Instrument groups produced by stem separation (the spec: groups first, exact GM programs later).
GROUPS = ("vocals", "drums", "bass", "piano", "guitar", "other")

#: Pitch-class names used for keys (sharps only).
PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

MODES = ("major", "minor")


@dataclass
class Note:
    """One note event.

    ``pitch`` is a MIDI note number; on drum tracks it is the General MIDI
    percussion key (36 kick, 38 snare, 42 closed hi-hat, ...).
    ``bend`` is an optional pitch curve: ``(seconds after onset, cents)``
    pairs relative to ``pitch``, sorted by time.
    """

    onset: float
    offset: float
    pitch: int
    velocity: int = 80
    confidence: float | None = None
    bend: list[tuple[float, float]] | None = None

    @property
    def duration(self) -> float:
        return self.offset - self.onset


@dataclass
class Control:
    """A MIDI controller change, e.g. sustain pedal (cc 64)."""

    time: float
    cc: int
    value: int


@dataclass
class Backend:
    """Which implementation produced a track, for comparisons and reruns."""

    stage: str
    name: str
    version: str = ""
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class Track:
    """Notes of one instrument.

    MIDI channels are assigned by the exporter, not stored here.
    ``program`` is a General MIDI program (0-127) or ``None`` when unknown;
    drum tracks have ``is_drum=True`` and no program.
    """

    id: str
    group: str
    name: str = ""
    stem: str | None = None
    program: int | None = None
    is_drum: bool = False
    backend: Backend | None = None
    notes: list[Note] = field(default_factory=list)
    controls: list[Control] = field(default_factory=list)

    def sort(self) -> None:
        """Sort notes by (onset, pitch) and controls by time, in place."""
        self.notes.sort(key=lambda n: (n.onset, n.pitch))
        self.controls.sort(key=lambda c: (c.time, c.cc))


@dataclass
class TempoMap:
    """Beat grid detected from the mix. All values are seconds.

    ``downbeats`` is the subset of ``beats`` that start a bar.
    ``time_signature`` is the dominant one, as (numerator, denominator).
    """

    beats: list[float]
    downbeats: list[float] = field(default_factory=list)
    time_signature: tuple[int, int] = (4, 4)


@dataclass
class Key:
    tonic: str
    mode: str


@dataclass
class Source:
    """The audio the notes were transcribed from."""

    path: str | None = None
    duration: float | None = None
    sample_rate: int | None = None


@dataclass
class NoteDocument:
    tracks: list[Track] = field(default_factory=list)
    source: Source | None = None
    tuning_hz: float = 440.0
    tempo: TempoMap | None = None
    key: Key | None = None

    def track(self, track_id: str) -> Track:
        """Return the track with the given id; raises ``KeyError`` if absent."""
        for t in self.tracks:
            if t.id == track_id:
                return t
        raise KeyError(track_id)
