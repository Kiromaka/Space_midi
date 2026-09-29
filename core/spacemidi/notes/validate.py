"""Semantic validation of a ``NoteDocument``.

Structural problems (missing keys, wrong JSON types) are caught while parsing
in ``io.from_dict``. This module checks values and cross-field rules and
reports every problem it finds, each prefixed with a path such as
``tracks[0].notes[3]``.
"""

from __future__ import annotations

import bisect
import math

from .model import GROUPS, MODES, PITCH_CLASSES, NoteDocument

#: Allowed time-signature denominators.
_DENOMINATORS = (1, 2, 4, 8, 16, 32)
#: Largest pitch-bend deviation we accept, in cents (two octaves).
MAX_BEND_CENTS = 2400.0
#: Tolerance when matching downbeats to beats, in seconds.
_BEAT_TOL = 1e-6


class NotesFormatError(ValueError):
    """Raised when a document is malformed or fails validation."""

    def __init__(self, problems: list[str]):
        self.problems = list(problems)
        lines = "\n".join(f"  {p}" for p in self.problems)
        super().__init__(f"invalid spacemidi-notes document:\n{lines}")


def _finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(x)


def validate(doc: NoteDocument) -> list[str]:
    """Return a list of problems; an empty list means the document is valid."""
    problems: list[str] = []

    def err(path: str, msg: str) -> None:
        problems.append(f"{path}: {msg}")

    if not (_finite(doc.tuning_hz) and 400.0 <= doc.tuning_hz <= 480.0):
        err("tuning_hz", "must be between 400 and 480")

    if doc.source is not None:
        s = doc.source
        if s.duration is not None and not (_finite(s.duration) and s.duration > 0):
            err("source.duration", "must be a positive number of seconds")
        if s.sample_rate is not None and s.sample_rate <= 0:
            err("source.sample_rate", "must be positive")

    if doc.tempo is not None:
        _validate_tempo(doc, err)

    if doc.key is not None:
        if doc.key.tonic not in PITCH_CLASSES:
            err("key.tonic", f"must be one of {', '.join(PITCH_CLASSES)}")
        if doc.key.mode not in MODES:
            err("key.mode", f"must be one of {', '.join(MODES)}")

    seen: set[str] = set()
    for ti, track in enumerate(doc.tracks):
        tp = f"tracks[{ti}]"
        if not track.id:
            err(f"{tp}.id", "must not be empty")
        elif track.id in seen:
            err(f"{tp}.id", f"duplicate id '{track.id}'")
        seen.add(track.id)
        if track.group not in GROUPS:
            err(f"{tp}.group", f"must be one of {', '.join(GROUPS)}")
        if track.program is not None and not 0 <= track.program <= 127:
            err(f"{tp}.program", "must be 0-127")
        if track.is_drum and track.program is not None:
            err(f"{tp}.program", "drum tracks have no program")
        _validate_notes(track, tp, doc, err)
        for ci, c in enumerate(track.controls):
            cp = f"{tp}.controls[{ci}]"
            if not (_finite(c.time) and c.time >= 0):
                err(f"{cp}.time", "must be >= 0")
            if not 0 <= c.cc <= 127:
                err(f"{cp}.cc", "must be 0-127")
            if not 0 <= c.value <= 127:
                err(f"{cp}.value", "must be 0-127")

    return problems


def _validate_tempo(doc: NoteDocument, err) -> None:
    t = doc.tempo
    assert t is not None
    for i, b in enumerate(t.beats):
        if not (_finite(b) and b >= 0):
            err(f"tempo.beats[{i}]", "must be >= 0")
    if any(b2 <= b1 for b1, b2 in zip(t.beats, t.beats[1:])):
        err("tempo.beats", "must be strictly increasing")
    if any(d2 <= d1 for d1, d2 in zip(t.downbeats, t.downbeats[1:])):
        err("tempo.downbeats", "must be strictly increasing")
    beats = sorted(t.beats)
    for i, d in enumerate(t.downbeats):
        if not any(abs(d - b) <= _BEAT_TOL for b in _near(beats, d)):
            err(f"tempo.downbeats[{i}]", "must also be listed in tempo.beats")
    num, den = t.time_signature
    if not 1 <= num <= 32:
        err("tempo.time_signature", "numerator must be 1-32")
    if den not in _DENOMINATORS:
        err("tempo.time_signature", f"denominator must be one of {_DENOMINATORS}")


def _near(sorted_values: list[float], x: float) -> list[float]:
    """The one or two values of a sorted list closest to ``x``."""
    i = bisect.bisect_left(sorted_values, x)
    return sorted_values[max(0, i - 1) : i + 1]


def _validate_notes(track, tp: str, doc: NoteDocument, err) -> None:
    duration = doc.source.duration if doc.source else None
    for ni, n in enumerate(track.notes):
        np_ = f"{tp}.notes[{ni}]"
        if not (_finite(n.onset) and n.onset >= 0):
            err(f"{np_}.onset", "must be >= 0")
        if not (_finite(n.offset) and _finite(n.onset) and n.offset > n.onset):
            err(f"{np_}.offset", "must be greater than onset")
        if duration is not None and _finite(n.onset) and n.onset >= duration:
            err(f"{np_}.onset", "is past the end of the source audio")
        if not 0 <= n.pitch <= 127:
            err(f"{np_}.pitch", "must be 0-127")
        if not 1 <= n.velocity <= 127:
            err(f"{np_}.velocity", "must be 1-127")
        if n.confidence is not None and not (_finite(n.confidence) and 0 <= n.confidence <= 1):
            err(f"{np_}.confidence", "must be between 0 and 1")
        if n.bend is not None:
            _validate_bend(n, np_, err)


def _validate_bend(n, np_: str, err) -> None:
    assert n.bend is not None
    prev = -math.inf
    for bi, (t, cents) in enumerate(n.bend):
        bp = f"{np_}.bend[{bi}]"
        if not _finite(t) or t < 0 or t > n.duration + _BEAT_TOL:
            err(bp, "time must be between 0 and the note duration")
        elif t <= prev:
            err(bp, "times must be strictly increasing")
        else:
            prev = t
        if not (_finite(cents) and abs(cents) <= MAX_BEND_CENTS):
            err(bp, f"cents must be within +/-{MAX_BEND_CENTS:g}")
