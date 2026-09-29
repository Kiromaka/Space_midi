"""NoteDocument -> Standard MIDI File (type 1).

Layout of the file:

* track 0 (conductor): tempo map from the detected beats, time and key
  signature, and a ``spacemidi`` text event with document metadata;
* one track per instrument, each on its own channel, drums on channel 10.

Each instrument track starts with its name, a ``spacemidi`` text event
(id, group, stem, backend), a Program Change and, if any note bends, an RPN
that sets the pitch-bend range. The reader uses the text events to restore
the document exactly; other programs ignore them.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from spacemidi import __version__
from spacemidi.notes import NoteDocument, NotesFormatError, Track, validate

from . import smf
from .tempo import BeatGrid, TickClock, clock_from_beats, pickup_signature, ticks_per_beat

DRUM_CHANNEL = 9
TEXT_PREFIX = "spacemidi "

#: General MIDI program (0-based) used when a track has none.
DEFAULT_PROGRAMS = {
    "vocals": 53,  # Voice Oohs
    "bass": 33,  # Electric Bass (finger)
    "piano": 0,  # Acoustic Grand Piano
    "guitar": 25,  # Acoustic Guitar (steel)
    "other": 48,  # String Ensemble 1
}

_MAJOR_KEY_SHARPS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}
_PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")

# Order of events that share a tick: set-up, note-offs, controllers,
# pitch bend, then note-ons (so a note starts with its bend already applied).
_SETUP, _NOTE_OFF, _CONTROL, _BEND, _NOTE_ON = range(5)


class MidiExportError(ValueError):
    """The document cannot be written as a MIDI file."""


@dataclass
class MidiOptions:
    ppq: int = 480
    grid: int | None = None  # grid subdivisions per beat, e.g. 4 = 16th notes in 4/4
    strength: float = 1.0  # 0 = as played, 1 = fully on the grid


def write_midi(doc: NoteDocument, path: str | Path, options: MidiOptions | None = None) -> None:
    Path(path).write_bytes(encode_midi(doc, options))


def encode_midi(doc: NoteDocument, options: MidiOptions | None = None) -> bytes:
    opts = options or MidiOptions()
    problems = validate(doc)
    if problems:
        raise NotesFormatError(problems)
    if not 0.0 <= opts.strength <= 1.0:
        raise MidiExportError("strength must be between 0 and 1")

    grid = _beat_grid(doc, opts.ppq)
    channels = assign_channels(doc.tracks)
    tracks = [_conductor_track(doc, grid)]
    for track in doc.tracks:
        tracks.append(_instrument_track(track, channels[track.id], grid, opts))
    return smf.encode_file(tracks, opts.ppq)


def assign_channels(tracks: list[Track]) -> dict[str, int]:
    """Drums share channel 10; every other track gets its own channel."""
    free = [c for c in range(16) if c != DRUM_CHANNEL]
    melodic = [t for t in tracks if not t.is_drum]
    if len(melodic) > len(free):
        raise MidiExportError(
            f"{len(melodic)} melodic tracks, but MIDI has only {len(free)} non-drum channels; "
            "merge some tracks first"
        )
    channels = {t.id: DRUM_CHANNEL for t in tracks if t.is_drum}
    channels.update({t.id: c for t, c in zip(melodic, free)})
    return channels


def _beat_grid(doc: NoteDocument, ppq: int) -> BeatGrid:
    if doc.tempo is None:
        tpb = ticks_per_beat(ppq, 4)
        return BeatGrid(TickClock(ppq), tpb, 0, 4 * tpb)  # 120 BPM, 4/4, bars from tick 0
    return clock_from_beats(doc.tempo.beats, doc.tempo.downbeats, doc.tempo.time_signature, ppq)


def _meta_text(payload: dict) -> bytes:
    return smf.text_event(smf.META_TEXT, TEXT_PREFIX + json.dumps(payload, separators=(",", ":")))


def _conductor_track(doc: NoteDocument, grid: BeatGrid) -> list[tuple[int, bytes]]:
    info = {"writer": f"spacemidi {__version__}", "tuning_hz": doc.tuning_hz, "lead_in": grid.lead_in}
    if doc.source is not None:
        info["source"] = {k: v for k, v in vars(doc.source).items() if v is not None}
    events = [
        (0, smf.text_event(smf.META_TRACK_NAME, "Space MIDI")),
        (0, _meta_text(info)),
    ]
    num, den = doc.tempo.time_signature if doc.tempo else (4, 4)
    if grid.pickup:  # a shorter first bar so that the first downbeat starts bar 2
        p_num, p_den = pickup_signature(grid.pickup, grid.ticks_per_beat, den)
        events.append((0, _time_signature(p_num, p_den)))
        events.append((grid.pickup, _time_signature(num, den)))
    else:
        events.append((0, _time_signature(num, den)))
    if doc.key is not None:
        tonic = _PITCH_CLASSES.index(doc.key.tonic)
        minor = doc.key.mode == "minor"
        sharps = _MAJOR_KEY_SHARPS[(tonic + 3) % 12 if minor else tonic]
        events.append((0, smf.meta(smf.META_KEY_SIGNATURE, bytes([sharps & 0xFF, int(minor)]))))
    events += [(tick, smf.tempo_event(us)) for tick, us in grid.clock.changes]
    events.sort(key=lambda e: e[0])
    # End the conductor track at the last detected beat, so the whole beat grid survives.
    events.append((max(grid.last_beat, events[-1][0]), smf.meta(smf.META_END_OF_TRACK, b"")))
    return events


def _time_signature(num: int, den: int) -> bytes:
    return smf.meta(smf.META_TIME_SIGNATURE, bytes([num, int(math.log2(den)), 24, 8]))


def _instrument_track(track: Track, ch: int, grid: BeatGrid, opts: MidiOptions) -> list[tuple[int, bytes]]:
    events: list[tuple[int, int, bytes]] = []  # (tick, order, data)

    def add(tick: int, order: int, data: bytes) -> None:
        events.append((tick, order, data))

    add(0, _SETUP, smf.text_event(smf.META_TRACK_NAME, track.name or track.id))
    info: dict = {"id": track.id, "group": track.group}
    if track.stem is not None:
        info["stem"] = track.stem
    if track.program is not None:
        info["program"] = track.program
    if track.backend is not None:
        info["backend"] = vars(track.backend)
    add(0, _SETUP, _meta_text(info))
    if not track.is_drum:
        program = track.program if track.program is not None else DEFAULT_PROGRAMS[track.group]
        add(0, _SETUP, smf.channel_message(smf.PROGRAM_CHANGE, ch, program))

    bend_range = _bend_range(track)
    if bend_range:
        for cc, value in ((101, 0), (100, 0), (6, bend_range), (38, 0), (101, 127), (100, 127)):
            add(0, _SETUP, smf.channel_message(smf.CONTROL_CHANGE, ch, cc, value))

    to_tick = _tick_converter(grid, opts)
    for note, on, off in _note_spans(track, to_tick):
        add(on, _NOTE_ON, smf.channel_message(smf.NOTE_ON, ch, note.pitch, note.velocity))
        add(off, _NOTE_OFF, smf.channel_message(smf.NOTE_OFF, ch, note.pitch, 0))
        if note.bend and bend_range:
            for t, cents in note.bend:
                tick = min(off, max(on, round(grid.clock.tick(note.onset + t))))
                add(tick, _BEND, _bend_message(ch, cents, bend_range))
            add(off, _BEND, _bend_message(ch, 0.0, bend_range))
    for c in track.controls:
        add(round(grid.clock.tick(c.time)), _CONTROL, smf.channel_message(smf.CONTROL_CHANGE, ch, c.cc, c.value))

    events.sort(key=lambda e: (e[0], e[1]))  # stable: equal keys keep insertion order
    return [(tick, data) for tick, _, data in events]


def _tick_converter(grid: BeatGrid, opts: MidiOptions):
    """Seconds -> integer tick, optionally pulled toward the beat grid."""
    step = grid.ticks_per_beat / opts.grid if opts.grid else None

    def to_tick(seconds: float) -> int:
        tick = grid.clock.tick(seconds)
        if step and tick >= grid.lead_in:
            snapped = grid.lead_in + round((tick - grid.lead_in) / step) * step
            tick += opts.strength * (snapped - tick)
        return max(0, round(tick))

    return to_tick


def _note_spans(track: Track, to_tick):
    """Notes with integer on/off ticks, fixed so that same-pitch notes never overlap.

    A note that would still sound when the next note of the same pitch starts
    is cut at that start; exact duplicates are dropped; every note lasts at
    least one tick.
    """
    spans = []
    by_pitch: dict[int, list] = {}
    for n in sorted(track.notes, key=lambda n: (n.onset, n.pitch)):
        by_pitch.setdefault(n.pitch, []).append([n, to_tick(n.onset), to_tick(n.offset)])
    for notes in by_pitch.values():
        for cur, nxt in zip(notes, notes[1:] + [None]):
            n, on, off = cur
            off = max(off, on + 1)
            if nxt is not None and nxt[1] < off:
                off = nxt[1]
            if off > on:
                spans.append((n, on, off))
    spans.sort(key=lambda s: (s[1], s[0].pitch))
    return spans


def _bend_range(track: Track) -> int:
    """Pitch-bend range in semitones for this track, or 0 if nothing bends."""
    cents = [abs(c) for n in track.notes if n.bend for _, c in n.bend]
    if not cents:
        return 0
    return min(24, max(2, math.ceil(max(cents) / 100)))


def _bend_message(ch: int, cents: float, bend_range: int) -> bytes:
    value = 8192 + round(cents / (bend_range * 100) * 8192)
    value = min(16383, max(0, value))
    return smf.channel_message(smf.PITCH_BEND, ch, value & 0x7F, value >> 7)
