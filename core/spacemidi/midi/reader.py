"""Standard MIDI File -> NoteDocument.

Works on any SMF (format 0 or 1, PPQ timing). Files written by
``spacemidi.midi.writer`` carry ``spacemidi`` text events, which restore
track ids, groups, stems, backends and the source; for other files these are
inferred (group from channel and General MIDI program).

Notes are split per (track, channel). Beats are reconstructed from the tempo
map and the main time signature (the one that covers most of the file): one
beat every ``ppq * 4 / denominator`` ticks from the first beat. Bar lines
follow every time-signature change, so a pickup bar is understood, and
downbeats are the beats on bar lines.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from spacemidi.notes import Backend, Control, Key, Note, NoteDocument, Source, TempoMap, Track

from . import smf
from .tempo import TickClock, ticks_per_beat
from .writer import DRUM_CHANNEL, TEXT_PREFIX

_PITCH_CLASSES = ("C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B")
_RPN_CCS = {6, 38, 98, 99, 100, 101}  # data entry and (N)RPN selection, not user controls


def read_midi(source: str | Path | bytes) -> NoteDocument:
    data = source if isinstance(source, bytes) else Path(source).read_bytes()
    f = smf.parse_file(data)

    tempo_changes: list[tuple[int, int]] = []
    signatures: list[tuple[int, tuple[int, int]]] = []
    key: Key | None = None
    info: dict = {}
    end_tick = 0
    for events in f.tracks:
        for tick, ev in events:
            end_tick = max(end_tick, tick)
            if ev[0] != 0xFF:
                continue
            kind, payload = ev[1], ev[2:]
            if kind == smf.META_TEMPO and len(payload) == 3:
                tempo_changes.append((tick, int.from_bytes(payload, "big")))
            elif kind == smf.META_TIME_SIGNATURE and len(payload) >= 2 and payload[0] > 0:
                signatures.append((tick, (payload[0], 2 ** payload[1])))
            elif kind == smf.META_KEY_SIGNATURE and key is None and len(payload) == 2:
                key = _key_from_signature(int.from_bytes(payload[:1], "big", signed=True), payload[1])
    clock = TickClock(f.ppq, tempo_changes)

    doc = NoteDocument(key=key)
    conductor = _spacemidi_info(f.tracks[0]) if f.tracks else {}
    if conductor:
        doc.tuning_hz = conductor.get("tuning_hz", 440.0)
        if "source" in conductor:
            doc.source = Source(**conductor["source"])
    doc.tempo = _beats(clock, signatures, conductor.get("lead_in", 0), end_tick)

    used_ids: set[str] = set()
    for index, events in enumerate(f.tracks):
        for track in _read_track(index, events, clock):
            base = track.id
            n = 2
            while track.id in used_ids:
                track.id = f"{base}-{n}"
                n += 1
            used_ids.add(track.id)
            doc.tracks.append(track)
    return doc


def _key_from_signature(sharps: int, minor: int) -> Key | None:
    if not -7 <= sharps <= 7:
        return None
    major_tonic = (sharps * 7) % 12
    if minor:
        return Key(_PITCH_CLASSES[(major_tonic + 9) % 12], "minor")
    return Key(_PITCH_CLASSES[major_tonic], "major")


def _spacemidi_info(events) -> dict:
    for _, ev in events:
        if ev[:2] == bytes([0xFF, smf.META_TEXT]):
            text = ev[2:].decode("utf-8", errors="replace")
            if text.startswith(TEXT_PREFIX):
                try:
                    return json.loads(text[len(TEXT_PREFIX) :])
                except json.JSONDecodeError:
                    return {}
    return {}


def _beats(clock: TickClock, signatures, lead_in: int, end_tick: int) -> TempoMap | None:
    spans = _signature_spans(signatures, end_tick)
    num, den = max(spans, key=lambda s: s[1] - s[0])[2]  # the signature covering most ticks
    try:
        tpb = ticks_per_beat(clock.ppq, den)
    except ValueError:
        return None
    if end_tick < lead_in + tpb:
        return None
    lines = set()
    for start, stop, (n, d) in spans:
        bar = n * clock.ppq * 4 / d
        x = float(start)
        while x < stop:
            lines.add(round(x))
            x += bar
    ticks = range(lead_in, end_tick + 1, tpb)
    beats = [round(clock.seconds(t), 6) for t in ticks]
    downbeats = [round(clock.seconds(t), 6) for t in ticks if t in lines]
    return TempoMap(beats=beats, downbeats=downbeats, time_signature=(num, den))


def _signature_spans(signatures, end_tick: int):
    """(start, stop, (num, den)) for each time signature in force; 4/4 if none."""
    sigs = sorted(signatures, key=lambda s: s[0])
    if not sigs or sigs[0][0] > 0:
        sigs.insert(0, (0, (4, 4)))
    spans = []
    for i, (tick, sig) in enumerate(sigs):
        stop = sigs[i + 1][0] if i + 1 < len(sigs) else max(end_tick + 1, tick + 1)
        if stop > tick:
            spans.append((tick, stop, sig))
    return spans


def _group_for(channel: int, program: int | None) -> str:
    if channel == DRUM_CHANNEL:
        return "drums"
    if program is None:
        return "other"
    if program <= 7:
        return "piano"
    if 24 <= program <= 31:
        return "guitar"
    if 32 <= program <= 39:
        return "bass"
    if 52 <= program <= 54:
        return "vocals"
    return "other"


class _Channel:
    def __init__(self) -> None:
        self.program: int | None = None
        self.notes: list[Note] = []
        self.controls: list[Control] = []
        self.active: dict[int, list[tuple[Note, float]]] = {}  # pitch -> [(note, onset seconds)]
        self.rpn = (127, 127)
        self.bend_range = 2.0  # semitones, the General MIDI default
        self.bend_cents = 0.0


def _read_track(index: int, events, clock: TickClock) -> list[Track]:
    name = ""
    info = _spacemidi_info(events)
    channels: dict[int, _Channel] = {}
    last_tick = 0

    for tick, ev in events:
        last_tick = max(last_tick, tick)
        status = ev[0]
        if status == 0xFF:
            if ev[1] == smf.META_TRACK_NAME and not name:
                name = ev[2:].decode("utf-8", errors="replace").strip()
            continue
        if status >= 0xF0:
            continue
        kind, ch = status & 0xF0, status & 0x0F
        c = channels.setdefault(ch, _Channel())
        t = clock.seconds(tick)
        if kind == smf.NOTE_ON and ev[2] > 0:
            note = Note(onset=t, offset=t, pitch=ev[1], velocity=ev[2])
            if c.bend_cents:
                note.bend = [(0.0, c.bend_cents)]
            c.active.setdefault(ev[1], []).append((note, t))
        elif kind in (smf.NOTE_OFF, smf.NOTE_ON):
            stack = c.active.get(ev[1])
            if stack:
                note, _ = stack.pop(0)  # first in, first out
                _finish(note, t, c)
        elif kind == smf.CONTROL_CHANGE:
            cc, value = ev[1], ev[2]
            if cc == 101:
                c.rpn = (value, c.rpn[1])
            elif cc == 100:
                c.rpn = (c.rpn[0], value)
            elif cc == 6 and c.rpn == (0, 0):
                c.bend_range = float(value) + (c.bend_range % 1)
            elif cc == 38 and c.rpn == (0, 0):
                c.bend_range = int(c.bend_range) + value / 100
            if cc not in _RPN_CCS:
                c.controls.append(Control(time=t, cc=cc, value=value))
        elif kind == smf.PROGRAM_CHANGE:
            if c.program is None:
                c.program = ev[1]
        elif kind == smf.PITCH_BEND:
            value = ev[1] | (ev[2] << 7)
            c.bend_cents = (value - 8192) / 8192 * c.bend_range * 100
            for stack in c.active.values():
                for note, onset in stack:
                    point = (t - onset, c.bend_cents)
                    if note.bend is None:
                        note.bend = [point]
                    elif note.bend[-1][0] == point[0]:
                        note.bend[-1] = point
                    else:
                        note.bend.append(point)

    end = clock.seconds(last_tick)
    for c in channels.values():  # notes never switched off end with the track
        for stack in c.active.values():
            for note, _ in stack:
                _finish(note, end, c)

    tracks = []
    used = [ch for ch, c in sorted(channels.items()) if c.notes or c.controls]
    for ch in used:
        c = channels[ch]
        suffix = f" ch{ch + 1}" if len(used) > 1 else ""
        track = Track(
            id=info.get("id") if len(used) == 1 and info.get("id") else _slug(name or f"track{index}") + suffix.replace(" ", "-"),
            name=(name or f"Track {index}") + suffix,
            group=info.get("group") if len(used) == 1 and info.get("group") else _group_for(ch, c.program),
            stem=info.get("stem") if len(used) == 1 else None,
            program=None if ch == DRUM_CHANNEL else (info.get("program") if len(used) == 1 and "program" in info else c.program),
            is_drum=ch == DRUM_CHANNEL,
            notes=c.notes,
            controls=c.controls,
        )
        if len(used) == 1 and "backend" in info:
            track.backend = Backend(**info["backend"])
        tracks.append(track)
    return tracks


def _finish(note: Note, t: float, c: _Channel) -> None:
    if t <= note.onset:
        return  # zero-length note: nothing to keep
    note.offset = t
    if note.bend is not None:
        duration = note.offset - note.onset
        note.bend = [(bt, cents) for bt, cents in note.bend if bt <= duration]
        if all(cents == 0 for _, cents in note.bend):
            note.bend = None
    c.notes.append(note)


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "track"
