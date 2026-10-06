"""Synthetic songs on a beat grid, for rhythm tests (not a test module itself)."""

from __future__ import annotations

import random

from spacemidi.notes import Note, NoteDocument, TempoMap, Track

# Chord roots (MIDI) cycled one per bar, so harmony changes on every downbeat.
_ROOTS = [48, 53, 55, 45, 50, 52]


def grid_song(
    bpm: float = 120.0,
    bars: int = 16,
    meter: int = 4,
    seed: int = 0,
    lead_in: float = 0.5,
    swing_ms: float = 0.0,
) -> NoteDocument:
    """A small band on a steady grid: drums, bass and chords, one chord per bar.

    Kick on beat 1 (and 3 in 4/4), snare on the other strong beats, hi-hat on
    eighths, a bass note per beat (root on 1), a triad held for the bar.
    ``swing_ms`` adds random timing jitter to every note. The document's
    ``tempo`` holds the true beats and downbeats.
    """
    rng = random.Random(seed)
    period = 60.0 / bpm
    beats = [lead_in + i * period for i in range(bars * meter + 1)]
    downbeats = beats[:-1:meter]
    drums, bass, chords = [], [], []

    def j() -> float:
        return rng.uniform(-swing_ms, swing_ms) / 1000.0 if swing_ms else 0.0

    for bar in range(bars):
        root = _ROOTS[bar % len(_ROOTS)]
        start = beats[bar * meter]
        chords += [Note(start + j(), start + meter * period * 0.95, root + 12 + i, 70) for i in (0, 4, 7)]
        for b in range(meter):
            t = beats[bar * meter + b]
            if b == 0:
                drums.append(Note(t + j(), t + 0.1, 36, 110))
            elif meter == 4 and b == 2:
                drums.append(Note(t + j(), t + 0.1, 36, 90))
            else:
                drums.append(Note(t + j(), t + 0.1, 38, 80))
            for half in (0.0, 0.5):
                h = t + half * period
                drums.append(Note(h + j(), h + 0.05, 42, 50))
            pitch = root if b == 0 else root + rng.choice([0, 7, 12])
            bass.append(Note(t + j(), t + period * 0.9, pitch - 12, 100 if b == 0 else 80))
    return NoteDocument(
        tracks=[
            Track(id="drums", group="drums", is_drum=True, notes=drums),
            Track(id="bass", group="bass", notes=bass),
            Track(id="keys", group="piano", notes=chords),
        ],
        tempo=TempoMap(beats=beats[:-1], downbeats=downbeats, time_signature=(meter, 4)),
    )
