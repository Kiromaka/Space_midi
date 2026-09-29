"""Tempo maps: converting between seconds and MIDI ticks.

``TickClock`` is a piecewise-constant tempo map, exactly as a MIDI player
reads it from Set Tempo events. ``clock_from_beats`` builds one from beat
times detected in the audio, so that every detected beat falls on a beat of
the MIDI grid and the first downbeat falls on a bar line.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field

DEFAULT_US_PER_QUARTER = 500_000  # 120 BPM, the MIDI default
_MAX_US_PER_QUARTER = 0xFFFFFF


@dataclass
class TickClock:
    """Tempo changes as ``(tick, microseconds per quarter note)``, sorted by tick."""

    ppq: int
    changes: list[tuple[int, int]] = field(default_factory=lambda: [(0, DEFAULT_US_PER_QUARTER)])
    _ticks: list[int] = field(init=False, repr=False)
    _seconds: list[float] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        changes = sorted(self.changes, key=lambda c: c[0])
        if not changes or changes[0][0] != 0:
            changes.insert(0, (0, DEFAULT_US_PER_QUARTER))
        merged: list[tuple[int, int]] = []
        for tick, us in changes:
            if merged and merged[-1][0] == tick:
                merged[-1] = (tick, us)  # the later event at the same tick wins
            elif not merged or merged[-1][1] != us:
                merged.append((tick, us))
        self.changes = merged
        self._ticks = [t for t, _ in merged]
        self._seconds = [0.0]
        for (t0, us), (t1, _) in zip(merged, merged[1:]):
            self._seconds.append(self._seconds[-1] + (t1 - t0) * us / 1e6 / self.ppq)

    def seconds(self, tick: float) -> float:
        """Time in seconds of a (possibly fractional) tick."""
        i = max(0, bisect.bisect_right(self._ticks, tick) - 1)
        return self._seconds[i] + (tick - self._ticks[i]) * self.changes[i][1] / 1e6 / self.ppq

    def tick(self, seconds: float) -> float:
        """Fractional tick at a time in seconds; round it for event positions."""
        i = max(0, bisect.bisect_right(self._seconds, seconds) - 1)
        return self._ticks[i] + (seconds - self._seconds[i]) * 1e6 * self.ppq / self.changes[i][1]


@dataclass
class BeatGrid:
    """Where the detected beats sit in the MIDI file."""

    clock: TickClock
    ticks_per_beat: int
    lead_in: int  # tick of the first detected beat
    bar: int  # ticks per bar
    pickup: int = 0  # length of a shortened first bar, 0 if bars start at tick 0
    last_beat: int = 0  # tick of the last detected beat


def ticks_per_beat(ppq: int, denominator: int) -> int:
    tpb = ppq * 4 / denominator
    if tpb != int(tpb):
        raise ValueError(f"ppq {ppq} cannot express 1/{denominator} notes in whole ticks")
    return int(tpb)


def _us(seconds_per_tick: float, ppq: int) -> int:
    return min(_MAX_US_PER_QUARTER, max(1, round(seconds_per_tick * ppq * 1e6)))


def clock_from_beats(
    beats: list[float],
    downbeats: list[float],
    time_signature: tuple[int, int],
    ppq: int,
) -> BeatGrid:
    """Build a tempo map that puts detected beat ``i`` at tick ``lead_in + i * tpb``.

    Each beat interval gets its own tempo, so notes stay aligned with the
    audio however much the tempo drifts. The stretch before the first beat
    (the lead-in) keeps the song's opening tempo, rounded to a 16th of a
    beat. If the first downbeat then does not fall on a bar line, the first
    bar becomes a shorter pickup bar (``pickup`` ticks) so that it does.
    """
    num, den = time_signature
    tpb = ticks_per_beat(ppq, den)
    bar = num * tpb
    if len(beats) < 2:
        return BeatGrid(TickClock(ppq), tpb, 0, bar)

    first_spb = beats[1] - beats[0]
    b0 = beats[0]
    unit = max(1, tpb // 4)
    lead_in = 0
    changes: list[tuple[int, int]] = []
    if b0 > 1e-3:
        lead_in = max(1, round(b0 / first_spb * tpb / unit)) * unit
        changes.append((0, _us(b0 / lead_in, ppq)))
    for i, (t0, t1) in enumerate(zip(beats, beats[1:])):
        changes.append((lead_in + i * tpb, _us((t1 - t0) / tpb, ppq)))

    first_downbeat = lead_in + _index_of_first_downbeat(beats, downbeats) * tpb
    pickup = first_downbeat % bar
    last_beat = lead_in + (len(beats) - 1) * tpb
    return BeatGrid(TickClock(ppq, changes), tpb, lead_in, bar, pickup, last_beat)


def pickup_signature(pickup: int, tpb: int, denominator: int) -> tuple[int, int]:
    """Time signature for a pickup bar of ``pickup`` ticks, e.g. 3/16 or 2/4."""
    for unit, den in ((tpb, denominator), (tpb // 2, denominator * 2), (tpb // 4, denominator * 4)):
        if unit and pickup % unit == 0:
            return pickup // unit, den
    raise ValueError(f"pickup of {pickup} ticks is not a whole number of 16ths of a beat")


def _index_of_first_downbeat(beats: list[float], downbeats: list[float]) -> int:
    if not downbeats:
        return 0
    d0 = min(downbeats)
    i = bisect.bisect_left(beats, d0)
    candidates = [j for j in (i - 1, i) if 0 <= j < len(beats)]
    return min(candidates, key=lambda j: abs(beats[j] - d0))
