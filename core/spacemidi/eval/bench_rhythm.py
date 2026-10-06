"""``spacemidi bench beats``: tempo, beat and downbeat tracking on Slakh2100 mixes.

The reference beats come from each track's MIDI tempo map (``all_src.mid``),
which is exactly the grid the audio was rendered on. Beats in the first 5 s
are ignored on both sides (mir_eval convention).

Per track: beat F-measure (+/-70 ms), the best F-measure at any metrical
level (double, half, off-beat), downbeat F-measure, tempo accuracy 1 and 2
(within 4 %, and allowing octave errors), and whether the bar length (3 or 4
beats) is right. Totals are means over tracks, as in the beat-tracking
literature.

Some Lakh MIDI files have a beat grid that does not line up with their own
notes (the notes sit a fraction of a beat off the grid), so the reference
itself is wrong there. :func:`grid_offset` measures this per track; means
are also reported over the "aligned" tracks only (offset within 0.06 beat).
"""

from __future__ import annotations

import cmath
import math
import time
from bisect import bisect_right
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .bench import BenchItem, expand_grid
from .metrics import beat_measures, tempo_accuracy, tempo_from_beats, trim_beats

#: Rhythm settings that change the precomputed features; the rest only change tracking.
FEATURE_KEYS = frozenset({"onset_preset", "low_hz"})

METRICS = ("beat_f1", "beat_f1_any", "downbeat_f1", "tempo_acc1", "tempo_acc2", "meter_ok")

#: A reference grid more than this many beats away from its own notes counts as misaligned.
MAX_GRID_OFFSET = 0.06


def grid_offset(doc) -> tuple[float, float]:
    """How far the notes of ``doc`` sit from its beat grid: (offset in beats, concentration 0..1).

    Note onsets are mapped to their phase within the beat, folded to the
    eighth-note level (so on-beat and "and" notes agree), and averaged on the
    circle. A well-aligned file gives an offset near 0; a file whose tempo
    map is shifted against its notes gives a clear non-zero offset.
    """
    beats = doc.tempo.beats if doc.tempo else []
    total, n = 0j, 0
    for track in doc.tracks:
        for note in track.notes:
            i = bisect_right(beats, note.onset) - 1
            if 0 <= i < len(beats) - 1:
                phase = (note.onset - beats[i]) / (beats[i + 1] - beats[i])
                total += cmath.exp(2j * math.pi * 2 * phase)
                n += 1
    if n == 0:
        return 0.0, 0.0
    z = total / n
    return cmath.phase(z) / (4 * math.pi), abs(z)


@dataclass
class RhythmPoint:
    params: dict
    rows: list[dict] = field(default_factory=list)

    def mean(self, key: str) -> float:
        values = [r[key] for r in self.rows if r.get(key) is not None]
        return float(np.mean(values)) if values else 0.0

    def aligned(self) -> RhythmPoint:
        """The same point restricted to tracks whose reference grid fits their notes."""
        return RhythmPoint(self.params, [r for r in self.rows if r.get("grid_aligned", True)])

    def to_dict(self) -> dict:
        levels: dict[str, int] = {}
        for r in self.rows:
            levels[r["best_level"]] = levels.get(r["best_level"], 0) + 1
        aligned = self.aligned()
        return {
            "params": self.params,
            "n_items": len(self.rows),
            "means": {k: round(self.mean(k), 4) for k in METRICS},
            "n_aligned": len(aligned.rows),
            "means_aligned": {k: round(aligned.mean(k), 4) for k in METRICS},
            "best_level_counts": levels,
        }


def score_rhythm(reference, rhythm, grid: tuple[float, float] = (0.0, 0.0)) -> dict | None:
    """Scores of one estimate against a reference TempoMap (None if the reference has too few beats).

    ``grid`` is :func:`grid_offset` of the reference document.
    """
    if reference is None or len(trim_beats(reference.beats)) < 2:
        return None
    beats = beat_measures(reference.beats, rhythm.beats)
    down = beat_measures(reference.downbeats, rhythm.downbeats)
    ref_bpm = tempo_from_beats(trim_beats(reference.beats))
    acc1, acc2 = tempo_accuracy(ref_bpm, rhythm.bpm) if ref_bpm else (False, False)
    return {
        "beat_f1": round(beats["f1"], 4),
        "beat_f1_any": round(beats["f1_any"], 4),
        "best_level": beats["best_level"],
        "downbeat_f1": round(down["f1"], 4),
        "ref_bpm": round(ref_bpm, 2) if ref_bpm else None,
        "est_bpm": round(rhythm.bpm, 2),
        "tempo_acc1": bool(acc1),
        "tempo_acc2": bool(acc2),
        "ref_meter": reference.time_signature[0],
        "est_meter": rhythm.time_signature[0],
        "meter_ok": reference.time_signature[0] == rhythm.time_signature[0],
        "grid_offset": round(grid[0], 3),
        "grid_aligned": abs(grid[0]) <= MAX_GRID_OFFSET,
    }


def _rhythm_item(job: tuple) -> dict:
    from spacemidi.algos.rhythm import RhythmConfig, rhythm_features, rhythm_from_features
    from spacemidi.dsp import load_audio

    item, points, audio_mode, sr = job
    started = time.perf_counter()
    if audio_mode == "synth":
        from .synth import render

        x = render(item.reference, sr, tail=0.5).astype(np.float64)
    else:
        x, _ = load_audio(item.audio, sr)
    features: dict[tuple, object] = {}
    grid = grid_offset(item.reference)
    scores = []
    for params in points:
        cfg = RhythmConfig().with_(**params)
        key = (cfg.onset_preset, cfg.low_hz)
        if key not in features:
            features[key] = rhythm_features(x, sr, cfg.onset_preset, cfg.low_hz)
        scores.append(score_rhythm(item.reference.tempo, rhythm_from_features(features[key], cfg), grid))
    return {
        "name": item.name,
        "duration": round(len(x) / sr, 2),
        "seconds": round(time.perf_counter() - started, 2),
        "scores": scores,
    }


def run_rhythm_bench(
    items: list[BenchItem],
    grid: dict[str, list] | None = None,
    *,
    audio: str = "real",
    sr: int = 22050,
    jobs: int = 1,
    progress=print,
) -> tuple[list[RhythmPoint], list[dict]]:
    from spacemidi.algos.rhythm import RhythmConfig

    points = expand_grid(grid or {})
    for params in points:
        RhythmConfig().with_(**params)  # fail early on a bad setting
    results = [RhythmPoint(params) for params in points]
    job_list = [(item, points, audio, sr) for item in items]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            rows = _collect(pool.map(_rhythm_item, job_list), results, len(items), progress)
    else:
        rows = _collect(map(_rhythm_item, job_list), results, len(items), progress)
    return results, rows


def _collect(outputs, results: list[RhythmPoint], n: int, progress) -> list[dict]:
    rows = []
    for k, row in enumerate(outputs, 1):
        for point, score in zip(results, row["scores"]):
            if score is not None:
                point.rows.append(score)
        first = row["scores"][0]
        if progress:
            if first is None:
                progress(f"[{k}/{n}] {row['name']}: no reference beats, skipped")
            else:
                progress(
                    f"[{k}/{n}] {row['name']} ({row['duration']:.0f} s): beat F {first['beat_f1']:.3f}, "
                    f"tempo {first['est_bpm']:.1f} vs {first['ref_bpm']}, downbeat F {first['downbeat_f1']:.3f}  "
                    f"{row['seconds']:.1f} s"
                )
        rows.append(row)
    return rows


def format_rhythm_results(results: list[RhythmPoint], top: int = 20) -> str:
    keys = sorted({k for p in results for k in p.params})
    head = [*keys, "beat F", "F any lvl", "downbeat F", "tempo acc1", "acc2", "meter ok", "n", "beat F aligned"]
    lines = [head]
    for p in sorted(results, key=lambda p: p.mean("beat_f1"), reverse=True)[:top]:
        aligned = p.aligned()
        lines.append(
            [str(p.params[k]) for k in keys]
            + [f"{p.mean(m):.3f}" for m in METRICS]
            + [str(len(p.rows)), f"{aligned.mean('beat_f1'):.3f} ({len(aligned.rows)})"]
        )
    widths = [max(len(r[i]) for r in lines) for i in range(len(head))]
    return "\n".join("  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in lines)
