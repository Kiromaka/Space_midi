"""Recompute our scores with mir_eval, the research standard, as a cross-check.

Used by ``spacemidi eval --mir-eval``. mir_eval is only a check here, never
part of the pipeline: it is in the dev dependency group, not required to run
Space MIDI.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np

from spacemidi.notes import Note, NoteDocument

from .metrics import Report, _notes_by_group, drum_class

_TOLERANCE = 1e-9


class MirEvalMissing(RuntimeError):
    pass


@dataclass
class Comparison:
    name: str
    ours: float
    theirs: float

    @property
    def agrees(self) -> bool:
        return abs(self.ours - self.theirs) <= _TOLERANCE


def _import():
    try:
        import mir_eval
    except ImportError as e:
        raise MirEvalMissing(
            "mir_eval is not installed. It is in the dev dependency group: run `uv sync` in core/."
        ) from e
    return mir_eval


def _arrays(notes: list[Note]):
    intervals = np.array([[n.onset, n.offset] for n in notes], dtype=float).reshape(-1, 2)
    pitches = np.array([440.0 * 2 ** ((n.pitch - 69) / 12) for n in notes], dtype=float)
    return intervals, pitches


def compare(reference: NoteDocument, estimate: NoteDocument, report: Report, onset_tolerance: float = 0.05) -> list[Comparison]:
    """F1 of every score in ``report``, next to mir_eval's F1 for the same comparison."""
    mir_eval = _import()
    ref_pitched, ref_drums = _notes_by_group(reference)
    est_pitched, est_drums = _notes_by_group(estimate)
    out: list[Comparison] = []

    def notes_f1(ref: list[Note], est: list[Note], offset_ratio: float | None) -> float:
        if not ref or not est:
            return 0.0
        (ri, rp), (ei, ep) = _arrays(ref), _arrays(est)
        return mir_eval.transcription.precision_recall_f1_overlap(
            ri, rp, ei, ep, onset_tolerance=onset_tolerance, offset_ratio=offset_ratio
        )[2]

    def events_f1(ref: list[float], est: list[float], window: float) -> float:
        if not ref or not est:
            return 0.0
        return mir_eval.onset.f_measure(np.array(sorted(ref)), np.array(sorted(est)), window=window)[0]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # mir_eval warns about empty or short inputs
        all_ref = [n for notes in ref_pitched.values() for n in notes]
        all_est = [n for notes in est_pitched.values() for n in notes]
        out.append(Comparison("ALL (any instr.)", report.agnostic.f1, notes_f1(all_ref, all_est, None)))
        for group in report.onset:
            r, e = ref_pitched.get(group, []), est_pitched.get(group, [])
            out.append(Comparison(group, report.onset[group].f1, notes_f1(r, e, None)))
            out.append(Comparison(f"{group} +offset", report.onset_offset[group].f1, notes_f1(r, e, 0.2)))
        for cls, score in report.drums.items():
            r = [n.onset for n in ref_drums if drum_class(n.pitch) == cls]
            e = [n.onset for n in est_drums if drum_class(n.pitch) == cls]
            out.append(Comparison(f"drums/{cls}", score.f1, events_f1(r, e, onset_tolerance)))
        if report.beats is not None and reference.tempo and estimate.tempo:
            r, e = reference.tempo.beats, estimate.tempo.beats
            theirs = mir_eval.beat.f_measure(np.array(r), np.array(e), f_measure_threshold=0.07) if r and e else 0.0
            out.append(Comparison("beats", report.beats.f1, theirs))
    return out


def format_comparison(rows: list[Comparison]) -> str:
    lines = [f"{'':<18}{'ours':>8}{'mir_eval':>10}", *(
        f"{r.name:<18}{r.ours:>8.4f}{r.theirs:>10.4f}  {'ok' if r.agrees else 'MISMATCH'}" for r in rows
    )]
    return "\n".join(lines)
