"""Scores for comparing estimated notes, onsets and beats against a reference.

All scores are counts of matched events (see ``matching``) turned into
precision, recall and F1. Conventions follow mir_eval, so the numbers are
comparable with published results:

* onsets: +/-50 ms;
* notes: same pitch, onsets +/-50 ms, and for "onset+offset" the offset
  within 20% of the reference duration (at least 50 ms);
* beats: +/-70 ms.
"""

from __future__ import annotations

from dataclasses import dataclass

from spacemidi.notes import GROUPS, Note, NoteDocument

from .matching import match_events, match_notes

#: General MIDI percussion keys grouped into the classes drum transcription reports.
DRUM_CLASSES: dict[str, frozenset[int]] = {
    "kick": frozenset({35, 36}),
    "snare": frozenset({37, 38, 39, 40}),
    "hihat": frozenset({42, 44, 46}),
    "tom": frozenset({41, 43, 45, 47, 48, 50}),
    "cymbal": frozenset({49, 51, 52, 53, 55, 57, 59}),
}


def drum_class(pitch: int) -> str:
    for name, keys in DRUM_CLASSES.items():
        if pitch in keys:
            return name
    return "other"


@dataclass
class Score:
    """Matched counts; precision, recall and F1 follow from them."""

    n_ref: int = 0
    n_est: int = 0
    n_match: int = 0

    @property
    def precision(self) -> float:
        return self.n_match / self.n_est if self.n_est else 0.0

    @property
    def recall(self) -> float:
        return self.n_match / self.n_ref if self.n_ref else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    def __add__(self, other: Score) -> Score:
        return Score(self.n_ref + other.n_ref, self.n_est + other.n_est, self.n_match + other.n_match)

    def to_dict(self) -> dict:
        return {
            "n_ref": self.n_ref,
            "n_est": self.n_est,
            "n_match": self.n_match,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def onset_score(reference: list[float], estimate: list[float], window: float = 0.05) -> Score:
    return Score(len(reference), len(estimate), len(match_events(reference, estimate, window)))


def beat_score(reference: list[float], estimate: list[float], window: float = 0.07) -> Score:
    return onset_score(reference, estimate, window)


def note_score(
    reference: list[Note],
    estimate: list[Note],
    onset_tolerance: float = 0.05,
    offset_ratio: float | None = 0.2,
) -> Score:
    ref = [(n.onset, n.offset, n.pitch) for n in reference]
    est = [(n.onset, n.offset, n.pitch) for n in estimate]
    pairs = match_notes(ref, est, onset_tolerance, offset_ratio)
    return Score(len(ref), len(est), len(pairs))


@dataclass
class Report:
    """Scores of one estimated document against its reference."""

    onset: dict[str, Score]  # per group, notes matched on onset + pitch
    onset_offset: dict[str, Score]  # per group, notes matched on onset + offset + pitch
    drums: dict[str, Score]  # per drum class, onsets
    agnostic: Score  # all pitched notes pooled, instrument ignored (onset + pitch)
    beats: Score | None = None
    downbeats: Score | None = None

    @property
    def multi_instrument(self) -> Score:
        """Onset score where a note only counts if its instrument group is right too."""
        total = Score()
        for score in self.onset.values():
            total += score
        return total + sum(self.drums.values(), Score())

    def to_dict(self) -> dict:
        out = {
            "multi_instrument": self.multi_instrument.to_dict(),
            "agnostic": self.agnostic.to_dict(),
            "onset": {g: s.to_dict() for g, s in self.onset.items()},
            "onset_offset": {g: s.to_dict() for g, s in self.onset_offset.items()},
            "drums": {c: s.to_dict() for c, s in self.drums.items()},
        }
        if self.beats is not None:
            out["beats"] = self.beats.to_dict()
        if self.downbeats is not None:
            out["downbeats"] = self.downbeats.to_dict()
        return out


def _notes_by_group(doc: NoteDocument) -> tuple[dict[str, list[Note]], list[Note]]:
    pitched: dict[str, list[Note]] = {}
    drums: list[Note] = []
    for track in doc.tracks:
        if track.is_drum or track.group == "drums":
            drums.extend(track.notes)
        else:
            pitched.setdefault(track.group, []).extend(track.notes)
    return pitched, drums


def evaluate(
    reference: NoteDocument,
    estimate: NoteDocument,
    onset_tolerance: float = 0.05,
    offset_ratio: float = 0.2,
) -> Report:
    """Compare two documents group by group (vocals, bass, piano, ...), plus drums by class."""
    ref_pitched, ref_drums = _notes_by_group(reference)
    est_pitched, est_drums = _notes_by_group(estimate)

    onset: dict[str, Score] = {}
    onset_offset: dict[str, Score] = {}
    for group in GROUPS:
        if group == "drums":
            continue
        r, e = ref_pitched.get(group, []), est_pitched.get(group, [])
        if r or e:
            onset[group] = note_score(r, e, onset_tolerance, None)
            onset_offset[group] = note_score(r, e, onset_tolerance, offset_ratio)

    drums: dict[str, Score] = {}
    classes = {drum_class(n.pitch) for n in ref_drums + est_drums}
    for cls in sorted(classes):
        r = [n.onset for n in ref_drums if drum_class(n.pitch) == cls]
        e = [n.onset for n in est_drums if drum_class(n.pitch) == cls]
        drums[cls] = onset_score(r, e, onset_tolerance)

    all_ref = [n for notes in ref_pitched.values() for n in notes]
    all_est = [n for notes in est_pitched.values() for n in notes]
    agnostic = note_score(all_ref, all_est, onset_tolerance, None)

    report = Report(onset, onset_offset, drums, agnostic)
    if reference.tempo is not None and estimate.tempo is not None:
        report.beats = beat_score(reference.tempo.beats, estimate.tempo.beats)
        report.downbeats = beat_score(reference.tempo.downbeats, estimate.tempo.downbeats)
    return report


def format_report(report: Report) -> str:
    """A plain-text table for the terminal."""
    lines = [f"{'':<18}{'ref':>6}{'est':>6}{'P':>8}{'R':>8}{'F1':>8}"]

    def row(name: str, s: Score) -> None:
        lines.append(f"{name:<18}{s.n_ref:>6}{s.n_est:>6}{s.precision:>8.3f}{s.recall:>8.3f}{s.f1:>8.3f}")

    row("ALL (instrument)", report.multi_instrument)
    row("ALL (any instr.)", report.agnostic)
    for group, s in report.onset.items():
        row(f"{group}", s)
        row(f"{group} +offset", report.onset_offset[group])
    for cls, s in report.drums.items():
        row(f"drums/{cls}", s)
    if report.beats is not None:
        row("beats", report.beats)
    if report.downbeats is not None:
        row("downbeats", report.downbeats)
    return "\n".join(lines)
