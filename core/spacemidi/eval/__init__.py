"""Evaluation: scoring against references, synthetic test audio, datasets."""

from .bench import BenchItem, reference_onsets, run_onset_bench, slakh_items
from .datasets import (
    DataError,
    MusdbSong,
    SlakhTrack,
    TestsetItem,
    data_root,
    describe,
    load_document,
    musdb_songs,
    slakh_tracks,
    list_testset,
)
from .matching import match_events, match_notes, maximum_matching
from .metrics import DRUM_CLASSES, Report, Score, beat_score, drum_class, evaluate, format_report, note_score, onset_score
from .synth import read_wav, render, write_wav

__all__ = [
    "BenchItem",
    "DRUM_CLASSES",
    "DataError",
    "MusdbSong",
    "Report",
    "Score",
    "SlakhTrack",
    "TestsetItem",
    "beat_score",
    "data_root",
    "describe",
    "drum_class",
    "evaluate",
    "format_report",
    "load_document",
    "match_events",
    "match_notes",
    "maximum_matching",
    "musdb_songs",
    "note_score",
    "onset_score",
    "read_wav",
    "reference_onsets",
    "render",
    "run_onset_bench",
    "slakh_items",
    "slakh_tracks",
    "list_testset",
    "write_wav",
]
