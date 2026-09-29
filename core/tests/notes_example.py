"""A small document that uses every field of the note format.

Shared by the tests and used to generate docs/formats/example.notes.json:
    uv run python tests/notes_example.py
"""

from pathlib import Path

from spacemidi.notes import Backend, Control, Key, Note, NoteDocument, Source, TempoMap, Track, save

EXAMPLE_PATH = Path(__file__).resolve().parents[2] / "docs" / "formats" / "example.notes.json"


def make_example() -> NoteDocument:
    beats = [0.5 * i for i in range(16)]  # 120 BPM, 8 seconds
    vocals = Track(
        id="vocals",
        name="Vocals",
        group="vocals",
        stem="vocals",
        program=52,
        backend=Backend(stage="pitch", name="pyin", version="0.1.0", params={"fmin": 65.0}),
        notes=[
            Note(0.52, 0.98, 69, 88, confidence=0.94, bend=[(0.0, -35.0), (0.08, 0.0)]),
            Note(1.0, 1.95, 72, 92, confidence=0.9),
            Note(2.0, 3.4, 71, 85, confidence=0.81, bend=[(0.9, 0.0), (1.1, 40.0), (1.3, -40.0)]),
        ],
    )
    drums = Track(
        id="drums",
        name="Drums",
        group="drums",
        stem="drums",
        is_drum=True,
        backend=Backend(stage="drums", name="band-onsets", version="0.1.0"),
        notes=[
            Note(0.5, 0.6, 36, 110),
            Note(0.5, 0.6, 42, 70),
            Note(1.0, 1.1, 42, 64),
            Note(1.5, 1.6, 38, 104),
            Note(1.5, 1.6, 42, 66),
        ],
    )
    piano = Track(
        id="piano",
        name="Piano",
        group="piano",
        stem="piano",
        program=0,
        backend=Backend(stage="poly", name="nmf-multif0", version="0.1.0", params={"rank": 88}),
        notes=[
            Note(0.5, 2.4, 57, 70, confidence=0.77),
            Note(0.5, 2.4, 60, 66, confidence=0.72),
            Note(0.5, 2.4, 64, 68, confidence=0.8),
        ],
        controls=[Control(0.45, 64, 127), Control(2.45, 64, 0)],
    )
    return NoteDocument(
        source=Source(path="example.wav", duration=8.0, sample_rate=44100),
        tuning_hz=440.0,
        tempo=TempoMap(beats=beats, downbeats=beats[::4], time_signature=(4, 4)),
        key=Key(tonic="A", mode="minor"),
        tracks=[vocals, drums, piano],
    )


if __name__ == "__main__":
    save(make_example(), EXAMPLE_PATH)
    print(f"wrote {EXAMPLE_PATH}")
