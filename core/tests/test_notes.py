import copy
import json

import jsonschema
import pytest

from notes_example import EXAMPLE_PATH, make_example
from spacemidi.cli import main
from spacemidi.notes import (
    Note,
    NoteDocument,
    NotesFormatError,
    Track,
    dumps,
    from_dict,
    load,
    loads,
    save,
    to_dict,
    validate,
)

SCHEMA_PATH = EXAMPLE_PATH.with_name("notes.schema.json")


@pytest.fixture(scope="module")
def schema():
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- happy path


def test_example_is_valid():
    assert validate(make_example()) == []


def test_round_trip_keeps_everything():
    doc = make_example()
    again = loads(dumps(doc))
    assert to_dict(again) == to_dict(doc)
    assert again.track("vocals").notes[0].bend == [(0.0, -35.0), (0.08, 0.0)]
    assert again.track("piano").controls[0].cc == 64
    assert again.tempo.time_signature == (4, 4)


def test_save_and_load_file(tmp_path):
    path = tmp_path / "song.notes.json"
    save(make_example(), path)
    assert to_dict(load(path)) == to_dict(make_example())


def test_notes_are_written_sorted_one_per_line():
    doc = NoteDocument(tracks=[Track(id="bass", group="bass")])
    doc.track("bass").notes = [Note(2.0, 2.5, 40), Note(1.0, 1.5, 43), Note(1.0, 1.5, 41)]
    text = dumps(doc)
    pitches = [n["pitch"] for n in json.loads(text)["tracks"][0]["notes"]]
    assert pitches == [41, 43, 40]
    note_lines = [line for line in text.splitlines() if '"onset"' in line]
    assert len(note_lines) == 3


def test_long_notes_stay_on_one_line():
    bend = [(i * 0.01, float(i % 7) * 10) for i in range(50)]
    doc = NoteDocument(tracks=[Track(id="v", group="vocals", notes=[Note(0.0, 1.0, 60, bend=bend)])])
    note_lines = [line for line in dumps(doc).splitlines() if '"onset"' in line]
    assert len(note_lines) == 1 and '"bend"' in note_lines[0]


def test_seconds_are_rounded():
    doc = NoteDocument(tracks=[Track(id="bass", group="bass", notes=[Note(0.1234567891, 0.5, 40)])])
    assert json.loads(dumps(doc))["tracks"][0]["notes"][0]["onset"] == 0.123457


def test_unknown_keys_are_ignored():
    data = to_dict(make_example())
    data["future_field"] = {"x": 1}
    data["tracks"][0]["notes"][0]["articulation"] = "legato"
    assert validate(from_dict(data)) == []


def test_optional_fields_get_defaults():
    doc = from_dict(
        {
            "format": "spacemidi-notes",
            "version": 1,
            "tracks": [{"id": "g", "group": "guitar", "notes": [{"onset": 0, "offset": 1, "pitch": 52}]}],
        }
    )
    assert doc.tuning_hz == 440.0
    note = doc.track("g").notes[0]
    assert note.velocity == 80 and note.confidence is None and note.bend is None


# ---------------------------------------------------------------- semantic errors


def _mutate(fn):
    doc = make_example()
    fn(doc)
    return doc


SEMANTIC_CASES = {
    "offset before onset": (lambda d: setattr(d.tracks[0].notes[0], "offset", 0.1), "offset"),
    "pitch too high": (lambda d: setattr(d.tracks[0].notes[0], "pitch", 128), "pitch"),
    "zero velocity": (lambda d: setattr(d.tracks[0].notes[0], "velocity", 0), "velocity"),
    "confidence above 1": (lambda d: setattr(d.tracks[0].notes[0], "confidence", 1.5), "confidence"),
    "bend past note end": (lambda d: setattr(d.tracks[0].notes[0], "bend", [(5.0, 0.0)]), "bend[0]"),
    "bend not increasing": (
        lambda d: setattr(d.tracks[0].notes[0], "bend", [(0.1, 0.0), (0.05, 0.0)]),
        "strictly increasing",
    ),
    "huge bend": (lambda d: setattr(d.tracks[0].notes[0], "bend", [(0.0, 5000.0)]), "cents"),
    "duplicate track id": (lambda d: setattr(d.tracks[1], "id", "vocals"), "duplicate"),
    "unknown group": (lambda d: setattr(d.tracks[0], "group", "kazoo"), "group"),
    "drum track with program": (lambda d: setattr(d.tracks[1], "program", 0), "no program"),
    "program too high": (lambda d: setattr(d.tracks[0], "program", 200), "program"),
    "beats not increasing": (lambda d: d.tempo.beats.reverse(), "strictly increasing"),
    "downbeat not a beat": (lambda d: d.tempo.downbeats.append(7.9), "downbeats"),
    "bad denominator": (lambda d: setattr(d.tempo, "time_signature", (4, 3)), "denominator"),
    "unknown tonic": (lambda d: setattr(d.key, "tonic", "H"), "tonic"),
    "tuning out of range": (lambda d: setattr(d, "tuning_hz", 300.0), "tuning_hz"),
    "note past end of audio": (lambda d: setattr(d.tracks[0].notes[2], "onset", 9.0), "past the end"),
    "cc value too high": (lambda d: setattr(d.tracks[2].controls[0], "value", 200), "value"),
}


@pytest.mark.parametrize("case", SEMANTIC_CASES)
def test_semantic_errors_are_reported(case):
    fn, expected = SEMANTIC_CASES[case]
    problems = validate(_mutate(fn))
    assert any(expected in p for p in problems), problems


def test_dumps_refuses_invalid_documents():
    doc = _mutate(lambda d: setattr(d.tracks[0].notes[0], "pitch", 200))
    with pytest.raises(NotesFormatError) as e:
        dumps(doc)
    assert "tracks[0].notes[0].pitch" in str(e.value)


def test_all_problems_are_reported_at_once():
    def break_two(d):
        d.tracks[0].notes[0].pitch = 200
        d.tracks[1].notes[0].velocity = 0

    assert len(validate(_mutate(break_two))) == 2


# ---------------------------------------------------------------- structural errors


def _broken(fn):
    data = copy.deepcopy(to_dict(make_example()))
    fn(data)
    return data


STRUCTURAL_CASES = {
    "missing onset": (lambda d: d["tracks"][0]["notes"][0].pop("onset"), "missing 'onset'"),
    "pitch as string": (lambda d: d["tracks"][0]["notes"][0].update(pitch="A4"), "expected integer"),
    "boolean velocity": (lambda d: d["tracks"][0]["notes"][0].update(velocity=True), "expected integer"),
    "wrong format name": (lambda d: d.update(format="other"), "format"),
    "missing tracks": (lambda d: d.pop("tracks"), "missing 'tracks'"),
    "note not an object": (lambda d: d["tracks"][0]["notes"].append(5), "expected object"),
    "bend point malformed": (lambda d: d["tracks"][0]["notes"][0].update(bend=[[0.1]]), "bend[0]"),
    "time signature malformed": (lambda d: d["tempo"].update(time_signature=[4]), "time_signature"),
}


@pytest.mark.parametrize("case", STRUCTURAL_CASES)
def test_structural_errors_are_reported(case):
    fn, expected = STRUCTURAL_CASES[case]
    with pytest.raises(NotesFormatError) as e:
        from_dict(_broken(fn))
    assert expected in str(e.value)


@pytest.mark.parametrize("case", STRUCTURAL_CASES)
def test_schema_rejects_structural_errors(case, schema):
    fn, _ = STRUCTURAL_CASES[case]
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(_broken(fn), schema)


def test_newer_version_is_rejected():
    with pytest.raises(NotesFormatError, match="supports up to 1"):
        from_dict(_broken(lambda d: d.update(version=2)))


def test_invalid_json_is_rejected():
    with pytest.raises(NotesFormatError, match="not valid JSON"):
        loads("{ not json")


# ---------------------------------------------------------------- schema and example file


def test_schema_is_valid_draft_2020_12(schema):
    jsonschema.Draft202012Validator.check_schema(schema)


def test_schema_accepts_writer_output(schema):
    jsonschema.validate(json.loads(dumps(make_example())), schema)


def test_example_file_is_current():
    """docs/formats/example.notes.json must equal what make_example() produces."""
    assert EXAMPLE_PATH.read_text(encoding="utf-8") == dumps(make_example())


# ---------------------------------------------------------------- CLI


def test_cli_validate(tmp_path, capsys):
    good = tmp_path / "good.notes.json"
    save(make_example(), good)
    bad = tmp_path / "bad.notes.json"
    bad.write_text(dumps(make_example()).replace('"pitch": 69', '"pitch": 169'), encoding="utf-8")

    assert main(["validate", str(good)]) == 0
    assert "OK" in capsys.readouterr().out
    assert main(["validate", str(good), str(bad)]) == 1
    assert "pitch" in capsys.readouterr().out
