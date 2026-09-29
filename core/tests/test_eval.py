import copy
import json
import random

import numpy as np
import pytest

from notes_example import make_example
from spacemidi.cli import main
from spacemidi.eval import (
    DataError,
    Score,
    data_root,
    describe,
    evaluate,
    format_report,
    match_events,
    match_notes,
    maximum_matching,
    musdb_songs,
    note_score,
    onset_score,
    read_wav,
    render,
    slakh_tracks,
    list_testset,
    write_wav,
)
from spacemidi.eval.datasets import ENV_VAR, parse_simple_yaml
from spacemidi.midi import write_midi
from spacemidi.notes import Note, NoteDocument, Track, save

# ---------------------------------------------------------------- matching


def test_maximum_matching_beats_greedy():
    # Closest-first would pair ref 1 with est 0 (distance 0) and leave ref 0 unmatched.
    assert match_events([0.0, 0.05], [0.05, 0.10], window=0.05) == [(0, 0), (1, 1)]


def test_maximum_matching_long_augmenting_path():
    # A chain where every left node can take its own right node or the next one.
    n = 3000
    adjacency = [[i, i + 1] if i + 1 < n else [i] for i in range(n)]
    adjacency[0] = [1]  # forces a shift along the whole chain
    pairs = maximum_matching(adjacency, n)
    assert len(pairs) == n - 1  # right node 0 is unreachable, all others get matched
    assert len({v for _, v in pairs}) == len(pairs)


def test_window_boundary_is_inclusive():
    assert len(match_events([1.0], [1.05], window=0.05)) == 1
    assert len(match_events([1.0], [1.0501], window=0.05)) == 0


def test_rounding_at_the_window_edge_follows_mir_eval():
    # Notes: distances are rounded to 4 decimals first, so 50.04 ms still counts.
    assert match_notes([(1.0, 2.0, 60)], [(1.05004, 2.0, 60)], offset_ratio=None) == [(0, 0)]
    assert match_notes([(1.0, 2.0, 60)], [(1.05006, 2.0, 60)], offset_ratio=None) == []
    # Onsets and beats: compared exactly, so 50.04 ms does not count.
    assert match_events([1.0], [1.05004], window=0.05) == []


def test_note_matching_rules():
    ref = [(0.0, 1.0, 60)]
    assert match_notes(ref, [(0.02, 1.15, 60)]) == [(0, 0)]  # offset within 20% of 1 s
    assert match_notes(ref, [(0.02, 1.30, 60)]) == []  # offset too far
    assert match_notes(ref, [(0.02, 1.30, 60)], offset_ratio=None) == [(0, 0)]  # onset only
    assert match_notes(ref, [(0.02, 1.0, 61)], offset_ratio=None) == []  # wrong pitch
    short = [(0.0, 0.1, 60)]  # 20% of 0.1 s is below the 50 ms minimum
    assert match_notes(short, [(0.0, 0.15, 60)]) == [(0, 0)]


def test_score_arithmetic():
    s = Score(n_ref=10, n_est=8, n_match=6)
    assert (s.precision, s.recall) == (0.75, 0.6)
    assert s.f1 == pytest.approx(2 * 0.75 * 0.6 / 1.35)
    assert (s + Score(2, 2, 2)).n_match == 8
    assert Score().f1 == 0.0


# ---------------------------------------------------------------- document evaluation


def _shifted(doc: NoteDocument, seconds: float) -> NoteDocument:
    out = copy.deepcopy(doc)
    for t in out.tracks:
        for n in t.notes:
            n.onset += seconds
            n.offset += seconds
    return out


def test_perfect_estimate_scores_one():
    doc = make_example()
    report = evaluate(doc, copy.deepcopy(doc))
    assert report.multi_instrument.f1 == 1.0
    assert report.agnostic.f1 == 1.0
    assert all(s.f1 == 1.0 for s in report.onset_offset.values())
    assert report.beats.f1 == 1.0


def test_shift_beyond_tolerance_scores_zero():
    doc = make_example()
    report = evaluate(doc, _shifted(doc, 0.1))
    assert report.multi_instrument.n_match == 0


def test_wrong_instrument_counts_only_in_agnostic_score():
    doc = make_example()
    est = copy.deepcopy(doc)
    est.track("piano").group = "guitar"
    report = evaluate(doc, est)
    assert report.agnostic.f1 == 1.0
    assert report.onset["piano"].recall == 0.0
    assert report.onset["guitar"].precision == 0.0
    assert report.multi_instrument.f1 < 1.0


def test_missing_track_and_drum_classes():
    doc = make_example()
    est = copy.deepcopy(doc)
    est.tracks = [t for t in est.tracks if t.id != "vocals"]
    report = evaluate(doc, est)
    assert report.onset["vocals"].recall == 0.0
    assert set(report.drums) == {"kick", "snare", "hihat"}
    assert all(s.f1 == 1.0 for s in report.drums.values())
    text = format_report(report)
    assert "drums/kick" in text and "vocals +offset" in text


def test_report_json_round_trip():
    data = evaluate(make_example(), make_example()).to_dict()
    assert json.loads(json.dumps(data))["multi_instrument"]["f1"] == 1.0


# ---------------------------------------------------------------- cross-check with mir_eval


def _random_notes(rng: random.Random, n: int) -> list[Note]:
    notes = []
    for _ in range(n):
        onset = round(rng.uniform(0, 20), 3)
        notes.append(Note(onset, onset + round(rng.uniform(0.05, 1.5), 3), rng.randint(55, 60)))
    return notes


def _jitter(rng: random.Random, notes: list[Note]) -> list[Note]:
    out = []
    for n in notes:
        if rng.random() < 0.15:
            continue  # missed note
        on = n.onset + rng.uniform(-0.08, 0.08)
        out.append(Note(on, max(on + 0.01, n.offset + rng.uniform(-0.3, 0.3)), n.pitch))
    out += _random_notes(rng, len(notes) // 10)  # false alarms
    return out


@pytest.mark.parametrize("seed", range(5))
def test_note_scores_match_mir_eval(seed):
    mir_eval = pytest.importorskip("mir_eval")
    rng = random.Random(seed)
    ref = _random_notes(rng, 300)
    est = _jitter(rng, ref)

    def arrays(notes):
        intervals = np.array([[n.onset, n.offset] for n in notes])
        pitches = np.array([440.0 * 2 ** ((n.pitch - 69) / 12) for n in notes])
        return intervals, pitches

    (ri, rp), (ei, ep) = arrays(ref), arrays(est)
    theirs = len(mir_eval.transcription.match_notes(ri, rp, ei, ep, offset_ratio=None))
    assert note_score(ref, est, offset_ratio=None).n_match == theirs
    theirs = len(mir_eval.transcription.match_notes(ri, rp, ei, ep, offset_ratio=0.2))
    assert note_score(ref, est, offset_ratio=0.2).n_match == theirs


@pytest.mark.parametrize("seed", range(5))
def test_onset_scores_match_mir_eval(seed):
    mir_eval = pytest.importorskip("mir_eval")
    rng = np.random.default_rng(seed)
    ref = np.sort(rng.uniform(0, 30, 200))
    keep = rng.random(200) > 0.2  # 20% missed
    hits = ref[keep] + rng.uniform(-0.07, 0.07, keep.sum())  # some beyond the 50 ms window
    est = np.sort(np.concatenate([hits, rng.uniform(0, 30, 20)]))  # plus false alarms
    f, p, r = mir_eval.onset.f_measure(ref, est, window=0.05)
    ours = onset_score(list(ref), list(est), window=0.05)
    assert (ours.f1, ours.precision, ours.recall) == pytest.approx((f, p, r))


# ---------------------------------------------------------------- synthetic audio


def test_render_is_deterministic_and_normalized():
    doc = make_example()
    a = render(doc, 16000, seed=3)
    b = render(doc, 16000, seed=3)
    assert np.array_equal(a, b)
    assert np.max(np.abs(a)) == pytest.approx(10 ** (-1 / 20), rel=1e-4)
    assert len(a) >= 8.0 * 16000  # at least the source duration


def test_note_onsets_are_audible():
    doc = NoteDocument(tracks=[Track(id="p", group="piano", notes=[Note(1.0, 1.5, 60)])])
    sr = 16000
    audio = render(doc, sr)
    before = np.sqrt(np.mean(audio[int(0.9 * sr) : int(0.99 * sr)] ** 2))
    after = np.sqrt(np.mean(audio[int(1.0 * sr) : int(1.1 * sr)] ** 2))
    assert before < 1e-6 < after


def test_every_drum_class_makes_sound():
    notes = [Note(0.5 * i, 0.5 * i + 0.1, p) for i, p in enumerate([36, 38, 42, 46, 45, 49, 70])]
    audio = render(NoteDocument(tracks=[Track(id="d", group="drums", is_drum=True, notes=notes)]), 16000)
    for i in range(len(notes)):
        chunk = audio[int(0.5 * i * 16000) : int((0.5 * i + 0.05) * 16000)]
        assert np.max(np.abs(chunk)) > 0.01


def test_noise_option_and_wav_round_trip(tmp_path):
    doc = make_example()
    clean = render(doc, 16000)
    noisy = render(doc, 16000, noise_snr_db=10)
    assert np.std(noisy[: int(0.4 * 16000)]) > np.std(clean[: int(0.4 * 16000)])
    path = tmp_path / "x.wav"
    write_wav(path, clean, 16000)
    back, sr = read_wav(path)
    assert sr == 16000 and np.max(np.abs(back - clean)) < 1e-4


# ---------------------------------------------------------------- datasets


def test_data_root_errors_explain_the_fix(monkeypatch, tmp_path):
    monkeypatch.delenv(ENV_VAR, raising=False)
    with pytest.raises(DataError, match="setx SPACEMIDI_DATA"):
        data_root()
    assert "not set" in describe()[0]
    monkeypatch.setenv(ENV_VAR, str(tmp_path / "missing"))
    with pytest.raises(DataError, match="not a folder"):
        data_root()


def test_musdb_layout(tmp_path):
    for split, song in (("train", "A - One"), ("test", "B - Two")):
        folder = tmp_path / "raw" / "musdb18hq" / split / song
        folder.mkdir(parents=True)
        for stem in ("mixture", "drums", "bass", "other", "vocals"):
            (folder / f"{stem}.wav").write_bytes(b"")
    songs = musdb_songs(tmp_path)
    assert [(s.name, s.split) for s in songs] == [("A - One", "train"), ("B - Two", "test")]
    assert songs[0].stem("vocals").name == "vocals.wav"
    assert musdb_songs(tmp_path, "test")[0].name == "B - Two"
    with pytest.raises(ValueError):
        songs[0].stem("guitar")


SLAKH_METADATA = """UUID: abc
normalized: true
overall_gain: 0.2
stems:
  S00:
    audio_rendered: true
    inst_class: Piano
    is_drum: false
    midi_program_name: Bright Acoustic Piano
    program_num: 1
  S01:
    audio_rendered: true
    inst_class: Strings (continued)
    is_drum: false
    midi_program_name: Synth Voice
    program_num: 54
  S02:
    audio_rendered: true
    inst_class: Drums
    is_drum: true
    midi_program_name: Drums
    program_num: 0
  S03:
    audio_rendered: false
    inst_class: Bass
    is_drum: false
    program_num: 33
"""


def test_simple_yaml():
    data = parse_simple_yaml(SLAKH_METADATA)
    assert data["normalized"] is True and data["overall_gain"] == 0.2
    assert data["stems"]["S01"]["inst_class"] == "Strings (continued)"
    assert data["stems"]["S03"]["program_num"] == 33


def test_slakh_reference(tmp_path):
    track = tmp_path / "raw" / "slakh2100_flac_redux" / "test" / "Track01881"
    (track / "MIDI").mkdir(parents=True)
    (track / "metadata.yaml").write_text(SLAKH_METADATA, encoding="utf-8")
    for stem, pitch, drum in (("S00", 60, False), ("S01", 72, False), ("S02", 36, True), ("S03", 40, False)):
        doc = NoteDocument(tracks=[Track(id="x", group="drums" if drum else "other", is_drum=drum,
                                         notes=[Note(0.5, 1.0, pitch), Note(1.0, 1.5, pitch)])])
        write_midi(doc, track / "MIDI" / f"{stem}.mid")
    tracks = slakh_tracks(tmp_path)
    assert [(t.name, t.split) for t in tracks] == [("Track01881", "test")]
    ref = tracks[0].reference()
    groups = {t.id: (t.group, t.is_drum, t.program) for t in ref.tracks}
    assert groups == {"S00": ("piano", False, 1), "S01": ("other", False, 54), "S02": ("drums", True, None)}
    assert all(len(t.notes) == 2 for t in ref.tracks)  # S03 was not rendered, so it is left out


def test_testset_items(tmp_path):
    item = tmp_path / "testset" / "song-one"
    item.mkdir(parents=True)
    (item / "mix.wav").write_bytes(b"")
    (item / "stem.wav").write_bytes(b"")
    save(make_example(), item / "reference.notes.json")
    (tmp_path / "testset" / "no-audio").mkdir()
    items = list_testset(tmp_path)
    assert [i.name for i in items] == ["song-one"]
    assert items[0].audio.name == "mix.wav"
    assert len(items[0].load_reference().tracks) == 3
    assert any("1 with a reference" in line for line in describe(tmp_path))


# ---------------------------------------------------------------- CLI


def test_cli_eval_and_synth(tmp_path, capsys):
    ref = tmp_path / "ref.notes.json"
    save(make_example(), ref)
    mid = tmp_path / "est.mid"
    write_midi(make_example(), mid)
    out = tmp_path / "scores.json"
    assert main(["eval", str(mid), str(ref), "--json", str(out)]) == 0
    assert "1.000" in capsys.readouterr().out
    assert json.loads(out.read_text())["multi_instrument"]["f1"] == 1.0
    wav = tmp_path / "ref.wav"
    assert main(["synth", str(ref), str(wav), "--sr", "16000"]) == 0
    assert read_wav(wav)[1] == 16000
    assert main(["eval", str(tmp_path / "missing.json"), str(ref)]) == 1


def test_cli_mir_eval_cross_check(tmp_path, capsys):
    pytest.importorskip("mir_eval")
    ref = tmp_path / "ref.notes.json"
    save(make_example(), ref)
    est = tmp_path / "est.notes.json"
    shifted = _shifted(make_example(), 0.03)
    shifted.track("piano").notes.pop()  # a missed note, so the scores are not all 1.0
    save(shifted, est)
    out = tmp_path / "scores.json"
    assert main(["eval", str(est), str(ref), "--mir-eval", "--json", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "Cross-check" in printed and "MISMATCH" not in printed
    assert all(row["agrees"] for row in json.loads(out.read_text())["mir_eval"].values())


def test_cli_mir_eval_missing(tmp_path, capsys, monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "mir_eval", None)  # makes `import mir_eval` fail
    ref = tmp_path / "ref.notes.json"
    save(make_example(), ref)
    assert main(["eval", str(ref), str(ref), "--mir-eval"]) == 1
    assert "uv sync" in capsys.readouterr().out
