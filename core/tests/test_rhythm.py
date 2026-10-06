"""Tempo, beats and downbeats (algos/rhythm), their metrics, benchmark and CLI."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rhythm_songs import grid_song
from spacemidi.algos.rhythm import (
    RhythmConfig,
    autocorrelation,
    choose_downbeats,
    estimate_rhythm,
    estimate_tempo,
    rhythm_features,
    track_beats,
)
from spacemidi.cli import main
from spacemidi.eval import (
    beat_measures,
    metrical_variants,
    render,
    tempo_accuracy,
    tempo_from_beats,
    trim_beats,
    write_wav,
)
from spacemidi.eval.bench import save_run, slakh_items
from spacemidi.eval.bench_rhythm import format_rhythm_results, grid_offset, run_rhythm_bench
from spacemidi.eval.datasets import ENV_VAR
from spacemidi.midi import read_midi, write_midi
from spacemidi.notes import NoteDocument

SR = 22050


def _audio(doc: NoteDocument, noise: float | None = 30.0) -> np.ndarray:
    return render(doc, SR, noise_snr_db=noise).astype(np.float64)


# ---------------------------------------------------------------- metrics


def test_trim_and_variants():
    beats = [i * 0.5 for i in range(20)]
    assert trim_beats(beats) == beats[10:]
    v = metrical_variants([0.0, 1.0, 2.0, 3.0])
    assert v["double"] == [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    assert v["half_odd"] == [0.0, 2.0] and v["half_even"] == [1.0, 3.0] and v["offbeat"] == [0.5, 1.5, 2.5]


def test_beat_measures_and_tempo():
    ref = [i * 0.5 for i in range(60)]
    assert beat_measures(ref, ref)["f1"] == 1.0
    half = beat_measures(ref, ref[::2])
    assert half["f1"] < 0.7 and half["f1_any"] == 1.0 and half["best_level"] == "half_odd"
    off = beat_measures(ref, [t + 0.25 for t in ref])
    assert off["f1"] == 0.0 and off["best_level"] == "offbeat"
    assert tempo_from_beats(ref) == pytest.approx(120.0)
    assert tempo_from_beats([1.0]) is None
    assert tempo_accuracy(120.0, 123.0) == (True, True)
    assert tempo_accuracy(120.0, 60.5) == (False, True)
    assert tempo_accuracy(120.0, 40.2) == (False, True)
    assert tempo_accuracy(120.0, 100.0) == (False, False)


# ---------------------------------------------------------------- building blocks


def test_autocorrelation_matches_direct_computation():
    x = np.random.default_rng(0).standard_normal(500)
    acf = autocorrelation(x, 40)
    y = x - x.mean()
    direct = [float(np.dot(y[: len(y) - k], y[k:])) for k in range(41)]
    assert np.allclose(acf, direct)


@pytest.mark.parametrize("bpm", [70.0, 100.0, 120.0, 140.0])
def test_tempo_of_a_pulse_train(bpm):
    fps = SR / 512
    env = np.zeros(int(30 * fps))
    for t in np.arange(0.3, 30.0, 60.0 / bpm):
        env[int(round(t * fps))] = 1.0
    est = estimate_tempo(env, fps)
    assert est.bpm == pytest.approx(bpm, rel=0.02)
    assert est.candidates[0][1] == 1.0


def test_tempo_prior_picks_the_metrical_level():
    fps = SR / 512
    env = np.zeros(int(30 * fps))
    for t in np.arange(0.3, 30.0, 60.0 / 120.0):
        env[int(round(t * fps))] = 1.0
    assert estimate_tempo(env, fps, start_bpm=60.0, std_octaves=0.3).bpm == pytest.approx(60.0, rel=0.02)


def test_dp_follows_jittered_pulses_and_bridges_gaps():
    fps = 100.0
    rng = np.random.default_rng(1)
    truth = np.arange(1.0, 29.0, 0.5)
    env = np.zeros(3000)
    for i, t in enumerate(truth):
        if i % 7 != 3:  # every 7th pulse missing
            env[int(round((t + rng.uniform(-0.01, 0.01)) * fps))] = 1.0
    frames = track_beats(env, fps, 120.0)
    assert beat_measures(truth.tolist(), (frames / fps).tolist(), trim=False)["f1"] > 0.95
    assert len(track_beats(np.zeros(500), fps, 120.0)) == 0


def test_downbeat_choice_prefers_four_unless_three_is_clearly_better():
    accent = np.tile([3.0, 1.0, 1.0, 1.0], 8)
    choice = choose_downbeats({"accent": np.roll(accent, 1)}, {"accent": 1.0})
    assert (choice.meter, choice.phase) == (4, 1)
    waltz = np.tile([3.0, 1.0, 1.0], 10)
    assert choose_downbeats({"accent": waltz}, {"accent": 1.0}).meter == 3
    assert choose_downbeats({"accent": waltz}, {"accent": 1.0}, meters=(4,)).meter == 4


# ---------------------------------------------------------------- whole pipeline on synthetic songs


@pytest.mark.parametrize("bpm,meter", [(120.0, 4), (90.0, 4), (150.0, 4), (100.0, 3)])
def test_rhythm_of_grid_songs(bpm, meter):
    doc = grid_song(bpm, bars=20, meter=meter, seed=2, swing_ms=10.0)
    r = estimate_rhythm(_audio(doc), SR)
    assert tempo_accuracy(bpm, r.bpm)[0], r.bpm
    assert beat_measures(doc.tempo.beats, r.beats)["f1"] > 0.95
    assert beat_measures(doc.tempo.downbeats, r.downbeats)["f1"] > 0.9
    assert r.time_signature == (meter, 4)
    assert r.tempo_map().beats == r.beats


def test_fast_song_is_tracked_at_half_tempo():
    # 180 BPM lies 0.6 octaves above the 120 BPM prior: the tracker picks 90,
    # an octave error that the "any metrical level" score forgives.
    doc = grid_song(180.0, bars=24, seed=3)
    r = estimate_rhythm(_audio(doc), SR)
    assert tempo_accuracy(180.0, r.bpm) == (False, True)
    assert beat_measures(doc.tempo.beats, r.beats)["f1_any"] > 0.95
    assert tempo_accuracy(180.0, estimate_rhythm(_audio(doc), SR, start_bpm=170.0).bpm)[0]


def test_config_and_silence():
    cfg = RhythmConfig().with_(tightness="400", triple="false", start_bpm="100")
    assert cfg.tightness == 400.0 and cfg.triple is False and cfg.start_bpm == 100.0
    with pytest.raises(KeyError):
        RhythmConfig().with_(tempo=1)
    r = estimate_rhythm(np.zeros(SR * 5), SR)
    assert r.beats == [] and r.downbeats == []
    feat = rhythm_features(_audio(grid_song(bars=4)), SR)
    assert feat.env.shape == feat.low_env.shape and feat.chroma.shape == (12, len(feat.env))


# ---------------------------------------------------------------- benchmark and CLI


SLAKH_META = """UUID: x
stems:
  S00:
    audio_rendered: true
    inst_class: Piano
    is_drum: false
    program_num: 0
  S01:
    audio_rendered: true
    inst_class: Bass
    is_drum: false
    program_num: 33
  S02:
    audio_rendered: true
    inst_class: Drums
    is_drum: true
    program_num: 0
"""


def fake_slakh_with_tempo(root, tempos=(120.0, 95.0)) -> list[NoteDocument]:
    docs = []
    for k, bpm in enumerate(tempos):
        doc = grid_song(bpm, bars=16, seed=k)
        folder = root / "raw" / "slakh2100_flac_redux" / "test" / f"Track0{k:04d}"
        (folder / "MIDI").mkdir(parents=True)
        (folder / "metadata.yaml").write_text(SLAKH_META, encoding="utf-8")
        for stem, track in zip(("S00", "S01", "S02"), (doc.track("keys"), doc.track("bass"), doc.track("drums"))):
            write_midi(NoteDocument(tracks=[track], tempo=doc.tempo), folder / "MIDI" / f"{stem}.mid")
        write_midi(doc, folder / "all_src.mid")
        write_wav(folder / "mix.flac", render(doc, SR, noise_snr_db=30.0), SR)  # WAV data; readers detect it by header
        docs.append(doc)
    return docs


def test_grid_offset_flags_a_shifted_reference():
    doc = grid_song(120.0, bars=8, seed=6)
    offset, conc = grid_offset(doc)
    assert abs(offset) < 0.01 and conc > 0.9
    shifted = grid_song(120.0, bars=8, seed=6)
    shifted.tempo.beats = [t + 0.1 for t in shifted.tempo.beats]  # grid 0.2 beat late
    offset, conc = grid_offset(shifted)
    assert offset == pytest.approx(-0.2, abs=0.02) and conc > 0.9
    assert grid_offset(NoteDocument()) == (0.0, 0.0)


def test_reference_tempo_survives_the_midi_round_trip(tmp_path):
    docs = fake_slakh_with_tempo(tmp_path)
    items = slakh_items("test", "mix", root=tmp_path)
    for doc, item in zip(docs, items):
        assert np.allclose(item.reference.tempo.beats[: len(doc.tempo.beats)], doc.tempo.beats, atol=1e-3)
        assert np.allclose(item.reference.tempo.downbeats[:4], doc.tempo.downbeats[:4], atol=1e-3)


@pytest.mark.parametrize("audio", ["real", "synth"])
def test_rhythm_bench_on_fake_slakh(tmp_path, audio):
    fake_slakh_with_tempo(tmp_path)
    items = slakh_items("test", "mix", root=tmp_path)
    results, rows = run_rhythm_bench(items, {"tightness": ["100", "400"]}, audio=audio, progress=None)
    assert [p.params for p in results] == [{"tightness": "100"}, {"tightness": "400"}]
    for p in results:
        assert len(p.rows) == 2
        assert p.mean("beat_f1") > 0.9 and p.mean("tempo_acc1") == 1.0 and p.mean("downbeat_f1") > 0.8
    path = save_run({"task": "beats", "preset": "rhythm", "dataset": "slakh", "split": "test", "source": "mix",
                     "audio": audio}, [p.to_dict() for p in results], rows, tmp_path / "runs")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["results"][0]["means"]["tempo_acc1"] == 1.0 and len(saved["items"]) == 2
    assert saved["results"][0]["n_aligned"] == 2
    assert "beat F" in format_rhythm_results(results)


def test_rhythm_bench_in_parallel(tmp_path):
    fake_slakh_with_tempo(tmp_path)
    items = slakh_items("test", "mix", root=tmp_path)
    serial, _ = run_rhythm_bench(items, progress=None)
    parallel, _ = run_rhythm_bench(items, jobs=2, progress=None)
    assert serial[0].rows == parallel[0].rows
    with pytest.raises(KeyError):
        run_rhythm_bench(items, {"tightnes": ["1"]}, progress=None)


def test_cli_beats(tmp_path, capsys):
    doc = grid_song(110.0, bars=16, seed=4)
    wav = tmp_path / "song.wav"
    write_wav(wav, render(doc, SR, noise_snr_db=30.0), SR)
    txt, mid = tmp_path / "beats.txt", tmp_path / "beats.mid"
    assert main(["beats", str(wav), "--out", str(txt), "--midi", str(mid)]) == 0
    assert "110." in capsys.readouterr().out
    lines = [line.split("\t") for line in txt.read_text().splitlines()]
    times = [float(a) for a, _ in lines]
    assert beat_measures(doc.tempo.beats, times)["f1"] > 0.95
    assert sum(b == "1" for _, b in lines) == len([t for t in doc.tempo.downbeats if t >= times[0] - 0.1])
    back = read_midi(mid)
    assert np.allclose(back.tempo.beats[: len(times)], times, atol=2e-3)
    assert len(back.tracks[0].notes) == len(times)
    assert main(["beats", str(tmp_path / "missing.wav")]) == 1


def test_cli_bench_beats(tmp_path, capsys, monkeypatch):
    fake_slakh_with_tempo(tmp_path, tempos=(120.0,))
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    out = tmp_path / "runs"
    assert main(["bench", "beats", "--set", "harmonics=1,2", "--out-dir", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "saved" in printed and "tempo acc1" in printed
    run = json.loads(next(out.glob("*-beats-*.json")).read_text(encoding="utf-8"))
    assert run["task"] == "beats" and len(run["results"]) == 2
    assert main(["bench", "beats", "--split", "train", "--out-dir", str(out)]) == 1


# ---------------------------------------------------------------- librosa cross-check


def test_tempo_and_beats_agree_with_librosa():
    librosa = pytest.importorskip("librosa")
    doc = grid_song(128.0, bars=24, seed=5, swing_ms=8.0)
    x = _audio(doc)
    ours = estimate_rhythm(x, SR)
    tempo, frames = librosa.beat.beat_track(y=x, sr=SR, hop_length=512)
    theirs = librosa.frames_to_time(frames, sr=SR, hop_length=512)
    assert tempo_accuracy(float(np.atleast_1d(tempo)[0]), ours.bpm)[0]
    # librosa delays its onset envelope by two frames; compare with a matching tolerance
    assert beat_measures(list(theirs), ours.beats, window=0.07)["f1"] > 0.9
