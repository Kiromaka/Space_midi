"""Onset detection, the onset benchmark and their CLI commands."""

from __future__ import annotations

import json
import random

import numpy as np
import pytest

from spacemidi.cli import main
from spacemidi.dsp import (
    ONSET_PRESETS,
    OnsetConfig,
    detect_onsets,
    envelope_from_bands,
    mel_filterbank,
    onset_config,
    onset_envelope,
    peak_pick,
    pick_onsets,
    power_to_db,
    stft,
)
from spacemidi.eval import onset_score, reference_onsets, render, write_wav
from spacemidi.eval.bench import (
    expand_grid,
    format_results,
    parse_grid,
    run_onset_bench,
    save_run,
    slakh_items,
)
from spacemidi.eval.datasets import ENV_VAR
from spacemidi.midi import read_midi, write_midi
from spacemidi.notes import Note, NoteDocument, Track

SR = 22050


def random_song(seed: int = 0, seconds: float = 30.0) -> tuple[NoteDocument, list[float]]:
    """Notes from all groups at random times at least 120 ms apart; returns the onset times too."""
    rng = random.Random(seed)
    groups = ["piano", "guitar", "bass", "other", "drums", "vocals"]
    notes: dict[str, list[Note]] = {g: [] for g in groups}
    onsets, t = [], 0.3
    while t < seconds:
        onsets.append(t)
        group = rng.choice(groups)
        if group == "drums":
            notes[group].append(Note(t, t + 0.1, rng.choice([36, 38, 42, 46, 49])))
        else:
            notes[group].append(Note(t, t + rng.uniform(0.1, 0.8), rng.randint(40, 84)))
        t += rng.uniform(0.12, 0.6)
    tracks = [Track(id=g, group=g, is_drum=g == "drums", notes=n) for g, n in notes.items() if n]
    return NoteDocument(tracks=tracks), onsets


# ---------------------------------------------------------------- peak picking


def test_peak_pick_rules():
    env = np.array([0, 1, 0, 0, 3, 2, 0, 0, 0, 2, 2.5, 0, 0, 0, 0])
    assert peak_pick(env, 1, 2, 2, 2, 0.0, 0).tolist() == [1, 4, 10]  # window [n-1, n+1]: 9 is not a max
    assert peak_pick(env, 1, 1, 2, 2, 0.0, 0).tolist() == [1, 4, 9, 10]  # window [n-1, n]: now it is
    assert peak_pick(env, 1, 2, 2, 2, 1.2, 0).tolist() == [4, 10]  # 1 is not 1.2 above its local mean
    assert peak_pick(env, 1, 4, 2, 2, 0.0, 0).tolist() == [4, 10]  # window [n-1, n+3] reaches frame 4
    assert peak_pick(env, 1, 2, 2, 2, 0.0, 5).tolist() == [1, 10]  # 4 comes too soon after 1
    assert peak_pick(np.zeros(10), 1, 1, 1, 1, 0.0, 0).tolist() == []
    assert peak_pick(np.zeros(0), 1, 1, 1, 1, 0.0, 0).tolist() == []
    with pytest.raises(ValueError):
        peak_pick(env, 1, 0, 1, 1, 0.0, 0)


def test_envelope_is_positive_flux_with_lag_and_max_filter():
    s = np.array([[0.0, 1.0, 1.0, 3.0], [2.0, 0.0, 1.0, 1.0]])
    assert np.allclose(envelope_from_bands(s, 1, 1), [0.0, 0.5, 0.5, 1.0])
    assert np.allclose(envelope_from_bands(s, 2, 1), [0.0, 0.0, 0.5, 1.5])
    # max filter over 3 bands: each band's reference is max(band 0, band 1) here
    assert np.allclose(envelope_from_bands(s, 1, 3), [0.0, 0.0, 0.0, 1.0])


# ---------------------------------------------------------------- detection


@pytest.mark.parametrize("preset", ["flux-stem", "superflux-stem", "librosa"])
@pytest.mark.parametrize("seed", [0, 1])
def test_presets_find_onsets_in_synthetic_songs(preset, seed):
    doc, onsets = random_song(seed)
    x = render(doc, SR).astype(np.float64)
    est = detect_onsets(x, SR, preset)
    score = onset_score(onsets, est.tolist())
    assert score.f1 > 0.9, score.to_dict()


@pytest.mark.parametrize("preset", ["flux", "superflux"])
def test_mix_presets_still_work_on_sparse_audio(preset):
    # The mix presets have very low absolute thresholds (tuned on dense Slakh
    # mixes), so on sparse, clean synthetic audio they fire too often; this
    # only checks that they stay usable (F1 measured: flux 0.84, superflux 0.78).
    doc, onsets = random_song(0)
    est = detect_onsets(render(doc, SR).astype(np.float64), SR, preset)
    score = onset_score(onsets, est.tolist())
    assert score.recall > 0.9 and score.f1 > 0.7, score.to_dict()


def test_offset_shifts_detections():
    doc, _ = random_song(0, 5.0)
    x = render(doc, SR).astype(np.float64)
    a = detect_onsets(x, SR, "librosa")
    b = detect_onsets(x, SR, "librosa", offset=0.02)
    assert np.allclose(b - a, 0.02)


def test_config_changes_and_presets():
    cfg = onset_config("superflux", delta="0.2", lag="3", fmax="none", normalize="false")
    assert cfg.delta == 0.2 and cfg.lag == 3 and cfg.fmax is None and cfg.normalize is False
    assert ONSET_PRESETS["superflux"].delta == 0.01  # presets are not modified
    with pytest.raises(KeyError, match="unknown onset preset"):
        onset_config("fluxx")
    with pytest.raises(KeyError):
        onset_config("flux", nonsense=1)
    assert isinstance(onset_config(OnsetConfig(), hop=256), OnsetConfig)
    assert onset_config("flux").to_dict()["bands"] == "mel"
    with pytest.raises(ValueError):
        onset_envelope(np.zeros(SR), SR, onset_config("flux", bands="bark"))


def test_silence_gives_no_onsets():
    for preset in ONSET_PRESETS:
        assert len(detect_onsets(np.zeros(SR * 2), SR, preset)) == 0


# ---------------------------------------------------------------- benchmark


SLAKH_META = """UUID: x
stems:
  S00:
    audio_rendered: true
    inst_class: Piano
    is_drum: false
    midi_program_name: Acoustic Grand Piano
    program_num: 0
  S01:
    audio_rendered: true
    inst_class: Drums
    is_drum: true
    midi_program_name: Drums
    program_num: 0
"""


def fake_slakh(root, n_tracks: int = 2) -> None:
    """A tiny Slakh layout. The .flac files hold WAV data, which every reader we use detects by header."""
    for k in range(n_tracks):
        doc, _ = random_song(10 + k, 8.0)
        piano = Track(id="S00", group="piano", notes=[n for t in doc.tracks if not t.is_drum for n in t.notes])
        drums = Track(id="S01", group="drums", is_drum=True, notes=doc.track("drums").notes)
        folder = root / "raw" / "slakh2100_flac_redux" / "test" / f"Track0{k:04d}"
        (folder / "MIDI").mkdir(parents=True)
        (folder / "stems").mkdir()
        (folder / "metadata.yaml").write_text(SLAKH_META, encoding="utf-8")
        for track in (piano, drums):
            single = NoteDocument(tracks=[track])
            write_midi(single, folder / "MIDI" / f"{track.id}.mid")
            write_wav(folder / "stems" / f"{track.id}.flac", render(single, SR), SR)
        both = NoteDocument(tracks=[piano, drums])
        write_midi(both, folder / "all_src.mid")
        write_wav(folder / "mix.flac", render(both, SR), SR)


def test_reference_onsets_merge_close_notes():
    doc = NoteDocument(tracks=[Track(id="a", group="piano", notes=[Note(1.0, 2.0, 60), Note(1.02, 2.0, 64)]),
                               Track(id="b", group="bass", notes=[Note(1.04, 2.0, 40), Note(3.0, 3.5, 40)])])
    assert reference_onsets(doc, 0.03) == [1.0, 1.04, 3.0]
    assert reference_onsets(doc, 0.0) == [1.0, 1.02, 1.04, 3.0]


def test_grid_helpers():
    grid = parse_grid(["delta=0.05, 0.1", "lag=2"])
    assert grid == {"delta": ["0.05", "0.1"], "lag": ["2"]}
    assert expand_grid(grid) == [{"delta": "0.05", "lag": "2"}, {"delta": "0.1", "lag": "2"}]
    assert expand_grid({}) == [{}]
    with pytest.raises(ValueError):
        parse_grid(["delta"])


@pytest.mark.parametrize("source", ["mix", "stems"])
@pytest.mark.parametrize("audio", ["real", "synth"])
def test_bench_on_fake_slakh(tmp_path, source, audio):
    fake_slakh(tmp_path)
    items = slakh_items("test", source, root=tmp_path)
    assert len(items) == (2 if source == "mix" else 4)
    results, rows = run_onset_bench(items, "superflux", {"delta": ["0.05", "0.1"]}, audio=audio, progress=None)
    assert [r.params for r in results] == [{"delta": "0.05"}, {"delta": "0.1"}]
    assert all(r.total.f1 > 0.8 for r in results)
    assert set(results[0].groups) == ({"mix"} if source == "mix" else {"piano", "drums"})
    path = save_run({"task": "onsets", "preset": "superflux", "dataset": "slakh", "split": "test",
                     "source": source, "audio": audio}, results, rows, tmp_path / "runs")
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["results"][0]["total"]["n_ref"] == results[0].total.n_ref
    assert "median_ms" in saved["results"][0]["timing"] and len(saved["items"]) == len(items)
    assert "F1" in format_results(results)


def test_bench_in_parallel_matches_serial(tmp_path):
    fake_slakh(tmp_path)
    items = slakh_items("test", "mix", root=tmp_path)
    serial, _ = run_onset_bench(items, "flux", progress=None)
    parallel, _ = run_onset_bench(items, "flux", jobs=2, progress=None)
    assert serial[0].total == parallel[0].total


def test_bench_skips_tracks_with_broken_midi(tmp_path):
    fake_slakh(tmp_path, 2)
    broken = tmp_path / "raw" / "slakh2100_flac_redux" / "test" / "Track00001" / "MIDI" / "S00.mid"
    broken.write_bytes(b"MThd" + bytes.fromhex("00000006 0000 0001 0060") + b"MTrk" + bytes.fromhex("00000004 00 3c 64 00"))
    skipped: list = []
    items = slakh_items("test", "mix", root=tmp_path, skipped=skipped)
    assert [i.name for i in items] == ["Track00000"]
    assert skipped and skipped[0][0] == "Track00001" and "status byte" in skipped[0][1]


def test_bench_rejects_unknown_setting(tmp_path):
    fake_slakh(tmp_path, 1)
    with pytest.raises(KeyError):
        run_onset_bench(slakh_items("test", "mix", root=tmp_path), "flux", {"deltaa": ["1"]}, progress=None)


# ---------------------------------------------------------------- CLI


def test_cli_onsets(tmp_path, capsys):
    doc, onsets = random_song(3, 6.0)
    wav = tmp_path / "song.wav"
    write_wav(wav, render(doc, SR), SR)
    txt, mid = tmp_path / "on.txt", tmp_path / "on.mid"
    assert main(["onsets", str(wav), "--preset", "flux-stem", "--out", str(txt), "--midi", str(mid)]) == 0
    times = [float(line) for line in txt.read_text().split()]
    assert onset_score(onsets, times).f1 > 0.9
    clicks = read_midi(mid).tracks[0]
    assert clicks.is_drum and len(clicks.notes) == len(times)
    assert abs(clicks.notes[0].onset - times[0]) < 0.002
    assert main(["onsets", str(wav), "--preset", "flux"]) == 0
    assert "onsets in" in capsys.readouterr().out
    assert main(["onsets", str(tmp_path / "missing.wav")]) == 1
    assert main(["onsets", str(wav), "--set", "bogus=1"]) == 1


def test_cli_bench(tmp_path, capsys, monkeypatch):
    fake_slakh(tmp_path, 1)
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    out = tmp_path / "runs"
    assert main(["bench", "onsets", "--limit", "1", "--set", "delta=0.05,0.1", "--out-dir", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "saved" in printed and "[1/1]" in printed
    run = json.loads(next(out.glob("*.json")).read_text(encoding="utf-8"))
    assert run["task"] == "onsets" and len(run["results"]) == 2 and run["grid"] == {"delta": ["0.05", "0.1"]}
    assert main(["bench", "onsets", "--split", "train", "--out-dir", str(out)]) == 1  # no train tracks
    monkeypatch.delenv(ENV_VAR)
    assert main(["bench", "onsets"]) == 1


# ---------------------------------------------------------------- librosa cross-checks


@pytest.fixture
def librosa():
    return pytest.importorskip("librosa")


def _mel_config(**changes) -> OnsetConfig:
    return onset_config("librosa", **changes)


def test_flux_envelope_matches_librosa(librosa):
    doc, _ = random_song(4, 10.0)
    x = render(doc, SR).astype(np.float64)
    for lag, max_size in ((1, 1), (2, 3)):
        ours = onset_envelope(x, SR, _mel_config(lag=lag, max_size=max_size))
        theirs = librosa.onset.onset_strength(y=x, sr=SR, lag=lag, max_size=max_size)
        # librosa delays its envelope by n_fft // (2 * hop) = 2 frames; ours is not delayed
        assert np.allclose(theirs[2:], ours[:-2], rtol=1e-4, atol=1e-4)


def test_peak_pick_matches_librosa(librosa):
    rng = np.random.default_rng(5)
    for trial in range(20):
        env = np.abs(rng.standard_normal(300)) * (rng.random(300) < 0.3)
        params = dict(pre_max=int(rng.integers(0, 4)), post_max=int(rng.integers(1, 4)),
                      pre_avg=int(rng.integers(0, 6)), post_avg=int(rng.integers(1, 6)),
                      delta=float(rng.uniform(0.0, 0.5)), wait=int(rng.integers(0, 4)))
        theirs = librosa.util.peak_pick(env, **params)
        assert peak_pick(env, **params).tolist() == list(theirs), (trial, params)


def test_flux_detection_matches_librosa(librosa):
    doc, _ = random_song(6, 10.0)
    x = render(doc, SR).astype(np.float64)
    env = librosa.onset.onset_strength(y=x, sr=SR)
    theirs = librosa.onset.onset_detect(onset_envelope=env, sr=SR, units="time")
    ours = pick_onsets(env, SR, onset_config("librosa"))
    assert np.allclose(ours, theirs)
    spec = power_to_db(mel_filterbank(SR, 2048) @ np.abs(stft(x)) ** 2)
    assert spec.shape[0] == 128
