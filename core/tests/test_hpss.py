"""Harmonic-percussive separation (dsp/hpss), SDR, the HPSS benchmark and CLI."""

from __future__ import annotations

import json

import numpy as np
import pytest

from rhythm_songs import grid_song
from spacemidi.algos.rhythm import estimate_rhythm
from spacemidi.cli import main
from spacemidi.dsp import detect_onsets, hpss, hpss_masks, load_audio, onset_config, softmask, stft
from spacemidi.dsp.onset import hpss_part
from spacemidi.eval import beat_measures, framewise_sdr, read_wav, render, sdr, write_wav
from spacemidi.eval.bench_hpss import (
    format_hpss_results,
    hpss_settings,
    musdb_items,
    run_hpss_bench,
    score_split,
)
from spacemidi.eval.datasets import ENV_VAR
from spacemidi.notes import NoteDocument

SR = 22050


def tone_and_clicks(seconds: float = 3.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """A held two-note chord and a train of short noise bursts, returned separately."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    tone = 0.3 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 330.0 * t)
    clicks = np.zeros_like(t)
    burst = int(0.01 * SR)
    for start in np.arange(0.25, seconds - 0.1, 0.25):
        i = int(start * SR)
        clicks[i : i + burst] += rng.standard_normal(burst) * np.exp(-np.arange(burst) / (0.002 * SR))
    return tone, 0.5 * clicks


# ---------------------------------------------------------------- masks


def test_softmask():
    x = np.array([0.0, 1.0, 3.0, 0.0])
    ref = np.array([0.0, 1.0, 1.0, 2.0])
    assert np.allclose(softmask(x, ref), [0.0, 0.5, 0.9, 0.0])
    assert np.allclose(softmask(x, ref, split_zeros=True), [0.5, 0.5, 0.9, 0.0])
    assert np.allclose(softmask(x, ref, power=1.0), [0.0, 0.5, 0.75, 0.0])
    assert softmask(x, ref, power=np.inf).tolist() == [0.0, 0.0, 1.0, 0.0]
    assert np.allclose(softmask([1e-200], [1e-200]), [0.5])  # tiny but non-zero: no underflow to 0/0
    with pytest.raises(ValueError):
        softmask([-1.0], [1.0])


def test_masks_sum_to_one_without_margin():
    mag = np.abs(stft(np.sum(tone_and_clicks(), axis=0)))
    mag[:, :5] = 0.0  # silent frames: the masks split them evenly
    mask_h, mask_p = hpss_masks(mag, 17, 17)
    assert np.allclose(mask_h + mask_p, 1.0)
    assert np.all((mask_h >= 0) & (mask_h <= 1))
    mask_h2, mask_p2 = hpss_masks(mag, 17, 17, margin=2.0)
    assert np.all(mask_h2 + mask_p2 <= 1.0 + 1e-12)
    assert np.all(mask_h2 <= mask_h + 1e-12) and np.all(mask_p2 <= mask_p + 1e-12)


def test_masks_reject_bad_settings():
    mag = np.ones((10, 10))
    for kwargs in ({"harm_frames": 0}, {"perc_bins": 0}, {"margin": 0.5}):
        with pytest.raises(ValueError):
            hpss_masks(mag, **kwargs)


def test_masks_on_lines():
    """A horizontal line is harmonic, a vertical line percussive."""
    mag = np.full((101, 101), 1e-3)
    mag[30, :] = 1.0
    mag[:, 70] = 1.0
    mask_h, mask_p = hpss_masks(mag, 15, 15)
    assert mask_h[30, 10] > 0.99 and mask_p[30, 10] < 0.01
    assert mask_p[80, 70] > 0.99 and mask_h[80, 70] < 0.01


# ---------------------------------------------------------------- separation


def test_hpss_reconstructs_the_input():
    x = np.sum(tone_and_clicks(), axis=0)
    parts = hpss(x)
    assert parts.harmonic.shape == x.shape
    assert np.max(np.abs(parts.harmonic + parts.percussive - x)) < 1e-9
    assert np.max(np.abs(parts.residual)) < 1e-9
    noisy = x + 0.05 * np.random.default_rng(3).standard_normal(len(x))  # noise is neither: residual
    wide = hpss(noisy, margin=3.0)
    assert np.sum(wide.residual**2) > 0.3 * 0.05**2 * len(x)
    assert np.allclose(wide.harmonic + wide.percussive + wide.residual, noisy)


def test_hpss_separates_tone_from_clicks():
    tone, clicks = tone_and_clicks()
    x = tone + clicks
    parts = hpss(x)
    assert sdr(tone, parts.harmonic) > sdr(tone, x) + 3.0
    assert sdr(clicks, parts.percussive) > sdr(clicks, x) + 20.0
    # a larger margin gives purer (if quieter) parts
    wide = hpss(x, margin=4.0)
    leak = np.sum(wide.harmonic * clicks) / np.sum(clicks**2)
    assert abs(leak) < abs(np.sum(parts.harmonic * clicks) / np.sum(clicks**2))


# ---------------------------------------------------------------- SDR


def test_sdr():
    rng = np.random.default_rng(1)
    s = rng.standard_normal(1000)
    assert sdr(s, s) > 90
    assert sdr(s, np.zeros_like(s)) == pytest.approx(0.0, abs=1e-6)
    assert sdr(s, 0.5 * s) == pytest.approx(20 * np.log10(2), abs=1e-6)
    with pytest.raises(ValueError):
        sdr(s, s[:-1])


def test_framewise_sdr():
    s = np.ones(4 * SR)
    e = s.copy()
    e[: SR] = 0.0  # one bad second: 0 dB there, perfect elsewhere
    assert framewise_sdr(s, e, SR) > 90
    s[2 * SR :] = 0.0  # silent windows are skipped
    e[2 * SR :] = 5.0
    perfect = sdr(np.ones(SR), np.ones(SR))
    assert framewise_sdr(s, e, SR) == pytest.approx(np.median([0.0, perfect]))
    assert np.isnan(framewise_sdr(np.zeros(SR), np.ones(SR), SR))
    with pytest.raises(ValueError):
        framewise_sdr(s, e[:-1], SR)


# ---------------------------------------------------------------- onsets and beats on the separated parts


def test_onset_source_option():
    tone, clicks = tone_and_clicks(4.0)
    x = tone + clicks
    expected = np.arange(0.25, 3.9, 0.25)
    found = detect_onsets(x, SR, onset_config("flux-stem", source="percussive"))
    hits = sum(np.min(np.abs(found - t)) < 0.05 for t in expected)
    assert hits >= len(expected) - 1
    assert len(found) <= len(expected) + 2
    mag = np.abs(stft(x))
    assert np.allclose(hpss_part(mag, "percussive") + hpss_part(mag, "harmonic"), mag)
    with pytest.raises(ValueError):
        hpss_part(mag, "vocals")
    with pytest.raises(ValueError):
        detect_onsets(x, SR, onset_config("flux", source="vocals"))


def test_rhythm_with_hpss():
    doc = grid_song(116.0, bars=16, seed=7, swing_ms=8.0)
    x = render(doc, SR, noise_snr_db=30.0).astype(np.float64)
    r = estimate_rhythm(x, SR, hpss=True)
    assert beat_measures(doc.tempo.beats, r.beats)["f1"] > 0.95
    assert beat_measures(doc.tempo.downbeats, r.downbeats)["f1"] > 0.9


# ---------------------------------------------------------------- benchmark and CLI


def fake_musdb(root, n_songs: int = 2) -> None:
    """A tiny MUSDB18-HQ layout: mixture.wav and drums.wav per song."""
    for k in range(n_songs):
        doc = grid_song(100.0 + 10 * k, bars=4, seed=k)
        drums = 0.4 * render(NoteDocument(tracks=[doc.track("drums")]), SR)
        rest = 0.4 * render(NoteDocument(tracks=[t for t in doc.tracks if not t.is_drum]), SR)
        n = min(len(drums), len(rest))
        folder = root / "raw" / "musdb18hq" / "test" / f"Song {k}"
        folder.mkdir(parents=True)
        write_wav(folder / "mixture.wav", drums[:n] + rest[:n], SR)
        write_wav(folder / "drums.wav", drums[:n], SR)


def test_hpss_settings():
    assert hpss_settings()["harm_frames"] == 31
    p = hpss_settings({"harm_frames": "17", "margin": "2"})
    assert p["harm_frames"] == 17 and isinstance(p["harm_frames"], int) and p["margin"] == 2.0
    with pytest.raises(KeyError):
        hpss_settings({"kernel": 3})
    with pytest.raises(ValueError):
        hpss_settings({"margin": 0.5})


def test_score_split():
    rng = np.random.default_rng(2)
    drums, rest = rng.standard_normal(SR * 3), rng.standard_normal(SR * 3)
    mix = drums + rest
    perfect = score_split(mix, drums, rest, drums, SR)
    assert perfect["sdr_p"] > 90 and perfect["gain_h"] > 90
    assert perfect["base_p"] == pytest.approx(0.0, abs=0.1)
    nothing = score_split(mix, drums, mix, mix, SR)
    assert nothing["gain_p"] == 0.0 and nothing["fgain_h"] == 0.0
    silent = score_split(rest, np.zeros_like(rest), rest, np.zeros_like(rest), SR)
    assert silent["sdr_p"] is None and silent["sdr_h"] > 90


def test_bench_on_fake_musdb(tmp_path):
    fake_musdb(tmp_path)
    items = musdb_items("test", root=tmp_path)
    assert [i.name for i in items] == ["Song 0", "Song 1"]
    results, rows = run_hpss_bench(items, {"margin": ["1", "2"]}, progress=None)
    assert len(results) == 2 and len(rows) == 2
    plain = results[0]
    assert plain.params == {"margin": "1"}
    assert plain.mean("gain_p") > 3.0  # synthetic drums are clearly percussive
    assert plain.mean("gain_h") > 0.0
    assert "SDR perc" in format_hpss_results(results)
    assert plain.to_dict()["n_items"] == 2
    short, _ = run_hpss_bench(items, duration=2.0, progress=None)
    assert rows[0]["duration"] > 2.0
    assert short[0].mean("sdr_p") != plain.mean("sdr_p")


def test_bench_in_parallel_matches_serial(tmp_path):
    fake_musdb(tmp_path)
    items = musdb_items("test", root=tmp_path)
    serial, _ = run_hpss_bench(items, progress=None)
    parallel, _ = run_hpss_bench(items, jobs=2, progress=None)
    assert serial[0].rows == parallel[0].rows


def test_bench_cli(tmp_path, monkeypatch):
    fake_musdb(tmp_path)
    monkeypatch.setenv(ENV_VAR, str(tmp_path))
    out = tmp_path / "runs"
    assert main(["bench", "hpss", "--limit", "1", "--set", "harm_frames=17,31", "--out-dir", str(out)]) == 0
    run = json.loads(next(out.glob("*-hpss-*.json")).read_text(encoding="utf-8"))
    assert run["dataset"] == "musdb" and len(run["results"]) == 2 and len(run["items"]) == 1
    assert main(["bench", "hpss", "--set", "kernel=3", "--out-dir", str(out)]) == 1
    assert main(["bench", "hpss", "--split", "train", "--out-dir", str(out)]) == 1  # no train songs


def test_hpss_cli(tmp_path):
    tone, clicks = tone_and_clicks(2.0)
    stereo = np.stack([tone + clicks, 0.5 * (tone + clicks)])
    wav = tmp_path / "song.wav"
    write_wav(wav, stereo, SR)
    x, _ = load_audio(wav, SR, mono=False)
    assert x.shape == (2, len(tone))
    out = tmp_path / "parts"
    assert main(["hpss", str(wav), "--out-dir", str(out)]) == 0
    assert sorted(p.name for p in out.iterdir()) == ["song.harmonic.wav", "song.percussive.wav"]
    harmonic, sr = read_wav(out / "song.harmonic.wav")
    assert sr == SR and len(harmonic) == len(tone)
    assert main(["hpss", str(wav), "--set", "margin=2"]) == 0  # next to the input, with a residual
    assert (tmp_path / "song.residual.wav").is_file()
    assert main(["hpss", str(tmp_path / "missing.wav")]) == 1
    assert main(["hpss", str(wav), "--set", "kernel=3"]) == 1


# ---------------------------------------------------------------- librosa cross-check


def test_masks_match_librosa():
    librosa = pytest.importorskip("librosa")
    mag = np.abs(stft(np.sum(tone_and_clicks(), axis=0)))
    for kernel, margin, power in ((31, 1.0, 2.0), ((17, 9), 1.0, 1.0), (31, 2.0, 2.0), ((9, 17), (3.0, 2.0), 2.0)):
        harm_frames, perc_bins = kernel if isinstance(kernel, tuple) else (kernel, kernel)
        m_h, m_p = margin if isinstance(margin, tuple) else (margin, margin)
        theirs_h, theirs_p = librosa.decompose.hpss(mag, kernel_size=kernel, margin=margin, power=power, mask=True)
        ours_h, _ = hpss_masks(mag, harm_frames, perc_bins, power=power, margin=m_h)
        _, ours_p = hpss_masks(mag, harm_frames, perc_bins, power=power, margin=m_p)
        assert np.allclose(ours_h, theirs_h, atol=1e-6), (kernel, margin, power)
        assert np.allclose(ours_p, theirs_p, atol=1e-6), (kernel, margin, power)
