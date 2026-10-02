"""Spectral core: STFT, mel, CQT, chroma, audio loading.

Self-checks run everywhere. Cross-checks against librosa (same conventions,
so the numbers must agree) run only where librosa is installed.
"""

from __future__ import annotations

import wave

import numpy as np
import pytest

from spacemidi.dsp import (
    amplitude_to_db,
    chroma_cqt,
    chroma_matrix,
    cqt,
    cqt_frequencies,
    hz_to_mel,
    istft,
    load_audio,
    log_filterbank,
    mel_filterbank,
    mel_spectrogram,
    mel_to_hz,
    midi_to_hz,
    n_frames,
    power_to_db,
    resample,
    stft,
    window,
)
from spacemidi.eval import write_wav

SR = 22050


def _noise(n: int = 12345, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(n)


def _tone(midi: float, seconds: float = 2.0, amp: float = 0.8, sr: int = SR) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    return amp * np.cos(2 * np.pi * float(midi_to_hz(midi)) * t)


# ---------------------------------------------------------------- STFT


def test_stft_matches_direct_frame_fft():
    x = _noise()
    s = stft(x, 2048, 512)
    assert s.shape == (1025, n_frames(len(x), 2048, 512)) == (1025, 1 + len(x) // 512)
    padded = np.pad(x, 1024)
    for t in (0, 7, s.shape[1] - 1):
        direct = np.fft.rfft(padded[t * 512 : t * 512 + 2048] * window("hann", 2048))
        assert np.allclose(s[:, t], direct, atol=1e-12)


def test_short_window_is_centred_in_the_frame():
    x = _noise()
    s = stft(x, 1024, 256, win_length=512)
    w = np.pad(window("hann", 512), 256)
    padded = np.pad(x, 512)
    assert np.allclose(s[:, 3], np.fft.rfft(padded[768 : 768 + 1024] * w), atol=1e-12)


def test_istft_reconstructs_the_signal():
    x = _noise()
    for n_fft, hop in ((2048, 512), (1024, 256), (512, 128)):
        y = istft(stft(x, n_fft, hop), hop, length=len(x))
        assert np.max(np.abs(y - x)) < 1e-10
    assert len(istft(stft(x), 512, length=len(x) + 100)) == len(x) + 100


def test_stft_rejects_stereo_and_short_input():
    with pytest.raises(ValueError):
        stft(np.zeros((2, 4096)))
    with pytest.raises(ValueError):
        stft(np.zeros(100), 2048, center=False)


def test_db_conversions():
    s = np.array([1.0, 0.1, 1e-3, 0.0])
    assert np.allclose(power_to_db(s, top_db=None), [0.0, -10.0, -30.0, -100.0])
    assert np.allclose(power_to_db(s, top_db=20.0), [0.0, -10.0, -20.0, -20.0])
    assert np.allclose(amplitude_to_db(np.array([1.0, 0.1]), top_db=None), [0.0, -20.0])
    assert np.allclose(power_to_db(s, ref=np.max, top_db=None)[:3], [0.0, -10.0, -30.0])


# ---------------------------------------------------------------- mel / log filterbanks


def test_mel_scale():
    assert hz_to_mel(1000.0) == pytest.approx(15.0)
    assert hz_to_mel(200.0) == pytest.approx(3.0)
    assert hz_to_mel(1000.0, htk=True) == pytest.approx(1000.0, abs=0.1)
    f = np.array([0.0, 50.0, 999.0, 1000.0, 4000.0, 11025.0])
    for htk in (False, True):
        assert np.allclose(mel_to_hz(hz_to_mel(f, htk), htk), f)


def test_mel_filterbank_shape_and_area():
    fb = mel_filterbank(SR, 2048, 128)
    assert fb.shape == (128, 1025) and fb.min() >= 0.0
    peaks = fb.argmax(axis=1)
    assert np.all(np.diff(peaks) >= 0)  # centres go up
    plain = mel_filterbank(SR, 2048, 40, norm=None)
    assert plain.max() <= 1.0 + 1e-12
    spec = np.abs(stft(_noise())) ** 2
    assert np.allclose(mel_spectrogram(_noise(), SR), fb @ spec)


def test_log_filterbank():
    fb = log_filterbank(SR, 1024, 24, 27.5, 16000.0)
    assert fb.shape[1] == 513 and 80 < fb.shape[0] < 140
    assert np.allclose(fb.sum(axis=1), 1.0)
    assert np.all(np.diff(fb.argmax(axis=1)) > 0)


# ---------------------------------------------------------------- CQT and chroma


@pytest.mark.parametrize("midi", [24, 36, 52, 69, 81, 100, 107])
def test_cqt_tone_lands_in_its_bin_with_half_amplitude(midi):
    c = np.abs(cqt(_tone(midi), SR))
    frame = c[:, c.shape[1] // 2]
    assert frame.argmax() == midi - 24
    assert frame.max() == pytest.approx(0.4, rel=0.03)


def test_cqt_frames_match_stft_and_tuning_shifts_bins():
    x = _tone(69.5)  # a quarter tone sharp
    c = cqt(x, SR, 512)
    assert c.shape == (84, 1 + len(x) // 512)
    tuned = np.abs(cqt(x, SR, 512, tuning=50.0))[:, c.shape[1] // 2]
    assert tuned.argmax() == 45 and tuned.max() == pytest.approx(0.4, rel=0.03)
    assert np.allclose(cqt_frequencies(3, 110.0, 12), [110.0, 110.0 * 2 ** (1 / 12), 110.0 * 2 ** (2 / 12)])


def test_cqt_rejects_bins_above_nyquist():
    with pytest.raises(ValueError, match="Nyquist"):
        cqt(_noise(4096), 8000, n_bins=96)


def test_chroma_of_a_major_triad():
    chord = _tone(60) + _tone(64) + _tone(67)
    ch = chroma_cqt(chord, SR)
    frame = ch[:, ch.shape[1] // 2]
    assert set(np.argsort(frame)[-3:]) == {0, 4, 7}
    assert frame.max() == pytest.approx(1.0)
    assert np.all(chroma_cqt(np.zeros(SR), SR) == 0.0)


def test_chroma_matrix_folds_three_bins_per_semitone():
    m = chroma_matrix(72, 36)
    assert m.shape == (12, 72) and np.all(m.sum(axis=0) == 1)
    assert m[0, 0] == m[0, 1] == m[1, 2] == m[0, 35] == 1


# ---------------------------------------------------------------- audio


def test_load_wav_resample_and_mixdown(tmp_path):
    x = 0.5 * np.sin(2 * np.pi * 440 * np.arange(44100) / 44100)
    path = tmp_path / "a.wav"
    write_wav(path, x, 44100)
    y, sr = load_audio(path, 22050)
    assert sr == 22050 and len(y) == 22050
    native, sr = load_audio(path, None)
    assert sr == 44100 and np.max(np.abs(native - x)) < 1e-4
    stereo = tmp_path / "s.wav"
    pcm = (np.stack([x, -x], axis=1) * 32767).astype("<i2")
    with wave.open(str(stereo), "wb") as f:
        f.setnchannels(2), f.setsampwidth(2), f.setframerate(44100), f.writeframes(pcm.tobytes())
    mono, _ = load_audio(stereo, None)
    assert np.max(np.abs(mono)) < 1e-4
    both, _ = load_audio(stereo, None, mono=False)
    assert both.shape == (2, 44100)
    assert len(resample(np.zeros(300), 48000, 16000)) == 100


# ---------------------------------------------------------------- librosa cross-checks


@pytest.fixture
def librosa():
    return pytest.importorskip("librosa")


def test_stft_and_istft_match_librosa(librosa):
    x = _noise(20000)
    for n_fft, hop, win in ((2048, 512, None), (1024, 256, 512)):
        ours = stft(x, n_fft, hop, win_length=win)
        theirs = librosa.stft(y=x, n_fft=n_fft, hop_length=hop, win_length=win)
        assert ours.shape == theirs.shape
        assert np.allclose(ours, theirs, atol=1e-8)
        assert np.allclose(istft(ours, hop, win_length=win, length=len(x)),
                           librosa.istft(theirs, hop_length=hop, win_length=win, length=len(x)), atol=1e-8)
    assert len(istft(stft(x))) == len(librosa.istft(librosa.stft(y=x)))


def test_mel_matches_librosa(librosa):
    f = np.linspace(0, 11025, 50)
    assert np.allclose(hz_to_mel(f), librosa.hz_to_mel(f))
    assert np.allclose(hz_to_mel(f, htk=True), librosa.hz_to_mel(f, htk=True))
    for kw in ({}, {"n_mels": 40, "fmin": 30.0, "fmax": 8000.0}, {"n_mels": 64, "htk": True}):
        ours = mel_filterbank(SR, 2048, **kw)
        theirs = librosa.filters.mel(sr=SR, n_fft=2048, **kw)
        assert np.allclose(ours, theirs, rtol=1e-5, atol=1e-7)
    x = _noise(20000)
    assert np.allclose(power_to_db(mel_spectrogram(x, SR)), librosa.power_to_db(librosa.feature.melspectrogram(y=x, sr=SR)),
                       atol=1e-3)


def test_cqt_and_chroma_agree_with_librosa(librosa):
    for midi in (30, 45, 60, 76, 95):
        x = _tone(midi)
        ours = np.abs(cqt(x, SR))
        theirs = np.abs(librosa.cqt(y=x, sr=SR, hop_length=512, n_bins=84, bins_per_octave=12, tuning=0.0))
        mid = min(ours.shape[1], theirs.shape[1]) // 2
        assert ours[:, mid].argmax() == theirs[:, mid].argmax() == midi - 24
    chord = _tone(57) + _tone(60) + _tone(64)  # A minor
    ours = chroma_cqt(chord, SR)
    theirs = librosa.feature.chroma_cqt(y=chord, sr=SR, tuning=0.0)
    mid = min(ours.shape[1], theirs.shape[1]) // 2
    assert set(np.argsort(ours[:, mid])[-3:]) == set(np.argsort(theirs[:, mid])[-3:]) == {9, 0, 4}
