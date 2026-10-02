# Stage 1.1 — spectral core

Own implementations of the time-frequency representations every later
stage uses. Code: `core/spacemidi/dsp/` (`spectral.py`, `mel.py`, `cqt.py`,
`audio.py`). Tests: `core/tests/test_dsp.py`.

## What is implemented

| Function | What it is | Checked against |
| --- | --- | --- |
| `stft`, `istft` | short-time Fourier transform and its inverse (weighted overlap-add) | direct per-frame FFT; perfect reconstruction (error < 1e-10); librosa (same values to 1e-8) |
| `mel_filterbank`, `mel_spectrogram` | Slaney or HTK mel scale, triangular filters, unit-area norm | librosa (to float32 precision) |
| `log_filterbank` | triangular filters, N bands per octave (SuperFlux front end) | unit sums, increasing centres |
| `power_to_db`, `amplitude_to_db` | dB with a floor (`amin`) and a dynamic range limit (`top_db`) | known values |
| `cqt` | constant-Q transform, direct spectral-kernel method | pure tones land in their bin with magnitude A/2 (±3 %) from C1 to B7; librosa (same peak bin) |
| `chroma_cqt` | 12 pitch classes from a 36-bins-per-octave CQT | triads give their three notes; librosa (same top three) |
| `load_audio`, `resample` | WAV/FLAC via soundfile, mono mixdown, polyphase resampling to 22050 Hz | round trip, lengths |

Conventions follow librosa on purpose (periodic Hann window, frames centred
on `t * hop`, shapes `(bins, frames)`), so features can be cross-checked and
compared with published work. The cross-checks run wherever librosa is
installed (the dev environment); they are skipped elsewhere.

## Design notes

- **CQT scaling.** Each bin's kernel is a Hann-windowed complex exponential
  normalized by the window sum, so a sinusoid of amplitude A reads as A/2 in
  its bin at any frequency. librosa instead scales by the square root of the
  window length; only the relative shape matters for chroma and key, and
  our scaling makes magnitudes interpretable across bins.
- **CQT method.** Direct kernel: one FFT size for all bins (the longest
  window, next power of two), sparse kernel matrix. Simple and exact, but
  the low bins force large FFTs.
- **Tuning.** `cqt(..., tuning=cents)` shifts all bins, for recordings not
  tuned to A440. Estimating the tuning is part of 1.6.

## Speed

Measured in the cloud workspace on 240 s of audio at 22050 Hz (one core):

| Operation | Time |
| --- | --- |
| STFT 2048/512 | ~0.3 s (2.8 s on the first call, memory warm-up) |
| CQT 84 bins, 12 per octave | 1.7 s |
| Chroma (252-bin CQT, 36 per octave) | 7.9 s |

## Open question

Is a recursive CQT (filter an octave, downsample by 2, repeat) worth it?
At 36 bins per octave the lowest kernel is 35k samples long, which forces
64k-point FFTs for every frame; chroma takes ~8 s per 4-minute song. That
is acceptable for now. Revisit if chroma or key detection (1.6) becomes a
bottleneck in the pipeline.
