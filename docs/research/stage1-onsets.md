# Stage 1.2 — onset detection

Finding the moments where notes start. Onsets feed beat tracking (1.3),
note segmentation (1.5), drum transcription and the quantizer (stage 2).
Code: `core/spacemidi/dsp/onset.py`. Tests: `core/tests/test_onsets.py`.
Benchmark: `core/spacemidi/eval/bench.py`.

## Method

Two steps: an **onset envelope** (how much new energy appears in each
frame), then **peak picking** (which bumps of the envelope are onsets).

**Envelope.** Compute an STFT, group it into bands, compress (dB or
`log10(1 + x)`), then for each band take the positive increase over the
frame `lag` frames earlier, and average over bands:

```latex
E(t) = \frac{1}{B} \sum_{b=1}^{B} \max\left(0,\; S_b(t) - \max_{|k| \le m} S_{b+k}(t - \text{lag})\right)
```

With `max_size = 1` (m = 0) this is plain **spectral flux**. With
`max_size = 3` the reference frame is max-filtered across neighbouring
bands first: a note whose pitch only wobbles (vibrato, a bend) stays
covered by its own neighbourhood and does not look like a new note. That
is **SuperFlux** (Böck & Widmer, DAFx 2013).

**Peak picking** (Böck, Krebs & Schedl, ISMIR 2012). Frame `n` is an onset if

1. it is the maximum of `E[n - pre_max : n + post_max]`;
2. it is at least `delta` above the mean of `E[n - pre_avg : n + post_avg]`;
3. it comes more than `wait` frames after the previous onset.

The envelope is scaled to 0..1 first, so `delta` is relative to the
loudest onset of the track. Window sizes are set in seconds.

## Presets

Result of the Slakh tuning below (rounds 1–5). Two envelopes:

| Envelope | `flux` | `superflux` |
| --- | --- | --- |
| STFT at 22050 Hz | 2048 / hop 512 (43 fps) | 1024 / hop 110 (200 fps) |
| Bands | 128 mel, power | 24 per octave, 27.5 Hz – 11 kHz, magnitude |
| Compression | dB, 80 dB range | log10(1 + 1000 x) |
| Lag / max filter | 1 / 1 | 2 / 3 bands |
| Peak windows (s) | pre_max 0.03, post_max 0, avg 0.10 / 0.10, wait 0.03 | pre_max 0.03, post_max 0.03, avg 0.10 / 0.07, wait 0.03 |

Five presets, all with an absolute threshold except `librosa`:

| Preset | Envelope | delta (absolute) | offset | Use | Slakh test F1 |
| --- | --- | --- | --- | --- | --- |
| `flux` (default) | flux | 0.3 dB | +6 ms | full mixes | 0.821 |
| `superflux` | superflux | 0.01 | −3 ms | full mixes | 0.817 |
| `flux-stem` | flux | 1.0 dB | +9 ms | one instrument | 0.840 |
| `superflux-stem` | superflux | 0.05 | 0 | one instrument | 0.838 |
| `librosa` | flux | 0.07 of the track maximum | 0 | cross-checks | 0.683 |

`librosa` is numerically the same as `librosa.onset.onset_detect` with
default settings (the tests check the envelope and the picked peaks against
librosa), except that librosa delays its envelope by two frames; ours is
not delayed. The offsets centre the median timing error measured on Slakh;
the F1 values above were measured before the offsets were added.

## How results are measured

- Reference onsets come from Slakh's MIDI: every note start of every
  rendered stem. Note starts closer than 30 ms are merged into one onset
  (`--combine`), since a chord or a kick under a bass note is one event.
- A detection counts when it is within ±50 ms of a reference onset
  (mir_eval convention, one-to-one matching).
- `--source mix` scores full mixes; `--source stems` scores every stem
  alone and reports F1 per instrument group.
- Every run also reports the median timing error of matched onsets
  (`dt med`), which is what `offset` should correct.

```
uv run spacemidi bench onsets --preset flux --jobs 6
uv run spacemidi bench onsets --preset superflux --jobs 6
uv run spacemidi bench onsets --source stems --jobs 6
uv run spacemidi bench onsets --set delta=0.02,0.04,0.07,0.1,0.15 --jobs 6
```

## Results

### Synthetic (preliminary, 2026-10-02)

10 random synthetic songs of 60 s from `random_song()` in the tests (1661
onsets, all six groups, at least 120 ms apart), our test synth, default
presets. These check that the detectors work; they say little about real
music.

| Audio | Preset | P | R | F1 | Median timing error |
| --- | --- | --- | --- | --- | --- |
| clean | flux | 0.984 | 0.904 | 0.942 | −20.9 ms |
| clean | superflux | 0.999 | 0.862 | 0.925 | −3.0 ms |
| white noise, 10 dB SNR | flux | 0.938 | 0.981 | 0.959 | −6.6 ms |
| white noise, 10 dB SNR | superflux | 0.917 | 0.940 | 0.928 | +0.7 ms |

First observations:

- Precision is near 1 and recall lower on clean audio: `delta = 0.07` is
  too strict when one loud onset sets the 0..1 scale. Noise raises the
  floor of the envelope, which changes the normalization and recall goes
  *up*. The threshold should probably adapt to the envelope's spread, not
  only its maximum.
- `flux` fires about 21 ms early. A centred 2048-sample frame (93 ms)
  starts to see a note well before its centre reaches it; SuperFlux's
  shorter frames and hop are almost unbiased.

### Slakh2100 test split

Full mixes, all 151 test tracks, default presets (2026-10-02, commit
b1ce931+dirty; runs `20261002-210418-onsets-flux-slakh-test-mix-real.json`
and `20261002-210517-onsets-superflux-slakh-test-mix-real.json`).

| Preset | P | R | F1 | Median timing error |
| --- | --- | --- | --- | --- |
| flux | 0.996 | 0.519 | 0.683 | −3.6 ms |
| superflux | 0.976 | 0.541 | 0.696 | +6.5 ms |

Superflux tracks split into quarters by reference onset density:

| Reference onsets per second | Mean recall | Mean precision | Detections per second |
| --- | --- | --- | --- |
| 2.7–4.0 | 0.64 | 0.98 | 2.2 |
| 4.0–5.1 | 0.63 | 0.97 | 3.0 |
| 5.1–6.1 | 0.51 | 0.98 | 2.9 |
| 6.1–11.7 | 0.50 | 0.98 | 3.8 |

- Almost every detection is a real onset, but only about half of the
  reference onsets are found. The detector fires 2–4 times per second
  whatever the music does, so recall falls as the music gets denser: the
  0.07 threshold (relative to the loudest onset) is far too strict for full
  mixes.
- Part of the missing recall may not be findable at all: the reference
  holds every MIDI note start, including very quiet or masked ones. The
  stems run (`--source stems`) and the synth run (`--audio synth`) will show
  how much.
- Per-track F1 spreads from 0.33 to 0.99.
- Timing errors on real audio are small for both presets; flux's −21 ms on
  synthetic audio came from our synth's instant attacks.

#### Round 2: threshold sweep, stems, synth (2026-10-02, commit b1ce931+dirty)

Threshold sweep on the 151 test mixes (runs `20261002-214337-…superflux…`
and `20261002-214419-…flux…`):

| delta | flux P | flux R | flux F1 | superflux P | superflux R | superflux F1 |
| --- | --- | --- | --- | --- | --- | --- |
| 0.005 | 0.781 | 0.795 | 0.788 | 0.496 | 0.885 | 0.636 |
| 0.01 | 0.886 | 0.757 | **0.816** | 0.562 | 0.853 | 0.678 |
| 0.02 | 0.964 | 0.698 | 0.810 | 0.710 | 0.775 | 0.741 |
| 0.03 | 0.982 | 0.655 | 0.786 | 0.829 | 0.706 | **0.763** |
| 0.05 | 0.993 | 0.582 | 0.734 | 0.943 | 0.611 | 0.741 |
| 0.07 (round 1) | 0.996 | 0.519 | 0.683 | 0.976 | 0.541 | 0.696 |

Each stem alone, superflux at default delta 0.07 (run
`20261002-215201-…stems…`, 1539 stems):

| Group | Reference onsets | P | R | F1 |
| --- | --- | --- | --- | --- |
| bass | 67 554 | 0.976 | 0.958 | 0.967 |
| piano | 80 280 | 0.986 | 0.902 | 0.942 |
| guitar | 132 055 | 0.938 | 0.864 | 0.899 |
| drums | 149 661 | 0.997 | 0.578 | 0.732 |
| other (strings, pads, organ, brass, synths) | 90 320 | 0.160 | 0.727 | 0.262 |

Same MIDI rendered with our test synth, superflux at delta 0.07 (run
`20261002-215648-…synth…`): P 0.997, R 0.541, F1 0.702.

Findings:

- **Tuned flux is the best L1 onset detector so far: F1 0.816 at delta 0.01.**
  Lowering the threshold trades precision for recall smoothly; both presets
  peak around P ≈ R.
- **SuperFlux falls apart at low thresholds** (precision 0.50 at delta
  0.005). Likely cause: madmom feeds SuperFlux int16-scaled samples, so its
  `log10(1 + X)` sees magnitudes ~32 000× larger than ours and really is a
  log. On our [-1, 1] float magnitudes `log10(1 + X)` is almost linear, so
  loud events dominate and the envelope is noisy at small scale. Test:
  sweep `log_mul` (1 … 10⁴).
- **On single instruments the detector is good** where notes are struck or
  plucked: bass 0.97, piano 0.94, guitar 0.90.
- **Drums alone: precision 1.00, recall 0.58.** Quiet hits (hi-hats, ghost
  notes) under a normalized threshold set by the loudest kick are missed:
  3.8 reference hits per second, 2.1 detections.
- **"other" alone: precision 0.16.** These stems are mostly sustained
  sounds (median 0.38 notes per second). Normalizing the envelope to its
  own maximum blows up the small fluctuations of a pad or a string section
  (vibrato, tremolo, chorus) into "onsets": one stem with a single note got
  57 detections. A max-normalized threshold cannot work on such material; it
  needs a floor relative to the signal level, or a spread-based threshold.
- **Synth vs real audio: the same recall (0.54 vs 0.54).** At delta 0.07 the
  missing half of the onsets is not caused by realistic instrument sounds;
  it is the threshold relative to the loudest events. (A synth sweep would
  give the ceiling of the method.)

#### Round 3: SuperFlux compression, flux on stems (2026-10-02, commit b1ce931+dirty)

SuperFlux on the test mixes with `log_mul` × `delta` (run `20261002-220415-…superflux…`), F1 (P / R):

| delta | log_mul 1 | log_mul 100 | log_mul 10 000 |
| --- | --- | --- | --- |
| 0.01 | 0.678 (0.56 / 0.85) | **0.797** (0.83 / 0.77) | 0.790 (0.93 / 0.69) |
| 0.03 | 0.763 (0.83 / 0.71) | 0.754 (0.98 / 0.61) | 0.676 (0.99 / 0.51) |
| 0.05 | 0.741 (0.94 / 0.61) | 0.685 (0.99 / 0.52) | 0.562 (0.99 / 0.39) |
| 0.1 | 0.630 (0.99 / 0.46) | 0.530 (1.00 / 0.36) | 0.353 (1.00 / 0.21) |

Flux on every stem alone (run `20261002-220915-…flux…stems…`), F1 (P / R):

| Group | delta 0.01 | delta 0.03 |
| --- | --- | --- |
| bass | 0.762 (0.62 / 0.99) | 0.828 (0.71 / 0.99) |
| piano | 0.898 (0.83 / 0.98) | 0.938 (0.91 / 0.97) |
| guitar | 0.752 (0.63 / 0.93) | 0.843 (0.79 / 0.91) |
| drums | **0.907** (0.97 / 0.85) | 0.877 (1.00 / 0.78) |
| other | 0.280 (0.17 / 0.85) | 0.447 (0.32 / 0.74) |

Findings:

- **The compression hypothesis holds.** With a real log (`log_mul` 100)
  SuperFlux goes from 0.763 to 0.797 and its timing error drops from +7 ms
  to +3 ms. With stronger compression the best delta moves below 0.01, so
  the true optimum is not in this grid yet.
- **Flux fixes the drums** (0.73 → 0.91 with delta 0.01): its dB scale is
  relative to each band's own level, so quiet hi-hats rise as many dB as a
  loud kick.
- **No single delta suits every instrument.** Mixes and drums want 0.01;
  bass and guitar alone want ≥ 0.03 (their sustained, bending notes produce
  false onsets at 0.01); "other" is poor at any delta. The reason is
  the 0..1 normalization: the threshold depends on the loudest event in the
  track, which differs by instrument and arrangement.
- **Next test, no code needed:** switch normalization off
  (`normalize=false`) so `delta` becomes an absolute rise (dB for flux,
  log units for SuperFlux). A pad that wobbles by a fraction of a dB then
  stays below the threshold, whatever its own maximum is.

#### Round 4: absolute thresholds (`normalize=false`, 2026-10-02, commit b1ce931+dirty)

`delta` is now an absolute rise: mean dB per band for flux, `log10` units
for SuperFlux with `log_mul` 1000 (0.05 ≈ 1 dB). F1 (P / R).

Test mixes (runs `20261002-221332-…flux…mix…`, `20261002-221859-…superflux…mix…`):

| flux delta (dB) | F1 | superflux delta | F1 |
| --- | --- | --- | --- |
| 0.5 | **0.804** (0.97 / 0.68) | 0.02 | **0.793** (0.97 / 0.67) |
| 1 | 0.747 (0.99 / 0.60) | 0.05 | 0.691 (0.99 / 0.53) |
| 1.5 | 0.690 | 0.1 | 0.543 |
| 2 | 0.634 | 0.15 | 0.412 |
| 3 | 0.519 | 0.2 | 0.313 |
| 5 | 0.314 | 0.3 | 0.172 |

Every stem alone (runs `20261002-221800-…flux…stems…`, `20261002-223108-…superflux…stems…`):

| Group | flux, 1 dB | superflux, 0.05 | best normalized so far |
| --- | --- | --- | --- |
| all stems | **0.840** (0.88 / 0.80) | 0.838 (0.90 / 0.79) | 0.761 (flux, 0.03) |
| piano | 0.936 | 0.924 | 0.942 |
| drums | 0.873 (0.900 at 0.5 dB) | 0.885 (0.910 at 0.02) | 0.907 |
| guitar | 0.864 | 0.868 | 0.899 |
| bass | 0.868 (0.906 at 2 dB) | 0.853 (0.884 at 0.1) | 0.967 |
| other | **0.597** | 0.568 | 0.447 |

Findings:

- **Absolute thresholds work across instruments.** On stems, one absolute
  delta scores 0.84 overall against 0.76 for the best normalized delta, and
  "other" (pads, strings) jumps from 0.45 to 0.60 because small wobbles no
  longer get scaled up.
- **Mixes want a much smaller absolute delta than stems** (best at the
  low edge of both grids, so the optimum is lower still). In a mix the
  dB rise of one new note is averaged over bands that other instruments
  keep full, so it is diluted. Since stage 2 separates stems before
  transcribing them, the stem numbers matter more for notes; the mix
  numbers matter for beat tracking (1.3).
- Bass alone is still best with the normalized threshold (0.967 with
  superflux at 0.07): sustained bass notes with slides need a higher
  absolute delta (2 dB) than other instruments.


#### Round 5: lower absolute thresholds on mixes (2026-10-02, commit b1ce931+dirty)

Runs `20261002-223403-…flux…mix…` and `20261002-223540-…superflux…mix…`, F1 (P / R):

| flux delta (dB) | F1 | superflux delta | F1 |
| --- | --- | --- | --- |
| 0.1 | 0.785 (0.77 / 0.80) | 0.005 | 0.739 (0.66 / 0.84) |
| 0.2 | 0.816 (0.88 / 0.76) | 0.01 | **0.817** (0.89 / 0.76) |
| 0.3 | **0.821** (0.94 / 0.73) | 0.015 | 0.812 (0.96 / 0.71) |
| 0.4 | 0.814 (0.96 / 0.71) | | |

### Summary of the tuning

| Setting (Slakh test) | Mixes F1 | Stems F1 |
| --- | --- | --- |
| librosa defaults (round 1) | 0.683 | — |
| best relative threshold | 0.816 (flux, 0.01) | 0.761 (flux, 0.03) |
| best absolute threshold | **0.821** (flux, 0.3 dB) | **0.840** (flux, 1 dB) |

- Absolute thresholds win, mostly through robustness: one setting per use
  instead of one per instrument, and no false-onset flood on pads.
- The two envelopes end up equal within 0.005; flux is the default because
  it is 5× cheaper (43 vs 200 frames per second). SuperFlux keeps the
  better timing (±3 ms vs ±6 ms before offsets).
- **Caveat found while testing:** the mix presets fire too often on sparse,
  clean audio. On our synthetic songs (one note at a time, digital silence
  between them) `flux` scores 0.84 and `superflux` 0.78, while the stem
  presets score 0.98 / 0.95; added white noise makes `superflux` worse
  still. An absolute threshold low enough for dense mixes is too low for a
  quiet intro or a solo passage. Possible fixes: a level gate, or a
  threshold that adapts to the local spread of the envelope.

## Open questions

1. A threshold that adapts to the material (local spread of the envelope,
   or a level gate), so one preset works on dense mixes and sparse passages
   alike. The mix/stem split is a workaround.
2. Drums alone still miss quiet hits at thresholds that keep pads clean;
   per-band or per-class detection (stage 2 drums) may be the real fix.
3. Bass alone is best with a higher threshold (2 dB) because of slides and
   sustained notes; check again once 1.5 (pitch tracking) can split notes
   by pitch change instead.
4. Reference onsets merge notes within 30 ms; check how many reference
   onsets are inaudible (very quiet MIDI notes) to know the real ceiling.
5. librosa 1.0 has a dynamic-programming peak picker; compare with ours.
