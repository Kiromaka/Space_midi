# Stage 1.4 — harmonic-percussive separation (HPSS)

Splitting a mix into a harmonic part (sustained, pitched sounds) and a
percussive part (short, broadband hits). It is the first, cheapest step
toward split instruments: drum transcription can work on the percussive
part, pitch tracking (YIN, later the transcription nets) on the harmonic
part, and the onset and beat trackers can each pick the part that suits
them. Code: `core/spacemidi/dsp/hpss.py`. Tests: `core/tests/test_hpss.py`.
Benchmark: `core/spacemidi/eval/bench_hpss.py`.

## Method

Fitzgerald, "Harmonic/percussive separation using median filtering"
(DAFx 2010), with the margin of Driedger, Müller and Disch (ISMIR 2014).

1. Magnitude STFT `S` (2048 / 512 at 22050 Hz: 93 ms windows, 23 ms hop).
2. In a spectrogram a held note is a horizontal line and a drum hit a
   vertical line. A median filter **along time** (`harm_frames` = 31 frames,
   0.7 s) keeps horizontal lines and removes vertical ones: `H`. A median
   filter **along frequency** (`perc_bins` = 31 bins, 330 Hz) does the
   opposite: `P`. The median, unlike a mean, ignores a short spike instead
   of smearing it.
3. Soft masks, Wiener-like:

```latex
M_h = \frac{H^p}{H^p + (m P)^p}, \qquad M_p = \frac{P^p}{P^p + (m H)^p}
```

4. The harmonic part is `istft(M_h · X)`, the percussive part
   `istft(M_p · X)`, with the mixture's phase.

With margin `m = 1` the masks sum to 1 and harmonic + percussive is exactly
the input. With `m > 1` a bin must be `m` times more harmonic than
percussive to count as harmonic; what is neither (noise, breaths, the
attack of a piano note) goes to a third, **residual** part. Power `p = 2`
gives soft masks, `p = ∞` binary ones.

`hpss_masks` returns the masks for an existing spectrogram, so the onset
and beat code apply them to the STFT it has already computed. Conventions
(reflected edges, zero bins split evenly) match `librosa.decompose.hpss`;
a test checks the masks against librosa for several kernel, margin and
power settings.

### Where it plugs in

| Setting | Effect |
| --- | --- |
| onsets: `--set source=percussive` (or `harmonic`) | onset envelope from that part only (`hpss_frames`, `hpss_bins` set the kernels) |
| beats: `--set hpss=true` | onset envelope from the percussive part, chroma (for downbeats) from the harmonic part; the low-end cue stays on the full mix, since it carries both kick and bass |

## How results are measured

`spacemidi bench hpss` on MUSDB18-HQ (test split, 50 songs; mono,
22050 Hz). MUSDB has four stems per song, and HPSS gives two parts, so the
closest references are:

- percussive part vs the **drums** stem;
- harmonic part vs **mixture − drums** (bass + other + vocals).

Neither is a perfect target: a hi-hat has a "harmonic" tail and a strummed
guitar has "percussive" attacks; vocals carry consonants. The scores say how
close the split gets to a drums / rest split, which is what later stages
want.

| Score | Meaning |
| --- | --- |
| SDR perc / harm | plain SDR in dB over the whole song, mean over songs (`eval.metrics.sdr`) |
| gain | SDR minus the SDR of the unprocessed mixture against the same reference: the improvement over doing nothing |
| fSDR | median SDR over 1 s windows (silent windows skipped), median over songs: the SiSEC / MUSDB convention |

This is plain SDR, not the BSS Eval v4 SDR of museval, which first lets
the estimate be filtered to fit the reference and reads a few dB higher.
Numbers are comparable between our runs, not directly with published
museval tables. With a margin > 1 the residual is not scored: the parts get
purer but lose energy, and SDR punishes the lost energy.

```
uv run spacemidi bench hpss --jobs 4
uv run spacemidi bench hpss --jobs 4 --set margin=1,2,4
uv run spacemidi bench hpss --jobs 4 --set harm_frames=17,31,63 --set perc_bins=17,31,63
uv run spacemidi bench hpss --jobs 4 --set n_fft=1024,2048,4096 --duration 60
uv run spacemidi hpss song.wav --set margin=2   # writes song.harmonic.wav, .percussive.wav, .residual.wav
```

Each worker needs about 1 GB of memory for a 4-minute song.

## Results

### Synthetic (2026-10-14)

A held chord plus short noise bursts every 250 ms (tests): harmonic SDR
against the chord improves by more than 3 dB over the mixture, percussive
SDR against the bursts by more than 20 dB. Onsets from the percussive part
find every burst despite the loud chord. A `grid_song` (drums, bass,
chords) still tracks with beat F > 0.95 with `hpss=true`.

### MUSDB18-HQ test split (2026-10-14, commit 201d6d1+dirty)

Run `20261014-234224-hpss-…` (whole songs, 50 songs, 816 s of worker time). SDR in dB, means over songs; fSDR = median over
songs of the per-song median over 1 s windows.

| margin | SDR perc | gain | SDR harm | gain | fSDR perc | gain | fSDR harm | gain |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (default) | 2.46 | +6.17 | 6.17 | +2.46 | 2.14 | +6.43 | 6.87 | +1.90 |
| 2 | 2.42 | +6.13 | 5.00 | +1.29 | 2.16 | +6.39 | 5.11 | +1.08 |
| 4 | 1.60 | +5.31 | 3.40 | −0.31 | 1.36 | +5.20 | 3.10 | −0.83 |

Kernel grid (`20261014-234838-hpss-…`, first 60 s of each song), gain of
the whole-segment SDR for percussive / harmonic:

| harm_frames ↓ / perc_bins → | 17 | 31 | 63 |
| --- | --- | --- | --- |
| 17 | +5.92 / +2.09 | **+6.17 / +2.34** | +5.56 / +1.73 |
| 31 | +5.66 / +1.83 | +6.08 / +2.25 | +5.63 / +1.80 |
| 63 | +5.19 / +1.36 | +5.70 / +1.87 | +5.35 / +1.52 |

Reading the numbers:

- **The split helps, but the percussive part is far from a clean drum
  stem.** Doing nothing scores −3.7 dB against the drums (drums are a
  small share of most mixes); the percussive part reaches +2.5 dB. The
  rest of the mix also gains, +2.5 dB.
- **At margin 1 the two scores are the same number.** harmonic +
  percussive = mixture and drums + rest = mixture, so the error of one part
  is minus the error of the other: percussive SDR = harmonic gain, and
  harmonic SDR = percussive gain. Only with a margin do they say different
  things.
- **A margin only costs SDR** (the residual is not scored and its energy
  counts as missing). It may still make cleaner input for later stages; to
  be judged by onset / pitch scores, not SDR.
- **Kernels barely matter**: the 3 × 3 grid spans 1 dB, and librosa's
  31 × 31 is 0.1 dB from the best (17 frames, 31 bins). Default stays
  31 × 31.
- Per song the gain ranges from −0.5 dB (PR – Happy Daze, where drums
  dominate the mix already) to +13.7 dB (Speak Softly – Like Horses, quiet
  drums under sustained parts).

### Onsets and beats on the separated parts (Slakh, 2026-10-14)

| Run | Setting | P | R | F1 |
| --- | --- | --- | --- | --- |
| `20261014-235435-onsets-flux-…` | source=mix (preset flux) | 0.938 | 0.728 | **0.820** |
| | source=percussive | 0.773 | 0.803 | 0.787 |
| `20261015-001054-onsets-flux-…` | source=percussive, delta 0.6 | 0.887 | 0.759 | 0.818 |
| | source=percussive, delta 1.0 | 0.945 | 0.718 | 0.816 |
| | source=percussive, delta 1.5 | 0.967 | 0.677 | 0.796 |

With the mix threshold (0.3 dB) the percussive part finds more notes
(recall +0.075) but fires far more often (precision −0.165): its envelope
has more small peaks. With the threshold re-tuned (0.6 dB) it ties the full
mix (0.818 vs 0.820) at a slightly different balance (P 0.89 / R 0.76 vs
0.94 / 0.73). Removing the sustained parts neither helps nor hurts onset
detection on mixes; default stays `source=mix`.

| Run | Setting | beat F | F any lvl | downbeat F | tempo acc1 | acc2 | beat F aligned (141) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `20261015-000034-beats-…` | hpss=false | **0.824** | 0.883 | **0.692** | **0.821** | 0.960 | **0.864** |
| | hpss=true | 0.815 | 0.889 | 0.690 | 0.795 | 0.967 | 0.855 |

HPSS does not help beat tracking on Slakh: 7 tracks improve, 13 get worse
(by more than 0.02 F). The loss is octave errors: 8 tracks move from the
right level to double tempo (5 move back). Without the sustained parts the
envelope is dominated by eighth-note hi-hats, and the tempo prior then
picks the eighth note. The downbeat chroma from the harmonic part changes
nothing (0.692 vs 0.690). Default stays `hpss=false`.

## Open questions

1. Kernel sizes in frames and bins depend on `n_fft`, `hop` and the sample
   rate. At 22050 Hz with 2048 / 512 the grid shows they hardly matter
   (1 dB spread); not worth more tuning.
2. Iterative / multi-resolution HPSS (Driedger 2014: a long window first
   for harmonics, then a short one on the rest for drums) as a cheap
   three-way split: harmonic, percussive, and vocals-ish residual.
3. Onsets: the percussive part ties the mix once its threshold is re-tuned
   (0.818 vs 0.820). Combining both envelopes (or picking per frequency
   band) might still help; not tried.
4. Reference (L3): Demucs drums stem vs our percussive part, once its
   adapter exists; and our own separation net in stage 3.
5. Beats: the percussive envelope pushes the tempo to the eighth note.
   Worth trying only together with a tempo fix (e.g. the full-mix envelope
   for the tempo, the percussive one for beat positions).
