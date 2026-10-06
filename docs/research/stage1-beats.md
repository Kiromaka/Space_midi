# Stage 1.3 — tempo, beats, downbeats

Finding the beat grid of a song: the tempo, where each beat falls, and which
beats start a bar. The MIDI export uses it directly (one tempo event per
beat, bar lines on downbeats), and later stages quantize notes to it.
Code: `core/spacemidi/algos/rhythm/`. Tests: `core/tests/test_rhythm.py`.
Benchmark: `core/spacemidi/eval/bench_rhythm.py`.

## Method

1. **Onset envelope** from the `flux` onset preset (mel spectral flux, 43
   frames per second). Only the envelope is used, not the picked onsets.
2. **Tempo** (`tempo.py`). The envelope is smoothed (Gaussian, 1 frame) and
   autocorrelated; a steady pulse correlates with itself one beat later.
   Every multiple and fraction of the beat correlates too, so each lag is
   weighted by a log-normal prior centred on 120 BPM, one octave wide
   (Ellis 2007; the same prior librosa uses). The best lag, refined by
   parabolic interpolation, gives the tempo. `harmonics` > 1 adds the
   correlation at 2, 3... times each lag, favouring lags whose multiples
   also line up.
3. **Beats** (`beats.py`): Ellis's dynamic programming. Each frame's score
   is its smoothed onset strength plus the best score of a predecessor
   between P/2 and 2P frames earlier, minus `tightness · log(gap / P)²`.
   Backtracking from the best last beat gives the globally best sequence;
   weak beats at the start and end (silence) are trimmed.
4. **Downbeats** (`downbeats.py`). For each beat: accent (onset strength),
   low end (onset strength below 150 Hz: kick and bass), and harmonic change
   (distance between the chroma of the beat before and after). The
   weighted z-scores are summed, and for 4 (and 3) beats per bar every phase
   is tried; the phase whose beats stand out most gives the downbeats. 3/4
   must beat 4/4 by `meter_bias`. The phase is assumed constant through
   the song.

```latex
C(t) = O(t) + \max_{t - 2P \le \tau \le t - P/2} \left( C(\tau) - \alpha \log^2 \frac{t - \tau}{P} \right)
```

Settings live in `RhythmConfig` (`spacemidi.algos.rhythm`); every one can be
changed from the command line with `--set`.

### Design note: smoothing before the autocorrelation

Without the 1-frame smoothing, a click track at 120 BPM came out at 60 BPM.
At 43 frames per second one beat is 21.5 frames, so the beat's correlation
is split between lags 21 and 22, while two beats (43.07 frames) land almost
exactly on lag 43. Smoothing spreads each peak over neighbouring frames so
a fractional period still correlates fully.

## How results are measured

`spacemidi bench beats` on Slakh mixes. The reference grid is read from
each track's `all_src.mid` (the tempo map the audio was rendered with).
Beats in the first 5 s are ignored on both sides (mir_eval convention).

| Score | Meaning |
| --- | --- |
| beat F | F-measure, ±70 ms, mean over tracks |
| F any lvl | best F-measure against the reference at double, half or off-beat level: separates "wrong level" from "lost" |
| downbeat F | F-measure of downbeats, ±70 ms |
| tempo acc1 / acc2 | estimated tempo within 4 % of the reference (acc2 also accepts ×2, ×3, ×½, ×⅓) |
| meter ok | 3 vs 4 beats per bar right |

What the Slakh references look like (test split, 151 tracks): 148 in 4/4,
one each in 2/4, 3/4 and 16/16 (beats on sixteenth notes, 448 "BPM");
median tempo 102 BPM, 42 tracks below 80 BPM and 5 above 160; 54 tracks
have tempo changes larger than 5 % (Lakh MIDI often comes from live
playing). Our tracker assumes one tempo per song, so those 54 are the
first place to look when scores are low.

```
uv run spacemidi bench beats --jobs 6
uv run spacemidi bench beats --jobs 6 --set tightness=50,100,200,400
uv run spacemidi bench beats --jobs 6 --set harmonics=1,2,4
uv run spacemidi bench beats --jobs 6 --audio synth
uv run spacemidi beats song.wav --midi beats.mid   # click track + tempo map, to check in a DAW
```

## Results

### Synthetic (2026-10-06)

`grid_song()` in the tests: drums, bass and a chord per bar on a steady
grid, ±12 ms random timing jitter, white noise at 25 dB SNR. 12 songs with
random tempo (70–170 BPM) and meter (3/4 or 4/4): beat F 1.000, tempo
acc1 1.000, downbeat F 1.000, meter 12/12. A 180 BPM song is tracked at
90 BPM (the prior's choice; F any lvl 1.0); setting `start_bpm=170` fixes it.

These songs are easy by design. They check the code, not the method.

### Slakh2100 test split

#### Round 1 (2026-10-06, commit dfbc6f0+dirty)

Runs `20261006-174954-beats-…-real`, `20261006-175427-beats-…-synth`,
`20261006-175643-beats-…-real` (grid). Means over 151 tracks:

| Setting | beat F | F any lvl | downbeat F | tempo acc1 | acc2 | meter ok |
| --- | --- | --- | --- | --- | --- | --- |
| defaults (tightness 100, harmonics 1) | 0.818 | 0.875 | 0.622 | 0.821 | 0.960 | 0.967 |
| defaults, synth audio | 0.804 | 0.883 | 0.606 | 0.775 | 0.967 | 0.954 |
| tightness 50 | 0.811 | 0.868 | 0.598 | 0.821 | 0.960 | 0.967 |
| tightness 200 | 0.819 | 0.877 | 0.625 | 0.821 | 0.960 | 0.967 |
| **tightness 400** | **0.824** | 0.883 | 0.637 | 0.821 | 0.960 | 0.967 |
| tightness 400, harmonics 2 | 0.817 | **0.889** | **0.652** | 0.801 | **0.974** | 0.967 |

Where the errors are (defaults):

| Tracks | n | beat F | F any lvl | downbeat F | acc1 | acc2 |
| --- | --- | --- | --- | --- | --- | --- |
| steady tempo | 100 | 0.835 | 0.893 | 0.646 | 0.83 | 0.98 |
| tempo changes > 5 % | 51 | 0.783 | 0.840 | 0.576 | 0.80 | 0.92 |
| reference < 80 BPM | 42 | 0.795 | 0.893 | 0.681 | 0.64 | 0.98 |
| reference ≥ 80 BPM | 109 | 0.827 | 0.868 | 0.600 | 0.89 | 0.95 |

Findings:

- **Beat tracking already works on most songs:** beat F 0.82, and at
  any metrical level 0.88. The tempo is right within 4 % for 82 % of songs,
  and right up to an octave for 96 %.
- **Octave errors are mostly a notation question.** In 24 tracks our beats
  match the reference at double speed: songs notated slowly (42 tracks
  below 80 BPM, tempo acc1 only 0.64 there) that we hear twice as fast. The
  reference level is arguably as debatable as ours.
- **Tempo changes cost about 0.05 beat F** (0.835 vs 0.783): the global
  tempo does not follow them; local tempo tracking is the obvious next step.
- **Real vs synth audio scores the same** (0.818 vs 0.804), so the realism
  of the sound is not the difficulty; the tempo maps and our model are.
- **A stricter tempo (tightness 400) helps a little;** `harmonics 2` trades
  exact tempo (acc1 −0.02) for fewer octave errors (acc2 +0.014) and better
  downbeats.
- **Downbeats are the weak part** (0.62–0.65). Next: find out which cue
  helps (accent, low end, harmony) by switching them off one by one.
- **Some references are wrong.** In 3 tracks (Track01895, Track01976,
  Track02038) the MIDI beat grid sits 0.18–0.20 beat away from the file's
  own notes, so a correct tracker scores 0 there. The bench now measures
  this per track (`grid_offset`) and also reports means over the aligned
  tracks only (`beat F aligned`, `means_aligned` in the JSON).

#### Round 2: tempo prior, downbeat cues, tightness (2026-10-06, commit dfbc6f0+dirty)

Runs `20261006-180839` (prior), `20261006-180955` (downbeat cues),
`20261006-181346` (tightness), all with tightness 400 unless varied.

Tempo prior (centre / width):

| start_bpm | std_octaves | beat F | F any lvl | acc1 | acc2 |
| --- | --- | --- | --- | --- | --- |
| **120** | **1** | **0.824** | **0.883** | **0.821** | 0.960 |
| 100 | 1 | 0.799 | 0.867 | 0.762 | 0.934 |
| 90 | 1 | 0.760 | 0.856 | 0.656 | 0.934 |
| 120 | 0.5 | 0.773 | 0.871 | 0.695 | 0.940 |
| 100 | 0.5 | 0.793 | 0.849 | 0.781 | 0.874 |

Downbeat cues (weights of accent / low end / harmony), downbeat F:

| accent | low | harmony | downbeat F |
| --- | --- | --- | --- |
| 0 | 0 | 0 | 0.511 (no cue: the first tracked beat is taken as "one") |
| 1 | 0 | 0 | 0.310 |
| 0 | 1 | 0 | 0.601 |
| 0 | 0 | 1 | 0.689 |
| 1 | 1 | 1 | 0.637 (round 1 default) |
| 1 | 1 | 2 | 0.667 |
| **0** | **1** | **2** | **0.692** |

Tightness 400 / 800 / 1600: beat F 0.824 / 0.823 / 0.821 (plateau).

Findings:

- **The 120 BPM, one-octave prior is right** for this material; a lower
  centre or a narrower prior only adds half-tempo errors.
- **Chord changes are the best downbeat cue** (0.689 alone), the low end
  (kick, bass) helps a little on top, and **onset strength hurts**: alone
  it scores 0.31, barely above a random phase (0.25), because the loudest
  hit in pop and rock is the snare backbeat on 2 and 4.
- Even with no cue at all, 51 % of downbeats are right: most songs start
  on a downbeat and our first tracked beat often lands there. A small prior
  for "the music starts on one" could be a cheap extra cue.
- **New defaults:** tightness 400, accent 0, low 1, harmony 2. Expected on
  the test split: beat F 0.824, downbeat F 0.692 (to be confirmed by one
  run on the committed code).

### Summary

Confirmation run with the final defaults: `20261006-183333-beats-…-real`.

| | beat F | F any lvl | downbeat F | tempo acc1 / acc2 | meter |
| --- | --- | --- | --- | --- | --- |
| first version | 0.818 | 0.875 | 0.622 | 0.821 / 0.960 | 0.967 |
| tuned defaults, all 151 tracks | **0.824** | 0.883 | **0.692** | 0.821 / 0.960 | 0.967 |
| tuned defaults, 141 tracks with an aligned reference grid | 0.864 | 0.924 | 0.728 | 0.830 / 0.972 | 0.972 |

The aligned subset drops 10 tracks whose MIDI grid is more than 0.06 beat
off its own notes (Track01895, 01898, 01903, 01905, 01930, 01976, 01987,
01995, 02026, 02038). Most of them score 0–0.39, as expected for a wrong
reference; Track02026 (offset 0.07, F 0.995) shows that the threshold also
catches a borderline case. The true score lies between the two rows.

For scale: published L1 trackers of this kind (Ellis 2007) typically land
around 0.6–0.7 beat F on hand-annotated pop datasets, and neural trackers
such as Beat This! around 0.85–0.9. Slakh's MIDI-rendered audio is
cleaner than real recordings, so our numbers are not directly comparable;
the L3 comparison on the same data is the fair test.

## Open questions

1. Tempo changes: a single global tempo cannot follow 54 of the 151 test
   tracks. Next step if they dominate the errors: a local tempogram
   (autocorrelation over a sliding window) and a DP over tempo and phase.
2. Octave errors: how often is the level wrong (F any lvl ≫ beat F), and
   does `harmonics` or a different prior centre fix it?
3. Downbeats: the constant-phase assumption breaks on songs with odd bars
   or pickups mid-song; and which cue (accent, low end, harmony) carries the
   most weight on real music?
4. SuperFlux envelope (200 fps) instead of flux for finer beat timing.
5. Reference: Beat This! (L3) for comparison once its adapter exists.
