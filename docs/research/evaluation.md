# Evaluation

How Space MIDI measures whether an algorithm is any good. Every algorithm
note in `docs/research/` reports numbers produced this way.

## Commands

```
uv run spacemidi eval estimate.notes.json reference.mid        # score table
uv run spacemidi eval estimate.mid reference.notes.json --json scores.json
uv run spacemidi eval estimate.mid reference.notes.json --mir-eval   # cross-check with mir_eval
uv run spacemidi synth song.notes.json song.wav                # test audio with a known answer
uv run spacemidi synth song.notes.json noisy.wav --noise-snr 10
uv run spacemidi info                                          # also shows which datasets are found
```

Both arguments of `eval` accept `.notes.json` or `.mid`.

## Scores

All scores count events matched one-to-one between the estimate and the
reference (maximum bipartite matching, Hopcroft-Karp), then report precision
(how many detected events are real), recall (how many real events were
found) and F1 (their harmonic mean).

| Score | A detected event counts when |
| --- | --- |
| Onset (per group) | same pitch, onset within ±50 ms |
| Onset + offset (per group) | as above, and offset within 20% of the note's length (at least ±50 ms) |
| Drums (per class) | onset within ±50 ms, same class: kick, snare, hi-hat, tom, cymbal, other |
| ALL (instrument) | onset score summed over all groups and drum classes: a note in the wrong group is a miss |
| ALL (any instr.) | onset score of all pitched notes pooled, ignoring the group |
| Beats / downbeats | within ±70 ms |

These are the conventions of [mir_eval](https://github.com/mir-evaluation/mir_eval),
the standard in music-transcription research, so our numbers compare
directly with published results. The scoring code is our own
(`core/spacemidi/eval/`); tests check that it gives exactly the same counts
as mir_eval on randomized data.

`--mir-eval` recomputes every score with mir_eval and prints both side by
side (exit code 2 and a `MISMATCH` line if they ever differ). Use it for any
number that will be compared with published results, and say so next to the
number. mir_eval is a development dependency only; it is installed by
`uv sync` in `core/`, not needed to run Space MIDI.

## Test material

| Source | Where | Correct answer |
| --- | --- | --- |
| Synthetic | `spacemidi synth` from any notes file | exact, by construction |
| Slakh2100 | `raw/slakh2100_flac_redux/` | per-stem MIDI; groups from `metadata.yaml` |
| MUSDB18-HQ | `raw/musdb18hq/` | stems only (no notes): for separation |
| Own test set | `testset/<name>/` | `reference.notes.json` or `reference.mid` next to the audio |

Paths are relative to the data folder, set once with:

```powershell
setx SPACEMIDI_DATA D:\SPACEMIDI_DATA
```

Synthetic audio uses simple built-in sounds (additive harmonics per group,
decaying noise for drums, optional white noise). It is not realistic; it is
exact. An algorithm that fails on it has a bug, and one that passes still
has to prove itself on Slakh and real recordings.

## Recording results

For each experiment, save the JSON (`--json`) and summarize it in the
algorithm's note in `docs/research/`: the data used, the scores, and the
comparison with the previous version and the L3 reference.
