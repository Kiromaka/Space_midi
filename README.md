# Space MIDI

A Windows desktop app that turns a full mixed song into multi-track MIDI,
one track per instrument.

The heart of the project is research: we implement source separation,
transcription and instrument recognition ourselves, and use off-the-shelf
models only as references to measure against.

> **Status:** early development, stage 1 (own DSP). Nothing converts yet; onset
> detection and beat tracking work and are benchmarked on Slakh2100.

## Planned features

- Split a mix into stems: vocals, drums, bass, piano, guitar, other
- Transcribe each stem into notes: pitch, onset, offset, velocity, pitch bend
- Detect tempo, beats and downbeats; quantize to the real beat grid
- Recognize instrument groups, later exact General MIDI programs
- Write a Standard MIDI File (type 1): one track and channel per instrument,
  drums on channel 10
- Export to the Space Station 13 instrument text format
- GUI: stem view, piano roll, A/B listening, re-running a single stage

## Approach

Every pipeline stage gets up to three interchangeable backends, compared on
the same metrics:

| Level | What | Example (transcription) |
| --- | --- | --- |
| L1 | Own classical DSP, no training | NMF with harmonic templates |
| L2 | Own neural network, own training | CRNN onsets + frames on CQT |
| L3 | Off-the-shelf model, reference only | Basic Pitch, Transkun |

"Own" means built on numpy, scipy and PyTorch primitives. librosa, mido,
mir_eval and ready-made models are used only in tests, metrics and as
references.

## Requirements

- Windows 10/11, x64
- NVIDIA GPU with a driver that supports CUDA 12.8 (developed on an
  RTX 3060, 12 GB). CPU works, but slowly.
- [uv](https://docs.astral.sh/uv/) for the Python core (it installs Python 3.12 itself)
- JDK 24 for the GUI

## Getting started

Install uv:

```powershell
winget install --id=astral-sh.uv -e
```

Set up and check the core:

```powershell
cd core
uv sync --extra nn       # dependencies + PyTorch with CUDA 12.8
uv run spacemidi info    # versions and CUDA status
uv run pytest
uv run spacemidi to-midi song.notes.json song.mid   # notes -> MIDI
uv run spacemidi from-midi song.mid song.notes.json # MIDI -> notes
uv run spacemidi eval estimate.notes.json reference.mid  # score against a reference
uv run spacemidi synth song.notes.json song.wav          # test audio with a known answer
uv run spacemidi onsets song.wav --midi onsets.mid       # detect onsets; clicks to check in a DAW
uv run spacemidi bench onsets --limit 20 --jobs 4        # score onset detection on Slakh
uv run spacemidi beats song.wav --midi beats.mid         # tempo, beats, bars; a click track with the tempo map
uv run spacemidi bench beats --jobs 4                    # score beat tracking on Slakh
```

Datasets live outside the repository; point `SPACEMIDI_DATA` at their folder
once (`setx SPACEMIDI_DATA D:\SPACEMIDI_DATA`), and `uv run spacemidi info`
shows what it finds.

Docs: [note JSON](docs/formats/notes.md), [MIDI export and import](docs/formats/midi.md),
[evaluation](docs/research/evaluation.md),
[spectral core](docs/research/stage1-spectral.md), [onset detection](docs/research/stage1-onsets.md),
[beat tracking](docs/research/stage1-beats.md).

Behind a proxy, set `HTTPS_PROXY`. If the proxy replaces TLS certificates,
also set `UV_NATIVE_TLS=1` so uv trusts the Windows certificate store.

### IntelliJ IDEA

Open the repository root. IntelliJ imports it as a Gradle project
(`settings.gradle.kts`) and fetches Gradle 8.14.3 through
`gradle/wrapper/gradle-wrapper.properties`. The project JDK is 24.

Shared run configurations:

| Configuration | Runs |
| --- | --- |
| GUI | `:gui:run` |
| Core: tests | `uv run pytest` in `core/` |
| Core: info | `uv run spacemidi info` |
| Core: sync dependencies | `uv sync --extra nn` |

The core tasks call `uv`, so it must be on the PATH IntelliJ started with.
Restart IntelliJ after installing uv.

## Layout

```
gui/             Java app (Gradle, JDK 24)
core/            Python core (uv)
  spacemidi/
    dsp/         own signal-processing building blocks
    algos/       own classical algorithms (L1)
    nn/          own neural networks (L2)
    refs/        reference model adapters (L3)
    notes/       intermediate note format
    export/      MIDI, SS13
    pipeline/    orchestrator, cache
    eval/        metrics, datasets, benchmarks
  tests/
experiments/     notebooks and training configs
docs/research/   algorithm notes and results
data/            datasets, test set, checkpoints (not in git)
```

## Roadmap

- [x] **0. Foundation:** repo layout ✓, note format ✓, own MIDI reader/writer ✓, evaluation ✓ (L3 reference adapters added when first needed)
- [ ] **1. Own DSP:** STFT/CQT/chroma ✓, onset detection ✓, tempo/beats/downbeats ✓ (benchmarking), HPSS, YIN/pYIN, key
- [ ] **2. Classical pipeline (L1) + minimal GUI**
- [ ] **3. Own neural networks (L2):** transcription, drums, F0, separation
- [ ] **4. Instrument detail:** timbre classifier, GM programs
- [ ] **5. Polish and packaging:** installer, batch mode, note editing
- [ ] **6. Space Station 13 export**

## License

Space MIDI is licensed under the GNU Affero General Public License v3.0 or
later; see [LICENSE-AGPL3.txt](LICENSE-AGPL3.txt). If you run a modified
version as a network service, you must offer its users the source code of
your version (AGPL section 13).

Reference models (L3) are optional, are not bundled, and keep their own
licenses. Some of them are GPL-3.0 (YourMT3) or non-commercial
CC BY-NC-SA 4.0 (ADTOF).
