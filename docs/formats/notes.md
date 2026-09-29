# Note format: `spacemidi-notes` v1

Every pipeline stage writes its result in this format, and every consumer
reads it: evaluation, the GUI piano roll, the MIDI writer and the SS13
exporter. MIDI is only an export target; this file is the source of truth.

Files use the extension `.notes.json`. The machine-readable schema is
[`notes.schema.json`](notes.schema.json); the reference implementation is
`core/spacemidi/notes`. [`example.notes.json`](example.notes.json) shows
every field.

## Conventions

- **Time is seconds** from the start of the source audio, as floats. Notes
  stay aligned with the audio whatever the tempo does. Beats and bars are
  derived from `tempo` at export time, never stored on notes.
- **Pitch is a MIDI note number** (60 = C4, 69 = A4). On drum tracks it is
  the General MIDI percussion key (36 kick, 38 snare, 42 closed hi-hat, ...).
- **Pitch bend is per note**: `[seconds after onset, cents]` pairs relative
  to the note's pitch. Exporters turn it into MIDI pitch-bend messages.
- **MIDI channels are not stored.** The MIDI exporter assigns them (drums on
  channel 10).
- **Notes are written sorted** by onset, then pitch. Readers must not rely
  on the order.
- **Unknown keys are ignored** on read, so adding an optional field does not
  break older readers.

## Versioning

`version` is an integer. Adding optional fields keeps the version. Removing
or renaming a field, or changing its meaning, increments it. Readers reject
files with a version newer than they support.

## Document

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `format` | string | yes | Always `"spacemidi-notes"` |
| `version` | integer | yes | Format version, currently `1` |
| `source` | object | no | `path`, `duration` (s), `sample_rate` (Hz) of the source audio |
| `tuning_hz` | number | no | Reference pitch of A4, 400–480, default 440 |
| `tempo` | object | no | Beat grid, see below |
| `key` | object | no | `tonic` (C, C#, ..., B) and `mode` (`major` / `minor`) |
| `tracks` | array | yes | One entry per instrument |

## Tempo

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `beats` | number[] | yes | Beat times in seconds, strictly increasing |
| `downbeats` | number[] | no | Beats that start a bar; each must also be in `beats` |
| `time_signature` | [int, int] | no | Dominant signature, default `[4, 4]` |

## Track

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `id` | string | yes | Unique within the document, e.g. `"vocals"`, `"other-2"` |
| `name` | string | no | Display name |
| `group` | string | yes | `vocals`, `drums`, `bass`, `piano`, `guitar` or `other` |
| `stem` | string | no | Stem the notes were transcribed from |
| `program` | integer | no | General MIDI program 0–127; absent when unknown or on drum tracks |
| `is_drum` | boolean | no | Drum track, default `false` |
| `backend` | object | no | Producer: `stage`, `name`, `version`, `params` |
| `notes` | array | yes | Notes, see below |
| `controls` | array | no | Controller changes: `time` (s), `cc`, `value` (0–127); sustain pedal is cc 64 |

## Note

| Field | Type | Required | Meaning |
| --- | --- | --- | --- |
| `onset` | number | yes | Start, seconds, ≥ 0 |
| `offset` | number | yes | End, seconds, > onset |
| `pitch` | integer | yes | MIDI note number 0–127 |
| `velocity` | integer | no | 1–127, default 80 |
| `confidence` | number | no | Backend confidence 0–1 |
| `bend` | [number, number][] | no | Pitch curve: seconds after onset (strictly increasing, within the note), cents (±2400) |

## Usage

Python:

```python
from spacemidi.notes import Note, NoteDocument, Track, load, save

doc = NoteDocument(tracks=[Track(id="bass", group="bass", program=33)])
doc.track("bass").notes.append(Note(onset=0.5, offset=0.9, pitch=40, velocity=96))
save(doc, "song.notes.json")      # validates, then writes
doc = load("song.notes.json")     # parses and validates
```

Command line:

```
uv run spacemidi validate song.notes.json
```

Java (GUI): validate against `notes.schema.json` with any JSON Schema
2020-12 library. The schema cannot express cross-field rules (offset > onset,
increasing beats, downbeats present in beats, unique track ids), so the GUI
gets files that the Python core has already validated.
