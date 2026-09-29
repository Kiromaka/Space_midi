# MIDI export and import

`spacemidi.midi` reads and writes Standard MIDI Files without any MIDI
library: byte encoding (`smf.py`), tempo maps (`tempo.py`), the writer
(`writer.py`) and the reader (`reader.py`). mido is used only in tests, as an
independent check.

```
uv run spacemidi to-midi song.notes.json song.mid              # as played
uv run spacemidi to-midi song.notes.json song.mid --grid 4     # snapped to 16th notes (in 4/4)
uv run spacemidi to-midi song.notes.json song.mid --grid 4 --strength 0.5
uv run spacemidi from-midi reference.mid reference.notes.json
```

[`example.mid`](example.mid) is [`example.notes.json`](example.notes.json)
written with default options.

## File layout

Standard MIDI File type 1, 480 ticks per quarter note by default.

| Track | Contents |
| --- | --- |
| 0, conductor | Name, `spacemidi` metadata, time signature(s), key signature, tempo map; ends at the last detected beat |
| 1 … n | One per instrument: name, `spacemidi` metadata, Program Change, pitch-bend range (if needed), notes, controllers, pitch bend |

- **Channels:** drums always on channel 10; every other track gets its own
  channel. More than 15 melodic tracks is an error; merge tracks first.
- **Programs:** the track's `program`, or a default per group: vocals 53
  (Voice Oohs), bass 33 (Electric Bass, finger), piano 0, guitar 25
  (Acoustic Guitar, steel), other 48 (String Ensemble 1). Numbers are 0-based.
- **Pitch bend:** the range is set per track with RPN 0, just wide enough for
  its largest bend (at least ±2, at most ±24 semitones). Bend curves are
  sent while the note sounds and reset to centre when it ends.
- **Overlapping notes of the same pitch** are cut where the next one starts,
  exact duplicates are dropped, and every note lasts at least one tick, so a
  note can never hang.
- **Metadata:** `spacemidi {json}` text events carry the source, tuning and
  per-track id, group, stem and backend. Other programs ignore them; our
  reader uses them to restore the document exactly.

## Timing

Times in the note format are seconds; MIDI needs ticks and tempo events.
With a beat grid in the document:

1. **Every detected beat becomes a MIDI beat.** Beat *i* sits at tick
   `lead_in + i × ticks_per_beat`, and each beat interval gets its own tempo
   event. Notes stay aligned with the audio whatever the tempo does, and in a
   DAW the bar lines follow the song.
2. **The intro before the first beat keeps the opening tempo**, rounded to a
   16th of a beat.
3. **The first downbeat starts a bar.** If it would not, the first bar is a
   shorter pickup bar (for example 1/4 or 3/16), followed by the song's own
   time signature.

Without a beat grid the file is 120 BPM in 4/4.

Rounding to whole ticks moves an event by at most half a tick: about 0.5 ms
at 120 BPM.

**Grid mode** (`--grid N`) pulls every note start and end toward the nearest
1/N of a beat by `--strength` (1 = fully on the grid). The intro before the
first beat is never snapped.

## Reading other MIDI files

Any format 0 or 1 file with tick-based timing works. Notes are split per
(track, channel); the group comes from the channel (10 = drums) and the
General MIDI program. Beats are rebuilt from the tempo map and the time
signature that covers most of the file, and downbeats follow every
time-signature change.
