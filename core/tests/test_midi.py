import io

import pytest

from notes_example import make_example
from spacemidi.cli import main
from spacemidi.midi import MidiExportError, MidiFormatError, MidiOptions, encode_midi, read_midi
from spacemidi.midi import smf
from spacemidi.midi.tempo import TickClock, clock_from_beats
from spacemidi.notes import Key, Note, NoteDocument, TempoMap, Track, load, save, validate

TICK_TOLERANCE = 0.0015  # seconds; one tick at 480 ppq and 120 BPM is ~1.04 ms


def _doc(*tracks, tempo=None, key=None):
    return NoteDocument(tracks=list(tracks), tempo=tempo, key=key)


def _note_events(data: bytes):
    """(track index, tick, status, pitch, velocity) of every note on/off in a file."""
    out = []
    for i, events in enumerate(smf.parse_file(data).tracks):
        for tick, ev in events:
            if ev[0] & 0xF0 in (smf.NOTE_ON, smf.NOTE_OFF):
                out.append((i, tick, ev[0], ev[1], ev[2]))
    return out


# ---------------------------------------------------------------- byte level


VLQ_CASES = [  # from the SMF 1.1 specification
    (0x00, "00"), (0x40, "40"), (0x7F, "7F"), (0x80, "81 00"), (0x2000, "C0 00"),
    (0x3FFF, "FF 7F"), (0x4000, "81 80 00"), (0x100000, "C0 80 00"), (0x1FFFFF, "FF FF 7F"),
    (0x200000, "81 80 80 00"), (0x8000000, "C0 80 80 00"), (0xFFFFFFF, "FF FF FF 7F"),
]


@pytest.mark.parametrize("value,hex_bytes", VLQ_CASES)
def test_vlq_matches_spec(value, hex_bytes):
    encoded = smf.encode_vlq(value)
    assert encoded == bytes.fromhex(hex_bytes)
    assert smf.decode_vlq(encoded + b"\x99", 0) == (value, len(encoded))


def test_vlq_rejects_out_of_range():
    with pytest.raises(ValueError):
        smf.encode_vlq(0x10000000)
    with pytest.raises(MidiFormatError):
        smf.decode_vlq(b"\x81\x80", 0)


def test_header_chunk():
    data = smf.encode_file([[]], ppq=480)
    assert data[:14] == b"MThd" + bytes.fromhex("00000006 0001 0001 01E0")
    assert data[14:18] == b"MTrk"


def test_running_status_and_meta_cancel_it():
    events = [
        (0, bytes([0x90, 60, 100])),
        (10, bytes([0x90, 64, 100])),  # same status: omitted
        (10, smf.meta(smf.META_TEXT, b"x")),
        (20, bytes([0x90, 67, 100])),  # after a meta event: status written again
    ]
    body = smf.encode_track(events)[8:]
    assert body.hex(" ") == "00 90 3c 64 0a 40 64 00 ff 01 01 78 0a 90 43 64 00 ff 2f 00"
    parsed = smf.parse_file(smf.encode_file([events], 96)).tracks[0]
    assert [e for e in parsed if e[1][0] == 0x90] == events[:2] + events[3:]


def test_rejects_non_midi():
    with pytest.raises(MidiFormatError):
        read_midi(b"RIFF....WAVE")
    smpte = b"MThd" + bytes.fromhex("00000006 0000 0001 E728")
    with pytest.raises(MidiFormatError, match="SMPTE"):
        read_midi(smpte)


# ---------------------------------------------------------------- round trip


def test_round_trip_of_the_example():
    doc = make_example()
    back = read_midi(encode_midi(doc))
    assert validate(back) == []
    assert back.key == Key("A", "minor")
    assert back.source == doc.source and back.tuning_hz == doc.tuning_hz
    assert [t.id for t in back.tracks] == [t.id for t in doc.tracks]
    for orig, got in zip(doc.tracks, back.tracks):
        assert (got.group, got.stem, got.program, got.is_drum) == (orig.group, orig.stem, orig.program, orig.is_drum)
        assert got.backend == orig.backend
        a = sorted(orig.notes, key=lambda n: (n.onset, n.pitch))
        b = sorted(got.notes, key=lambda n: (n.onset, n.pitch))
        assert [(n.pitch, n.velocity) for n in a] == [(n.pitch, n.velocity) for n in b]
        for x, y in zip(a, b):
            assert abs(x.onset - y.onset) < TICK_TOLERANCE
            assert abs(x.offset - y.offset) < TICK_TOLERANCE
        assert [(c.cc, c.value) for c in orig.controls] == [(c.cc, c.value) for c in got.controls]
        for x, y in zip(orig.controls, got.controls):
            assert abs(x.time - y.time) < TICK_TOLERANCE
    vocal = back.track("vocals").notes
    assert [round(c) for _, c in vocal[0].bend] == [-35, 0]
    assert [round(c) for _, c in vocal[2].bend] == [0, 40, -40]
    n = len(doc.tempo.beats)
    assert back.tempo.beats[:n] == pytest.approx(doc.tempo.beats, abs=1e-4)


def test_drums_on_channel_10_and_melodic_channels_distinct():
    data = encode_midi(make_example())
    channels = {}
    for track, _, status, _, _ in _note_events(data):
        channels.setdefault(track, set()).add(status & 0x0F)
    assert channels[2] == {9}  # drums are the second instrument track (file track 2)
    melodic = [channels[1], channels[3]]
    assert all(len(c) == 1 and 9 not in c for c in melodic)
    assert melodic[0] != melodic[1]


def test_note_ons_and_offs_balance():
    track = Track(id="p", group="piano", notes=[Note(0.0, 1.0, 60), Note(0.5, 1.5, 60), Note(0.5, 1.5, 60)])
    events = _note_events(encode_midi(_doc(track)))
    ons = [e for e in events if e[2] & 0xF0 == 0x90 and e[4] > 0]
    offs = [e for e in events if e[2] & 0xF0 == 0x80]
    assert len(ons) == len(offs) == 2  # the exact duplicate is dropped
    assert offs[0][1] == ons[1][1]  # the first note is cut where the second starts


# ---------------------------------------------------------------- tempo


def test_no_tempo_means_120_bpm():
    track = Track(id="b", group="bass", notes=[Note(1.0, 1.5, 40)])
    events = _note_events(encode_midi(_doc(track)))
    assert events[0][1] == 960  # 1 s at 120 BPM = 2 beats of 480 ticks


def test_tempo_follows_detected_beats():
    intervals = [0.6 - 0.01 * i for i in range(20)]  # accelerando
    beats = [0.0]
    for d in intervals:
        beats.append(beats[-1] + d)
    notes = [Note(b, b + 0.1, 60) for b in beats[:-1]]
    doc = _doc(Track(id="p", group="piano", notes=notes), tempo=TempoMap(beats=beats, downbeats=beats[::4]))
    data = encode_midi(doc)
    ons = [e[1] for e in _note_events(data) if e[2] & 0xF0 == 0x90]
    assert ons == [i * 480 for i in range(len(notes))]  # every beat on the MIDI beat grid
    back = read_midi(data).track("p").notes
    assert [n.onset for n in back] == pytest.approx([n.onset for n in notes], abs=TICK_TOLERANCE)


def test_first_downbeat_lands_on_a_bar_line():
    """Beats from 0.3 s, first downbeat on the second beat: a pickup bar keeps the intro tempo sane."""
    beats = [0.3 + 0.5 * i for i in range(12)]
    grid = clock_from_beats(beats, beats[1::4], (4, 4), 480)
    lead_in_bpm = 60e6 / grid.clock.changes[0][1]
    assert 90 <= lead_in_bpm <= 160  # near the song's 120 BPM (intro rounded to 16th notes)
    assert (grid.lead_in + 480 - grid.pickup) % grid.bar == 0

    doc = _doc(Track(id="p", group="piano", notes=[Note(0.8, 1.0, 60)]),
               tempo=TempoMap(beats=beats, downbeats=beats[1::4]))
    data = encode_midi(doc)
    sigs = [(t, ev[2], 2 ** ev[3]) for t, ev in smf.parse_file(data).tracks[0] if ev[1] == smf.META_TIME_SIGNATURE]
    assert sigs[-1][1:] == (4, 4) and len(sigs) == 2  # a pickup bar, then 4/4
    back = read_midi(data).tempo
    assert back.time_signature == (4, 4)
    assert back.downbeats[:3] == pytest.approx(beats[1::4][:3], abs=1e-4)


def test_downbeat_on_first_beat_needs_no_pickup():
    beats = [0.5 * i for i in range(9)]
    grid = clock_from_beats(beats, beats[::4], (4, 4), 480)
    assert grid.lead_in == 0 and grid.pickup == 0


def test_tick_clock_is_invertible():
    clock = TickClock(480, [(0, 500_000), (960, 400_000), (1920, 650_000)])
    for tick in (0, 100, 960, 1500, 1920, 5000):
        assert clock.tick(clock.seconds(tick)) == pytest.approx(tick)
    assert clock.seconds(960) == pytest.approx(1.0)


def test_grid_quantize():
    beats = [0.5 * i for i in range(9)]
    notes = [Note(0.52, 0.7, 60), Note(1.13, 1.3, 62)]
    doc = _doc(Track(id="p", group="piano", notes=notes), tempo=TempoMap(beats=beats, downbeats=beats[::4]))
    full = [e[1] for e in _note_events(encode_midi(doc, MidiOptions(grid=4))) if e[2] & 0xF0 == 0x90]
    assert full == [480, 1080]  # 16th-note grid = 120 ticks
    half = [e[1] for e in _note_events(encode_midi(doc, MidiOptions(grid=4, strength=0.5))) if e[2] & 0xF0 == 0x90]
    raw = [e[1] for e in _note_events(encode_midi(doc)) if e[2] & 0xF0 == 0x90]
    assert all(abs(h - (r + f) / 2) <= 1 for h, r, f in zip(half, raw, full))


# ---------------------------------------------------------------- metadata and limits


@pytest.mark.parametrize("key,sharps", [(Key("E", "major"), 4), (Key("F", "major"), -1), (Key("C#", "minor"), 4), (Key("A#", "major"), -2)])
def test_key_signature(key, sharps):
    data = encode_midi(_doc(Track(id="p", group="piano", notes=[Note(0, 1, 60)]), key=key))
    sig = [ev for _, ev in smf.parse_file(data).tracks[0] if ev[1] == smf.META_KEY_SIGNATURE][0]
    assert int.from_bytes(sig[2:3], "big", signed=True) == sharps
    assert read_midi(data).key == key


def test_too_many_melodic_tracks():
    tracks = [Track(id=f"t{i}", group="other", notes=[Note(0, 1, 60)]) for i in range(16)]
    with pytest.raises(MidiExportError, match="16 melodic tracks"):
        encode_midi(_doc(*tracks))


def test_reads_a_foreign_format_0_file():
    """One track, two channels, running status and note-on velocity 0 as note-off."""
    events = [
        (0, smf.text_event(smf.META_TRACK_NAME, "Band")),
        (0, bytes([0xC1, 33])),  # bass program on channel 2
        (0, bytes([0x91, 40, 90])),
        (0, bytes([0x99, 36, 110])),  # kick on channel 10
        (240, bytes([0x99, 36, 0])),
        (480, bytes([0x91, 40, 0])),
        (480, bytes([0x91, 43, 80])),
        (960, bytes([0x91, 43, 0])),
    ]
    doc = read_midi(smf.encode_file([events], 480, fmt=0))
    assert validate(doc) == []
    by_group = {t.group: t for t in doc.tracks}
    assert set(by_group) == {"bass", "drums"}
    assert [(n.pitch, n.onset, n.offset) for n in by_group["bass"].notes] == [(40, 0.0, 0.5), (43, 0.5, 1.0)]
    assert by_group["drums"].is_drum and by_group["drums"].notes[0].offset == 0.25


def test_mido_reads_our_files_the_same_way():
    mido = pytest.importorskip("mido")
    data = encode_midi(make_example())
    ours = sorted((round(n.onset, 3), n.pitch) for t in read_midi(data).tracks for n in t.notes)
    now, theirs = 0.0, []
    for msg in mido.MidiFile(file=io.BytesIO(data)):
        now += msg.time
        if msg.type == "note_on" and msg.velocity > 0:
            theirs.append((round(now, 3), msg.note))
    assert sorted(theirs) == ours


def test_cli_round_trip(tmp_path, capsys):
    notes_in = tmp_path / "in.notes.json"
    save(make_example(), notes_in)
    mid = tmp_path / "out.mid"
    assert main(["to-midi", str(notes_in), str(mid)]) == 0
    assert mid.read_bytes()[:4] == b"MThd"
    notes_out = tmp_path / "back.notes.json"
    assert main(["from-midi", str(mid), str(notes_out)]) == 0
    assert len(load(notes_out).track("drums").notes) == 5
    assert main(["from-midi", str(notes_in), str(notes_out)]) == 1
    assert "FAIL" in capsys.readouterr().out
