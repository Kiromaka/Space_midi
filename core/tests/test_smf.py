"""Unit tests for spacemidi.midi.smf (Standard MIDI File byte format).

Expected values come from "Standard MIDI-File Format Spec. 1.1" (MIDI
Manufacturers Association), so each test checks the module against the
published format rather than against its own output. One group of tests per
function; each group covers normal values, edge values and invalid input.
"""

import pytest

from spacemidi.midi import smf
from spacemidi.midi.smf import MidiFormatError

END_OF_TRACK = "00 ff 2f 00"  # delta 0 + End of Track meta event


def hexb(text: str) -> bytes:
    """bytes.fromhex that also accepts spaces between bytes."""
    return bytes.fromhex(text)


def raw_file(track_body: str, ntracks: int = 1, fmt: int = 0, ppq: int = 96) -> bytes:
    """A file built by hand (header + one track), bypassing our encoder."""
    body = hexb(track_body)
    header = b"MThd" + (6).to_bytes(4, "big") + fmt.to_bytes(2, "big") + ntracks.to_bytes(2, "big") + ppq.to_bytes(2, "big")
    return header + b"MTrk" + len(body).to_bytes(4, "big") + body


# ---------------------------------------------------------------- 1. encode_vlq

# The example table of the SMF 1.1 specification ("Variable Length Quantities").
SPEC_VLQ = [
    (0x00000000, "00"),
    (0x00000040, "40"),
    (0x0000007F, "7f"),
    (0x00000080, "81 00"),
    (0x00002000, "c0 00"),
    (0x00003FFF, "ff 7f"),
    (0x00004000, "81 80 00"),
    (0x00100000, "c0 80 00"),
    (0x001FFFFF, "ff ff 7f"),
    (0x00200000, "81 80 80 00"),
    (0x08000000, "c0 80 80 00"),
    (0x0FFFFFFF, "ff ff ff 7f"),
]


@pytest.mark.parametrize("value,expected", SPEC_VLQ)
def test_encode_vlq_spec_table(value, expected):
    assert smf.encode_vlq(value) == hexb(expected)


@pytest.mark.parametrize("value", [-1, 0x10000000])
def test_encode_vlq_rejects_out_of_range(value):
    with pytest.raises(ValueError, match="out of range"):
        smf.encode_vlq(value)


# ---------------------------------------------------------------- 2. decode_vlq


@pytest.mark.parametrize("value,encoded", SPEC_VLQ)
def test_decode_vlq_spec_table(value, encoded):
    data = hexb(encoded)
    assert smf.decode_vlq(data, 0) == (value, len(data))


def test_decode_vlq_starts_at_pos_and_stops_after_last_byte():
    data = hexb("aa bb 81 00 99")  # junk, VLQ 0x80, junk
    assert smf.decode_vlq(data, 2) == (0x80, 4)


def test_decode_vlq_truncated():
    with pytest.raises(MidiFormatError, match="truncated"):
        smf.decode_vlq(hexb("81 80"), 0)  # continuation bit set on the last byte


def test_decode_vlq_longer_than_four_bytes():
    with pytest.raises(MidiFormatError, match="longer than 4 bytes"):
        smf.decode_vlq(hexb("81 80 80 80 00"), 0)


# ---------------------------------------------------------------- 3. meta


def test_meta_builds_ff_type_payload():
    assert smf.meta(smf.META_TEXT, b"hi") == hexb("ff 01 68 69")


def test_meta_without_payload():
    assert smf.meta(smf.META_END_OF_TRACK) == hexb("ff 2f")


# ---------------------------------------------------------------- 4. channel_message


def test_channel_message_note_on():
    # Note On, channel 1 (index 0), middle C (60), velocity 100.
    assert smf.channel_message(smf.NOTE_ON, 0, 60, 100) == hexb("90 3c 64")


def test_channel_message_puts_channel_in_low_nibble():
    assert smf.channel_message(smf.NOTE_OFF, 9, 38, 0) == hexb("89 26 00")


def test_channel_message_with_one_data_byte():
    assert smf.channel_message(smf.PROGRAM_CHANGE, 2, 33) == hexb("c2 21")


@pytest.mark.parametrize("channel", [-1, 16])
def test_channel_message_rejects_bad_channel(channel):
    with pytest.raises(ValueError, match="channel"):
        smf.channel_message(smf.NOTE_ON, channel, 60, 100)


@pytest.mark.parametrize("data", [(128, 100), (60, -1)])
def test_channel_message_rejects_bad_data_byte(data):
    with pytest.raises(ValueError, match="data byte"):
        smf.channel_message(smf.NOTE_ON, 0, *data)


# ---------------------------------------------------------------- 5. tempo_event


def test_tempo_event_120_bpm():
    # 120 BPM = 500 000 microseconds per quarter note = 0x07A120 (spec example).
    assert smf.tempo_event(500_000) == hexb("ff 51 07 a1 20")


@pytest.mark.parametrize("tempo", [0, 0x1000000])
def test_tempo_event_rejects_out_of_range(tempo):
    with pytest.raises(ValueError, match="tempo"):
        smf.tempo_event(tempo)


# ---------------------------------------------------------------- 6. text_event


def test_text_event_ascii():
    assert smf.text_event(smf.META_TRACK_NAME, "Bass") == hexb("ff 03") + b"Bass"


def test_text_event_is_utf8():
    assert smf.text_event(smf.META_TEXT, "Бас") == hexb("ff 01") + "Бас".encode("utf-8")


# ---------------------------------------------------------------- 7. encode_track


def test_encode_track_empty_gets_end_of_track():
    assert smf.encode_track([]) == b"MTrk" + hexb("00 00 00 04") + hexb(END_OF_TRACK)


def test_encode_track_writes_delta_times():
    events = [(0, hexb("90 3c 64")), (200, hexb("80 3c 00"))]
    body = smf.encode_track(events)[8:]
    assert body == hexb("00 90 3c 64  81 48 80 3c 00") + hexb(END_OF_TRACK)  # 200 = 81 48


def test_encode_track_chunk_length():
    track = smf.encode_track([(0, hexb("90 3c 64"))])
    assert int.from_bytes(track[4:8], "big") == len(track) - 8


def test_encode_track_running_status_and_meta_cancels_it():
    events = [
        (0, hexb("90 3c 64")),
        (10, hexb("90 40 64")),  # same status: omitted
        (10, smf.meta(smf.META_TEXT, b"x")),
        (20, hexb("90 43 64")),  # after a meta event: written again
    ]
    body = smf.encode_track(events)[8:]
    assert body == hexb("00 90 3c 64  0a 40 64  00 ff 01 01 78  0a 90 43 64") + hexb(END_OF_TRACK)


def test_encode_track_sysex_gets_length():
    body = smf.encode_track([(0, hexb("f0 7e 7f 09 01 f7"))])[8:]
    assert body == hexb("00 f0 05 7e 7f 09 01 f7") + hexb(END_OF_TRACK)


def test_encode_track_keeps_existing_end_of_track():
    body = smf.encode_track([(5, smf.meta(smf.META_END_OF_TRACK))])[8:]
    assert body == hexb("05 ff 2f 00")


def test_encode_track_rejects_unsorted_events():
    with pytest.raises(ValueError, match="sorted"):
        smf.encode_track([(10, hexb("90 3c 64")), (5, hexb("80 3c 00"))])


def test_encode_track_rejects_event_after_end_of_track():
    with pytest.raises(ValueError, match="after End of Track"):
        smf.encode_track([(0, smf.meta(smf.META_END_OF_TRACK)), (1, hexb("90 3c 64"))])


# ---------------------------------------------------------------- 8. encode_file


def test_encode_file_header():
    data = smf.encode_file([[], []], ppq=480)
    # "MThd", length 6, format 1, 2 tracks, 480 ticks per quarter note.
    assert data[:14] == b"MThd" + hexb("00 00 00 06  00 01  00 02  01 e0")
    assert data.count(b"MTrk") == 2


def test_encode_file_format_0():
    data = smf.encode_file([[]], ppq=96, fmt=0)
    assert data[8:10] == hexb("00 00")


def test_encode_file_format_0_needs_one_track():
    with pytest.raises(ValueError, match="exactly one track"):
        smf.encode_file([[], []], ppq=96, fmt=0)


@pytest.mark.parametrize("ppq", [0, 0x8000])
def test_encode_file_rejects_bad_ppq(ppq):
    with pytest.raises(ValueError, match="ticks per quarter"):
        smf.encode_file([[]], ppq=ppq)


# ---------------------------------------------------------------- 9. parse_file


def test_parse_file_round_trip():
    tracks = [
        [(0, smf.text_event(smf.META_TRACK_NAME, "Tempo")), (0, smf.tempo_event(500_000))],
        [(0, hexb("90 3c 64")), (480, hexb("80 3c 00")), (480, hexb("90 40 64")), (960, hexb("80 40 00"))],
    ]
    parsed = smf.parse_file(smf.encode_file(tracks, ppq=480))
    assert (parsed.format, parsed.ppq) == (1, 480)
    eot = (lambda tick: (tick, smf.meta(smf.META_END_OF_TRACK)))
    assert parsed.tracks == [tracks[0] + [eot(0)], tracks[1] + [eot(960)]]


def test_parse_file_hand_made_file_with_running_status():
    data = raw_file("00 90 3c 64  60 3c 00  " + END_OF_TRACK)  # second note on uses running status
    parsed = smf.parse_file(data)
    assert (parsed.format, parsed.ppq) == (0, 96)
    assert parsed.tracks[0][:2] == [(0, hexb("90 3c 64")), (96, hexb("90 3c 00"))]


def test_parse_file_skips_unknown_chunks():
    data = smf.encode_file([[(0, hexb("90 3c 64"))]], ppq=96)
    alien = b"XFIH" + (3).to_bytes(4, "big") + b"abc"
    with_alien = data[:14] + alien + data[14:]
    assert smf.parse_file(with_alien).tracks == smf.parse_file(data).tracks


def test_parse_file_rejects_non_midi():
    with pytest.raises(MidiFormatError, match="MThd"):
        smf.parse_file(b"RIFF\x00\x00\x00\x00WAVEfmt ")


def test_parse_file_rejects_smpte_division():
    data = b"MThd" + hexb("00 00 00 06  00 00  00 01  e7 28")
    with pytest.raises(MidiFormatError, match="SMPTE"):
        smf.parse_file(data)


def test_parse_file_rejects_missing_track():
    data = raw_file(END_OF_TRACK, ntracks=2)
    with pytest.raises(MidiFormatError, match="announces 2 tracks"):
        smf.parse_file(data)


def test_parse_file_rejects_truncated_chunk():
    data = raw_file("00 90 3c 64  " + END_OF_TRACK)[:-2]
    with pytest.raises(MidiFormatError, match="truncated"):
        smf.parse_file(data)


def test_parse_file_rejects_data_byte_without_status():
    with pytest.raises(MidiFormatError, match="without a status byte"):
        smf.parse_file(raw_file("00 3c 64  " + END_OF_TRACK))
