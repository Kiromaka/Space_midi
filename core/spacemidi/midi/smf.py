"""Standard MIDI File (SMF) byte format: encoding and parsing, from scratch.

Reference: "Standard MIDI-File Format Spec. 1.1" (MIDI Manufacturers
Association). This module knows nothing about notes or tempo maps; it turns
lists of timed events into bytes and back.

An event is ``(tick, data)`` where ``tick`` is absolute and ``data`` is:

* a channel message: its status byte and data bytes, e.g. ``b"\\x90\\x3c\\x64"``;
* a meta event: ``b"\\xff" + type + payload`` (the length is added on write);
* a sysex event: ``b"\\xf0" + payload`` (the length is added on write).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

MAX_VLQ = 0x0FFFFFFF

# Meta event types used by Space MIDI.
META_TEXT = 0x01
META_TRACK_NAME = 0x03
META_END_OF_TRACK = 0x2F
META_TEMPO = 0x51
META_TIME_SIGNATURE = 0x58
META_KEY_SIGNATURE = 0x59

# Channel message kinds (upper nibble of the status byte).
NOTE_OFF = 0x80
NOTE_ON = 0x90
CONTROL_CHANGE = 0xB0
PROGRAM_CHANGE = 0xC0
PITCH_BEND = 0xE0

#: Number of data bytes that follow each channel message kind.
_DATA_BYTES = {0x80: 2, 0x90: 2, 0xA0: 2, 0xB0: 2, 0xC0: 1, 0xD0: 1, 0xE0: 2}


class MidiFormatError(ValueError):
    """Raised for bytes that are not a valid Standard MIDI File."""


# ---------------------------------------------------------------- variable-length quantities


def encode_vlq(value: int) -> bytes:
    """Encode a non-negative integer as a MIDI variable-length quantity.

    Seven bits per byte, most significant first; every byte except the last
    has its top bit set. At most four bytes, so values up to 0x0FFFFFFF.
    """
    if not 0 <= value <= MAX_VLQ:
        raise ValueError(f"VLQ value out of range: {value}")
    out = [value & 0x7F]
    value >>= 8
    while value:
        out.append((value & 0x7F) | 0x80)
        value >>= 7
    return bytes(reversed(out))


def decode_vlq(data: bytes, pos: int) -> tuple[int, int]:
    """Decode a VLQ starting at ``pos``; returns ``(value, next_pos)``."""
    value = 0
    for i in range(4):
        if pos >= len(data):
            raise MidiFormatError("truncated variable-length quantity")
        byte = data[pos]
        pos += 1
        value = (value << 7) | (byte & 0x7F)
        if not byte & 0x80:
            return value, pos
    raise MidiFormatError("variable-length quantity longer than 4 bytes")


# ---------------------------------------------------------------- event constructors


def meta(kind: int, payload: bytes = b"") -> bytes:
    return bytes([0xFF, kind]) + payload


def channel_message(kind: int, channel: int, *data: int) -> bytes:
    if not 0 <= channel <= 15:
        raise ValueError(f"channel out of range: {channel}")
    if any(not 0 <= d <= 127 for d in data):
        raise ValueError(f"data byte out of range: {data}")
    return bytes([kind | channel, *data])


def tempo_event(us_per_quarter: int) -> bytes:
    if not 1 <= us_per_quarter <= 0xFFFFFF:
        raise ValueError(f"tempo out of range: {us_per_quarter} us per quarter note")
    return meta(META_TEMPO, us_per_quarter.to_bytes(3, "big"))


def text_event(kind: int, text: str) -> bytes:
    return meta(kind, text.encode("utf-8"))


# ---------------------------------------------------------------- writing


def encode_track(events: list[tuple[int, bytes]]) -> bytes:
    """Encode one track chunk. Events must be sorted by tick.

    Uses running status for channel messages (a repeated status byte is
    omitted); meta and sysex events cancel running status, as the spec says.
    An End of Track event is appended if missing.
    """
    body = bytearray()
    last_tick = 0
    running: int | None = None
    ended = False
    for tick, data in events:
        if ended:
            raise ValueError("event after End of Track")
        if tick < last_tick:
            raise ValueError("events must be sorted by tick")
        body += encode_vlq(tick - last_tick)
        last_tick = tick
        status = data[0]
        if status == 0xFF:
            body += data[:2] + encode_vlq(len(data) - 2) + data[2:]
            running = None
            ended = data[1] == META_END_OF_TRACK
        elif status in (0xF0, 0xF7):
            body += data[:1] + encode_vlq(len(data) - 1) + data[1:]
            running = None
        else:
            if status != running:
                body.append(status)
                running = status
            body += data[1:]
    if not ended:
        body += encode_vlq(0) + meta(META_END_OF_TRACK) + b"\x00"  # length 0
    return b"MTrk" + struct.pack(">I", len(body)) + bytes(body)


def encode_file(tracks: list[list[tuple[int, bytes]]], ppq: int, fmt: int = 1) -> bytes:
    """Encode a whole file: header chunk plus one chunk per track."""
    if not 1 <= ppq <= 0x7FFF:
        raise ValueError(f"ticks per quarter note out of range: {ppq}")
    if fmt == 0 and len(tracks) != 1:
        raise ValueError("format 0 files have exactly one track")
    header = b"MThd" + struct.pack(">IHHH", 6, fmt, len(tracks), ppq)
    return header + b"".join(encode_track(t) for t in tracks)


# ---------------------------------------------------------------- reading


@dataclass
class SmfFile:
    """A parsed file: absolute-tick events per track, in file order."""

    format: int
    ppq: int
    tracks: list[list[tuple[int, bytes]]] = field(default_factory=list)


def parse_file(data: bytes) -> SmfFile:
    """Parse SMF bytes. Meta and sysex events come back without their length field."""
    if data[:4] != b"MThd" or len(data) < 14:
        raise MidiFormatError("not a Standard MIDI File (missing MThd header)")
    length = struct.unpack(">I", data[4:8])[0]
    fmt, ntracks, division = struct.unpack(">HHH", data[8:14])
    if division & 0x8000:
        raise MidiFormatError("SMPTE time division is not supported")
    smf = SmfFile(format=fmt, ppq=division)
    pos = 8 + length
    while pos + 8 <= len(data) and len(smf.tracks) < ntracks:
        kind = data[pos : pos + 4]
        size = struct.unpack(">I", data[pos + 4 : pos + 8])[0]
        chunk = data[pos + 8 : pos + 8 + size]
        if len(chunk) < size:
            raise MidiFormatError("truncated chunk")
        pos += 8 + size
        if kind == b"MTrk":
            smf.tracks.append(_parse_track(chunk))
        # Unknown chunk types are skipped, as the spec requires.
    if len(smf.tracks) != ntracks:
        raise MidiFormatError(f"header announces {ntracks} tracks, found {len(smf.tracks)}")
    return smf


def _parse_track(chunk: bytes) -> list[tuple[int, bytes]]:
    # The spec says meta and sysex events cancel running status, but real files
    # (Slakh's Track01998/all_src.mid, for one) keep using it after a meta event.
    # Like most readers we accept that: running status survives meta and sysex.
    # A data byte where no channel message came before is still an error.
    events: list[tuple[int, bytes]] = []
    pos = 0
    tick = 0
    running: int | None = None
    while pos < len(chunk):
        delta, pos = decode_vlq(chunk, pos)
        tick += delta
        if pos >= len(chunk):
            raise MidiFormatError("truncated event")
        status = chunk[pos]
        if status == 0xFF:
            if pos + 2 > len(chunk):
                raise MidiFormatError("truncated meta event")
            kind = chunk[pos + 1]
            n, start = decode_vlq(chunk, pos + 2)
            events.append((tick, bytes([0xFF, kind]) + chunk[start : start + n]))
            pos = start + n
            if kind == META_END_OF_TRACK:
                break
        elif status in (0xF0, 0xF7):
            n, start = decode_vlq(chunk, pos + 1)
            events.append((tick, bytes([status]) + chunk[start : start + n]))
            pos = start + n
        else:
            if status & 0x80:
                running = status
                pos += 1
            elif running is None:
                raise MidiFormatError("data byte without a status byte")
            if running & 0xF0 == 0xF0:
                raise MidiFormatError(f"unexpected system message 0x{running:02X} in a track")
            count = _DATA_BYTES[running & 0xF0]
            payload = chunk[pos : pos + count]
            if len(payload) < count:
                raise MidiFormatError("truncated channel message")
            events.append((tick, bytes([running]) + payload))
            pos += count
    return events
