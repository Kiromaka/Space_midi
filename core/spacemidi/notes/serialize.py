"""Reading and writing ``spacemidi-notes`` JSON files.

``to_dict`` / ``from_dict`` convert between the model and plain JSON data;
``save`` / ``load`` add file handling and validation. Unknown keys are
ignored on read, so newer optional fields do not break older readers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import (
    FORMAT_NAME,
    FORMAT_VERSION,
    Backend,
    Control,
    Key,
    Note,
    NoteDocument,
    Source,
    TempoMap,
    Track,
)
from .validate import NotesFormatError, validate

_SECONDS_DIGITS = 6
_LINE_WIDTH = 100


# ---------------------------------------------------------------- writing


def to_dict(doc: NoteDocument) -> dict[str, Any]:
    """Convert a document to JSON-ready data. Notes are sorted by (onset, pitch)."""
    out: dict[str, Any] = {"format": FORMAT_NAME, "version": FORMAT_VERSION}
    if doc.source is not None:
        src = {
            "path": doc.source.path,
            "duration": _sec(doc.source.duration),
            "sample_rate": doc.source.sample_rate,
        }
        out["source"] = {k: v for k, v in src.items() if v is not None}
    out["tuning_hz"] = doc.tuning_hz
    if doc.tempo is not None:
        out["tempo"] = {
            "beats": [_sec(b) for b in doc.tempo.beats],
            "downbeats": [_sec(b) for b in doc.tempo.downbeats],
            "time_signature": list(doc.tempo.time_signature),
        }
    if doc.key is not None:
        out["key"] = {"tonic": doc.key.tonic, "mode": doc.key.mode}
    out["tracks"] = [_track_to_dict(t) for t in doc.tracks]
    return out


def _track_to_dict(t: Track) -> dict[str, Any]:
    d: dict[str, Any] = {"id": t.id, "name": t.name, "group": t.group}
    if t.stem is not None:
        d["stem"] = t.stem
    if t.program is not None:
        d["program"] = t.program
    d["is_drum"] = t.is_drum
    if t.backend is not None:
        d["backend"] = {
            "stage": t.backend.stage,
            "name": t.backend.name,
            "version": t.backend.version,
            "params": t.backend.params,
        }
    d["notes"] = [_note_to_dict(n) for n in sorted(t.notes, key=lambda n: (n.onset, n.pitch))]
    if t.controls:
        d["controls"] = [
            {"time": _sec(c.time), "cc": c.cc, "value": c.value}
            for c in sorted(t.controls, key=lambda c: (c.time, c.cc))
        ]
    return d


def _note_to_dict(n: Note) -> dict[str, Any]:
    d: dict[str, Any] = {
        "onset": _sec(n.onset),
        "offset": _sec(n.offset),
        "pitch": n.pitch,
        "velocity": n.velocity,
    }
    if n.confidence is not None:
        d["confidence"] = round(n.confidence, 4)
    if n.bend is not None:
        d["bend"] = [[_sec(t), round(c, 2)] for t, c in n.bend]
    return d


def _sec(x: float | None) -> float | None:
    return None if x is None else round(x, _SECONDS_DIGITS)


def dumps(doc: NoteDocument, *, check: bool = True) -> str:
    """Serialize to text. Short objects stay on one line, so each note is one line."""
    if check:
        problems = validate(doc)
        if problems:
            raise NotesFormatError(problems)
    return _format(to_dict(doc), 0) + "\n"


def save(doc: NoteDocument, path: str | Path, *, check: bool = True) -> None:
    Path(path).write_text(dumps(doc, check=check), encoding="utf-8")


#: Lists whose items are always written one per line, however long.
_ONE_LINE_ITEMS = ("notes", "controls")


def _compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(", ", ": "))


def _format(value: Any, indent: int) -> str:
    compact = _compact(value)
    if not isinstance(value, (dict, list)) or not value or indent + len(compact) <= _LINE_WIDTH:
        return compact
    pad = " " * (indent + 2)
    end = "\n" + " " * indent
    if isinstance(value, dict):
        items = []
        for k, v in value.items():
            key = json.dumps(k, ensure_ascii=False)
            if k in _ONE_LINE_ITEMS and isinstance(v, list) and v:
                inner = ",\n".join(pad + "  " + _compact(item) for item in v)
                items.append(f"{pad}{key}: [\n{inner}\n{pad}]")
            else:
                items.append(f"{pad}{key}: {_format(v, indent + 2)}")
        return "{\n" + ",\n".join(items) + end + "}"
    if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in value):
        return "[\n" + ",\n".join(_wrap_numbers(value, pad)) + end + "]"
    return "[\n" + ",\n".join(pad + _format(v, indent + 2) for v in value) + end + "]"


def _wrap_numbers(values: list[float], pad: str) -> list[str]:
    lines, line = [], ""
    for v in values:
        s = json.dumps(v, allow_nan=False)
        if line and len(pad) + len(line) + len(s) + 2 > _LINE_WIDTH:
            lines.append(pad + line)
            line = ""
        line = f"{line}, {s}" if line else s
    if line:
        lines.append(pad + line)
    return lines


# ---------------------------------------------------------------- reading


def from_dict(data: Any) -> NoteDocument:
    """Build a document from parsed JSON. Raises ``NotesFormatError`` on structural errors."""
    r = _Reader()
    doc = r.document(data)
    if r.problems:
        raise NotesFormatError(r.problems)
    return doc


def loads(text: str, *, check: bool = True) -> NoteDocument:
    try:
        data = json.loads(text)
    except json.JSONDecodeError as e:
        raise NotesFormatError([f"not valid JSON: {e}"]) from e
    doc = from_dict(data)
    if check:
        problems = validate(doc)
        if problems:
            raise NotesFormatError(problems)
    return doc


def load(path: str | Path, *, check: bool = True) -> NoteDocument:
    return loads(Path(path).read_text(encoding="utf-8"), check=check)


_INT = "integer"
_NUM = "number"
_STR = "string"
_BOOL = "boolean"
_OBJ = "object"
_ARR = "array"


def _kind_ok(value: Any, kind: str) -> bool:
    if kind == _BOOL:
        return isinstance(value, bool)
    if isinstance(value, bool):
        return False
    return {
        _INT: lambda v: isinstance(v, int),
        _NUM: lambda v: isinstance(v, (int, float)),
        _STR: lambda v: isinstance(v, str),
        _OBJ: lambda v: isinstance(v, dict),
        _ARR: lambda v: isinstance(v, list),
    }[kind](value)


class _Reader:
    """Collects structural problems instead of stopping at the first one."""

    def __init__(self) -> None:
        self.problems: list[str] = []

    def get(self, obj: dict, key: str, kind: str, path: str, *, required: bool = True, default=None):
        if key not in obj or obj[key] is None:
            if required:
                self.problems.append(f"{path}: missing '{key}'")
            return default
        value = obj[key]
        if not _kind_ok(value, kind):
            self.problems.append(f"{path}.{key}: expected {kind}, got {_json_type(value)}")
            return default
        return value

    def document(self, data: Any) -> NoteDocument:
        if not isinstance(data, dict):
            self.problems.append(f"document: expected object, got {_json_type(data)}")
            return NoteDocument()
        fmt = self.get(data, "format", _STR, "document")
        if fmt is not None and fmt != FORMAT_NAME:
            self.problems.append(f"format: expected '{FORMAT_NAME}', got '{fmt}'")
        version = self.get(data, "version", _INT, "document")
        if version is not None and version > FORMAT_VERSION:
            self.problems.append(
                f"version: file is version {version}, this reader supports up to {FORMAT_VERSION}"
            )
        doc = NoteDocument(tuning_hz=self.get(data, "tuning_hz", _NUM, "document", required=False, default=440.0))
        src = self.get(data, "source", _OBJ, "document", required=False)
        if src is not None:
            doc.source = Source(
                path=self.get(src, "path", _STR, "source", required=False),
                duration=self.get(src, "duration", _NUM, "source", required=False),
                sample_rate=self.get(src, "sample_rate", _INT, "source", required=False),
            )
        tempo = self.get(data, "tempo", _OBJ, "document", required=False)
        if tempo is not None:
            doc.tempo = self.tempo(tempo)
        key = self.get(data, "key", _OBJ, "document", required=False)
        if key is not None:
            doc.key = Key(
                tonic=self.get(key, "tonic", _STR, "key", default=""),
                mode=self.get(key, "mode", _STR, "key", default=""),
            )
        tracks = self.get(data, "tracks", _ARR, "document", default=[])
        for i, t in enumerate(tracks):
            track = self.track(t, f"tracks[{i}]")
            if track is not None:
                doc.tracks.append(track)
        return doc

    def numbers(self, obj: dict, key: str, path: str, *, required: bool) -> list[float]:
        values = self.get(obj, key, _ARR, path, required=required, default=[])
        good = []
        for i, v in enumerate(values):
            if _kind_ok(v, _NUM):
                good.append(v)
            else:
                self.problems.append(f"{path}.{key}[{i}]: expected number, got {_json_type(v)}")
        return good

    def tempo(self, obj: dict) -> TempoMap:
        ts = self.get(obj, "time_signature", _ARR, "tempo", required=False, default=[4, 4])
        if len(ts) != 2 or not all(_kind_ok(v, _INT) for v in ts):
            self.problems.append("tempo.time_signature: expected [numerator, denominator] integers")
            ts = [4, 4]
        return TempoMap(
            beats=self.numbers(obj, "beats", "tempo", required=True),
            downbeats=self.numbers(obj, "downbeats", "tempo", required=False),
            time_signature=(ts[0], ts[1]),
        )

    def track(self, obj: Any, path: str) -> Track | None:
        if not isinstance(obj, dict):
            self.problems.append(f"{path}: expected object, got {_json_type(obj)}")
            return None
        track = Track(
            id=self.get(obj, "id", _STR, path, default=""),
            group=self.get(obj, "group", _STR, path, default=""),
            name=self.get(obj, "name", _STR, path, required=False, default=""),
            stem=self.get(obj, "stem", _STR, path, required=False),
            program=self.get(obj, "program", _INT, path, required=False),
            is_drum=self.get(obj, "is_drum", _BOOL, path, required=False, default=False),
        )
        backend = self.get(obj, "backend", _OBJ, path, required=False)
        if backend is not None:
            bp = f"{path}.backend"
            track.backend = Backend(
                stage=self.get(backend, "stage", _STR, bp, default=""),
                name=self.get(backend, "name", _STR, bp, default=""),
                version=self.get(backend, "version", _STR, bp, required=False, default=""),
                params=self.get(backend, "params", _OBJ, bp, required=False, default={}),
            )
        for i, n in enumerate(self.get(obj, "notes", _ARR, path, default=[])):
            note = self.note(n, f"{path}.notes[{i}]")
            if note is not None:
                track.notes.append(note)
        for i, c in enumerate(self.get(obj, "controls", _ARR, path, required=False, default=[])):
            cp = f"{path}.controls[{i}]"
            if not isinstance(c, dict):
                self.problems.append(f"{cp}: expected object, got {_json_type(c)}")
                continue
            track.controls.append(
                Control(
                    time=self.get(c, "time", _NUM, cp, default=0.0),
                    cc=self.get(c, "cc", _INT, cp, default=0),
                    value=self.get(c, "value", _INT, cp, default=0),
                )
            )
        return track

    def note(self, obj: Any, path: str) -> Note | None:
        if not isinstance(obj, dict):
            self.problems.append(f"{path}: expected object, got {_json_type(obj)}")
            return None
        note = Note(
            onset=self.get(obj, "onset", _NUM, path, default=0.0),
            offset=self.get(obj, "offset", _NUM, path, default=0.0),
            pitch=self.get(obj, "pitch", _INT, path, default=0),
            velocity=self.get(obj, "velocity", _INT, path, required=False, default=80),
            confidence=self.get(obj, "confidence", _NUM, path, required=False),
        )
        bend = self.get(obj, "bend", _ARR, path, required=False)
        if bend is not None:
            points = []
            for i, p in enumerate(bend):
                if isinstance(p, list) and len(p) == 2 and all(_kind_ok(v, _NUM) for v in p):
                    points.append((p[0], p[1]))
                else:
                    self.problems.append(f"{path}.bend[{i}]: expected [seconds, cents]")
            note.bend = points
        return note


def _json_type(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return _BOOL
    if isinstance(value, int):
        return _INT
    if isinstance(value, float):
        return _NUM
    if isinstance(value, str):
        return _STR
    if isinstance(value, list):
        return _ARR
    if isinstance(value, dict):
        return _OBJ
    return type(value).__name__
