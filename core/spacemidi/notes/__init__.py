"""Intermediate note format (JSON), shared by every stage.

See ``docs/formats/notes.md`` for the format description.
"""

from .model import (
    FORMAT_NAME,
    FORMAT_VERSION,
    GROUPS,
    MODES,
    PITCH_CLASSES,
    Backend,
    Control,
    Key,
    Note,
    NoteDocument,
    Source,
    TempoMap,
    Track,
)
from .serialize import dumps, from_dict, load, loads, save, to_dict
from .validate import NotesFormatError, validate

__all__ = [
    "FORMAT_NAME",
    "FORMAT_VERSION",
    "GROUPS",
    "MODES",
    "PITCH_CLASSES",
    "Backend",
    "Control",
    "Key",
    "Note",
    "NoteDocument",
    "NotesFormatError",
    "Source",
    "TempoMap",
    "Track",
    "dumps",
    "from_dict",
    "load",
    "loads",
    "save",
    "to_dict",
    "validate",
]
