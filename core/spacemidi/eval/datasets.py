"""Where the data lives and how the datasets are laid out.

The data folder is set with the ``SPACEMIDI_DATA`` environment variable,
for example ``D:\\SPACEMIDI_DATA``. Inside it::

    raw/musdb18hq/{train,test}/<song>/{mixture,drums,bass,other,vocals}.wav
    raw/slakh2100_flac_redux/{train,validation,test,omitted}/TrackNNNNN/
        mix.flac  all_src.mid  metadata.yaml  MIDI/Sxx.mid  stems/Sxx.flac
    testset/<name>/  one audio file (mix.* preferred) + reference.notes.json or reference.mid

The code only reads these folders; unpacking the downloads is done by hand.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from spacemidi.notes import NoteDocument, Source, Track, load

ENV_VAR = "SPACEMIDI_DATA"
AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg")
MUSDB_STEMS = ("mixture", "drums", "bass", "other", "vocals")
SLAKH_SPLITS = ("train", "validation", "test")
_SLAKH_GROUPS = {"Piano": "piano", "Bass": "bass", "Guitar": "guitar", "Drums": "drums"}


class DataError(RuntimeError):
    """The data folder is missing or not set up."""


def data_root(path: str | Path | None = None) -> Path:
    """The data folder: ``path`` if given, else ``$SPACEMIDI_DATA``."""
    value = path if path is not None else os.environ.get(ENV_VAR)
    if not value:
        raise DataError(
            f"{ENV_VAR} is not set. Point it at your data folder, e.g. in PowerShell:\n"
            f"    setx {ENV_VAR} D:\\SPACEMIDI_DATA\n"
            "then open a new terminal."
        )
    root = Path(value)
    if not root.is_dir():
        raise DataError(f"{ENV_VAR} points to {root}, which is not a folder")
    return root


def load_document(path: str | Path) -> NoteDocument:
    """A notes document from ``.notes.json`` / ``.json`` or ``.mid`` / ``.midi``."""
    from spacemidi.midi import read_midi

    p = Path(path)
    if p.suffix.lower() in (".mid", ".midi"):
        return read_midi(p)
    return load(p)


# ---------------------------------------------------------------- MUSDB18-HQ


@dataclass
class MusdbSong:
    name: str
    split: str
    folder: Path

    def stem(self, name: str) -> Path:
        if name not in MUSDB_STEMS:
            raise ValueError(f"unknown MUSDB stem '{name}', expected one of {MUSDB_STEMS}")
        return self.folder / f"{name}.wav"


def musdb_songs(root: str | Path | None = None, split: str | None = None) -> list[MusdbSong]:
    base = data_root(root) / "raw" / "musdb18hq"
    songs = []
    for sp in (split,) if split else ("train", "test"):
        folder = base / sp
        if not folder.is_dir():
            continue
        for song in sorted(p for p in folder.iterdir() if p.is_dir()):
            if (song / "mixture.wav").is_file():
                songs.append(MusdbSong(song.name, sp, song))
    return songs


# ---------------------------------------------------------------- Slakh2100


@dataclass
class SlakhTrack:
    name: str
    split: str
    folder: Path

    @property
    def mix(self) -> Path:
        return self.folder / "mix.flac"

    def stems_info(self) -> dict[str, dict]:
        """Per-stem metadata (inst_class, is_drum, program_num, ...) from metadata.yaml."""
        meta = parse_simple_yaml((self.folder / "metadata.yaml").read_text(encoding="utf-8"))
        stems = meta.get("stems", {})
        return stems if isinstance(stems, dict) else {}

    def reference(self) -> NoteDocument:
        """The correct answer: one track per rendered stem, groups from Slakh's instrument class."""
        from spacemidi.midi import read_midi

        tracks = []
        for stem_id, info in sorted(self.stems_info().items()):
            midi = self.folder / "MIDI" / f"{stem_id}.mid"
            if not (info.get("audio_rendered") and midi.is_file()):
                continue
            notes = sorted(
                (n for t in read_midi(midi).tracks for n in t.notes), key=lambda n: (n.onset, n.pitch)
            )
            is_drum = bool(info.get("is_drum"))
            program = info.get("program_num")
            tracks.append(
                Track(
                    id=stem_id,
                    name=str(info.get("midi_program_name") or stem_id),
                    group="drums" if is_drum else _SLAKH_GROUPS.get(str(info.get("inst_class")), "other"),
                    stem=stem_id,
                    program=None if is_drum or not isinstance(program, int) else program,
                    is_drum=is_drum,
                    notes=notes,
                )
            )
        full = self.folder / "all_src.mid"
        tempo = read_midi(full).tempo if full.is_file() else None
        return NoteDocument(tracks=tracks, tempo=tempo, source=Source(path=str(self.mix)))


def slakh_tracks(
    root: str | Path | None = None, split: str | None = None, include_omitted: bool = False
) -> list[SlakhTrack]:
    base = data_root(root) / "raw" / "slakh2100_flac_redux"
    splits = (split,) if split else SLAKH_SPLITS + (("omitted",) if include_omitted else ())
    tracks = []
    for sp in splits:
        folder = base / sp
        if not folder.is_dir():
            continue
        for track in sorted(p for p in folder.iterdir() if p.is_dir()):
            if (track / "metadata.yaml").is_file():
                tracks.append(SlakhTrack(track.name, sp, track))
    return tracks


def parse_simple_yaml(text: str) -> dict:
    """Parse the YAML subset Slakh uses: nested ``key: value`` maps by indentation.

    Scalars become bool, int, float, None or str. Lists and multi-line values
    are not supported (Slakh's metadata has none); such lines are skipped.
    """
    root: dict = {}
    stack: list[tuple[int, dict]] = [(-1, root)]
    for raw in text.splitlines():
        line = raw.split(" #", 1)[0].rstrip()
        if not line.strip() or line.lstrip().startswith(("#", "- ")) or ":" not in line:
            continue
        indent = len(line) - len(line.lstrip(" "))
        key, _, value = line.strip().partition(":")
        while stack[-1][0] >= indent:
            stack.pop()
        parent = stack[-1][1]
        if value.strip() == "":
            child: dict = {}
            parent[key.strip()] = child
            stack.append((indent, child))
        else:
            parent[key.strip()] = _scalar(value.strip())
    return root


def _scalar(value: str):
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    lowered = value.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    if lowered in ("null", "~"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


# ---------------------------------------------------------------- own test set


@dataclass
class TestsetItem:
    name: str
    folder: Path
    audio: Path
    reference: Path | None

    def load_reference(self) -> NoteDocument | None:
        return load_document(self.reference) if self.reference else None


def list_testset(root: str | Path | None = None) -> list[TestsetItem]:
    base = data_root(root) / "testset"
    if not base.is_dir():
        return []
    items = []
    for folder in sorted(p for p in base.iterdir() if p.is_dir()):
        audio_files = sorted(p for p in folder.iterdir() if p.suffix.lower() in AUDIO_EXTENSIONS)
        if not audio_files:
            continue
        audio = next((p for p in audio_files if p.stem.lower() == "mix"), audio_files[0])
        reference = next(
            (folder / n for n in ("reference.notes.json", "reference.mid", "reference.midi") if (folder / n).is_file()),
            None,
        )
        items.append(TestsetItem(folder.name, folder, audio, reference))
    return items


# ---------------------------------------------------------------- overview


def describe(root: str | Path | None = None) -> list[str]:
    """Human-readable lines about the data folder and what is found in it."""
    try:
        base = data_root(root)
    except DataError as e:
        return [str(e).splitlines()[0]]
    lines = [f"data folder: {base}"]
    musdb = musdb_songs(base)
    if musdb:
        lines.append(f"  MUSDB18-HQ: {len(musdb)} songs")
    else:
        lines.append("  MUSDB18-HQ: not found (unzip musdb18hq.zip into raw/musdb18hq/)")
    slakh = slakh_tracks(base)
    if slakh:
        by_split = {sp: sum(t.split == sp for t in slakh) for sp in SLAKH_SPLITS}
        lines.append("  Slakh2100: " + ", ".join(f"{sp} {n}" for sp, n in by_split.items()))
    else:
        lines.append("  Slakh2100: not found (untar slakh2100_flac_redux.tar.gz into raw/)")
    tests = list_testset(base)
    with_ref = sum(t.reference is not None for t in tests)
    lines.append(f"  own test set: {len(tests)} items, {with_ref} with a reference")
    return lines
