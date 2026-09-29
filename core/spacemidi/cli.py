"""Core command line. Usage: `uv run spacemidi <command>`."""

from __future__ import annotations

import argparse
import platform
import sys
from importlib import metadata

from spacemidi import __version__


def _cmd_info(_: argparse.Namespace) -> int:
    print(f"spacemidi {__version__}")
    print(f"python     {platform.python_version()} ({sys.platform})")
    for pkg in ("numpy", "scipy", "soundfile", "torch"):
        try:
            print(f"{pkg:<11}{metadata.version(pkg)}")
        except metadata.PackageNotFoundError:
            print(f"{pkg:<11}not installed")
    try:
        import torch

        if torch.cuda.is_available():
            print(f"cuda       {torch.version.cuda}, {torch.cuda.get_device_name(0)}")
        else:
            print("cuda       not available")
    except ImportError:
        pass
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    from spacemidi.notes import NotesFormatError, load

    failed = 0
    for path in args.files:
        try:
            doc = load(path)
        except (OSError, NotesFormatError) as e:
            failed += 1
            print(f"FAIL {path}\n{e}")
            continue
        notes = sum(len(t.notes) for t in doc.tracks)
        print(f"OK   {path}: {len(doc.tracks)} tracks, {notes} notes")
    return 1 if failed else 0


def _cmd_to_midi(args: argparse.Namespace) -> int:
    from spacemidi.midi import MidiExportError, MidiOptions, write_midi
    from spacemidi.notes import NotesFormatError, load

    try:
        doc = load(args.notes)
        write_midi(doc, args.midi, MidiOptions(ppq=args.ppq, grid=args.grid, strength=args.strength))
    except (OSError, NotesFormatError, MidiExportError) as e:
        print(f"FAIL {args.notes}\n{e}")
        return 1
    notes = sum(len(t.notes) for t in doc.tracks)
    print(f"OK   {args.midi}: {len(doc.tracks)} tracks, {notes} notes")
    return 0


def _cmd_from_midi(args: argparse.Namespace) -> int:
    from spacemidi.midi import MidiFormatError, read_midi
    from spacemidi.notes import NotesFormatError, save

    try:
        doc = read_midi(args.midi)
        save(doc, args.notes)
    except (OSError, MidiFormatError, NotesFormatError) as e:
        print(f"FAIL {args.midi}\n{e}")
        return 1
    notes = sum(len(t.notes) for t in doc.tracks)
    print(f"OK   {args.notes}: {len(doc.tracks)} tracks, {notes} notes")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spacemidi", description="Space MIDI core")
    parser.add_argument("--version", action="version", version=f"spacemidi {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="show package versions and CUDA status").set_defaults(func=_cmd_info)
    p_val = sub.add_parser("validate", help="check .notes.json files against the note format")
    p_val.add_argument("files", nargs="+", help="files to check")
    p_val.set_defaults(func=_cmd_validate)

    p_to = sub.add_parser("to-midi", help="write a .notes.json file as a Standard MIDI File")
    p_to.add_argument("notes", help="input .notes.json")
    p_to.add_argument("midi", help="output .mid")
    p_to.add_argument("--ppq", type=int, default=480, help="ticks per quarter note (default 480)")
    p_to.add_argument("--grid", type=int, default=None, help="snap to N subdivisions per beat, e.g. 4")
    p_to.add_argument("--strength", type=float, default=1.0, help="snap strength 0-1 (default 1)")
    p_to.set_defaults(func=_cmd_to_midi)

    p_from = sub.add_parser("from-midi", help="read a MIDI file into a .notes.json file")
    p_from.add_argument("midi", help="input .mid")
    p_from.add_argument("notes", help="output .notes.json")
    p_from.set_defaults(func=_cmd_from_midi)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
