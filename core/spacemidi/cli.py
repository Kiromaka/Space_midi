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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spacemidi", description="Space MIDI core")
    parser.add_argument("--version", action="version", version=f"spacemidi {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="show package versions and CUDA status").set_defaults(func=_cmd_info)
    p_val = sub.add_parser("validate", help="check .notes.json files against the note format")
    p_val.add_argument("files", nargs="+", help="files to check")
    p_val.set_defaults(func=_cmd_validate)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
