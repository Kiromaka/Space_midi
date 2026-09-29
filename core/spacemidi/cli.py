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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spacemidi", description="Space MIDI core")
    parser.add_argument("--version", action="version", version=f"spacemidi {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("info", help="show package versions and CUDA status").set_defaults(func=_cmd_info)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
