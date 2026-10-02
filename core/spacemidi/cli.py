"""Core command line. Usage: `uv run spacemidi <command>`."""

from __future__ import annotations

import argparse
import platform
import sys
from importlib import metadata

from spacemidi import __version__
from spacemidi.dsp.onset import ONSET_PRESETS


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
    from spacemidi.eval.datasets import describe

    for line in describe():
        print(line)
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


def _cmd_eval(args: argparse.Namespace) -> int:
    import json

    from spacemidi.eval import evaluate, format_report, load_document
    from spacemidi.midi import MidiFormatError
    from spacemidi.notes import NotesFormatError

    try:
        estimate = load_document(args.estimate)
        reference = load_document(args.reference)
    except (OSError, NotesFormatError, MidiFormatError) as e:
        print(f"FAIL\n{e}")
        return 1
    report = evaluate(reference, estimate, onset_tolerance=args.onset_tol)
    print(f"estimate:  {args.estimate}\nreference: {args.reference}\n")
    print(format_report(report))
    payload = {"estimate": str(args.estimate), "reference": str(args.reference), **report.to_dict()}
    status = 0
    if args.mir_eval:
        from spacemidi.eval.mireval import MirEvalMissing, compare, format_comparison

        try:
            rows = compare(reference, estimate, report, onset_tolerance=args.onset_tol)
        except MirEvalMissing as e:
            print(f"\n{e}")
            return 1
        print("\nCross-check (F1):\n" + format_comparison(rows))
        payload["mir_eval"] = {r.name: {"ours": r.ours, "mir_eval": r.theirs, "agrees": r.agrees} for r in rows}
        if not all(r.agrees for r in rows):
            print("\nMISMATCH: our scores differ from mir_eval. Please report this.")
            status = 2
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
    return status


def _cmd_synth(args: argparse.Namespace) -> int:
    from spacemidi.eval import render, write_wav
    from spacemidi.notes import NotesFormatError

    try:
        from spacemidi.eval import load_document

        doc = load_document(args.notes)
    except (OSError, NotesFormatError, ValueError) as e:
        print(f"FAIL {args.notes}\n{e}")
        return 1
    audio = render(doc, args.sr, noise_snr_db=args.noise_snr, seed=args.seed)
    write_wav(args.wav, audio, args.sr)
    print(f"OK   {args.wav}: {len(audio) / args.sr:.1f} s at {args.sr} Hz")
    return 0


def _cmd_onsets(args: argparse.Namespace) -> int:
    from spacemidi.dsp import detect_onsets, load_audio, onset_config
    from spacemidi.eval.bench import parse_grid

    try:
        changes = {k: v[0] for k, v in parse_grid(args.set).items()}
        cfg = onset_config(args.preset, **changes)
        x, sr = load_audio(args.audio, args.sr)
    except (OSError, ValueError, KeyError, RuntimeError) as e:
        print(f"FAIL {args.audio}\n{e}")
        return 1
    times = detect_onsets(x, sr, cfg)
    print(f"OK   {args.audio}: {len(times)} onsets in {len(x) / sr:.1f} s ({args.preset})")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.writelines(f"{t:.4f}\n" for t in times)
    if args.midi:
        from spacemidi.midi import write_midi
        from spacemidi.notes import Note, NoteDocument, Track

        clicks = [Note(float(t), float(t) + 0.05, 37, velocity=100) for t in times if t >= 0]
        write_midi(NoteDocument(tracks=[Track(id="onsets", group="drums", name="Onsets", is_drum=True, notes=clicks)]), args.midi)
    if not args.out and not args.midi:
        print(" ".join(f"{t:.3f}" for t in times))
    return 0


def _cmd_bench_onsets(args: argparse.Namespace) -> int:
    from spacemidi.eval.bench import format_results, parse_grid, run_onset_bench, save_run, slakh_items
    from spacemidi.eval.datasets import DataError

    skipped: list[tuple[str, str]] = []
    try:
        grid = parse_grid(args.set)
        items = slakh_items(args.split, args.source, args.limit, skipped=skipped)
    except (ValueError, DataError) as e:
        print(f"FAIL\n{e}")
        return 1
    for name, reason in skipped:
        print(f"skip {name}: {reason}")
    if not items:
        print(f"FAIL no Slakh tracks found for split '{args.split}' (check `spacemidi info`)")
        return 1
    print(f"onsets: {len(items)} items, preset {args.preset}, audio {args.audio}, grid {grid or '-'}")
    try:
        results, rows = run_onset_bench(
            items, args.preset, grid, audio=args.audio, sr=args.sr, combine=args.combine, jobs=args.jobs
        )
    except (KeyError, ValueError) as e:
        print(f"FAIL\n{e}")
        return 1
    meta = {
        "task": "onsets", "preset": args.preset, "dataset": "slakh", "split": args.split, "source": args.source,
        "audio": args.audio, "limit": args.limit, "sr": args.sr, "combine": args.combine, "window": 0.05,
        "grid": grid, "skipped": [name for name, _ in skipped],
    }
    path = save_run(meta, results, rows, args.out_dir)
    print("\n" + format_results(results))
    print(f"\nsaved {path}")
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

    p_eval = sub.add_parser("eval", help="score estimated notes against a reference")
    p_eval.add_argument("estimate", help="estimated notes: .notes.json or .mid")
    p_eval.add_argument("reference", help="correct notes: .notes.json or .mid")
    p_eval.add_argument("--onset-tol", type=float, default=0.05, help="onset tolerance in seconds (default 0.05)")
    p_eval.add_argument("--json", help="also write the scores to this JSON file")
    p_eval.add_argument("--mir-eval", action="store_true", help="also compute the scores with mir_eval and compare")
    p_eval.set_defaults(func=_cmd_eval)

    p_synth = sub.add_parser("synth", help="render notes to a WAV file with simple test sounds")
    p_synth.add_argument("notes", help="input .notes.json or .mid")
    p_synth.add_argument("wav", help="output .wav")
    p_synth.add_argument("--sr", type=int, default=44100, help="sample rate (default 44100)")
    p_synth.add_argument("--noise-snr", type=float, default=None, help="add white noise at this SNR in dB")
    p_synth.add_argument("--seed", type=int, default=0, help="random seed for noise and drums")
    p_synth.set_defaults(func=_cmd_synth)

    p_on = sub.add_parser("onsets", help="detect note onsets in an audio file")
    p_on.add_argument("audio", help="input audio (.wav, .flac, ...)")
    p_on.add_argument("--preset", default="flux", choices=list(ONSET_PRESETS), help="detector settings (default flux; *-stem for one instrument)")
    p_on.add_argument("--set", action="append", metavar="KEY=VALUE", help="change one setting, e.g. --set delta=0.1")
    p_on.add_argument("--sr", type=int, default=22050, help="analysis sample rate (default 22050)")
    p_on.add_argument("--out", help="write onset times (seconds, one per line) to this file")
    p_on.add_argument("--midi", help="write a MIDI file with a click at every onset, to check by ear in a DAW")
    p_on.set_defaults(func=_cmd_onsets)

    p_bench = sub.add_parser("bench", help="run an algorithm over a dataset and save the scores")
    bench_sub = p_bench.add_subparsers(dest="task", required=True)
    p_bo = bench_sub.add_parser("onsets", help="onset detection on Slakh2100")
    p_bo.add_argument("--preset", default="flux", choices=list(ONSET_PRESETS), help="detector settings (default flux; *-stem for one instrument)")
    p_bo.add_argument("--set", action="append", metavar="KEY=V1[,V2...]",
                      help="setting to change; several values make a grid, e.g. --set delta=0.05,0.1")
    p_bo.add_argument("--split", default="test", choices=["train", "validation", "test"], help="Slakh split (default test)")
    p_bo.add_argument("--source", default="mix", choices=["mix", "stems"], help="full mixes or every stem alone")
    p_bo.add_argument("--audio", default="real", choices=["real", "synth"],
                      help="Slakh's audio, or the same MIDI rendered with our simple test synth")
    p_bo.add_argument("--limit", type=int, default=None, help="only the first N tracks")
    p_bo.add_argument("--jobs", type=int, default=1, help="parallel worker processes (default 1)")
    p_bo.add_argument("--sr", type=int, default=22050, help="analysis sample rate (default 22050)")
    p_bo.add_argument("--combine", type=float, default=0.03, help="merge reference onsets closer than this (s)")
    p_bo.add_argument("--out-dir", default=None, help="where to save the run (default experiments/runs)")
    p_bo.set_defaults(func=_cmd_bench_onsets)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
