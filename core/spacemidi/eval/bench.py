"""Benchmarks: run an algorithm over a dataset split and score every item.

``spacemidi bench onsets`` runs onset detection on Slakh2100, either on the
full mixes or on every rendered stem, with a grid of settings. Each run is
saved as JSON under ``experiments/runs/`` (settings, git commit, per-item and
total scores) so a number in the research doc can always be traced back.

Reference onsets come from Slakh's MIDI. Notes that start within
``combine`` seconds of each other (a chord, a drum hit with a bass note)
count as one onset, since a listener and an onset detector hear one event.
"""

from __future__ import annotations

import datetime as _dt
import itertools
import json
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from spacemidi.notes import NoteDocument

from .datasets import slakh_tracks
from .matching import match_events
from .metrics import Score

#: Onset settings that change the envelope; the rest only change peak picking.
ENVELOPE_KEYS = frozenset(
    {"n_fft", "hop", "bands", "n_bands", "fmin", "fmax", "compression", "log_mul", "lag", "max_size"}
)

REPO_ROOT = Path(__file__).resolve().parents[3]


def reference_onsets(doc: NoteDocument, combine: float = 0.03) -> list[float]:
    """Sorted note onsets of all tracks; onsets within ``combine`` s of the previous kept one are merged."""
    times = sorted(n.onset for t in doc.tracks for n in t.notes)
    kept: list[float] = []
    for t in times:
        if not kept or t - kept[-1] >= combine:
            kept.append(t)
    return kept


@dataclass
class BenchItem:
    """One piece of audio with its correct answer."""

    name: str
    group: str  # instrument group for stems, "mix" for full mixes
    audio: Path | None
    reference: NoteDocument


def slakh_items(
    split: str = "test",
    source: str = "mix",
    limit: int | None = None,
    root=None,
    skipped: list[tuple[str, str]] | None = None,
) -> list[BenchItem]:
    """Bench items from Slakh: one per track (``source="mix"``) or one per rendered stem (``"stems"``).

    A track whose MIDI cannot be read is left out; pass a list as ``skipped``
    to collect ``(track name, reason)`` for each one.
    """
    from spacemidi.midi import MidiFormatError

    if source not in ("mix", "stems"):
        raise ValueError("source must be 'mix' or 'stems'")
    tracks = slakh_tracks(root, split)
    if limit:
        tracks = tracks[:limit]
    items = []
    for track in tracks:
        try:
            ref = track.reference()
        except (MidiFormatError, OSError, ValueError) as e:
            if skipped is not None:
                skipped.append((track.name, str(e)))
            continue
        if source == "mix":
            items.append(BenchItem(track.name, "mix", track.mix, ref))
            continue
        for t in ref.tracks:
            if t.notes:
                single = NoteDocument(tracks=[t], tempo=ref.tempo)
                items.append(BenchItem(f"{track.name}/{t.id}", t.group, track.folder / "stems" / f"{t.id}.flac", single))
    return items


@dataclass
class GridPoint:
    params: dict
    total: Score = field(default_factory=Score)
    groups: dict[str, Score] = field(default_factory=dict)
    deviations: list[float] = field(default_factory=list)  # estimate - reference for matched onsets


def expand_grid(grid: dict[str, list]) -> list[dict]:
    keys = list(grid)
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))] or [{}]


def _onset_item(job: tuple) -> dict:
    """Worker: score one item for every grid point (runs in a separate process)."""
    from spacemidi.dsp import load_audio, onset_config, onset_envelope, pick_onsets

    item, preset, points, audio_mode, sr, combine, window = job
    started = time.perf_counter()
    if audio_mode == "synth":
        from .synth import render

        x = render(item.reference, sr, tail=0.5).astype(np.float64)
    else:
        x, _ = load_audio(item.audio, sr)
    ref = reference_onsets(item.reference, combine)
    envelopes: dict[tuple, np.ndarray] = {}
    scores, deviations = [], []
    for params in points:
        cfg = onset_config(preset, **params)
        env_key = tuple(sorted((k, str(v)) for k, v in params.items() if k in ENVELOPE_KEYS))
        if env_key not in envelopes:
            envelopes[env_key] = onset_envelope(x, sr, cfg)
        est = pick_onsets(envelopes[env_key], sr, cfg).tolist()
        pairs = match_events(ref, est, window)
        scores.append((len(ref), len(est), len(pairs)))
        deviations.append([est[j] - ref[i] for i, j in pairs])
    return {
        "name": item.name,
        "group": item.group,
        "duration": round(len(x) / sr, 2),
        "seconds": round(time.perf_counter() - started, 2),
        "scores": scores,
        "deviations": deviations,
    }


def run_onset_bench(
    items: list[BenchItem],
    preset: str = "flux",
    grid: dict[str, list] | None = None,
    *,
    audio: str = "real",
    sr: int = 22050,
    combine: float = 0.03,
    window: float = 0.05,
    jobs: int = 1,
    progress=print,
) -> tuple[list[GridPoint], list[dict]]:
    """Score every item at every grid point. Returns the grid points (totals) and per-item rows."""
    from spacemidi.dsp import onset_config

    points = expand_grid(grid or {})
    for params in points:
        onset_config(preset, **params)  # fail early on a bad setting name or value
    results = [GridPoint(params) for params in points]
    jobs_list = [(item, preset, points, audio, sr, combine, window) for item in items]
    rows = []
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            outputs = pool.map(_onset_item, jobs_list)
            rows = _collect(outputs, results, len(items), progress)
    else:
        rows = _collect(map(_onset_item, jobs_list), results, len(items), progress)
    return results, rows


def _collect(outputs, results: list[GridPoint], n: int, progress) -> list[dict]:
    rows = []
    for k, row in enumerate(outputs, 1):
        for point, (n_ref, n_est, n_match), devs in zip(results, row["scores"], row["deviations"]):
            score = Score(n_ref, n_est, n_match)
            point.total = point.total + score
            point.groups[row["group"]] = point.groups.get(row["group"], Score()) + score
            point.deviations.extend(devs)
        best = max(Score(*s).f1 for s in row["scores"])
        if progress:
            progress(f"[{k}/{n}] {row['name']} ({row['group']}, {row['duration']:.0f} s): F1 {best:.3f}  {row['seconds']:.1f} s")
        row.pop("deviations")
        rows.append(row)
    return rows


def deviation_stats(devs: list[float]) -> dict:
    if not devs:
        return {}
    d = np.asarray(devs) * 1000.0
    return {
        "median_ms": round(float(np.median(d)), 1),
        "mean_ms": round(float(np.mean(d)), 1),
        "p10_ms": round(float(np.percentile(d, 10)), 1),
        "p90_ms": round(float(np.percentile(d, 90)), 1),
    }


def git_commit(repo: Path = REPO_ROOT) -> str | None:
    """Short commit hash of the repo, with ``+dirty`` if there are uncommitted changes."""
    try:
        sha = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "--no-optional-locks", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"],
            capture_output=True, text=True, timeout=10, check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    return sha + ("+dirty" if dirty else "")


def default_out_dir() -> Path:
    base = REPO_ROOT if (REPO_ROOT / "experiments").is_dir() else Path.cwd()
    return base / "experiments" / "runs"


def _serialize(point) -> dict:
    if isinstance(point, dict):
        return point
    return {
        "params": point.params,
        "total": point.total.to_dict(),
        "groups": {g: s.to_dict() for g, s in sorted(point.groups.items())},
        "timing": deviation_stats(point.deviations),
    }


def save_run(meta: dict, results: list, rows: list[dict], out_dir: Path | None = None) -> Path:
    """Write one benchmark run as JSON; ``results`` are onset grid points or ready-made dicts."""
    out_dir = Path(out_dir) if out_dir else default_out_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    name = f"{stamp}-{meta['task']}-{meta['preset']}-{meta['dataset']}-{meta['split']}-{meta['source']}-{meta['audio']}.json"
    payload = {
        **meta,
        "created": _dt.datetime.now().isoformat(timespec="seconds"),
        "commit": git_commit(),
        "results": [_serialize(p) for p in results],
        "items": rows,
    }
    path = out_dir / name
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    return path


def format_results(results: list[GridPoint], top: int = 20) -> str:
    """A table of grid points, best F1 first, with per-group F1 and the median timing error."""
    groups = sorted({g for p in results for g in p.groups})
    ranked = sorted(results, key=lambda p: p.total.f1, reverse=True)[:top]
    param_keys = sorted({k for p in results for k in p.params})
    head = [*param_keys, "P", "R", "F1", *[f"F1 {g}" for g in groups], "dt med ms"]
    lines = [head]
    for p in ranked:
        timing = deviation_stats(p.deviations).get("median_ms", "")
        lines.append(
            [str(p.params[k]) for k in param_keys]
            + [f"{p.total.precision:.3f}", f"{p.total.recall:.3f}", f"{p.total.f1:.3f}"]
            + [f"{p.groups[g].f1:.3f}" if g in p.groups else "" for g in groups]
            + [str(timing)]
        )
    widths = [max(len(row[i]) for row in lines) for i in range(len(head))]
    return "\n".join("  ".join(cell.rjust(w) for cell, w in zip(row, widths)) for row in lines)


def parse_grid(settings: list[str] | None) -> dict[str, list[str]]:
    """``["delta=0.05,0.1", "lag=2"]`` -> ``{"delta": ["0.05", "0.1"], "lag": ["2"]}``."""
    grid: dict[str, list[str]] = {}
    for item in settings or []:
        key, sep, values = item.partition("=")
        if not sep or not key.strip() or not values.strip():
            raise ValueError(f"expected key=value[,value...], got '{item}'")
        grid[key.strip()] = [v.strip() for v in values.split(",") if v.strip()]
    return grid
