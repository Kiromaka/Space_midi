"""``spacemidi bench hpss``: harmonic-percussive separation on MUSDB18-HQ.

MUSDB gives every song as a mixture plus four stems (drums, bass, other,
vocals). HPSS splits the mixture into two parts, so the closest references
are:

- percussive part vs the **drums** stem;
- harmonic part vs **mixture - drums** (bass + other + vocals).

Neither is a perfect target: a hi-hat or a guitar strum is partly
"harmonic" and partly "percussive" in the spectrogram, and vocals carry
consonants. The numbers measure how close the split gets to a drums / rest
split, which is what the later stages need (drum transcription on one side,
pitch on the other).

Scores per song, all plain SDR in dB (``eval.metrics.sdr``) on mono audio:

- ``sdr_p`` / ``sdr_h``: over the whole song;
- ``fsdr_p`` / ``fsdr_h``: median over 1 s windows (the SiSEC convention);
- ``base_*``: the same with the unprocessed mixture as the estimate;
- ``gain_*``: ``sdr - base``, the improvement over doing nothing.

Totals are means over songs, and medians of the framewise values (as in
SiSEC). With ``margin`` > 1 part of the signal goes to the residual, which
is not scored: the parts get purer but lose energy, and SDR punishes that.
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np

from .bench import expand_grid
from .metrics import framewise_sdr, sdr

#: HPSS settings a benchmark grid may change, with their defaults.
HPSS_DEFAULTS: dict[str, int | float] = {
    "n_fft": 2048,
    "hop": 512,
    "harm_frames": 31,
    "perc_bins": 31,
    "power": 2.0,
    "margin": 1.0,
}

METRICS = ("sdr_p", "sdr_h", "gain_p", "gain_h", "fsdr_p", "fsdr_h", "fgain_p", "fgain_h")

#: A stem quieter than this (mean square) counts as silent and is not scored.
SILENT = 1e-8


def hpss_settings(params: dict | None = None) -> dict:
    """``HPSS_DEFAULTS`` with ``params`` applied; values given as strings are converted."""
    out = dict(HPSS_DEFAULTS)
    for key, value in (params or {}).items():
        if key not in HPSS_DEFAULTS:
            raise KeyError(f"unknown HPSS setting '{key}', expected one of {sorted(HPSS_DEFAULTS)}")
        kind = type(HPSS_DEFAULTS[key])
        out[key] = kind(float(value)) if kind is int else kind(value)
    if out["margin"] < 1:
        raise ValueError("margin must be at least 1")
    if min(out["n_fft"], out["hop"], out["harm_frames"], out["perc_bins"]) < 1:
        raise ValueError("n_fft, hop and kernel sizes must be positive")
    return out


@dataclass
class HpssItem:
    """One song: mixture and drums files."""

    name: str
    mixture: object  # path
    drums: object  # path


def musdb_items(split: str = "test", limit: int | None = None, root=None) -> list[HpssItem]:
    from .datasets import musdb_songs

    songs = musdb_songs(root, split)
    if limit is not None:
        songs = songs[:limit]
    return [HpssItem(s.name, s.stem("mixture"), s.stem("drums")) for s in songs]


@dataclass
class HpssPoint:
    params: dict
    rows: list[dict] = field(default_factory=list)

    def values(self, key: str) -> list[float]:
        return [r[key] for r in self.rows if r.get(key) is not None]

    def mean(self, key: str) -> float:
        v = self.values(key)
        return float(np.mean(v)) if v else float("nan")

    def median(self, key: str) -> float:
        v = self.values(key)
        return float(np.median(v)) if v else float("nan")

    def to_dict(self) -> dict:
        return {
            "params": self.params,
            "n_items": len(self.rows),
            "means": {k: round(self.mean(k), 3) for k in METRICS},
            "medians": {k: round(self.median(k), 3) for k in METRICS},
        }


def _round(v: float) -> float | None:
    return None if v is None or not np.isfinite(v) else round(float(v), 3)


def score_split(mix: np.ndarray, drums: np.ndarray, harmonic: np.ndarray, percussive: np.ndarray, sr: int) -> dict:
    """Scores of one separation (see the module docstring); silent references give None."""
    rest = mix - drums
    out: dict[str, float | None] = {}
    for tag, ref, est in (("p", drums, percussive), ("h", rest, harmonic)):
        if np.mean(ref**2) < SILENT:
            out.update({f"{k}_{tag}": None for k in ("sdr", "base", "gain", "fsdr", "fbase", "fgain")})
            continue
        s, b = sdr(ref, est), sdr(ref, mix)
        fs, fb = framewise_sdr(ref, est, sr), framewise_sdr(ref, mix, sr)
        out.update({
            f"sdr_{tag}": _round(s), f"base_{tag}": _round(b), f"gain_{tag}": _round(s - b),
            f"fsdr_{tag}": _round(fs), f"fbase_{tag}": _round(fb), f"fgain_{tag}": _round(fs - fb),
        })
    return out


def _load_pair(item: HpssItem, sr: int, duration: float | None) -> tuple[np.ndarray, np.ndarray]:
    from spacemidi.dsp import load_audio

    mix, _ = load_audio(item.mixture, sr)
    drums, _ = load_audio(item.drums, sr)
    n = min(len(mix), len(drums))
    if duration is not None:
        n = min(n, int(duration * sr))
    return mix[:n], drums[:n]


def _hpss_item(job: tuple) -> dict:
    from spacemidi.dsp import istft, masks_from_parts, median_parts, stft

    item, points, sr, duration = job
    started = time.perf_counter()
    mix, drums = _load_pair(item, sr, duration)
    specs: dict[tuple, np.ndarray] = {}
    parts: dict[tuple, tuple] = {}
    scores = []
    for params in points:
        p = hpss_settings(params)
        fft_key = (p["n_fft"], p["hop"])
        if fft_key not in specs:
            specs[fft_key] = stft(mix, p["n_fft"], p["hop"])
        spec = specs[fft_key]
        med_key = (*fft_key, p["harm_frames"], p["perc_bins"])
        if med_key not in parts:
            parts[med_key] = median_parts(np.abs(spec), p["harm_frames"], p["perc_bins"])
        mask_h, mask_p = masks_from_parts(*parts[med_key], power=p["power"], margin=p["margin"])
        harmonic = istft(spec * mask_h, p["hop"], length=len(mix))
        percussive = istft(spec * mask_p, p["hop"], length=len(mix))
        scores.append(score_split(mix, drums, harmonic, percussive, sr))
    return {
        "name": item.name,
        "duration": round(len(mix) / sr, 2),
        "seconds": round(time.perf_counter() - started, 2),
        "scores": scores,
    }


def run_hpss_bench(
    items: list[HpssItem],
    grid: dict[str, list] | None = None,
    *,
    sr: int = 22050,
    duration: float | None = None,
    jobs: int = 1,
    progress=print,
) -> tuple[list[HpssPoint], list[dict]]:
    points = expand_grid(grid or {})
    for params in points:
        hpss_settings(params)  # fail early on a bad setting
    results = [HpssPoint(params) for params in points]
    job_list = [(item, points, sr, duration) for item in items]
    if jobs > 1:
        with ProcessPoolExecutor(max_workers=jobs) as pool:
            rows = _collect(pool.map(_hpss_item, job_list), results, len(items), progress)
    else:
        rows = _collect(map(_hpss_item, job_list), results, len(items), progress)
    return results, rows


def _collect(outputs, results: list[HpssPoint], n: int, progress) -> list[dict]:
    rows = []
    for k, row in enumerate(outputs, 1):
        for point, score in zip(results, row["scores"]):
            point.rows.append(score)
        if progress:
            s = row["scores"][0]
            progress(
                f"[{k}/{n}] {row['name']} ({row['duration']:.0f} s): "
                f"perc {_fmt(s.get('sdr_p'))} dB ({_fmt(s.get('gain_p'), True)}), "
                f"harm {_fmt(s.get('sdr_h'))} dB ({_fmt(s.get('gain_h'), True)})  {row['seconds']:.1f} s"
            )
        rows.append(row)
    return rows


def _fmt(v: float | None, signed: bool = False) -> str:
    if v is None:
        return "-"
    return f"{v:+.2f}" if signed else f"{v:.2f}"


def format_hpss_results(results: list[HpssPoint], top: int = 20) -> str:
    keys = sorted({k for p in results for k in p.params})
    head = [*keys, "SDR perc", "gain", "SDR harm", "gain", "fSDR perc", "gain", "fSDR harm", "gain", "n"]
    lines = [head]
    ranked = sorted(results, key=lambda p: np.nan_to_num(p.mean("gain_p") + p.mean("gain_h"), nan=-1e9), reverse=True)
    for p in ranked[:top]:
        lines.append(
            [str(p.params[k]) for k in keys]
            + [_fmt(p.mean("sdr_p")), _fmt(p.mean("gain_p"), True), _fmt(p.mean("sdr_h")), _fmt(p.mean("gain_h"), True)]
            + [_fmt(p.median("fsdr_p")), _fmt(p.median("fgain_p"), True)]
            + [_fmt(p.median("fsdr_h")), _fmt(p.median("fgain_h"), True)]
            + [str(len(p.rows))]
        )
    widths = [max(len(r[i]) for r in lines) for i in range(len(head))]
    return "\n".join("  ".join(c.rjust(w) for c, w in zip(r, widths)) for r in lines)
