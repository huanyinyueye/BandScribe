"""Separation SDR on the MUSDB18-HQ test set (spec section 6, M2 / M3).

Usage: uv run python benchmarks/bench_separation.py [--models a b c] [--limit N]

Metric: the MSST "sdr" used for the published checkpoint numbers:
    10 * log10(sum(ref^2) / sum((ref - est)^2)) over the whole track and both channels,
then the mean over tracks. This is NOT museval's median-of-frames SDR, so it is comparable
with MSST-published numbers only.

6-stem models are scored on the 4 MUSDB stems with other = guitar + piano + other.
Results are written per track to benchmarks/results/separation_musdb18hq.json so an
interrupted run resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from bandscribe.separation import get_separator

DATA = Path.home() / ".cache" / "bandscribe" / "datasets" / "musdb18hq" / "test"
RESULTS = Path(__file__).parent / "results" / "separation_musdb18hq.json"
STEMS = ("vocals", "drums", "bass", "other")
DEFAULT_MODELS = ("bs_roformer_sw", "scnet_xl_ihf", "htdemucs_6s", "subroformer")
# Ensembles scored on the fly: name -> {model: weight}; applied to vocals / drums / bass only.
ENSEMBLES = {
    "avg(sw,scnet)": {"bs_roformer_sw": 0.5, "scnet_xl_ihf": 0.5},
    "avg(sw,subroformer)": {"bs_roformer_sw": 0.5, "subroformer": 0.5},
    "avg(scnet,subroformer)": {"scnet_xl_ihf": 0.5, "subroformer": 0.5},
    "avg(sw,scnet,subroformer)": {"bs_roformer_sw": 1 / 3, "scnet_xl_ihf": 1 / 3,
                                  "subroformer": 1 / 3},
}


def sdr(ref: np.ndarray, est: np.ndarray) -> float:
    n = min(ref.shape[-1], est.shape[-1])
    ref, est = ref[..., :n].astype(np.float64), est[..., :n].astype(np.float64)
    num = np.sum(ref ** 2) + 1e-8
    den = np.sum((ref - est) ** 2) + 1e-8
    return float(10 * np.log10(num / den))


def read(path: Path) -> np.ndarray:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    if sr != 44100:
        raise RuntimeError(f"{path}: unexpected sample rate {sr}")
    return np.ascontiguousarray(audio.T)


def four_stem(stems: dict) -> dict[str, np.ndarray]:
    out = {k: stems[k].audio for k in ("vocals", "drums", "bass")}
    out["other"] = sum(stems[k].audio for k in stems if k not in out)
    return out


def summarize(tracks: dict) -> dict:
    systems: dict[str, dict[str, list[float]]] = {}
    perf: dict[str, dict[str, list[float]]] = {}
    for t in tracks.values():
        for system, scores in t["sdr"].items():
            for stem, value in scores.items():
                systems.setdefault(system, {}).setdefault(stem, []).append(value)
        for model, p in t["perf"].items():
            for key, value in p.items():
                if isinstance(value, int | float):
                    perf.setdefault(model, {}).setdefault(key, []).append(value)
    table = {}
    for system, by_stem in systems.items():
        row = {stem: round(float(np.mean(v)), 3) for stem, v in by_stem.items()}
        if all(s in row for s in STEMS):
            row["avg"] = round(float(np.mean([row[s] for s in STEMS])), 3)
        row["tracks"] = len(next(iter(by_stem.values())))
        table[system] = row
    perf_table = {m: {"peak_vram_gb": round(max(p["peak_vram_gb"]), 2),
                      "s_per_audio_minute": round(float(np.mean(p["s_per_audio_minute"])), 2),
                      "oom_retries": int(sum(p["oom_retries"]))} for m, p in perf.items()}
    return {"sdr": table, "perf": perf_table}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    track_dirs = sorted(p for p in DATA.iterdir() if (p / "mixture.wav").is_file())
    if args.limit:
        track_dirs = track_dirs[: args.limit]
    state = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.is_file() else {"tracks": {}}
    tracks = state["tracks"]

    for n, tdir in enumerate(track_dirs, 1):
        entry = tracks.setdefault(tdir.name, {"sdr": {}, "perf": {}})
        todo = [m for m in args.models if m not in entry["sdr"]]
        need_ens = [e for e, w in ENSEMBLES.items()
                    if e not in entry["sdr"] and all(m in args.models or m in entry["sdr"] for m in w)]
        if not todo and not need_ens:
            continue
        mix = read(tdir / "mixture.wav")
        refs = {s: read(tdir / f"{s}.wav") for s in STEMS}
        minutes = mix.shape[-1] / 44100 / 60
        # Estimates are kept in RAM only for models an ensemble still needs.
        ens_models = {m for e in need_ens for m in ENSEMBLES[e]}
        est: dict[str, dict[str, np.ndarray]] = {}
        for model_id in sorted(set(todo) | ens_models, key=args.models.index):
            started = time.perf_counter()
            res = get_separator(model_id).separate(mix, 44100)
            stems4 = four_stem(res.stems)
            entry["sdr"][model_id] = {s: round(sdr(refs[s], stems4[s]), 4) for s in STEMS}
            entry["perf"][model_id] = {
                "peak_vram_gb": round(res.peak_vram_gb, 3),
                "s_per_audio_minute": round(res.seconds / minutes, 3),
                "oom_retries": res.oom_retries, "device": res.device,
                "batch_size": res.batch_size, "chunk_size": res.chunk_size,
                "notes": list(res.notes)}
            if model_id in ens_models:
                est[model_id] = {s: stems4[s] for s in ("vocals", "drums", "bass")}
            print(f"[{n}/{len(track_dirs)}] {tdir.name[:40]:40s} {model_id:16s} "
                  f"{entry['sdr'][model_id]} {time.perf_counter() - started:.0f}s", flush=True)
        for e in need_ens:
            w = ENSEMBLES[e]
            entry["sdr"][e] = {s: round(sdr(refs[s], sum(est[m][s] * wt for m, wt in w.items())), 4)
                               for s in ("vocals", "drums", "bass")}
        state["summary"] = summarize(tracks)
        RESULTS.parent.mkdir(exist_ok=True)
        RESULTS.write_text(json.dumps(state, indent=1), encoding="utf-8")
    state["summary"] = summarize(tracks)
    RESULTS.write_text(json.dumps(state, indent=1), encoding="utf-8")
    print(json.dumps(state["summary"], indent=2))


if __name__ == "__main__":
    main()
