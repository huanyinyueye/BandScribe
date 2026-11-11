"""Separation benchmark on MUSDB18-HQ test (QA_PLAN 2.1, spec section 6).

Metric: global SDR per stem per track, 10*log10(sum(s^2) / sum((s - s_hat)^2)), averaged over
tracks. This is the "SDR avg" used by Music-Source-Separation-Training and the MVSep
leaderboards (not museval's chunked median BSSEval SDR), so numbers are comparable with
MSST's MUSDB table. For 6-stem models, MUSDB's `other` is compared with
other + guitar + piano (MUSDB has no guitar / piano references).

Also recorded per model: peak VRAM, seconds per audio minute, OOM retries, device.
Ensembles (primary 6-stem + secondary 4-stem, vocals / drums / bass blended with weight w)
are scored from saved float16 estimates, for w in WEIGHTS.

Resumable: per-track results are appended to results/separation_musdb_tracks.jsonl.

    py -m uv run python benchmarks/bench_separation.py [--models a,b] [--limit N]
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from bandscribe.separation import get_separator

DATA = Path.home() / ".cache" / "bandscribe" / "datasets" / "musdb18hq" / "test"
EST = Path.home() / ".cache" / "bandscribe" / "bench" / "sep_est"
RESULTS = Path(__file__).parent / "results"
TRACKS = RESULTS / "separation_musdb_tracks.jsonl"
MUSDB_STEMS = ("vocals", "drums", "bass", "other")
BLEND = ("vocals", "drums", "bass")
WEIGHTS = (0.25, 0.5, 0.75)
DEFAULT_MODELS = ("bs_roformer_sw", "scnet_xl_ihf", "subroformer", "htdemucs_6s")
SAVE_FOR_ENSEMBLE = {"bs_roformer_sw", "scnet_xl_ihf", "subroformer"}


def sdr(ref: np.ndarray, est: np.ndarray, eps: float = 1e-8) -> float:
    n = min(ref.shape[-1], est.shape[-1])
    ref, est = ref[..., :n].astype(np.float64), est[..., :n].astype(np.float64)
    return float(10 * np.log10((np.sum(ref ** 2) + eps) / (np.sum((ref - est) ** 2) + eps)))


def load(path: Path) -> np.ndarray:
    y, sr = sf.read(path, dtype="float32", always_2d=True)
    assert sr == 44100, path
    return y.T  # (2, n)


def to_musdb(stems: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    out = {k: stems[k] for k in ("vocals", "drums", "bass") if k in stems}
    other = [stems[k] for k in stems if k not in ("vocals", "drums", "bass")]
    out["other"] = np.sum(other, axis=0) if other else None
    return out


def done_pairs() -> set[tuple[str, str]]:
    if not TRACKS.exists():
        return set()
    return {(r["model"], r["track"]) for r in map(json.loads, TRACKS.read_text("utf-8").splitlines())
            if r}


def run_model(model_id: str, tracks: list[Path]) -> None:
    done = done_pairs()
    todo = [t for t in tracks if (model_id, t.name) not in done]
    if not todo:
        return
    sep = get_separator(model_id)
    for tdir in todo:
        mix = load(tdir / "mixture.wav")
        res = sep.separate(mix, sr=44100)
        est = to_musdb({k: s.audio for k, s in res.stems.items()})
        row = {"model": model_id, "track": tdir.name, "seconds": res.seconds,
               "audio_seconds": mix.shape[-1] / 44100, "peak_vram_gb": res.peak_vram_gb,
               "device": res.device, "oom_retries": res.oom_retries,
               "chunk_size": res.chunk_size, "batch_size": res.batch_size, "sdr": {}}
        for stem in MUSDB_STEMS:
            ref = load(tdir / f"{stem}.wav")
            row["sdr"][stem] = sdr(ref, est[stem]) if est.get(stem) is not None else None
        if model_id in SAVE_FOR_ENSEMBLE:
            d = EST / model_id
            d.mkdir(parents=True, exist_ok=True)
            np.savez(d / f"{tdir.name}.npz", **{k: est[k].astype(np.float16) for k in BLEND})
        with TRACKS.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        print(model_id, tdir.name, {k: round(v, 2) for k, v in row["sdr"].items() if v is not None},
              f"{row['seconds']:.1f}s", flush=True)
        del res, est, mix
        gc.collect()
    del sep
    gc.collect()


def summarise(tracks: list[Path]) -> dict:
    rows = [json.loads(x) for x in TRACKS.read_text("utf-8").splitlines() if x]
    names = {t.name for t in tracks}
    rows = [r for r in rows if r["track"] in names]
    out: dict = {"models": {}, "ensembles": {}}
    for model in sorted({r["model"] for r in rows}):
        mr = [r for r in rows if r["model"] == model]
        s = {st: float(np.mean([r["sdr"][st] for r in mr if r["sdr"].get(st) is not None]))
             for st in MUSDB_STEMS}
        s["avg"] = float(np.mean([s[st] for st in MUSDB_STEMS]))
        out["models"][model] = {
            "tracks": len(mr), "sdr": s,
            "peak_vram_gb": max(r["peak_vram_gb"] for r in mr),
            "s_per_audio_minute": 60 * sum(r["seconds"] for r in mr) / sum(r["audio_seconds"] for r in mr),
            "oom_retries": sum(r["oom_retries"] for r in mr),
            "devices": sorted({r["device"] for r in mr}),
        }

    # Ensembles from saved estimates.
    for prim in ("bs_roformer_sw",):
        for sec in ("scnet_xl_ihf", "subroformer"):
            pd, sd = EST / prim, EST / sec
            common = [t for t in tracks if (pd / f"{t.name}.npz").exists()
                      and (sd / f"{t.name}.npz").exists()]
            if not common:
                continue
            acc = {w: {st: [] for st in BLEND} for w in WEIGHTS}
            for t in common:
                a, b = np.load(pd / f"{t.name}.npz"), np.load(sd / f"{t.name}.npz")
                for st in BLEND:
                    ref = load(t / f"{st}.wav")
                    ea, eb = a[st].astype(np.float32), b[st].astype(np.float32)
                    n = min(ea.shape[-1], eb.shape[-1])
                    for w in WEIGHTS:
                        acc[w][st].append(sdr(ref, (1 - w) * ea[..., :n] + w * eb[..., :n]))
            out["ensembles"][f"{prim}+{sec}"] = {
                "tracks": len(common),
                "sdr_by_weight": {str(w): {st: float(np.mean(v)) for st, v in acc[w].items()}
                                  for w in WEIGHTS}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    tracks = sorted(p for p in DATA.iterdir() if p.is_dir())
    if args.limit:
        tracks = tracks[: args.limit]
    RESULTS.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    for m in args.models.split(","):
        try:
            run_model(m, tracks)
        except Exception as exc:  # noqa: BLE001 - record and continue with the next model
            print(f"{m} FAILED: {type(exc).__name__}: {exc}", flush=True)
            with TRACKS.open("a", encoding="utf-8") as f:
                f.write("")
    summary = summarise(tracks)
    summary.update({"dataset": f"MUSDB18-HQ test, {len(tracks)} tracks",
                    "metric": "global SDR (MSST 'SDR avg'), mean over tracks",
                    "wall_seconds": time.perf_counter() - t0})
    name = "separation_musdb" + (f"_n{len(tracks)}" if args.limit else "") + ".json"
    (RESULTS / name).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
