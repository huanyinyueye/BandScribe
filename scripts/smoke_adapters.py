"""Run every model adapter once on a short clip and record that it works on this GPU.

Usage: uv run python scripts/smoke_adapters.py [audio file] [--seconds 30]

This is an adapter smoke test, not a quality benchmark: it records device, peak VRAM,
seconds, OOM retries and output shape for each adapter in docs/adapter_smoke.json.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from bandscribe.analysis.beats import track_beats
from bandscribe.io.audio_load import load_audio
from bandscribe.runtime.vram import gpu_slot
from bandscribe.separation import get_separator
from bandscribe.transcription import get_transcriber
from bandscribe.transcription.base import TranscriberUnavailable

DEFAULT_CLIP = Path.home() / ".cache/bandscribe/datasets/gtzan/genres/metal/metal.00010.wav"
OUT = Path(__file__).resolve().parent.parent / "docs" / "adapter_smoke.json"
SEPARATORS = [("bs_roformer_sw", {}), ("scnet_xl_ihf", {}), ("htdemucs_6s", {}),
              ("subroformer", {}), ("mvsep_mega53", {}),
              ("mvsep_mega53", {"chunk_size": 441000}), ("mvsep_mega53", {"chunk_size": 220500})]
TRANSCRIBERS = [("muscriptor_medium", "guitar"), ("basic_pitch", "guitar"), ("adtof", "drums")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", nargs="?", type=Path, default=DEFAULT_CLIP)
    ap.add_argument("--seconds", type=float, default=30.0)
    args = ap.parse_args()
    audio = load_audio(args.audio)
    n = int(args.seconds * audio.sr)
    mix = audio.samples[:, :n]
    minutes = mix.shape[-1] / audio.sr / 60
    report: dict = {"clip": args.audio.name, "seconds": round(mix.shape[-1] / audio.sr, 2),
                    "separators": [], "transcribers": [], "beats": None}

    started = time.perf_counter()
    beats = track_beats(mix.mean(axis=0), audio.sr)
    rec = gpu_slot.records[-1]
    report["beats"] = {"model": "beat_this", "ok": True, "device": beats.device,
                       "beats": len(beats.beats), "peak_vram_gb": round(rec.peak_gb, 3),
                       "seconds": round(time.perf_counter() - started, 2)}
    print(report["beats"], flush=True)

    stems = {}
    for model_id, opts in SEPARATORS:
        entry = {"model": model_id, "options": opts}
        try:
            res = get_separator(model_id, **opts).separate(mix, audio.sr)
            total = sum(s.audio for s in res.stems.values())
            m = total.shape[-1]
            resid = float(np.mean((mix[:, :m] - total) ** 2))
            entry.update(ok=True, device=res.device, stems=len(res.stems),
                         batch_size=res.batch_size, chunk_size=res.chunk_size,
                         peak_vram_gb=round(res.peak_vram_gb, 2),
                         seconds=round(res.seconds, 2),
                         s_per_audio_minute=round(res.seconds / minutes, 1),
                         oom_retries=res.oom_retries, notes=list(res.notes),
                         stems_sum_vs_mix_db=round(10 * np.log10(
                             float(np.mean(mix[:, :m] ** 2)) / max(resid, 1e-20)), 1),
                         files_sha256=res.files_sha256)
            if model_id == "bs_roformer_sw":
                stems = {k: s.audio for k, s in res.stems.items()}
        except Exception as exc:  # noqa: BLE001 - a smoke test records every failure
            entry.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:300]}")
        report["separators"].append(entry)
        print(entry, flush=True)

    for model_id, family in TRANSCRIBERS:
        entry = {"model": model_id, "input": f"{family} stem (bs_roformer_sw)"}
        try:
            tr = get_transcriber(model_id)
            notes = tr.transcribe(stems[family], audio.sr, family)
            entry.update(ok=True, notes=len(notes), sha256=tr.model_sha256, **(tr.last_run or {}))
        except TranscriberUnavailable as exc:
            entry.update(ok=None, skipped=str(exc).splitlines()[0])  # not a failure
        except Exception as exc:  # noqa: BLE001
            entry.update(ok=False, error=f"{type(exc).__name__}: {str(exc)[:300]}")
        report["transcribers"].append(entry)
        print(entry, flush=True)

    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
