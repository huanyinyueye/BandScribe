"""Beat / downbeat F-measure of Beat This! on GTZAN (spec section 6, M1).

Usage: uv run python benchmarks/bench_beats.py [--limit N] [--data DIR]

Audio: GTZAN `genres/<genre>/<genre>.NNNNN.wav`. Annotations: CPJKU/beat_this_annotations
`gtzan/annotations/beats/gtzan_<genre>_NNNNN.beats` (time <tab> beat number, 1 = downbeat).
Metric: mir_eval.beat.f_measure, 70 ms window, no trimming of the first seconds.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import mir_eval
import numpy as np

from bandscribe.analysis.beats import BeatTracker
from bandscribe.io.audio_load import AudioLoadError, load_audio
from bandscribe.runtime.vram import gpu_slot

DEFAULT_DATA = Path.home() / ".cache" / "bandscribe" / "datasets" / "gtzan"
RESULTS = Path(__file__).parent / "results" / "beats_gtzan.json"


def read_annotation(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    times = np.array([float(r[0]) for r in rows])
    if any(len(r) < 2 for r in rows):  # beat times only: no downbeat annotation
        return times, np.array([])
    numbers = np.array([int(float(r[1])) for r in rows])
    return times, times[numbers == 1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--limit", type=int, default=0, help="evaluate only the first N tracks")
    args = ap.parse_args()

    ann_dir = args.data / "beat_this_annotations-main" / "gtzan" / "annotations" / "beats"
    ann_files = sorted(ann_dir.glob("*.beats"))
    if args.limit:
        step = max(1, len(ann_files) // args.limit)
        ann_files = ann_files[::step][: args.limit]

    beat_f, down_f, skipped, audio_seconds = [], [], [], 0.0
    per_genre: dict[str, list[float]] = {}
    started = time.perf_counter()
    with BeatTracker() as tracker:
        for n, ann in enumerate(ann_files, 1):
            _, genre, num = ann.stem.split("_")
            wav = args.data / "genres" / genre / f"{genre}.{num}.wav"
            try:
                audio = load_audio(wav)
            except AudioLoadError as exc:
                skipped.append({"file": wav.name, "reason": str(exc)})
                continue
            ref_beats, ref_down = read_annotation(ann)
            res = tracker.track(audio)
            audio_seconds += audio.duration
            f = mir_eval.beat.f_measure(ref_beats, res.beats, f_measure_threshold=0.07)
            beat_f.append(f)
            per_genre.setdefault(genre, []).append(f)
            if len(ref_down):
                down_f.append(mir_eval.beat.f_measure(ref_down, res.downbeats,
                                                      f_measure_threshold=0.07))
            if n % 100 == 0:
                print(f"{n}/{len(ann_files)} beat F {np.mean(beat_f):.4f}", flush=True)
        device, sha = tracker.device, tracker.sha256
    elapsed = time.perf_counter() - started

    result = {
        "dataset": "GTZAN (annotations: CPJKU/beat_this_annotations)",
        "model": "beat_this final0", "model_sha256": sha, "device": device,
        "tracks": len(beat_f), "tracks_with_downbeats": len(down_f), "skipped": skipped,
        "beat_f": round(float(np.mean(beat_f)), 4),
        "downbeat_f": round(float(np.mean(down_f)), 4),
        "beat_f_per_genre": {g: round(float(np.mean(v)), 4) for g, v in sorted(per_genre.items())},
        "peak_vram_gb": round(gpu_slot.records[-1].peak_gb, 3),
        "seconds_per_audio_minute": round(elapsed / (audio_seconds / 60), 3),
        "includes_audio_decoding": True,
    }
    RESULTS.parent.mkdir(exist_ok=True)
    RESULTS.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
