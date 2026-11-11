"""Grid search of DPFretAssigner cost weights on GuitarSet ground-truth pitches.

Tune on players 00-02, report on held-out players 03-05 (no tuning on the test half).
Writes benchmarks/results/fret_dp_tuning.json.

    py -m uv run python benchmarks/tune_fret_dp.py
"""

from __future__ import annotations

import itertools
import json
import sys
from dataclasses import asdict, replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from bench_guitarset import DATA, RESULTS, load_reference, tab_accuracy  # noqa: E402

from bandscribe.tab.fret_dp import DPFretAssigner, FretOptions  # noqa: E402
from bandscribe.tab.tunings import TUNINGS  # noqa: E402

TUNING = TUNINGS["guitar_standard"]


def accuracy(opt: FretOptions, refs: list[list]) -> float:
    dp = DPFretAssigner(opt)
    c = t = 0
    for ref in refs:
        res = dp.assign([replace(n, string=None, fret=None) for n in ref], TUNING)
        ci, ti = tab_accuracy(ref, res.notes)
        c += ci
        t += ti
    return c / max(1, t)


def main() -> None:
    files = sorted((DATA / "annotation").glob("*.jams"))
    train = [load_reference(p) for p in files if p.stem[:2] in ("00", "01", "02")]
    test = [load_reference(p) for p in files if p.stem[:2] in ("03", "04", "05")]

    base = FretOptions()
    result = {"default": asdict(base), "default_train": accuracy(base, train),
              "default_test": accuracy(base, test)}
    print("default", result["default_train"], result["default_test"], flush=True)

    grid = {
        "open_bonus": [0.0, 0.25, 0.5],
        "high_fret_weight": [0.0, 0.02, 0.05],
        "span_weight": [0.3, 0.6, 1.0],
        "move_weight": [0.5, 1.0, 2.0, 4.0],
        "string_jump_weight": [0.0, 0.15, 0.5],
    }
    best, best_acc = base, result["default_train"]
    rows = []
    keys = list(grid)
    for vals in itertools.product(*(grid[k] for k in keys)):
        opt = replace(base, **dict(zip(keys, vals, strict=True)))
        acc = accuracy(opt, train)
        rows.append({**dict(zip(keys, vals, strict=True)), "train": acc})
        if acc > best_acc:
            best, best_acc = opt, acc
            print("better", dict(zip(keys, vals, strict=True)), round(acc, 4), flush=True)

    result.update({"best": asdict(best), "best_train": best_acc,
                   "best_test": accuracy(best, test), "grid": grid,
                   "rows": sorted(rows, key=lambda r: -r["train"])[:30],
                   "train_tracks": len(train), "test_tracks": len(test)})
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "fret_dp_tuning.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("default_train", "default_test", "best",
                                             "best_train", "best_test")}, indent=1))


if __name__ == "__main__":
    main()
