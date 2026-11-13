"""Score a pipeline output directory against a reference Guitar Pro transcription.

The reference notes come from benchmarks/gp_reference.mjs (alphaTab playback, repeats expanded).
Score time is mapped to audio time with the file's tempo map and first sync point, then a
global offset search (+-150 ms) checks that alignment.

    node benchmarks/gp_reference.mjs ref.gp ref.json
    py -m uv run python benchmarks/score_vs_gp.py ref.json out/<song> [--map guitar=GT1,GT2]

Metrics (mir_eval, onset tolerance 50 ms):
  note F1      pitch + onset
  chroma F1    pitch class + onset (octave errors forgiven)
  onset F1     onsets only, any pitch (rhythm)
  drums        onset F1 per coarse class (kick / snare / hihat / tom / cymbal)
  beats        mir_eval beat F-measure against the reference quarter-note grid
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mir_eval
import numpy as np

from bandscribe.transcription.drums import coarse_class

TOL = 0.05
DEFAULT_MAP = {"guitar": ["GT1", "GT2", "GT3", "GT4"], "bass": ["Ba"], "piano": ["KB.2"],
               "keys_all": ["KB.1", "KB.2"], "vocals": ["Vo."], "drums": ["Drumkit"]}
GP_OUTPUT_MAP = {"guitar": ["Guitar (combined)"], "bass": ["Bass"], "piano": ["Piano"],
                 "vocals": ["Vocals"], "drums": ["Drums"]}


def tick_to_seconds(ref: dict):
    ppq = ref["ppq"]
    tempos = sorted(ref["tempos"], key=lambda t: t["tick"])
    offset = ((ref["syncPoints"][0]["millisecondOffset"] / 1000)
              if ref["syncPoints"] else 0.0) + ref.get("audio_offset_seconds", 0.0)
    # Cumulative seconds at each tempo change.
    starts, secs, acc = [], [], 0.0
    for i, t in enumerate(tempos):
        if i:
            prev = tempos[i - 1]
            acc += (t["tick"] - prev["tick"]) / ppq * 60 / prev["bpm"]
        starts.append(t["tick"])
        secs.append(acc)

    def conv(tick: float) -> float:
        i = max(0, np.searchsorted(starts, tick, side="right") - 1)
        return offset + secs[i] + (tick - starts[i]) / ppq * 60 / tempos[i]["bpm"]

    return conv


def ref_notes(ref: dict, names: list[str]) -> np.ndarray:
    conv = tick_to_seconds(ref)
    rows = []
    for t in ref["tracks"]:
        if t["name"] in names:
            rows += [(conv(n["tick"]), conv(n["tick"] + n["length"]), n["key"]) for n in t["notes"]]
    rows.sort()
    # Doubled parts (e.g. two guitars playing the same riff) count once.
    out: list[tuple[float, float, int]] = []
    for r in rows:
        if not any(abs(r[0] - o[0]) < 0.03 and r[2] == o[2] for o in out[-12:]):
            out.append(r)
    return np.array(out, dtype=float).reshape(-1, 3)


def est_notes(path: Path) -> np.ndarray:
    if not path.exists():
        return np.zeros((0, 3))
    d = json.loads(path.read_text("utf-8"))
    return np.array([(n["onset"], max(n["offset"], n["onset"] + 0.01), n["pitch"]) for n in d],
                    dtype=float).reshape(-1, 3)


def prf(ref: np.ndarray, est: np.ndarray, *, chroma: bool = False, onset_only: bool = False,
        shift: float = 0.0) -> dict:
    if len(ref) == 0 or len(est) == 0:
        return {"p": 0.0, "r": 0.0, "f": 0.0, "n_ref": len(ref), "n_est": len(est)}
    ri, ei = ref[:, :2].copy(), est[:, :2].copy() + shift
    rp, ep = ref[:, 2].copy(), est[:, 2].copy()
    if chroma:
        rp, ep = 60 + rp % 12, 60 + ep % 12
    if onset_only:
        rp, ep = np.full_like(rp, 60), np.full_like(ep, 60)
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(
        ri, mir_eval.util.midi_to_hz(rp), ei, mir_eval.util.midi_to_hz(ep),
        onset_tolerance=TOL, offset_ratio=None)
    return {"p": round(p, 3), "r": round(r, 3), "f": round(f, 3), "n_ref": len(ref), "n_est": len(est)}


def best_shift(ref: np.ndarray, est: np.ndarray) -> tuple[float, float]:
    best = (0.0, -1.0)
    for s in np.arange(-0.15, 0.1501, 0.005):
        f = prf(ref, est, onset_only=True, shift=float(s))["f"]
        if f > best[1]:
            best = (round(float(s), 3), f)
    return best


def drum_scores(ref: np.ndarray, est: np.ndarray) -> dict:
    out = {}
    for cls in ("kick", "snare", "hihat", "tom", "cymbal"):
        r = np.array(sorted({round(x[0], 3) for x in ref if coarse_class(int(x[2])) == cls}))
        e = np.array(sorted({round(x[0], 3) for x in est if coarse_class(int(x[2])) == cls}))
        if len(r) == 0 and len(e) == 0:
            continue
        f, p, rr = mir_eval.onset.f_measure(r, e, window=TOL) if len(r) and len(e) else (0, 0, 0)
        out[cls] = {"p": round(p, 3), "r": round(rr, 3), "f": round(f, 3), "n_ref": len(r),
                    "n_est": len(e)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("out_dir")
    ap.add_argument("--map", action="append", default=[], help="family=TrackA,TrackB")
    ap.add_argument("--audio-offset", type=float, default=0.0,
                    help="Seconds the supplied audio lags the GP embedded audio (negative if earlier).")
    ap.add_argument("--gp-playback", help="JSON from gp_reference.mjs for the exported GP file")
    args = ap.parse_args()
    ref = json.loads(Path(args.ref).read_text("utf-8"))
    ref["audio_offset_seconds"] = args.audio_offset
    out = Path(args.out_dir)
    mapping = dict(DEFAULT_MAP)
    for m in args.map:
        k, v = m.split("=", 1)
        mapping[k] = v.split(",")
    notes_dir = out / "notes"
    report: dict = {"audio_offset_seconds": args.audio_offset, "families": {}, "raw": {}}

    for fam, names in mapping.items():
        r = ref_notes(ref, names)
        if len(r) == 0:
            continue
        est_fam = "piano" if fam == "keys_all" else fam
        e = est_notes(notes_dir / f"final_{est_fam}.json")
        row: dict = {"reference_tracks": names}
        if fam == "drums":
            row["by_class"] = drum_scores(r, e)
            row["onset"] = prf(r, e, onset_only=True)
        else:
            row["note"] = prf(r, e)
            row["chroma"] = prf(r, e, chroma=True)
            row["onset"] = prf(r, e, onset_only=True)
        row["best_shift_s"], row["onset_f_at_best_shift"] = best_shift(r, e) if len(e) else (None, None)
        report["families"][fam] = row
        print(fam, json.dumps({k: v for k, v in row.items() if k != "reference_tracks"}))

    # Per-model raw outputs, for the same reference families.
    for p in sorted(notes_dir.glob("raw_*.json")):
        fam = p.stem.split("_")[-1]
        names = mapping.get(fam)
        if not names:
            continue
        r, e = ref_notes(ref, names), est_notes(p)
        if len(r) == 0:
            continue  # no reference part: do not report a misleading zero score
        if fam == "drums":
            report["raw"][p.stem] = {"by_class": drum_scores(r, e)}
        else:
            report["raw"][p.stem] = {"note": prf(r, e), "chroma": prf(r, e, chroma=True)}
        print(p.stem, json.dumps(report["raw"][p.stem]))

    # Beats and tempo.
    an = out / "analysis.json"
    if an.exists():
        a = json.loads(an.read_text("utf-8"))
        if args.gp_playback:
            playback = json.loads(Path(args.gp_playback).read_text("utf-8"))
            # GP playback begins at the score's first bar, which may precede audio time zero.
            playback["audio_offset_seconds"] = a["bars"][0]["start_time"]
            report["exported_gp"] = {}
            for fam, names in mapping.items():
                if fam not in GP_OUTPUT_MAP:
                    continue
                r, e = ref_notes(ref, names), ref_notes(playback, GP_OUTPUT_MAP[fam])
                if len(r) == 0:
                    continue
                report["exported_gp"][fam] = (drum_scores(r, e) if fam == "drums"
                                              else prf(r, e))
                print("exported_gp", fam, json.dumps(report["exported_gp"][fam]))
        conv = tick_to_seconds(ref)
        end = max(conv(n["tick"]) for t in ref["tracks"] for n in t["notes"])
        ref_beats = np.array([conv(k * ref["ppq"]) for k in range(10_000)
                              if conv(k * ref["ppq"]) <= end])
        est_beats = np.array([b if isinstance(b, (int, float)) else b["time"] for b in a["beats"]])
        down = a.get("downbeats") or []
        est_down = np.array([b if isinstance(b, (int, float)) else b["time"] for b in down])
        report["beats"] = {
            "est_bpm": a.get("global_bpm"), "ref_bpm": ref["tempos"][0]["bpm"],
            "beat_f": round(mir_eval.beat.f_measure(ref_beats, est_beats), 4),
            "downbeat_f": round(mir_eval.beat.f_measure(ref_beats[::4], est_down), 4)
            if len(est_down) else None,
            "meter": a.get("meter"), "key": a.get("key"),
        }
        print("beats", json.dumps(report["beats"]))

    dest = out / "score_vs_reference.json"
    dest.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
