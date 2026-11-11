"""Guitar transcription and TAB benchmark on GuitarSet (QA_PLAN 2.2 / 2.3, spec section 6).

Transcription: note precision / recall / F1 with `mir_eval.transcription`, both onset-only
(50 ms) and onset + offset (50 ms, offset ratio 0.2), mean over tracks, for
  - muscriptor_medium (instrument="guitar")
  - basic_pitch (instrument="guitar")
  - consensus(muscriptor, basic_pitch) with all notes kept, and with unconfirmed notes dropped.

TAB: the ground-truth pitches are given to the DP fret assigner (standard tuning), so the score
measures string/fret choice alone, independent of transcription errors. Accuracy = share of
ground-truth notes whose (string, fret) equals the annotation; unplaced notes count as wrong.
Also reported: the same with the transcribed notes that matched a reference note.

Data: GuitarSet (zenodo 3371780) under ~/.cache/bandscribe/datasets/guitarset:
  annotation/<id>.jams, audio_mono-mic/<id>_mic.wav, audio_mono-pickup_mix/<id>_mix.wav.
JAMS `note_midi` annotations carry `data_source` 0..5 = low E .. high E string.

    py -m uv run python benchmarks/bench_guitarset.py [--audio mic|mix] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import mir_eval
import numpy as np
import soundfile as sf

from bandscribe.tab.fret_dp import DPFretAssigner
from bandscribe.tab.tunings import TUNINGS
from bandscribe.transcription import get_transcriber
from bandscribe.transcription.base import Note
from bandscribe.transcription.consensus import consensus, match_notes

DATA = Path.home() / ".cache" / "bandscribe" / "datasets" / "guitarset"
RESULTS = Path(__file__).parent / "results"
OPEN_LOW_TO_HIGH = (40, 45, 50, 55, 59, 64)  # GuitarSet is in standard tuning


def load_reference(jams_path: Path) -> list[Note]:
    d = json.loads(jams_path.read_text(encoding="utf-8"))
    notes = []
    for ann in d["annotations"]:
        if ann["namespace"] != "note_midi":
            continue
        ds = int(ann["annotation_metadata"]["data_source"])
        for ev in ann["data"]:
            pitch = int(round(ev["value"]))
            notes.append(Note(onset=float(ev["time"]), offset=float(ev["time"] + ev["duration"]),
                              pitch=pitch, instrument="guitar", sources=("ref",),
                              string=6 - ds, fret=pitch - OPEN_LOW_TO_HIGH[ds]))
    return sorted(notes, key=lambda n: (n.onset, n.pitch))


def _arrays(notes: list[Note]) -> tuple[np.ndarray, np.ndarray]:
    if not notes:
        return np.zeros((0, 2)), np.zeros(0)
    iv = np.array([[n.onset, max(n.offset, n.onset + 1e-3)] for n in notes])
    hz = mir_eval.util.midi_to_hz(np.array([n.pitch for n in notes], dtype=float))
    return iv, hz


def note_scores(ref: list[Note], est: list[Note]) -> dict[str, float]:
    ri, rp = _arrays(ref)
    ei, ep = _arrays(est)
    if len(ei) == 0:
        return {"p": 0.0, "r": 0.0, "f": 0.0, "p_off": 0.0, "r_off": 0.0, "f_off": 0.0}
    p, r, f, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep, offset_ratio=None)
    po, ro, fo, _ = mir_eval.transcription.precision_recall_f1_overlap(ri, rp, ei, ep)
    return {"p": p, "r": r, "f": f, "p_off": po, "r_off": ro, "f_off": fo}


def tab_accuracy(ref: list[Note], assigned: list[Note]) -> tuple[int, int]:
    """(correct, total) where assigned notes are matched to ref by onset and pitch."""
    pairs = match_notes(ref, assigned, tolerance=0.05, same_pitch=True)
    correct = sum(1 for i, j in pairs
                  if (assigned[j].string, assigned[j].fret) == (ref[i].string, ref[i].fret))
    return correct, len(ref)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DATA)
    ap.add_argument("--audio", choices=("mic", "mix"), default="mic")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ids = sorted(p.stem for p in (args.data / "annotation").glob("*.jams"))
    if args.limit:
        ids = ids[:: max(1, len(ids) // args.limit)][: args.limit]
    adir = args.data / f"audio_mono-{'mic' if args.audio == 'mic' else 'pickup_mix'}"

    refs, audios = {}, {}
    for tid in ids:
        refs[tid] = load_reference(args.data / "annotation" / f"{tid}.jams")
        y, sr = sf.read(adir / f"{tid}_{args.audio}.wav", dtype="float32", always_2d=False)
        audios[tid] = (y, sr)

    est: dict[str, dict[str, list[Note]]] = {tid: {} for tid in ids}
    timing: dict[str, float] = {}
    run_info: dict[str, dict] = {}

    t0 = time.perf_counter()
    mus = get_transcriber("muscriptor_medium")
    outs = mus.transcribe_many([(audios[t][0], audios[t][1], "guitar") for t in ids])
    for tid, notes in zip(ids, outs, strict=True):
        est[tid]["muscriptor_medium"] = notes
    timing["muscriptor_medium"] = time.perf_counter() - t0
    run_info["muscriptor_medium"] = {"last_run": mus.last_run, "sha256": mus.model_sha256}
    del mus

    t0 = time.perf_counter()
    bp = get_transcriber("basic_pitch")
    for tid in ids:
        est[tid]["basic_pitch"] = bp.transcribe(audios[tid][0], audios[tid][1], "guitar")
    timing["basic_pitch"] = time.perf_counter() - t0
    run_info["basic_pitch"] = {"last_run": bp.last_run, "sha256": bp.model_sha256}

    agreement = []
    for tid in ids:
        keep, stats = consensus(est[tid]["muscriptor_medium"], est[tid]["basic_pitch"])
        drop, _ = consensus(est[tid]["muscriptor_medium"], est[tid]["basic_pitch"],
                            drop_unconfirmed=True)
        est[tid]["consensus_keep"] = keep
        est[tid]["consensus_drop"] = drop
        est[tid]["consensus_conf>=0.5"] = [n for n in keep if n.conf >= 0.5]
        agreement.append(stats.agreement)

    systems = list(est[ids[0]].keys())
    per_track = {s: [] for s in systems}
    for tid in ids:
        for s in systems:
            per_track[s].append({"id": tid, **note_scores(refs[tid], est[tid][s]),
                                 "n_est": len(est[tid][s]), "n_ref": len(refs[tid])})

    summary = {}
    for s in systems:
        rows = per_track[s]
        summary[s] = {k: float(np.mean([r[k] for r in rows])) for k in
                      ("p", "r", "f", "p_off", "r_off", "f_off")}
        for style in ("comp", "solo"):
            sub = [r for r in rows if r["id"].endswith(style)]
            if sub:
                summary[s][f"f_{style}"] = float(np.mean([r["f"] for r in sub]))

    # TAB: oracle pitches, then transcribed notes.
    dp = DPFretAssigner()
    tuning = TUNINGS["guitar_standard"]
    oracle_c = oracle_t = 0
    oracle_unplaced = 0
    for tid in ids:
        bare = [replace(n, string=None, fret=None) for n in refs[tid]]
        res = dp.assign(bare, tuning)
        c, t = tab_accuracy(refs[tid], res.notes)
        oracle_c += c
        oracle_t += t
        oracle_unplaced += len(res.unplaced)
    tab = {"oracle_pitch_accuracy": oracle_c / max(1, oracle_t), "oracle_notes": oracle_t,
           "oracle_unplaced": oracle_unplaced}
    for s in ("muscriptor_medium", "consensus_keep"):
        c_tot = m_tot = 0
        for tid in ids:
            res = dp.assign([replace(n, string=None, fret=None) for n in est[tid][s]], tuning)
            pairs = match_notes(refs[tid], res.notes, tolerance=0.05, same_pitch=True)
            m_tot += len(pairs)
            c_tot += sum(1 for i, j in pairs if (res.notes[j].string, res.notes[j].fret)
                         == (refs[tid][i].string, refs[tid][i].fret))
        tab[f"{s}_matched_notes_accuracy"] = c_tot / max(1, m_tot)
        tab[f"{s}_matched_notes"] = m_tot

    out = {
        "dataset": f"GuitarSet ({args.audio}), {len(ids)} tracks",
        "metric": "mir_eval.transcription, onset tol 50 ms; *_off adds offset ratio 0.2",
        "summary": summary,
        "consensus_agreement_mean": float(np.mean(agreement)),
        "tab": tab,
        "seconds": timing,
        "audio_seconds": float(sum(len(a[0]) / a[1] for a in audios.values())),
        "run_info": run_info,
        "per_track": per_track,
    }
    RESULTS.mkdir(parents=True, exist_ok=True)
    name = f"guitarset_{args.audio}" + (f"_n{len(ids)}" if args.limit else "") + ".json"
    (RESULTS / name).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("dataset", "summary", "consensus_agreement_mean",
                                          "tab", "seconds", "audio_seconds")}, indent=1))


if __name__ == "__main__":
    main()
