"""Compare the Go exporters (cmd/bandscribe-export) with the Python / alphaTab reference.

    py -m uv run python scripts/parity_exports.py [--stress 300] [--seed 1] [--formats midi,musicxml]

Inputs: every out/*/score.json plus generated stress scores (all note values, dots, tuplets,
ties across bars, per-beat tempo, empty bars, drums, 4-8 strings, F clef, keys, sections,
Unicode). The reference is the Python exporter in this repository (and alphaTab through Node
for `.gp`). MIDI, MusicXML and `.gp5` must be byte-identical; `.gp` must have identical zip
entries and an identical score.gpif element tree. Results go to .cache/parity/report.json.
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import sys
import traceback
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from bandscribe.notation.score_model import BarSpec, Score, ScoreNote, Track  # noqa: E402

CACHE = ROOT / ".cache" / "parity"
GO_EXE = CACHE / "bandscribe-export.exe"

SIGNATURES = [(4, 4), (3, 4), (2, 4), (5, 4), (6, 8), (12, 8), (9, 8), (7, 8), (3, 8), (2, 2),
              (6, 4), (1, 4), (4, 8), (15, 16)]
GRIDS = (2, 3, 4, 6, 8)
TUNINGS = {
    "guitar6": ((64, 59, 55, 50, 45, 40), "Standard"), "guitar_drop_d": ((64, 59, 55, 50, 45, 38), "Drop D"),
    "guitar7": ((64, 59, 55, 50, 45, 40, 35), "7-string"), "guitar8": ((64, 59, 55, 50, 45, 40, 35, 30), ""),
    "bass4": ((43, 38, 33, 28), "Bass"), "bass5": ((43, 38, 33, 28, 23), "5-string bass"),
    "bass6": ((48, 43, 38, 33, 28, 23), ""),
}
DRUM_KEYS = (35, 36, 38, 39, 40, 42, 43, 44, 46, 47, 49, 50, 51, 52, 53, 55, 57, 81, 0, 127)
TEXTS = ["", "Song", "ネオアスペクト", "A & B <test> \"q\" 'a'", "Café déjà", "x" * 300, "tab\there"]
SECTIONS = [None, None, None, "", "Verse", "サビ", "&<>\"'", "Chorus 2"]


def stress_score(rng: random.Random, idx: int) -> Score:
    n_bars = rng.choice([1, 2, 3, rng.randint(4, 40)])
    bars = []
    start_q, t = Fraction(0), rng.uniform(0, 3)
    sig = rng.choice(SIGNATURES)
    bpm = round(rng.uniform(40, 250), 2)
    for i in range(n_bars):
        if rng.random() < 0.2:
            sig = rng.choice(SIGNATURES)
        r = rng.random()
        if r < 0.15:
            bpm = round(bpm + rng.uniform(-0.04, 0.04), 3)
        elif r < 0.3:
            bpm = round(bpm + rng.uniform(-0.99, 0.99), 2)
        elif r < 0.5:
            bpm = round(rng.uniform(30, 300), rng.choice([0, 1, 2, 4]))
        num, den = sig
        length = Fraction(4, den) * num
        dur = float(60 * length / bpm)
        mode = rng.random()
        if mode < 0.15:
            beat_times = ()
        elif mode < 0.2:
            beat_times = tuple(round(t + k * 0.3, 4) for k in range(num))  # wrong length
        else:
            pts = sorted(rng.uniform(0, dur) for _ in range(num - 1))
            jitter = [t] + [t + p for p in pts] + [t + dur]
            beat_times = tuple(round(x, 4) for x in jitter)
        bars.append(BarSpec(i, num, den, i * num, start_q, bpm, round(t, 4), round(t + dur, 4),
                            rng.choice(SECTIONS), rng.random() > 0.1, beat_times))
        start_q += length
        t += dur
    tracks = []
    templates = ["guitar6", "guitar_drop_d", "guitar7", "guitar8", "bass4", "bass5", "bass6",
                 "vocals", "piano", "other", "bass_staff", "drums"]
    chosen = rng.sample(templates, rng.randint(1, min(6, len(templates))))
    for k, name in enumerate(chosen):
        tracks.append(stress_track(rng, name, k, bars))
    key_r = rng.random()
    key = None if key_r < 0.25 else ({"fifths": None} if key_r < 0.3 else
                                     {"name": "x", "fifths": rng.randint(-7, 7),
                                      "mode": rng.choice(["major", "minor"])})
    if key and rng.random() < 0.1:
        key.pop("mode", None)
    title = rng.choice(TEXTS) if rng.random() < 0.5 else f"Stress {idx}"
    meta = {"artist": rng.choice(TEXTS)} if rng.random() < 0.3 else {}
    return Score(title, bars, tracks, key, meta)


def stress_track(rng: random.Random, name: str, k: int, bars: list[BarSpec]) -> Track:
    if name.startswith(("guitar", "bass")) and name != "bass_staff":
        tuning, tname = TUNINGS[name]
        family = "guitar" if name.startswith("guitar") else "bass"
        track = Track(f"{name}{k}", f"{family.title()} {k}", family, "tab",
                      rng.choice([30, 29, 27, 25, 33, 34, 105, 42, 40, 0]), name, tuning,
                      info={"tuning_name": tname} if rng.random() < 0.8 else {})
    elif name == "drums":
        track = Track(f"drums{k}", "Drums", "drums", "drums", 0)
    elif name == "bass_staff":
        track = Track(f"bass{k}", "Bass (staff)", "bass", "staff", 33)
    else:
        track = Track(f"{name}{k}", name.title(), name, "staff",
                      {"vocals": 52, "piano": 0, "other": 48}[name])
    density = rng.choice([0.0, 0.2, 1.0, 3.0, 6.0])
    notes = []
    for bar in bars:
        bq = Fraction(4, bar.denominator)
        count = int(rng.expovariate(1 / density)) if density else 0
        if rng.random() < 0.1:
            count = 0  # empty bar
        for _ in range(count):
            c = rng.randrange(bar.numerator)
            s = track.grid.get((bar.index, c))
            if s is None:
                s = rng.choice(GRIDS)
                if rng.random() < 0.8:
                    track.grid[(bar.index, c)] = s
            j = rng.randrange(s)
            pos = c * bq + j * bq / s
            dur = rng.randint(1, 3 * s) * bq / s
            if rng.random() < 0.05:
                dur = bq * rng.randint(4, 20)
            chord = rng.choice([1, 1, 1, 2, 3, 6])
            for _ in range(chord):
                notes.append(stress_note(rng, track, bar.index, pos, dur))
    track.notes = notes
    return track


def stress_note(rng: random.Random, track: Track, bar: int, pos: Fraction, dur: Fraction) -> ScoreNote:
    string = fret = None
    if track.kind == "tab":
        if rng.random() < 0.95:
            string = rng.randint(1, len(track.tuning))
            fret = rng.choice([0, 0, rng.randint(0, 24)])
            pitch = track.tuning[string - 1] + fret
        else:
            pitch = rng.randint(20, 90)
    elif track.kind == "drums":
        pitch = rng.choice(DRUM_KEYS)
    else:
        pitch = rng.choice([rng.randint(21, 108), rng.randint(0, 127)])
    vel = rng.choice([rng.randint(1, 127), 0, 140, 80])
    return ScoreNote(bar, pos, dur, pitch, vel, round(rng.random(), 4), rng.random() < 0.2,
                     ("muscriptor_medium",), string, fret, (), 0.0, 0.0)


REPORT_KEYS = {"gp": ("verified", "mismatches", "notes_written"),
               "gp5": ("notes_written", "notes_dropped", "warnings")}


def python_exports(score: Score, out: Path, formats: set[str],
                   reports: dict[str, dict]) -> dict[str, str | None]:
    from bandscribe.notation import export_gp, export_gp5, export_midi, export_musicxml

    errors: dict[str, str | None] = {}
    jobs = {"midi": lambda: export_midi.export_midi(score, out / "score.mid"),
            "musicxml": lambda: export_musicxml.export_musicxml(score, out / "score.musicxml"),
            "gp5": lambda: export_gp5.export_gp5(score, out / "score.gp5"),
            "gp": lambda: export_gp.export_gp(score, out / "score.gp")}
    for fmt in formats:
        try:
            rep = jobs[fmt]()
            if fmt in REPORT_KEYS:
                reports[fmt] = {k: getattr(rep, k) for k in REPORT_KEYS[fmt]}
            errors[fmt] = None
        except Exception as exc:  # noqa: BLE001 - record and compare failures too
            errors[fmt] = f"{type(exc).__name__}: {exc}"[:300]
    return errors


FILES = {"midi": "score.mid", "musicxml": "score.musicxml", "gp5": "score.gp5", "gp": "score.gp"}


def go_exports(score_path: Path, out: Path, formats: set[str],
               reports: dict[str, dict]) -> dict[str, str | None]:
    args = [str(GO_EXE), "--score", str(score_path)]
    for fmt in formats:
        args += [f"--{fmt}", str(out / FILES[fmt])]
    proc = subprocess.run(args, capture_output=True, text=True, encoding="utf-8", check=False)
    if proc.returncode != 0:
        return {fmt: f"exit {proc.returncode}: {proc.stderr[-300:]}" for fmt in formats}
    results = json.loads(proc.stdout)["results"]
    res = {r["format"]: (None if r["ok"] else r.get("error", "failed")) for r in results}
    for r in results:
        if r["ok"] and isinstance(r.get("report"), dict):
            reports[r["format"]] = {k: r["report"].get(k) for k in REPORT_KEYS[r["format"]]}
    return {fmt: res.get(fmt, "missing") for fmt in formats}


def compare_gp(a: Path, b: Path) -> str | None:
    from gp_compare import compare  # scripts/gp_compare.py

    return compare(a, b)


def compare(fmt: str, py: Path, go: Path) -> str | None:
    if fmt == "gp":
        return compare_gp(py, go)
    a, b = py.read_bytes(), go.read_bytes()
    if a == b:
        return None
    i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
    return f"bytes differ at {i} (python {len(a)} bytes, go {len(b)} bytes)"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stress", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--formats", default="midi,musicxml,gp5,gp")
    ap.add_argument("--no-real", action="store_true")
    args = ap.parse_args()
    formats = set(args.formats.split(","))
    sys.path.insert(0, str(ROOT / "scripts"))

    cases: list[tuple[str, Path]] = []
    if not args.no_real:
        cases += [(p.parent.name, p) for p in sorted((ROOT / "out").glob("*/score.json"))]
    rng = random.Random(args.seed)
    for i in range(args.stress):
        d = CACHE / "stress" / f"{i:04d}"
        d.mkdir(parents=True, exist_ok=True)
        stress_score(rng, i).save(d / "score.json")
        cases.append((f"stress-{i:04d}", d / "score.json"))

    report = {"cases": len(cases), "formats": sorted(formats), "failures": [],
              "deviations": [], "passed": {f: 0 for f in formats}}
    for name, path in cases:
        work = CACHE / "runs" / name
        (work / "py").mkdir(parents=True, exist_ok=True)
        (work / "go").mkdir(parents=True, exist_ok=True)
        score = Score.load(path)
        py_rep: dict[str, dict] = {}
        go_rep: dict[str, dict] = {}
        try:
            py_err = python_exports(score, work / "py", formats, py_rep)
        except Exception:  # noqa: BLE001
            py_err = {f: traceback.format_exc()[-300:] for f in formats}
        go_err = go_exports(path, work / "go", formats, go_rep)
        for fmt in sorted(formats):
            if py_err[fmt] and not go_err[fmt]:
                report["deviations"].append({"case": name, "format": fmt, "python": py_err[fmt]})
                continue
            if py_err[fmt] or go_err[fmt]:
                if py_err[fmt] and go_err[fmt]:
                    report["passed"][fmt] += 1  # both refuse the input
                    continue
                report["failures"].append({"case": name, "format": fmt, "python": py_err[fmt],
                                           "go": go_err[fmt]})
                continue
            diff = compare(fmt, work / "py" / FILES[fmt], work / "go" / FILES[fmt])
            if not diff and py_rep.get(fmt) != go_rep.get(fmt):
                diff = f"report differs: python {py_rep.get(fmt)} go {go_rep.get(fmt)}"
            if diff:
                report["failures"].append({"case": name, "format": fmt, "diff": diff})
            else:
                report["passed"][fmt] += 1
    (CACHE / "report.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                       encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("cases", "passed")}),
          f"failures={len(report['failures'])} deviations={len(report['deviations'])}")
    for f in report["failures"][:15]:
        print(" ", json.dumps(f, ensure_ascii=False)[:400])
    kinds = sorted({(d["format"], d["python"].split(":")[0]) for d in report["deviations"]})
    for fmt, kind in kinds:
        print(f"  deviation {fmt}: python raised {kind}, go wrote the file")
    return 1 if report["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
