"""Guitar Pro 8 `.gp` export through alphaTab's Gp7Exporter (spec 12.1), with read-back check.

Python builds the engraving layout; the Node script ``exporter/export_gp.mjs`` turns it into
alphaTab's model and writes the file, then re-reads the bytes and reports what it found.
:func:`export_gp` compares that report with the layout: bar count, time signatures, and for
every track the sequence of (bar, pitch, string, fret, tie).
"""

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from bandscribe.notation.layout import layout_to_json
from bandscribe.notation.score_model import Score

EXPORTER_DIR = Path(__file__).resolve().parent.parent.parent / "exporter"
SCRIPT = EXPORTER_DIR / "export_gp.mjs"
NOTICE = "Automatic transcription by BandScribe. '?' marks low-confidence notes. " \
         "Non-commercial use only."


class GpExportError(RuntimeError):
    pass


@dataclass
class GpExportReport:
    path: Path
    bytes: int
    alphatab_version: str
    verified: bool
    mismatches: list[str] = field(default_factory=list)
    notes_written: int = 0
    summary: dict = field(default_factory=dict)


def _node() -> str:
    exe = shutil.which("node")
    if not exe:
        raise GpExportError("Node.js not found; run `bandscribe doctor`")
    if not (EXPORTER_DIR / "node_modules" / "@coderline" / "alphatab").is_dir():
        raise GpExportError(f"alphaTab is not installed; run `npm install` in {EXPORTER_DIR}")
    return exe


def build_payload(score: Score) -> dict:
    """The JSON handed to the Node exporter."""
    tracks = []
    for t in score.tracks:
        tracks.append({
            "name": t.name, "kind": t.kind, "program": t.midi_program,
            "tuning": list(t.tuning), "tuning_name": t.info.get("tuning_name", ""),
            "clef": "F4" if t.family == "bass" and t.kind != "tab" else None,
            "bars": layout_to_json(t, score.bars)})
    key = score.key or {}
    return {
        "title": score.title, "artist": score.meta.get("artist", ""), "notice": NOTICE,
        "key_fifths": key.get("fifths"), "key_minor": key.get("mode") == "minor",
        "bars": [{"num": b.numerator, "den": b.denominator, "bpm": b.bpm, "section": b.section}
                 for b in score.bars],
        "tracks": tracks}


def _expected_notes(track_payload: dict) -> list[list]:
    out = []
    stringed = track_payload["kind"] == "tab"
    for bi, beats in enumerate(track_payload["bars"]):
        for b in beats:
            for n in b["notes"]:
                out.append([bi, n["p"], n["s"] if stringed else None,
                            n["f"] if stringed else None, 1 if b["tie"] else 0])
    return out


def verify(payload: dict, summary: dict) -> list[str]:
    """Differences between what was sent and what alphaTab read back (empty = identical)."""
    issues = []
    sent = [(b["num"], b["den"]) for b in payload["bars"]]
    got = [(b["num"], b["den"]) for b in summary["masterBars"]]
    if sent != got:
        issues.append(f"time signatures differ: sent {len(sent)} bars, read {len(got)}")
    sections_sent = [b["section"] for b in payload["bars"]]
    sections_got = [b["section"] for b in summary["masterBars"]]
    if sections_sent != sections_got:
        issues.append("section markers differ")
    if len(payload["tracks"]) != len(summary["tracks"]):
        issues.append(f"track count differs: {len(payload['tracks'])} vs {len(summary['tracks'])}")
        return issues
    for tp, ts in zip(payload["tracks"], summary["tracks"], strict=True):
        name = tp["name"]
        if ts["bars"] != len(tp["bars"]):
            issues.append(f"{name}: {ts['bars']} bars read, {len(tp['bars'])} sent")
        if tp["kind"] == "tab" and ts["tuning"] != tp["tuning"]:
            issues.append(f"{name}: tuning differs: {ts['tuning']} vs {tp['tuning']}")
        if (tp["kind"] == "drums") != bool(ts["percussion"]):
            issues.append(f"{name}: percussion flag differs")
        exp = sorted(_expected_notes(tp), key=lambda x: (x[0], x[1], x[2] or 0))
        got_notes = ts["notes"]
        if tp["kind"] != "tab":  # string/fret are meaningless on non-tab staves
            got_notes = [[n[0], n[1], None, None, n[4]] for n in got_notes]
        # Strip before sorting, and sort on every field: otherwise two same-pitch notes in a bar
        # (one tied, one not) can come out in a different order and look like a mismatch.
        exp = sorted(exp, key=lambda x: (x[0], x[1], x[2] or 0, x[3] or 0, x[4]))
        got_notes = sorted(got_notes, key=lambda x: (x[0], x[1], x[2] or 0, x[3] or 0, x[4]))
        if exp != got_notes:
            diff = next((i for i, (a, b) in enumerate(zip(exp, got_notes, strict=False))
                         if a != b), min(len(exp), len(got_notes)))
            issues.append(f"{name}: notes differ ({len(exp)} sent, {len(got_notes)} read; "
                          f"first difference at index {diff})")
    return issues


def export_gp(score: Score, path: Path) -> GpExportReport:
    payload = build_payload(score)
    # Absolute: the exporter runs with cwd=EXPORTER_DIR, so a relative path would land there.
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "score.json"
        src.write_text(json.dumps(payload), encoding="utf-8")
        proc = subprocess.run([_node(), str(SCRIPT), str(src), str(path)], cwd=EXPORTER_DIR,
                              check=False, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0 or not path.is_file():
        raise GpExportError(proc.stderr.strip()[-1500:] or "exporter produced no file")
    summary = json.loads(proc.stdout)
    issues = verify(payload, summary)
    return GpExportReport(path, summary["bytes"], summary.get("alphatab", "?"), not issues,
                          issues, sum(len(t["notes"]) for t in summary["tracks"]), summary)


def read_gp(path: Path) -> dict:
    """Summary of an existing `.gp` file as alphaTab reads it."""
    proc = subprocess.run([_node(), str(SCRIPT), "--read", str(Path(path).resolve())],
                          cwd=EXPORTER_DIR,
                          check=False, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise GpExportError(proc.stderr.strip()[-1500:])
    return json.loads(proc.stdout)
