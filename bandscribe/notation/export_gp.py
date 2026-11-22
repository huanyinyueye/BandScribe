"""Guitar Pro 8 `.gp` export with read-back check (spec 12.1), written by the Go exporter.

``internal/export/gp`` is a translation of alphaTab 1.9.0's Gp7Exporter (MPL-2.0): it writes the
file, reads it back the way alphaTab does and compares bar count, time signatures, and for every
track the sequence of (bar, pitch, string, fret, tie). Node and alphaTab are not needed at run
time; :func:`read_gp` still uses them to read files in tests and benchmarks.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from bandscribe.notation import go_export
from bandscribe.notation.score_model import Score

EXPORTER_DIR = Path(__file__).resolve().parent.parent.parent / "exporter"
SCRIPT = EXPORTER_DIR / "export_gp.mjs"


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


def export_gp(score: Score, path: Path) -> GpExportReport:
    path = Path(path).resolve()
    try:
        rep = go_export.export(score, "gp", path)
    except go_export.ExportError as exc:
        raise GpExportError(str(exc)) from exc
    return GpExportReport(path, rep["bytes"], rep["alphatab"], rep["verified"],
                          rep["mismatches"] or [], rep["notes_written"])


def _node() -> str:
    exe = shutil.which("node")
    if not exe:
        raise GpExportError("Node.js not found (needed only to read .gp files back in tests)")
    if not (EXPORTER_DIR / "node_modules" / "@coderline" / "alphatab").is_dir():
        raise GpExportError(f"alphaTab is not installed; run `npm install` in {EXPORTER_DIR}")
    return exe


def read_gp(path: Path) -> dict:
    """Summary of an existing `.gp` file as alphaTab reads it (tests and benchmarks)."""
    proc = subprocess.run([_node(), str(SCRIPT), "--read", str(Path(path).resolve())],
                          cwd=EXPORTER_DIR,
                          check=False, capture_output=True, text=True, encoding="utf-8")
    if proc.returncode != 0:
        raise GpExportError(proc.stderr.strip()[-1500:])
    return json.loads(proc.stdout)
