"""Guitar Pro 5 `.gp5` export: the fallback when `.gp` export fails (spec 4.2).

Written by the Go exporter (``internal/export/gp5``). GP5 only knows fretted and percussion
tracks; staff tracks (vocals, piano, other) are written on a synthetic 7-string tuning in octaves
so every pitch is kept; they are for reference only. Tracks with more than 7 strings are skipped.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from bandscribe.notation import go_export
from bandscribe.notation.score_model import Score


@dataclass
class Gp5ExportReport:
    path: Path
    notes_written: int = 0
    notes_dropped: int = 0  # chord notes that did not fit the GP5 string model
    warnings: list[str] = field(default_factory=list)


def export_gp5(score: Score, path: Path) -> Gp5ExportReport:
    rep = go_export.export(score, "gp5", Path(path))
    return Gp5ExportReport(Path(path), rep["notes_written"], rep["notes_dropped"],
                           rep["warnings"] or [])
