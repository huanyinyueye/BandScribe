"""MusicXML 4.0 export (score-partwise), written by the Go exporter from the same layout as the
`.gp` export.

One staff per part. Fretted tracks carry string / fret as <technical> elements; a separate
TAB staff is not written (Guitar Pro `.gp` is the format that carries the tablature).
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from bandscribe.notation import go_export
from bandscribe.notation.score_model import Score

DIVISIONS = 960


def export_musicxml(score: Score, path: Path) -> Path:
    go_export.export(score, "musicxml", Path(path))
    return Path(path)


def measure_durations(path: Path) -> list[list[Fraction]]:
    """Per part, the summed note duration of each measure in quarter notes (for tests)."""
    root = ET.parse(path).getroot()
    out = []
    for part in root.findall("part"):
        rows = []
        for measure in part.findall("measure"):
            total = 0
            for note in measure.findall("note"):
                if note.find("chord") is None:
                    total += int(note.findtext("duration"))
            rows.append(Fraction(total, DIVISIONS))
        out.append(rows)
    return out
