"""MIDI export (Go exporter): one track per instrument, tempo per bar, time signatures and
section markers."""

from __future__ import annotations

from pathlib import Path

from bandscribe.notation import go_export
from bandscribe.notation.score_model import Score


def export_midi(score: Score, path: Path) -> Path:
    go_export.export(score, "midi", Path(path))
    return Path(path)
