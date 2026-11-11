"""Quality-assurance hook called by the pipeline before export.

Interface (stable, see docs/INTERFACES.md):

    run_checks(score, context) -> dict

``score`` is a :class:`bandscribe.notation.score_model.Score`. ``context`` is a dict with:
    config        PipelineConfig
    out_dir       pathlib.Path of the run directory
    notes         dict[family, list[Note]]  final notes per instrument family
    activity      dict[stem, StemActivity]
    confidence    dict[stem, StemConfidence]
    attribution   AttributionLog (removed / promoted notes with reasons)
    manifest      dict, the run manifest built so far (models, sha256, VRAM, timings)

The returned dict must be JSON-serialisable; it is written to ``qa.json`` and embedded in the
manifest. Two optional keys are interpreted by the pipeline:
    "score"   a Score that replaces the input score for export (e.g. after removing notes)
    "block"   True to stop before export; the pipeline raises QABlocked with "reason"

This default implementation checks nothing. The rules of BANDSCRIBE_SPEC.md section 5 are
implemented by the QA owner in this package.
"""

from __future__ import annotations

from typing import Any


def run_checks(score, context: dict[str, Any]) -> dict[str, Any]:  # noqa: ARG001
    return {"status": "not-implemented", "issues": []}
