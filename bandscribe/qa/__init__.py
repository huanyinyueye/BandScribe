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

Returns a JSON-serialisable dict (written to ``qa.json``):
    status        pass | warn | fail   (fail = an error-level rule fired)
    rules         {rule id: {"errors": n, "warnings": n}}
    tracks        per track: notes, low_conf count and ratio, flagged notes, removed by QA
    stems         per stem: energy and activity from the separation stage
    issues        list of issues (capped at MAX_ISSUES; counts in ``rules`` are complete)
    removed       number of notes removed for breaking a hard rule
    report        path of report.html when out_dir is given
    score         (only when notes were removed) the cleaned Score for export

Rules are in :mod:`bandscribe.qa.rules`. QA never blocks the export: notes that break a hard
rule are removed and reported, so the user still gets the rest of the score.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any

from bandscribe.qa.rules import RULES, Issue

MAX_ISSUES = 500


def _activity_dict(activity) -> dict[str, Any]:
    out = {}
    for name, a in (activity or {}).items():
        out[name] = {"present": bool(a.present), "rms_db": round(float(a.rms_db), 2),
                     "relative_db": round(float(a.relative_db), 2),
                     "active_fraction": round(float(a.active_fraction), 3)}
    return out


def run_checks(score, context: dict[str, Any]) -> dict[str, Any]:
    issues: list[Issue] = []
    for rid, rule in RULES.items():
        try:
            issues += rule(score, context)
        except Exception as exc:  # noqa: BLE001 - a broken rule is reported, not fatal
            issues.append(Issue(rid, "error", f"rule crashed: {type(exc).__name__}: {exc}"))

    to_remove = {(i.track, i.onset, i.pitch) for i in issues if i.remove}
    new_score = None
    removed_by_track: Counter = Counter()
    if to_remove:
        tracks = []
        for t in score.tracks:
            keep = []
            for n in t.notes:
                if (t.id, n.onset, n.pitch) in to_remove:
                    removed_by_track[t.id] += 1
                else:
                    keep.append(n)
            tracks.append(replace(t, notes=keep))
        new_score = replace(score, tracks=tracks)

    final = new_score or score
    tracks_info = {}
    for t in final.tracks:
        n = len(t.notes)
        low = sum(1 for x in t.notes if x.low_conf)
        flags = Counter(f.split(":")[0] for x in t.notes for f in x.flags)
        tracks_info[t.id] = {"name": t.name, "family": t.family, "kind": t.kind, "notes": n,
                             "low_conf": low, "low_conf_ratio": round(low / n, 3) if n else 0.0,
                             "flags": dict(flags), "removed_by_qa": removed_by_track[t.id]}

    rules = {rid: {"errors": 0, "warnings": 0} for rid in RULES}
    for i in issues:
        bucket = rules.setdefault(i.rule, {"errors": 0, "warnings": 0})
        if i.severity == "error":
            bucket["errors"] += 1
        elif i.severity == "warning":
            bucket["warnings"] += 1
    status = ("fail" if any(r["errors"] for r in rules.values())
              else "warn" if any(r["warnings"] for r in rules.values()) else "pass")

    result: dict[str, Any] = {
        "status": status,
        "rules": rules,
        "tracks": tracks_info,
        "stems": _activity_dict(context.get("activity")),
        "removed": sum(removed_by_track.values()),
        "issues": [i.to_dict() for i in issues[:MAX_ISSUES]],
        "issues_truncated": max(0, len(issues) - MAX_ISSUES),
    }
    out_dir = context.get("out_dir")
    if out_dir is not None:
        from bandscribe.qa.report import write_report

        path = write_report(Path(out_dir) / "report.html", final, result, context)
        result["report"] = str(path)
    if new_score is not None:
        result["score"] = new_score
    return result
