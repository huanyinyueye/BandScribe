"""QA rules of BANDSCRIBE_SPEC.md section 5.

Each rule takes the score and the pipeline context and returns a list of ``Issue``. Rules
never add, move or re-pitch notes. The only change any rule may ask for is the removal of a
note that breaks a hard rule (severity "error"); everything else is reported.

Rule ids:
  R1 other-to-guitar      a guitar note that came from the `other` stem without the
                          mix-confirmed promotion, or with confidence above the 0.5 cap
  R2 untraceable-note     a score note with no matching final transcription note
                          (= created after transcription)
  R3 confidence           conf outside [0, 1], missing sources, low_conf flag inconsistent
                          with the threshold, low-confidence notes left in strict mode
  R4 range                tab string / fret arithmetic wrong or out of range; staff notes
                          outside the family range
  R5 playability          more simultaneous notes than strings, two notes on one string,
                          fret span wider than max_span
  R6 bass-polyphony       overlapping bass notes (bass is treated as monophonic)
  R7 limb-overflow        more than 4 simultaneous drum events
  R8 separation-conf      high-confidence notes where the separation models disagreed
  R9 provenance           a model in the manifest without version or weight hash
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from bandscribe.transcription.base import FAMILY_RANGE

TAB_MAX_FRET = 24
SIMULTANEOUS = 0.03  # seconds: onsets closer than this sound together
LIMB_LIMIT = 4
OTHER_PROMOTED_CAP = 0.5
SEPARATION_LOW = 0.5  # stem confidence below this counts as disagreement
HIGH_CONF = 0.6


@dataclass
class Issue:
    rule: str
    severity: str  # error | warning | info
    message: str
    track: str | None = None
    bar: int | None = None
    onset: float | None = None
    pitch: int | None = None
    remove: bool = False  # the note breaks a hard rule and must not be exported
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = {"rule": self.rule, "severity": self.severity, "message": self.message}
        for k in ("track", "bar", "onset", "pitch"):
            v = getattr(self, k)
            if v is not None:
                d[k] = round(v, 3) if isinstance(v, float) else v
        if self.remove:
            d["removed"] = True
        if self.extra:
            d.update(self.extra)
        return d


def _cfg(context: dict, name: str, default):
    cfg = context.get("config")
    return getattr(cfg, name, default) if cfg is not None else default


def _note_issue(rule, severity, msg, track, n, remove=False, **extra) -> Issue:
    return Issue(rule, severity, msg, track.id, n.bar, n.onset, n.pitch, remove, extra)


def _groups(notes, window=SIMULTANEOUS):
    """Group notes whose onsets lie within ``window`` of the group's first onset."""
    out, cur = [], []
    for n in sorted(notes, key=lambda n: n.onset):
        if cur and n.onset - cur[0].onset > window:
            out.append(cur)
            cur = []
        cur.append(n)
    if cur:
        out.append(cur)
    return out


# --- R1 -------------------------------------------------------------------------------------

def rule_other_to_guitar(score, context) -> list[Issue]:
    """A guitar note flagged ``from-other-stem`` must be one that the attribution stage
    promoted to guitar (logged in ``attribution.promoted``, which only happens with a matching
    mix-transcription note while the other stem is active)."""
    log = context.get("attribution")
    promoted = {(round(p["onset"], 3), p["pitch"]) for p in getattr(log, "promoted", [])
                if p.get("to") == "guitar"}
    issues = []
    for t in score.tracks:
        if t.family != "guitar":
            continue
        for n in t.notes:
            if "from-other-stem" not in n.flags:
                continue
            if (round(n.onset, 3), n.pitch) not in promoted:
                issues.append(_note_issue("R1", "error", "guitar note from the other stem that "
                                          "the mix-confirmed promotion did not produce", t, n,
                                          remove=True))
            elif n.conf > OTHER_PROMOTED_CAP + 1e-9:
                issues.append(_note_issue("R1", "error", f"note promoted from the other stem has "
                                          f"conf {n.conf:.2f} > {OTHER_PROMOTED_CAP}", t, n,
                                          remove=True))
    return issues


# --- R2 -------------------------------------------------------------------------------------

def rule_traceable(score, context, tolerance: float = 0.002) -> list[Issue]:
    finals = context.get("notes")
    if not finals:
        return [Issue("R2", "warning", "final notes not available; traceability not checked")]
    index: dict[tuple[str, int], list[float]] = defaultdict(list)
    for family, notes in finals.items():
        for n in notes:
            index[(family, n.pitch)].append(n.onset)
    for v in index.values():
        v.sort()
    used: dict[tuple[str, int], int] = defaultdict(int)
    issues = []
    for t in score.tracks:
        for n in t.notes:
            onsets = index.get((t.family, n.pitch), [])
            if not any(abs(o - n.onset) <= tolerance for o in onsets):
                issues.append(_note_issue("R2", "error", "score note has no source transcription "
                                          "note (created after transcription)", t, n, remove=True))
            else:
                used[(t.family, n.pitch)] += 1
    for key, count in used.items():
        if count > len(index[key]):
            issues.append(Issue("R2", "error", f"{count} score notes but only {len(index[key])} "
                                "transcribed notes", key[0], pitch=key[1]))
    return issues


# --- R3 -------------------------------------------------------------------------------------

def rule_confidence(score, context) -> list[Issue]:
    thr = _cfg(context, "low_confidence", 0.5)
    strict = _cfg(context, "strict", False)
    issues = []
    for t in score.tracks:
        for n in t.notes:
            if not 0.0 <= n.conf <= 1.0:
                issues.append(_note_issue("R3", "error", f"conf {n.conf} outside [0, 1]", t, n))
            if not n.sources:
                issues.append(_note_issue("R3", "error", "note has no sources", t, n))
            if n.conf < thr and not n.low_conf:
                issues.append(_note_issue("R3", "warning", "below threshold but not marked "
                                          "low_conf", t, n))
            if n.low_conf and n.conf >= thr:
                issues.append(_note_issue("R3", "warning", "marked low_conf above threshold", t, n))
            if strict and n.conf < thr:
                issues.append(_note_issue("R3", "error", "low-confidence note left in strict "
                                          "mode", t, n, remove=True))
    return issues


# --- R4 -------------------------------------------------------------------------------------

def rule_range(score, context) -> list[Issue]:
    issues = []
    for t in score.tracks:
        if t.kind == "tab":
            for n in t.notes:
                if n.string is None or n.fret is None:
                    issues.append(_note_issue("R4", "error", "tab note without string / fret", t,
                                              n))
                    continue
                if not 1 <= n.string <= len(t.tuning):
                    issues.append(_note_issue("R4", "error", f"string {n.string} not on a "
                                              f"{len(t.tuning)}-string instrument", t, n))
                    continue
                if not 0 <= n.fret <= TAB_MAX_FRET:
                    issues.append(_note_issue("R4", "error", f"fret {n.fret} out of range", t, n))
                if t.tuning[n.string - 1] + n.fret != n.pitch:
                    issues.append(_note_issue("R4", "error", "string + fret does not give the "
                                              "written pitch", t, n))
        elif t.kind == "staff" and t.family in FAMILY_RANGE:
            lo, hi = FAMILY_RANGE[t.family]
            for n in t.notes:
                if not lo <= n.pitch <= hi:
                    issues.append(_note_issue("R4", "warning", f"pitch outside {t.family} range "
                                              f"{lo}-{hi}", t, n))
    return issues


# --- R5 -------------------------------------------------------------------------------------

def rule_playability(score, context) -> list[Issue]:
    max_span = _cfg(context, "max_span", 5)
    issues = []
    for t in score.tracks:
        if t.kind != "tab":
            continue
        for g in _groups([n for n in t.notes if n.string is not None]):
            first = g[0]
            if len(g) > len(t.tuning):
                issues.append(_note_issue("R5", "warning", f"{len(g)} notes at once on "
                                          f"{len(t.tuning)} strings", t, first))
            strings = [n.string for n in g]
            if len(set(strings)) < len(strings):
                issues.append(_note_issue("R5", "warning", "two notes on one string at once", t,
                                          first))
            fretted = [n.fret for n in g if n.fret]
            if fretted and max(fretted) - min(fretted) > max_span:
                issues.append(_note_issue("R5", "warning", f"fret span "
                                          f"{max(fretted) - min(fretted)} > {max_span}", t, first))
    return issues


# --- R6 -------------------------------------------------------------------------------------

def rule_bass_polyphony(score, context) -> list[Issue]:
    issues = []
    for t in score.tracks:
        if t.family != "bass":
            continue
        for g in _groups(t.notes):
            if len({n.pitch for n in g}) > 1:
                issues.append(_note_issue("R6", "warning", f"{len(g)} simultaneous bass notes",
                                          t, g[0], pitches=sorted(n.pitch for n in g)))
    return issues


# --- R7 -------------------------------------------------------------------------------------

def rule_limbs(score, context) -> list[Issue]:
    issues = []
    for t in score.tracks:
        if t.kind != "drums":
            continue
        for g in _groups(t.notes):
            pieces = {n.pitch for n in g}
            if len(pieces) > LIMB_LIMIT:
                issues.append(_note_issue("R7", "warning", f"{len(pieces)} drum pieces at once "
                                          f"(> {LIMB_LIMIT} limbs)", t, g[0],
                                          pieces=sorted(pieces)))
    return issues


# --- R8 -------------------------------------------------------------------------------------

def rule_separation_confidence(score, context) -> list[Issue]:
    conf_by_stem = context.get("confidence") or {}
    issues = []
    for t in score.tracks:
        sc = conf_by_stem.get(t.family)
        if sc is None:
            continue
        for n in t.notes:
            if sc.at(n.onset) < SEPARATION_LOW and n.conf >= HIGH_CONF:
                issues.append(_note_issue("R8", "warning", "high-confidence note where the "
                                          "separation models disagree", t, n,
                                          stem_conf=round(sc.at(n.onset), 3)))
    return issues


# --- R9 -------------------------------------------------------------------------------------

def rule_provenance(score, context) -> list[Issue]:
    manifest = context.get("manifest") or {}
    models = manifest.get("models") or {}
    issues = []
    if not models:
        issues.append(Issue("R9", "warning", "manifest lists no models"))
    for mid, m in models.items():
        if not m.get("version"):
            issues.append(Issue("R9", "warning", f"model {mid} has no version", extra={"model": mid}))
        if not (m.get("sha256") or m.get("files_sha256")):
            issues.append(Issue("R9", "warning", f"model {mid} has no weight hash",
                                extra={"model": mid}))
    return issues


RULES = {
    "R1": rule_other_to_guitar,
    "R2": rule_traceable,
    "R3": rule_confidence,
    "R4": rule_range,
    "R5": rule_playability,
    "R6": rule_bass_polyphony,
    "R7": rule_limbs,
    "R8": rule_separation_confidence,
    "R9": rule_provenance,
}
