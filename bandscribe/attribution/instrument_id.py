"""Instrument attribution: is a stem really that instrument, and where do notes belong?

Implements spec 4.1 step 5 and the hard rules 5.1 (other is never guitar), 5.4 (range) and
5.8 (separation disagreement lowers confidence). Nothing here creates notes.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from bandscribe.separation.base import INSTRUMENT_STEMS, Stem
from bandscribe.separation.ensemble import StemConfidence
from bandscribe.transcription.base import FAMILY_RANGE, Note, sort_notes
from bandscribe.transcription.consensus import ONSET_TOLERANCE, match_notes

ACTIVITY_WINDOW = 0.5  # seconds
ACTIVE_DB = -50.0  # window RMS above this (dBFS) counts as active
MIN_ACTIVE_SECONDS = 2.0
MIN_RELATIVE_DB = -40.0  # stem energy relative to the mix below this -> treated as empty
OTHER_PROMOTED_CONF_CAP = 0.5


@dataclass(frozen=True)
class StemActivity:
    name: str
    rms_db: float
    relative_db: float  # stem energy relative to the mix
    active_seconds: float
    active_fraction: float
    present: bool
    active: np.ndarray  # bool per ACTIVITY_WINDOW

    def is_active_at(self, t: float) -> bool:
        if len(self.active) == 0:
            return False
        return bool(self.active[min(len(self.active) - 1, max(0, int(t / ACTIVITY_WINDOW)))])


def _db(power: float) -> float:
    return float(10 * np.log10(max(power, 1e-20)))


def stem_activity(stem: Stem, mix_power: float) -> StemActivity:
    """Energy-based activity. A stem that is (almost) silent is not transcribed (spec 4.1.5)."""
    audio = stem.audio.astype(np.float64)
    power = float(np.mean(audio ** 2)) if audio.size else 0.0
    hop = max(1, int(ACTIVITY_WINDOW * stem.sr))
    mono_sq = np.mean(audio ** 2, axis=0) if audio.size else np.zeros(0)
    frames = int(np.ceil(len(mono_sq) / hop)) if len(mono_sq) else 0
    win = np.array([mono_sq[i * hop:(i + 1) * hop].mean() for i in range(frames)])
    active = win > 10 ** (ACTIVE_DB / 10) if frames else np.zeros(0, dtype=bool)
    seconds = float(active.sum() * ACTIVITY_WINDOW)
    rel = _db(power) - _db(mix_power)
    present = seconds >= MIN_ACTIVE_SECONDS and rel >= MIN_RELATIVE_DB
    return StemActivity(stem.name, _db(power), rel, seconds,
                        float(active.mean()) if frames else 0.0, present, active)


def analyze_stems(stems: dict[str, Stem], mix: np.ndarray) -> dict[str, StemActivity]:
    mix_power = float(np.mean(mix.astype(np.float64) ** 2)) if mix.size else 0.0
    return {name: stem_activity(stem, mix_power) for name, stem in stems.items()}


# --- note-level rules ----------------------------------------------------------------------

@dataclass
class AttributionLog:
    """What was removed or re-labelled and why; goes into the QA report."""

    removed: list[dict] = field(default_factory=list)
    promoted: list[dict] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def remove(self, note: Note, reason: str) -> None:
        self.removed.append({"instrument": note.instrument, "onset": round(note.onset, 3),
                             "pitch": note.pitch, "reason": reason})


def apply_range(notes: list[Note], family: str, log: AttributionLog,
                bounds: tuple[int, int] | None = None) -> list[Note]:
    """Spec 5.4: notes outside the playable range are removed and logged."""
    lo, hi = bounds or FAMILY_RANGE.get(family, (0, 127))
    out = []
    for n in notes:
        if lo <= n.pitch <= hi:
            out.append(n)
        else:
            log.remove(n, f"out of range for {family} ({lo}-{hi})")
    return out


def drop_inactive(notes: list[Note], activity: StemActivity, log: AttributionLog) -> list[Note]:
    """Remove notes whose onset falls where the stem is silent (transcriber hallucination)."""
    out = []
    for n in notes:
        if activity.is_active_at(n.onset):
            out.append(n)
        else:
            log.remove(n, "stem silent at onset")
    return out


def apply_stem_confidence(notes: list[Note], conf: StemConfidence | None) -> list[Note]:
    """Spec 5.8: where the separators disagree, every note of that stem loses confidence.

    The factor is 0.5 + 0.5 * stem confidence, so total disagreement halves the note
    confidence instead of zeroing it; such notes are flagged ``separation-disagreement``.
    """
    if conf is None:
        return notes
    out = []
    for n in notes:
        c = conf.at(n.onset)
        scaled = n.with_conf(n.conf * (0.5 + 0.5 * c))
        out.append(scaled.flagged("separation-disagreement") if c < 0.5 else scaled)
    return out


def resolve_leakage(by_stem: dict[str, list[Note]], stems: dict[str, Stem],
                    log: AttributionLog, tolerance: float = ONSET_TOLERANCE) -> dict[str, list[Note]]:
    """A note transcribed from two stems is leakage in one of them.

    The copy on the stem that is quieter around the onset is removed and logged; the other is
    kept unchanged. Drums and vocals are not compared (their "pitches" are not comparable).
    """
    names = [n for n in by_stem if n not in ("drums", "vocals")]
    drop: dict[str, set[int]] = {n: set() for n in names}

    def local_power(stem_name: str, t: float) -> float:
        stem = stems[stem_name]
        a = max(0, int((t - 0.02) * stem.sr))
        b = min(stem.audio.shape[-1], int((t + 0.15) * stem.sr))
        seg = stem.audio[..., a:b]
        return float(np.mean(seg.astype(np.float64) ** 2)) if seg.size else 0.0

    for x in range(len(names)):
        for y in range(x + 1, len(names)):
            a, b = names[x], names[y]
            for i, j in match_notes(by_stem[a], by_stem[b], tolerance):
                pa = local_power(a, by_stem[a][i].onset)
                pb = local_power(b, by_stem[b][j].onset)
                if pa >= pb:
                    drop[b].add(j)
                else:
                    drop[a].add(i)
    out = {}
    for name, notes in by_stem.items():
        kept = []
        for i, n in enumerate(notes):
            if name in drop and i in drop[name]:
                log.remove(n, "leakage: same note is stronger in another stem")
            else:
                kept.append(n)
        out[name] = kept
    return out


def vote_with_mix(notes: list[Note], family: str, mix_notes: list[Note] | None,
                  log: AttributionLog, tolerance: float = ONSET_TOLERANCE) -> list[Note]:
    """Cross-check a stem's notes against an unconditioned transcription of the full mix.

    - The mix transcription assigns the same note to this family: flag ``mix-confirmed``.
    - It assigns it to a different pitched family: likely leakage; confidence is halved and
      the note is flagged ``mix-disagrees:<family>``.
    - The mix transcription has no such note: unchanged.
    Without a mix transcription (``None``) the notes are returned unchanged.
    """
    if mix_notes is None:
        return notes
    pitched = [m for m in mix_notes if m.instrument != "drums"]
    pairs = dict(match_notes(notes, pitched, tolerance))
    out = []
    for i, n in enumerate(notes):
        if i not in pairs:
            out.append(n)
            continue
        other = pitched[pairs[i]].instrument
        if other == family:
            out.append(n.flagged("mix-confirmed"))
        elif other == "other":
            out.append(n)
        else:
            out.append(n.with_conf(n.conf * 0.5).flagged(f"mix-disagrees:{other}"))
    return out


def promote_from_other(other_notes: list[Note], mix_notes: list[Note] | None,
                       activity: dict[str, StemActivity], log: AttributionLog,
                       tolerance: float = ONSET_TOLERANCE) -> tuple[list[Note], dict[str, list[Note]]]:
    """Spec 5.1: the only way a note from the ``other`` stem can become an instrument note.

    A note transcribed from ``other`` is re-labelled to an instrument family only when BOTH:
      1. the unconditioned mix transcription contains the same note labelled as that family,
      2. the ``other`` stem is actually active at that time.
    Without a mix transcription nothing is promoted. Promoted notes are capped at confidence
    0.5 and flagged ``from-other-stem``. Returns (remaining other notes, promoted by family).
    """
    promoted: dict[str, list[Note]] = {}
    if mix_notes is None or not other_notes:
        return list(other_notes), promoted
    act = activity.get("other")
    candidates = [m for m in mix_notes if m.instrument in INSTRUMENT_STEMS
                  and m.instrument not in ("drums", "vocals")]
    pairs = dict(match_notes(other_notes, candidates, tolerance))
    remaining = []
    for i, n in enumerate(other_notes):
        if i not in pairs or act is None or not act.is_active_at(n.onset):
            remaining.append(n)
            continue
        family = candidates[pairs[i]].instrument
        moved = replace(n, instrument=family, conf=min(n.conf, OTHER_PROMOTED_CONF_CAP),
                        sources=tuple(dict.fromkeys((*n.sources, *candidates[pairs[i]].sources))))
        promoted.setdefault(family, []).append(moved.flagged("from-other-stem"))
        log.promoted.append({"to": family, "onset": round(n.onset, 3), "pitch": n.pitch})
    return remaining, {k: sort_notes(v) for k, v in promoted.items()}


def assert_no_unsupported_other_to_guitar(notes_by_family: dict[str, list[Note]]) -> None:
    """Invariant check used by the pipeline and the tests (spec 5.1).

    Every guitar note must come from the guitar stem, or be an explicitly promoted note that
    carries the ``from-other-stem`` flag together with a mix-transcription source.
    """
    for n in notes_by_family.get("guitar", []):
        if "from-other-stem" in n.flags and not any(s.startswith("muscriptor") for s in n.sources):
            raise AssertionError("guitar note promoted from the other stem without mix support")
