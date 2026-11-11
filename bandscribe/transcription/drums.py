"""Drum hit voting and mapping to the General MIDI kit (spec 3.3, 5.7)."""

from __future__ import annotations

from dataclasses import replace

from bandscribe.transcription.base import Note, sort_notes
from bandscribe.transcription.consensus import (
    MATCHED_BASE,
    PRIMARY_ONLY,
    SECONDARY_ONLY,
    SINGLE_BACKEND,
    ConsensusStats,
)

ONSET_TOLERANCE = 0.03

# General MIDI percussion key -> (kit piece written to the score, coarse voting class, limb).
GM_DRUMS: dict[int, tuple[str, str, str]] = {
    35: ("kick", "kick", "foot"), 36: ("kick", "kick", "foot"),
    37: ("snare", "snare", "hand"), 38: ("snare", "snare", "hand"), 40: ("snare", "snare", "hand"),
    42: ("hihat_closed", "hihat", "hand"), 44: ("hihat_pedal", "hihat", "foot"),
    46: ("hihat_open", "hihat", "hand"),
    41: ("tom_low", "tom", "hand"), 43: ("tom_low", "tom", "hand"),
    45: ("tom_mid", "tom", "hand"), 47: ("tom_mid", "tom", "hand"),
    48: ("tom_high", "tom", "hand"), 50: ("tom_high", "tom", "hand"),
    49: ("crash", "cymbal", "hand"), 57: ("crash", "cymbal", "hand"),
    51: ("ride", "cymbal", "hand"), 53: ("ride", "cymbal", "hand"), 59: ("ride", "cymbal", "hand"),
    52: ("china", "cymbal", "hand"), 55: ("splash", "cymbal", "hand"),
}
# Canonical GM key written for each kit piece.
PIECE_KEY = {"kick": 36, "snare": 38, "hihat_closed": 42, "hihat_pedal": 44, "hihat_open": 46,
             "tom_low": 43, "tom_mid": 47, "tom_high": 50, "crash": 49, "ride": 51,
             "china": 52, "splash": 55}
MAX_LIMBS = 4


def coarse_class(pitch: int) -> str | None:
    entry = GM_DRUMS.get(pitch)
    return entry[1] if entry else None


def canonical(note: Note) -> Note | None:
    """Map a drum note to the canonical key of its kit piece; None if it is not a kit sound."""
    entry = GM_DRUMS.get(note.pitch)
    if entry is None:
        return None
    return replace(note, pitch=PIECE_KEY[entry[0]], instrument="drums")


def _match_by_class(a: list[Note], b: list[Note], tolerance: float) -> list[tuple[int, int]]:
    cands = []
    for i, x in enumerate(a):
        cx = coarse_class(x.pitch)
        for j, y in enumerate(b):
            d = abs(x.onset - y.onset)
            if d <= tolerance and cx == coarse_class(y.pitch):
                cands.append((d, i, j))
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs = []
    for _d, i, j in sorted(cands):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j))
    return pairs


def vote_drums(detailed: list[Note] | None, coarse: list[Note] | None, *,
               tolerance: float = ONSET_TOLERANCE,
               drop_unconfirmed: bool = False) -> tuple[list[Note], ConsensusStats, int]:
    """Vote a fine-grained drum transcription (MuScriptor) against a coarse one (ADTOF).

    Hits agree when they fall within ``tolerance`` and share a coarse class (kick, snare, tom,
    hi-hat, cymbal). The kit piece of an agreed hit comes from the detailed model, because the
    coarse model cannot tell open from closed hi-hat or crash from ride. With only one model
    available its hits are kept, flagged ``no-second-opinion``.
    Returns (notes, stats, number of hits discarded because they are not kit sounds).
    """
    discarded = 0

    def clean(notes: list[Note] | None) -> list[Note] | None:
        nonlocal discarded
        if notes is None:
            return None
        out = []
        for n in notes:
            c = canonical(n)
            if c is None:
                discarded += 1
            else:
                out.append(c)
        return out

    a, b = clean(detailed), clean(coarse)
    if a is None and b is None:
        return [], ConsensusStats(0, 0, 0), discarded
    if a is None or b is None:
        only = a if a is not None else b
        out = [n.with_conf(n.conf * SINGLE_BACKEND).flagged("no-second-opinion") for n in only]
        return sort_notes(out), ConsensusStats(0, len(only), 0), discarded
    pairs = _match_by_class(a, b, tolerance)
    ma, mb = {i for i, _ in pairs}, {j for _, j in pairs}
    out = []
    for i, j in pairs:
        conf = MATCHED_BASE + (1 - MATCHED_BASE) * (a[i].conf + b[j].conf) / 2
        out.append(replace(a[i], conf=min(1.0, conf), velocity=b[j].velocity,
                           sources=tuple(dict.fromkeys((*a[i].sources, *b[j].sources)))))
    if not drop_unconfirmed:
        out += [n.with_conf(n.conf * PRIMARY_ONLY).flagged("single-source")
                for i, n in enumerate(a) if i not in ma]
        out += [n.with_conf(n.conf * SECONDARY_ONLY).flagged("single-source")
                for j, n in enumerate(b) if j not in mb]
    return sort_notes(out), ConsensusStats(len(pairs), len(a) - len(ma), len(b) - len(mb)), discarded


def flag_limb_overload(notes: list[Note], window: float = 0.02) -> list[Note]:
    """Spec 5.7: more than four simultaneous hits, or more than two foot-played hits, is flagged."""
    notes = sort_notes(notes)
    out = list(notes)
    i = 0
    while i < len(notes):
        j = i
        while j + 1 < len(notes) and notes[j + 1].onset - notes[i].onset <= window:
            j += 1
        group = range(i, j + 1)
        feet = sum(1 for k in group if GM_DRUMS[notes[k].pitch][2] == "foot")
        hands = len(group) - feet
        if len(group) > MAX_LIMBS or feet > 2 or hands > 2:
            for k in group:
                out[k] = out[k].flagged("limb-overload")
        i = j + 1
    return out
