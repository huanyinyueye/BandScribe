"""Rule-based string/fret assignment by dynamic programming (spec 3.2, 5.5).

Notes are grouped into chords (near-simultaneous onsets). For each chord every playable
fingering is enumerated; a Viterbi pass picks the sequence that minimises
    fingering cost (hand span, high positions, open-string bonus)
  + transition cost (hand movement between consecutive chords, relaxed after a pause).
Notes that cannot be placed (below the lowest string, more notes than strings, span too wide)
are returned separately so the caller can flag them; they are never moved to another pitch.
"""

from __future__ import annotations

import itertools
from abc import ABC, abstractmethod
from dataclasses import dataclass, replace

from bandscribe.tab.tunings import MAX_FRET, Tuning
from bandscribe.transcription.base import Note, sort_notes

CHORD_WINDOW = 0.03  # onsets closer than this belong to one chord
MAX_CANDIDATES = 24  # fingerings kept per chord


@dataclass(frozen=True)
class FretOptions:
    max_span: int = 5  # frets between lowest and highest fretted note of a chord
    max_fret: int = MAX_FRET
    open_bonus: float = 0.5
    high_fret_weight: float = 0.05
    span_weight: float = 0.6
    move_weight: float = 1.0
    string_jump_weight: float = 0.15
    pause_reset: float = 1.0  # seconds of silence after which hand movement is free


@dataclass(frozen=True)
class TabResult:
    notes: list[Note]  # every input note that could be placed, with string/fret set
    unplaced: list[Note]  # notes that have no playable position, flagged ``unplayable``


class FretAssigner(ABC):
    """Pluggable backend interface (rule DP is the default; ML backends can replace it)."""

    @abstractmethod
    def assign(self, notes: list[Note], tuning: Tuning) -> TabResult: ...


Fingering = tuple[tuple[int, int], ...]  # per note: (string index 0 = highest, fret)


def _positions(pitch: int, tuning: Tuning, max_fret: int) -> list[tuple[int, int]]:
    return [(s, pitch - open_pitch) for s, open_pitch in enumerate(tuning.strings)
            if 0 <= pitch - open_pitch <= max_fret]


def _fingering_cost(f: Fingering, opt: FretOptions) -> float:
    fretted = [fret for _, fret in f if fret > 0]
    span = (max(fretted) - min(fretted)) if fretted else 0
    cost = opt.span_weight * span + opt.high_fret_weight * sum(fretted)
    cost -= opt.open_bonus * sum(1 for _, fret in f if fret == 0)
    return cost


def _hand(f: Fingering) -> float | None:
    fretted = [fret for _, fret in f if fret > 0]
    return min(fretted) if fretted else None


def chord_fingerings(pitches: list[int], tuning: Tuning, opt: FretOptions) -> list[Fingering]:
    """All fingerings for a chord: distinct strings, fretted span <= max_span."""
    options = [_positions(p, tuning, opt.max_fret) for p in pitches]
    if any(not o for o in options) or len(pitches) > tuning.n_strings:
        return []
    out = []
    for combo in itertools.product(*options):
        if len({s for s, _ in combo}) != len(combo):
            continue
        fretted = [fret for _, fret in combo if fret > 0]
        if fretted and max(fretted) - min(fretted) > opt.max_span:
            continue
        out.append(tuple(combo))
    out.sort(key=lambda f: _fingering_cost(f, opt))
    return out[:MAX_CANDIDATES]


def group_chords(notes: list[Note], window: float = CHORD_WINDOW) -> list[list[Note]]:
    groups: list[list[Note]] = []
    for n in sort_notes(notes):
        if groups and n.onset - groups[-1][0].onset <= window:
            groups[-1].append(n)
        else:
            groups.append([n])
    return groups


def _reduce_chord(chord: list[Note], tuning: Tuning,
                  opt: FretOptions) -> tuple[list[Note], list[Fingering], list[Note]]:
    """Drop the least confident notes until the chord is playable. Returns kept, fingerings, dropped."""
    kept = sorted(chord, key=lambda n: n.pitch)
    dropped: list[Note] = []
    # Duplicate pitches inside one chord cannot both be placed meaningfully: keep the best.
    seen: dict[int, Note] = {}
    for n in kept:
        if n.pitch not in seen or n.conf > seen[n.pitch].conf:
            if n.pitch in seen:
                dropped.append(seen[n.pitch])
            seen[n.pitch] = n
        else:
            dropped.append(n)
    kept = sorted(seen.values(), key=lambda n: n.pitch)
    while kept:
        f = chord_fingerings([n.pitch for n in kept], tuning, opt)
        if f:
            return kept, f, dropped
        # Remove notes with no position at all first, else the least confident one.
        impossible = [n for n in kept if not _positions(n.pitch, tuning, opt.max_fret)]
        victim = impossible[0] if impossible else min(kept, key=lambda n: (n.conf, -n.pitch))
        kept.remove(victim)
        dropped.append(victim)
    return [], [], dropped


class DPFretAssigner(FretAssigner):
    def __init__(self, options: FretOptions | None = None) -> None:
        self.options = options or FretOptions()

    def _transition(self, a: Fingering, b: Fingering, gap: float) -> float:
        opt = self.options
        ha, hb = _hand(a), _hand(b)
        cost = 0.0
        if ha is not None and hb is not None and gap < opt.pause_reset:
            cost += opt.move_weight * abs(ha - hb)
        sa = sum(s for s, _ in a) / len(a)
        sb = sum(s for s, _ in b) / len(b)
        return cost + opt.string_jump_weight * abs(sa - sb)

    def assign(self, notes: list[Note], tuning: Tuning) -> TabResult:
        opt = self.options
        chords, cands, unplaced = [], [], []
        for chord in group_chords(notes):
            kept, fingerings, dropped = _reduce_chord(chord, tuning, opt)
            unplaced += [n.flagged("unplayable") for n in dropped]
            if kept:
                chords.append(kept)
                cands.append(fingerings)
        if not chords:
            return TabResult([], sort_notes(unplaced))

        # Viterbi over chords.
        cost = [_fingering_cost(f, opt) for f in cands[0]]
        back: list[list[int]] = []
        for k in range(1, len(chords)):
            gap = chords[k][0].onset - max(n.offset for n in chords[k - 1])
            new_cost, ptr = [], []
            for f in cands[k]:
                best_j, best = 0, float("inf")
                for j, prev in enumerate(cands[k - 1]):
                    c = cost[j] + self._transition(prev, f, gap)
                    if c < best:
                        best_j, best = j, c
                new_cost.append(best + _fingering_cost(f, opt))
                ptr.append(best_j)
            cost = new_cost
            back.append(ptr)
        idx = min(range(len(cost)), key=cost.__getitem__)
        chosen = [idx]
        for ptr in reversed(back):
            idx = ptr[idx]
            chosen.append(idx)
        chosen.reverse()

        placed = []
        for chord, fingerings, c in zip(chords, cands, chosen, strict=True):
            for n, (s, fret) in zip(chord, fingerings[c], strict=True):
                placed.append(replace(n, string=s + 1, fret=fret))
        return TabResult(sort_notes(placed), sort_notes(unplaced))
