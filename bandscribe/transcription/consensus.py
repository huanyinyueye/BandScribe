"""Align and vote the notes of several transcribers (spec 3.1, 5.3).

No note is ever created here: every output note is one of the input notes, possibly with a
changed confidence and merged sources. Offsets are never extended.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from bandscribe.transcription.base import Note, sort_notes

ONSET_TOLERANCE = 0.05  # seconds
MATCHED_BASE = 0.6  # agreed notes land in [0.6, 1.0]
PRIMARY_ONLY = 0.45  # notes only the primary found land in [0, 0.45]
SECONDARY_ONLY = 0.35  # notes only the secondary found land in [0, 0.35]
SINGLE_BACKEND = 0.8  # no second opinion at all: backend confidence scaled by this


@dataclass(frozen=True)
class ConsensusStats:
    matched: int
    primary_only: int
    secondary_only: int

    @property
    def agreement(self) -> float:
        total = self.matched + self.primary_only + self.secondary_only
        return self.matched / total if total else 1.0


def match_notes(a: list[Note], b: list[Note], tolerance: float = ONSET_TOLERANCE,
                same_pitch: bool = True) -> list[tuple[int, int]]:
    """Greedy one-to-one matching of notes by pitch and onset proximity.

    Returns index pairs (i in a, j in b). Closest onsets are paired first.
    """
    cands = []
    b_sorted = sorted(range(len(b)), key=lambda j: b[j].onset)
    start = 0
    for i in sorted(range(len(a)), key=lambda i: a[i].onset):
        while start < len(b_sorted) and b[b_sorted[start]].onset < a[i].onset - tolerance:
            start += 1
        k = start
        while k < len(b_sorted) and b[b_sorted[k]].onset <= a[i].onset + tolerance:
            j = b_sorted[k]
            if not same_pitch or a[i].pitch == b[j].pitch:
                cands.append((abs(a[i].onset - b[j].onset), i, j))
            k += 1
    used_a: set[int] = set()
    used_b: set[int] = set()
    pairs = []
    for _d, i, j in sorted(cands):
        if i not in used_a and j not in used_b:
            used_a.add(i)
            used_b.add(j)
            pairs.append((i, j))
    return sorted(pairs)


def consensus(primary: list[Note], secondary: list[Note] | None, *,
              tolerance: float = ONSET_TOLERANCE,
              drop_unconfirmed: bool = False) -> tuple[list[Note], ConsensusStats]:
    """Vote two transcriptions of the same stem.

    - Both found the note: the primary's timing and pitch are kept, sources are merged and
      conf = 0.6 + 0.4 * mean(conf_a, conf_b).
    - Only one found it: kept with a low confidence (below the default 0.5 threshold) and the
      flag ``single-source``; removed when ``drop_unconfirmed`` is set.
    - ``secondary is None`` means there was no second backend: notes keep their own
      confidence scaled by 0.8 and carry the flag ``no-second-opinion``.
    """
    if secondary is None:
        out = [n.with_conf(n.conf * SINGLE_BACKEND).flagged("no-second-opinion") for n in primary]
        return sort_notes(out), ConsensusStats(0, len(primary), 0)
    pairs = match_notes(primary, secondary, tolerance)
    matched_a = {i for i, _ in pairs}
    matched_b = {j for _, j in pairs}
    out: list[Note] = []
    for i, j in pairs:
        a, b = primary[i], secondary[j]
        conf = MATCHED_BASE + (1 - MATCHED_BASE) * (a.conf + b.conf) / 2
        out.append(replace(a, conf=min(1.0, conf),
                           sources=tuple(dict.fromkeys((*a.sources, *b.sources))),
                           velocity=max(a.velocity, b.velocity) if a.velocity == 80 else a.velocity))
    if not drop_unconfirmed:
        out += [n.with_conf(n.conf * PRIMARY_ONLY).flagged("single-source")
                for i, n in enumerate(primary) if i not in matched_a]
        out += [n.with_conf(n.conf * SECONDARY_ONLY).flagged("single-source")
                for j, n in enumerate(secondary) if j not in matched_b]
    stats = ConsensusStats(len(pairs), len(primary) - len(matched_a),
                           len(secondary) - len(matched_b))
    return sort_notes(out), stats
