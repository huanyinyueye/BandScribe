"""Bars from the tracked beats, and quantisation of notes onto a per-beat grid (spec 4.1.8).

Notes are only moved in time (snapped to the grid) and deduplicated; none is created. Each
tracked beat picks its own subdivision (straight or triplet), which is how tuplets are found.
"""

from __future__ import annotations

import bisect
import math
from collections import defaultdict
from dataclasses import dataclass, replace
from fractions import Fraction

import numpy as np

from bandscribe.analysis.meter import Meter
from bandscribe.analysis.sections import Section
from bandscribe.analysis.tempo import TempoMap
from bandscribe.notation.score_model import BarSpec, ScoreNote
from bandscribe.transcription.base import Note

# Candidate subdivisions of one tracked beat and their complexity penalty (in beats per note).
GRIDS = {
    4: {4: 0.0, 3: 0.015, 6: 0.03, 8: 0.035},  # beat = quarter: 16ths, triplets, sextuplets, 32nds
    8: {2: 0.0, 3: 0.02, 4: 0.03},             # beat = eighth: 16ths, 16th triplets, 32nds
}


# A bar within this fraction of its tempo segment is written at the segment tempo.
TEMPO_MARK_TOLERANCE = 0.08


def default_grid(denominator: int) -> int:
    return 2 if denominator >= 8 else 4


def _grids_for(denominator: int) -> dict[int, float]:
    return GRIDS.get(denominator, {default_grid(denominator): 0.0})


def build_bars(tempo_map: TempoMap, meter: Meter, first_time: float, last_time: float,
               sections: list[Section] | None = None) -> list[BarSpec]:
    """Lay out bars over the tracked beats, covering [first_time, last_time].

    Beats before the first downbeat become a pickup bar with its own (shorter) time signature.
    A final bar cut short by the end of the audio is completed to the previous bar's length;
    the extra space is rests. Without any downbeat the dominant meter (4/4) starts at beat 0.
    """
    n_beats = len(tempo_map.beats)
    num, den = meter.dominant
    b_min = min(0, math.floor(float(tempo_map.time_to_beat(first_time)))) if n_beats else 0
    b_max = max(1, math.ceil(float(tempo_map.time_to_beat(last_time)) + 1e-6)) if n_beats else num

    spans: list[tuple[int, int, int, bool]] = []  # (start_beat, n_beats, denominator, confident)
    first_db = meter.bars[0].start_beat if meter.bars else 0
    first_den = meter.bars[0].denominator if meter.bars else den
    # Pickup: split from the right so only the very first bar is short.
    remaining = first_db - b_min
    pickup: list[tuple[int, int, int, bool]] = []
    end = first_db
    while remaining > 0:
        size = min(num, remaining)
        pickup.append((end - size, size, first_den, True))
        end -= size
        remaining -= size
    spans += reversed(pickup)

    for i, bar in enumerate(meter.bars):
        count = bar.numerator
        if i == len(meter.bars) - 1:
            prev = meter.bars[i - 1].numerator if i > 0 else num
            if count < prev:
                count = prev  # cut off by the end of the audio
            elif count > max(prev, num):  # downbeats stopped being detected: use the last meter
                start = bar.start_beat
                while count > prev:
                    spans.append((start, prev, bar.denominator, bar.confident))
                    start += prev
                    count -= prev
                spans.append((start, prev, bar.denominator, bar.confident))
                continue
        spans.append((bar.start_beat, count, bar.denominator, bar.confident))
    if not spans:
        spans.append((0, num, den, True))
    last_num, last_den = spans[-1][1], spans[-1][2]
    while spans[-1][0] + spans[-1][1] < b_max:
        spans.append((spans[-1][0] + spans[-1][1], last_num, last_den, True))

    bars: list[BarSpec] = []
    q = Fraction(0)
    for i, (start, count, d, confident) in enumerate(spans):
        t0 = float(tempo_map.beat_to_time(start))
        t1 = float(tempo_map.beat_to_time(start + count))
        length_q = Fraction(4 * count, d)
        bpm = 60.0 * float(length_q) / (t1 - t0) if t1 > t0 else tempo_map.global_bpm or 120.0
        # Written tempo: the tempo segment's value unless this bar really deviates from it.
        # Bar start times stay exact; this only keeps tempo marks from changing every bar.
        seg_bpm = tempo_map.bpm_at((t0 + t1) / 2) * 4 / d
        if seg_bpm > 0 and abs(bpm - seg_bpm) / seg_bpm <= TEMPO_MARK_TOLERANCE:
            bpm = seg_bpm
        bars.append(BarSpec(i, count, d, start, q, round(bpm, 2), t0, t1, None, confident))
        q += length_q
    for s in sections or []:
        i = min(range(len(bars)), key=lambda k: abs(bars[k].start_time - s.start_time))
        bars[i] = replace(bars[i], section=s.name)
    return bars


@dataclass(frozen=True)
class QuantizeStats:
    notes_in: int
    notes_out: int
    duplicates_removed: int
    mean_shift_ms: float
    max_shift_ms: float


def quantize_notes(notes: list[Note], bars: list[BarSpec], tempo_map: TempoMap, *,
                   drums: bool = False, low_confidence: float = 0.5,
                   ) -> tuple[list[ScoreNote], dict[tuple[int, int], int], QuantizeStats]:
    """Snap notes to the grid. Returns (score notes, per-beat grid, stats)."""
    if not notes:
        return [], {}, QuantizeStats(0, 0, 0, 0.0, 0.0)
    starts = [b.start_beat for b in bars]
    last = bars[-1]
    lo, hi = bars[0].start_beat, last.start_beat + last.numerator

    def locate(bpos: float) -> tuple[int, int, float]:
        bpos = min(max(bpos, lo), hi - 1e-9)
        i = max(0, bisect.bisect_right(starts, bpos) - 1)
        rel = bpos - bars[i].start_beat
        cell = min(bars[i].numerator - 1, int(math.floor(rel)))
        return i, cell, rel - cell

    on_b = np.atleast_1d(tempo_map.time_to_beat(np.array([n.onset for n in notes])))
    off_b = np.atleast_1d(tempo_map.time_to_beat(np.array([n.offset for n in notes])))
    located = [locate(float(b)) for b in on_b]

    cells: dict[tuple[int, int], list[float]] = defaultdict(list)
    for bar_i, cell, frac in located:
        cells[(bar_i, cell)].append(frac)
    grid: dict[tuple[int, int], int] = {}
    for (bar_i, cell), fracs in cells.items():
        best, best_cost = None, float("inf")
        for s, penalty in _grids_for(bars[bar_i].denominator).items():
            cost = sum(abs(f - round(f * s) / s) for f in fracs) + penalty * len(fracs)
            if cost < best_cost - 1e-12:
                best, best_cost = s, cost
        grid[(bar_i, cell)] = best

    def snap(bar_i: int, cell: int, frac: float) -> Fraction:
        """Absolute quarter position of a (bar, beat, fraction) snapped to that beat's grid."""
        bar = bars[bar_i]
        s = grid.get((bar_i, cell), default_grid(bar.denominator))
        k = round(frac * s)
        return bar.start_q + (cell + Fraction(k, s)) * bar.beat_q

    def bar_of(abs_q: Fraction) -> int:
        i = max(0, bisect.bisect_right([b.start_q for b in bars], abs_q) - 1)
        return min(i, len(bars) - 1)

    end_q = last.end_q
    out: dict[tuple, ScoreNote] = {}
    shifts = []
    duplicates = 0
    for n, (bar_i, cell, frac), ob in zip(notes, located, off_b, strict=True):
        start = snap(bar_i, cell, frac)
        bar = bars[bar_i]
        unit = bar.beat_q / grid[(bar_i, cell)]
        if start >= end_q:  # rounded past the last bar: keep it inside
            start = end_q - unit
        if drums:
            dur = unit
        else:
            stop = snap(*locate(float(ob)))
            dur = max(stop - start, unit)
        dur = min(dur, end_q - start)
        b = bar_of(start)
        note = ScoreNote(b, start - bars[b].start_q, dur, n.pitch, n.velocity, n.conf,
                         n.conf < low_confidence, n.sources, n.string, n.fret, n.flags,
                         n.onset, n.offset)
        key = (b, note.pos_q, n.string) if n.string is not None else (b, note.pos_q, n.pitch)
        if key in out:
            duplicates += 1
            if out[key].conf >= note.conf:
                continue
        out[key] = note
        grid_time = float(tempo_map.beat_to_time(
            bars[bar_i].start_beat + float((start - bars[bar_i].start_q) / bars[bar_i].beat_q)))
        shifts.append(abs(grid_time - n.onset) * 1000)
    result = sorted(out.values(), key=lambda x: (x.bar, x.pos_q, x.pitch))
    stats = QuantizeStats(len(notes), len(result), duplicates,
                          float(np.mean(shifts)) if shifts else 0.0,
                          float(np.max(shifts)) if shifts else 0.0)
    return result, grid, stats
