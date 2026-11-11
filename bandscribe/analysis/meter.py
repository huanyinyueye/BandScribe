"""Bars and time signatures from beats and downbeats, including odd and changing meters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FAST_BEAT_BPM = 200.0  # a tracked pulse faster than this is notated as eighth notes


@dataclass(frozen=True)
class Bar:
    index: int
    start_time: float
    end_time: float
    start_beat: int  # index of the first beat of the bar in the beat array
    numerator: int
    denominator: int
    confident: bool  # False for a lone bar that disagrees with both (agreeing) neighbours


@dataclass(frozen=True)
class MeterChange:
    bar_index: int
    numerator: int
    denominator: int


@dataclass(frozen=True)
class Meter:
    bars: tuple[Bar, ...]
    changes: tuple[MeterChange, ...]
    pickup_beats: int  # beats before the first downbeat

    @property
    def dominant(self) -> tuple[int, int]:
        if not self.bars:
            return (4, 4)
        sigs = [(b.numerator, b.denominator) for b in self.bars]
        return max(sorted(set(sigs)), key=sigs.count)


def _denominator(bar_beats: np.ndarray) -> int:
    if len(bar_beats) < 2:
        return 4
    bpm = 60.0 / float(np.median(np.diff(bar_beats)))
    return 8 if bpm >= FAST_BEAT_BPM else 4


def analyze_meter(beats: np.ndarray, downbeats: np.ndarray, *, tol: float = 0.05) -> Meter:
    """Count beats between consecutive downbeats.

    The counts are taken as tracked: nothing is "corrected" to 4/4. A single bar whose count
    differs from two agreeing neighbours is kept but marked ``confident=False`` for QA.
    """
    beats = np.asarray(beats, dtype=float)
    downbeats = np.asarray(downbeats, dtype=float)
    if len(beats) == 0 or len(downbeats) == 0:
        return Meter((), (), len(beats))
    db_idx: list[int] = []
    for d in downbeats:  # snap each downbeat to its nearest beat
        i = int(np.argmin(np.abs(beats - d)))
        if abs(beats[i] - d) <= tol and (not db_idx or i > db_idx[-1]):
            db_idx.append(i)
    if not db_idx:
        return Meter((), (), len(beats))

    last_ibi = float(beats[-1] - beats[-2]) if len(beats) > 1 else 0.5
    raw = []
    for n, start in enumerate(db_idx):
        end = db_idx[n + 1] if n + 1 < len(db_idx) else len(beats)
        end_time = float(beats[end]) if end < len(beats) else float(beats[-1] + last_ibi)
        raw.append((start, end - start, float(beats[start]), end_time,
                    _denominator(beats[start:end + 1])))
    counts = [r[1] for r in raw]
    bars = []
    for n, (start, count, t0, t1, den) in enumerate(raw):
        prev_c = counts[n - 1] if n > 0 else None
        next_c = counts[n + 1] if n + 1 < len(counts) else None
        lone = (prev_c is not None and next_c is not None and prev_c == next_c
                and count != prev_c)
        bars.append(Bar(n, t0, t1, start, count, den, not lone))
    changes: list[MeterChange] = []
    for b in bars:
        sig = (b.numerator, b.denominator)
        if not changes or (changes[-1].numerator, changes[-1].denominator) != sig:
            changes.append(MeterChange(b.index, *sig))
    return Meter(tuple(bars), tuple(changes), db_idx[0])
