"""Tempo map derived from tracked beats. Tempo changes are allowed (spec 3.4)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TempoSegment:
    start_time: float
    start_beat: int  # index into the beat array
    bpm: float


@dataclass(frozen=True)
class TempoMap:
    beats: np.ndarray  # beat times in seconds
    segments: tuple[TempoSegment, ...]
    global_bpm: float

    def bpm_at(self, t: float) -> float:
        bpm = self.segments[0].bpm if self.segments else self.global_bpm
        for seg in self.segments:
            if seg.start_time <= t:
                bpm = seg.bpm
        return bpm

    def _fallback(self) -> tuple[float, float]:
        period = 60.0 / self.global_bpm if self.global_bpm > 0 else 0.5
        origin = float(self.beats[0]) if len(self.beats) else 0.0
        return origin, period

    def time_to_beat(self, t: float | np.ndarray) -> np.ndarray:
        """Fractional beat position of time ``t`` (beat 0 = first tracked beat).

        Uses the tracked beat times themselves, so it follows rubato exactly; outside the
        tracked range it extrapolates with the first / last inter-beat interval.
        """
        b = self.beats
        t = np.asarray(t, dtype=float)
        if len(b) < 2:
            origin, period = self._fallback()
            return (t - origin) / period
        idx = np.arange(len(b), dtype=float)
        out = np.interp(t, b, idx)
        first, last = b[1] - b[0], b[-1] - b[-2]
        out = np.where(t < b[0], (t - b[0]) / first, out)
        return np.where(t > b[-1], idx[-1] + (t - b[-1]) / last, out)

    def beat_to_time(self, beat: float | np.ndarray) -> np.ndarray:
        b = self.beats
        beat = np.asarray(beat, dtype=float)
        if len(b) < 2:
            origin, period = self._fallback()
            return origin + beat * period
        idx = np.arange(len(b), dtype=float)
        out = np.interp(beat, idx, b)
        first, last = b[1] - b[0], b[-1] - b[-2]
        out = np.where(beat < 0, b[0] + beat * first, out)
        return np.where(beat > idx[-1], b[-1] + (beat - idx[-1]) * last, out)


def _median_filter(x: np.ndarray, width: int) -> np.ndarray:
    half = width // 2
    padded = np.pad(x, half, mode="edge")
    return np.array([np.median(padded[i:i + width]) for i in range(len(x))])


def build_tempo_map(beats: np.ndarray, *, change_ratio: float = 0.04, min_beats: int = 8,
                    smooth: int = 9) -> TempoMap:
    """Split the beat sequence into constant-tempo segments.

    A new segment starts when the smoothed local tempo stays more than ``change_ratio`` away
    from the current segment tempo for ``min_beats`` consecutive beats. Shorter deviations
    (fills, tracker jitter) do not create tempo changes.
    """
    beats = np.asarray(beats, dtype=float)
    if len(beats) < 2:
        return TempoMap(beats, (), 0.0)
    ibi = np.diff(beats)
    if np.any(ibi <= 0):
        raise ValueError("beats must be strictly increasing")
    width = min(smooth, len(ibi))
    width -= 1 - width % 2  # odd window
    local = 60.0 / _median_filter(ibi, max(width, 1))
    global_bpm = float(60.0 / np.median(ibi))

    bounds = [0]
    seg_start, run = 0, 0
    for i in range(len(local)):
        stop = max(seg_start + 1, i - run)
        ref = float(np.median(local[seg_start:stop]))
        if abs(local[i] - ref) / ref > change_ratio:
            run += 1
            if run >= min_beats:
                seg_start = i - run + 1
                bounds.append(seg_start)
                run = 0
        else:
            run = 0
    bounds.append(len(ibi))
    segments: list[TempoSegment] = []
    for a, b in zip(bounds, bounds[1:], strict=False):
        bpm = float(60.0 / np.median(ibi[a:b]))
        if segments and abs(bpm - segments[-1].bpm) / segments[-1].bpm <= change_ratio:
            continue  # not a real change after all
        segments.append(TempoSegment(float(beats[a]), a, round(bpm, 2)))
    return TempoMap(beats, tuple(segments), round(global_bpm, 2))
