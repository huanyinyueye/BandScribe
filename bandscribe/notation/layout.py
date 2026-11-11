"""Engraving layout: turn a track's notes into per-bar sequences of notated beats.

One voice per track. Each onset group (a chord) lasts until its own end or the next onset,
whichever comes first; the rest of the time is rests. Segments are cut at bar lines and beat
boundaries and written as plain, dotted or tuplet values; continuation pieces are tied.
All exporters (Guitar Pro, MusicXML) consume this layout, so they agree with each other.
"""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from fractions import Fraction

from bandscribe.notation.quantize import default_grid
from bandscribe.notation.score_model import BarSpec, ScoreNote, Track

LOW_CONF_TEXT = "?"


@dataclass(frozen=True)
class LBeat:
    duration: int  # note value: 1 whole, 2 half, 4 quarter, 8, 16, 32, 64
    dots: int
    tuplet: tuple[int, int] | None  # (3, 2) = three in the time of two
    rest: bool
    tied: bool  # every note continues the previous beat's notes
    notes: tuple[ScoreNote, ...]
    text: str | None
    length_q: Fraction


def _value_length(duration: int, dots: int, tuplet: tuple[int, int] | None) -> Fraction:
    length = Fraction(4, duration)
    if dots:
        length *= Fraction(3, 2)
    if tuplet:
        length *= Fraction(tuplet[1], tuplet[0])
    return length


def _split_cells(k: int, offset: int, bar: BarSpec) -> list[tuple[int, int]]:
    """Split ``k`` whole beats starting at beat ``offset`` into (duration, dots) values."""
    den = bar.denominator
    compound = den == 8 and bar.numerator % 3 == 0
    out = []
    while k > 0:
        size, value, dots = 1, den, 0
        if compound:
            if offset % 6 == 0 and k >= 6:
                size, value, dots = 6, 2, 1  # dotted half
            elif offset % 3 == 0 and k >= 3:
                size, value, dots = 3, 4, 1  # dotted quarter
        elif den == 4 and bar.numerator == 3 and offset == 0 and k == 3:
            size, value, dots = 3, 2, 1  # dotted half filling a 3/4 bar
        else:
            for cand in (8, 4, 2):
                if cand <= k and offset % cand == 0 and den % cand == 0:
                    size, value = cand, den // cand
                    break
        out.append((value, dots))
        k -= size
        offset += size
    return out


_BINARY = ((8, 4, 0, 8), (6, 8, 1, 2), (4, 8, 0, 2), (3, 16, 1, 1), (2, 16, 0, 2), (1, 32, 0, 1))


def _split_binary(start: int, length: int) -> list[tuple[int, int]]:
    """Split a span inside one beat, measured in 32nd notes, into (duration, dots) values."""
    out = []
    while length > 0:
        for size, value, dots, align in _BINARY:
            if size <= length and start % align == 0:
                out.append((value, dots))
                start += size
                length -= size
                break
    return out


def _split_tuplet(start: int, length: int, unit_value: int) -> list[int]:
    """Split a span measured in tuplet units into note values (each a 3:2 tuplet)."""
    out = []
    while length > 0:
        for size in (4, 2, 1):
            if size <= length and start % size == 0 and unit_value // size >= 1:
                out.append(unit_value // size)
                start += size
                length -= size
                break
    return out


def layout_track(track: Track, bars: list[BarSpec]) -> list[list[LBeat]]:
    """Return, for every bar, the beats of ``track``. Durations in a bar sum to its length."""
    drums = track.kind == "drums"
    starts_q = [b.start_q for b in bars]
    total_end = bars[-1].end_q

    def cell_end(abs_q: Fraction) -> Fraction:
        i = max(0, bisect.bisect_right(starts_q, abs_q) - 1)
        bar = bars[i]
        c = (abs_q - bar.start_q) // bar.beat_q
        return bar.start_q + (c + 1) * bar.beat_q

    groups: dict[Fraction, list[ScoreNote]] = {}
    for n in track.notes:
        groups.setdefault(bars[n.bar].start_q + n.pos_q, []).append(n)
    onsets = sorted(groups)

    # Timeline of segments (start, end, notes or None for a rest).
    segments: list[tuple[Fraction, Fraction, tuple[ScoreNote, ...] | None]] = []
    cursor = bars[0].start_q
    for i, s in enumerate(onsets):
        if s >= total_end:
            break
        nxt = onsets[i + 1] if i + 1 < len(onsets) else total_end
        end = min(nxt, s + max(n.dur_q for n in groups[s]), total_end)
        if drums:
            end = min(end, cell_end(s))
        if s > cursor:
            segments.append((cursor, s, None))
        notes = tuple(sorted(groups[s], key=lambda n: (n.string or 0, n.pitch)))
        segments.append((s, end, notes))
        cursor = end
    if cursor < total_end:
        segments.append((cursor, total_end, None))

    out: list[list[LBeat]] = []
    seg_i = 0
    emitted: set[int] = set()  # segments that already produced their first (untied) beat
    for bar in bars:
        beats: list[LBeat] = []
        den, bq = bar.denominator, bar.beat_q
        # Chunks per beat cell: (segment index, start offset in cell, end offset in cell).
        cells: list[list[tuple[int, Fraction, Fraction]]] = []
        for c in range(bar.numerator):
            c0, c1 = bar.start_q + c * bq, bar.start_q + (c + 1) * bq
            while seg_i < len(segments) and segments[seg_i][1] <= c0:
                seg_i += 1
            chunks = []
            k = seg_i
            while k < len(segments) and segments[k][0] < c1:
                a, b = max(segments[k][0], c0), min(segments[k][1], c1)
                if b > a:
                    chunks.append((k, a - c0, b - c0))
                k += 1
            cells.append(chunks)

        def emit(k: int, duration: int, dots: int, tuplet: tuple[int, int] | None) -> None:
            notes = segments[k][2]
            first = k not in emitted
            emitted.add(k)
            text = None
            if notes and first and any(n.low_conf for n in notes):
                text = LOW_CONF_TEXT
            beats.append(LBeat(duration, dots, tuplet, notes is None,  # noqa: B023
                               bool(notes) and not first, notes or (), text,
                               _value_length(duration, dots, tuplet)))

        c = 0
        while c < len(cells):
            chunks = cells[c]
            full = len(chunks) == 1 and chunks[0][1] == 0 and chunks[0][2] == bq
            if full:
                k = chunks[0][0]
                run = 1
                while (c + run < len(cells) and len(cells[c + run]) == 1
                       and cells[c + run][0] == (k, Fraction(0), bq)):
                    run += 1
                for duration, dots in _split_cells(run, c, bar):
                    emit(k, duration, dots, None)
                c += run
                continue
            s = track.grid.get((bar.index, c), default_grid(den))
            if s in (3, 6):
                unit = bq / s
                unit_value = 2 * den * s // 3
                for k, a, b in chunks:
                    for value in _split_tuplet(int(a / unit), int((b - a) / unit), unit_value):
                        emit(k, value, 0, (3, 2))
            else:
                unit = Fraction(1, 8)  # a 32nd note
                for k, a, b in chunks:
                    for duration, dots in _split_binary(int(a / unit), int((b - a) / unit)):
                        emit(k, duration, dots, None)
            c += 1
        out.append(beats)
    return out


def layout_to_json(track: Track, bars: list[BarSpec]) -> list[list[dict]]:
    """Layout in the JSON form consumed by the Node Guitar Pro exporter."""
    result = []
    for beats in layout_track(track, bars):
        row = []
        for b in beats:
            row.append({
                "d": b.duration, "dots": b.dots, "tu": list(b.tuplet) if b.tuplet else None,
                "rest": b.rest, "tie": b.tied, "text": b.text,
                "notes": [{"p": n.pitch, "s": n.string, "f": n.fret, "v": n.velocity,
                           "lc": n.low_conf} for n in b.notes]})
        result.append(row)
    return result
