"""Internal score representation, independent of any export format (spec section 4).

Positions and durations are exact fractions of a quarter note. The JSON form (``to_dict``)
is the stable artefact written to ``score.json``; fractions are stored as "n/d" strings.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
KINDS = ("tab", "staff", "drums")


def _f(x: Fraction) -> str:
    return f"{x.numerator}/{x.denominator}"


def _pf(s: str) -> Fraction:
    n, d = s.split("/")
    return Fraction(int(n), int(d))


@dataclass(frozen=True)
class BarSpec:
    index: int
    numerator: int
    denominator: int
    start_beat: int  # index of the bar's first tracked beat (negative before the first beat)
    start_q: Fraction  # quarter notes from the start of the score
    bpm: float  # quarter-note tempo over this bar
    start_time: float  # seconds in the audio
    end_time: float
    section: str | None = None
    confident: bool = True  # False: meter of this bar is doubtful (see analysis.meter)
    # Audio time of each tracked beat in the bar plus the bar end (numerator + 1 values);
    # lets exporters follow tempo drift inside a bar. Empty for older scores.
    beat_times: tuple[float, ...] = ()

    @property
    def beat_q(self) -> Fraction:
        """Length of one tracked beat in quarter notes."""
        return Fraction(4, self.denominator)

    @property
    def length_q(self) -> Fraction:
        return self.beat_q * self.numerator

    @property
    def end_q(self) -> Fraction:
        return self.start_q + self.length_q


@dataclass(frozen=True)
class ScoreNote:
    bar: int
    pos_q: Fraction  # offset from the bar start, in quarter notes
    dur_q: Fraction
    pitch: int
    velocity: int = 80
    conf: float = 1.0
    low_conf: bool = False
    sources: tuple[str, ...] = ()
    string: int | None = None  # 1 = highest string
    fret: int | None = None
    flags: tuple[str, ...] = ()
    onset: float = 0.0  # original time in the audio, seconds
    offset: float = 0.0


@dataclass
class Track:
    id: str
    name: str
    family: str
    kind: str  # tab | staff | drums
    midi_program: int = 0
    tuning_key: str | None = None
    tuning: tuple[int, ...] = ()  # open strings, string 1 (highest) first
    notes: list[ScoreNote] = field(default_factory=list)
    # Subdivisions per tracked beat chosen by quantisation: (bar, beat) -> 2/3/4/6/8.
    grid: dict[tuple[int, int], int] = field(default_factory=dict)
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"unknown track kind {self.kind!r}")


@dataclass
class Score:
    title: str
    bars: list[BarSpec]
    tracks: list[Track]
    key: dict[str, Any] | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    def note_count(self) -> int:
        return sum(len(t.notes) for t in self.tracks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION, "title": self.title, "key": self.key,
            "meta": self.meta,
            "bars": [{"index": b.index, "numerator": b.numerator, "denominator": b.denominator,
                      "start_beat": b.start_beat, "start_q": _f(b.start_q), "bpm": b.bpm,
                      "start_time": b.start_time, "end_time": b.end_time, "section": b.section,
                      "confident": b.confident,
                      "beat_times": [round(t, 4) for t in b.beat_times]} for b in self.bars],
            "tracks": [{
                "id": t.id, "name": t.name, "family": t.family, "kind": t.kind,
                "midi_program": t.midi_program, "tuning_key": t.tuning_key,
                "tuning": list(t.tuning), "info": t.info,
                "grid": [[bar, beat, s] for (bar, beat), s in sorted(t.grid.items())],
                "notes": [{"bar": n.bar, "pos_q": _f(n.pos_q), "dur_q": _f(n.dur_q),
                           "pitch": n.pitch, "velocity": n.velocity, "conf": round(n.conf, 4),
                           "low_conf": n.low_conf, "sources": list(n.sources),
                           "string": n.string, "fret": n.fret, "flags": list(n.flags),
                           "onset": round(n.onset, 4), "offset": round(n.offset, 4)}
                          for n in t.notes],
            } for t in self.tracks],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Score:
        if d.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported score schema version")
        bars = [BarSpec(b["index"], b["numerator"], b["denominator"], b["start_beat"],
                        _pf(b["start_q"]), b["bpm"], b["start_time"], b["end_time"],
                        b.get("section"), b.get("confident", True),
                        tuple(b.get("beat_times", ()))) for b in d["bars"]]
        tracks = []
        for t in d["tracks"]:
            notes = [ScoreNote(n["bar"], _pf(n["pos_q"]), _pf(n["dur_q"]), n["pitch"],
                               n["velocity"], n["conf"], n["low_conf"], tuple(n["sources"]),
                               n["string"], n["fret"], tuple(n["flags"]), n["onset"], n["offset"])
                     for n in t["notes"]]
            tracks.append(Track(t["id"], t["name"], t["family"], t["kind"], t["midi_program"],
                                t["tuning_key"], tuple(t["tuning"]), notes,
                                {(b, c): s for b, c, s in t["grid"]}, t.get("info", {})))
        return cls(d["title"], bars, tracks, d.get("key"), d.get("meta", {}))

    def save(self, path: Path) -> None:
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> Score:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
