"""MusicXML 4.0 export (score-partwise), written from the same layout as the `.gp` export.

One staff per part. Fretted tracks carry string / fret as <technical> elements; a separate
TAB staff is not written (Guitar Pro `.gp` is the format that carries the tablature).
"""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path
from xml.etree import ElementTree as ET

from bandscribe.notation.layout import LBeat, layout_track
from bandscribe.notation.score_model import Score, Track

DIVISIONS = 960
TYPES = {1: "whole", 2: "half", 4: "quarter", 8: "eighth", 16: "16th", 32: "32nd", 64: "64th"}
STEPS = (("C", 0), ("C", 1), ("D", 0), ("D", 1), ("E", 0), ("F", 0), ("F", 1), ("G", 0),
         ("G", 1), ("A", 0), ("A", 1), ("B", 0))
# GM drum key -> (display step, display octave, notehead) on a 5-line percussion staff.
DRUM_DISPLAY = {36: ("F", 4, None), 38: ("C", 5, None), 42: ("G", 5, "x"), 44: ("D", 4, "x"),
                46: ("G", 5, "circle-x"), 43: ("A", 4, None), 47: ("D", 5, None),
                50: ("E", 5, None), 49: ("A", 5, "x"), 51: ("F", 5, "x"), 52: ("B", 5, "x"),
                55: ("B", 5, "x")}


def _sub(parent: ET.Element, tag: str, text: str | int | None = None, **attrs: str) -> ET.Element:
    el = ET.SubElement(parent, tag, attrs)
    if text is not None:
        el.text = str(text)
    return el


def _clef(track: Track) -> tuple[str, int, int | None]:
    if track.kind == "drums":
        return "percussion", 2, None
    if track.family == "bass":
        return "F", 4, -1 if track.kind == "tab" else None
    if track.family == "guitar":
        return "G", 2, -1
    return "G", 2, None


def _note(measure: ET.Element, track: Track, beat: LBeat, note, chord: bool, tie_start: bool,
          octave_shift: int) -> None:
    el = _sub(measure, "note")
    if chord:
        _sub(el, "chord")
    if track.kind == "drums":
        step, octave, head = DRUM_DISPLAY.get(note.pitch, ("C", 5, None))
        up = _sub(el, "unpitched")
        _sub(up, "display-step", step)
        _sub(up, "display-octave", octave)
    else:
        head = None
        step, alter = STEPS[note.pitch % 12]
        p = _sub(el, "pitch")
        _sub(p, "step", step)
        if alter:
            _sub(p, "alter", alter)
        _sub(p, "octave", note.pitch // 12 - 1 + octave_shift)
    _sub(el, "duration", int(beat.length_q * DIVISIONS))
    if beat.tied:
        _sub(el, "tie", type="stop")
    if tie_start:
        _sub(el, "tie", type="start")
    _sub(el, "voice", 1)
    _sub(el, "type", TYPES[beat.duration])
    for _ in range(beat.dots):
        _sub(el, "dot")
    if beat.tuplet:
        tm = _sub(el, "time-modification")
        _sub(tm, "actual-notes", beat.tuplet[0])
        _sub(tm, "normal-notes", beat.tuplet[1])
    if head:
        _sub(el, "notehead", head)
    technical = track.kind == "tab" and note.string is not None
    if beat.tied or tie_start or technical:
        notations = _sub(el, "notations")
        if beat.tied:
            _sub(notations, "tied", type="stop")
        if tie_start:
            _sub(notations, "tied", type="start")
        if technical:
            tech = _sub(notations, "technical")
            _sub(tech, "string", note.string)
            _sub(tech, "fret", note.fret)


def build_musicxml(score: Score) -> ET.ElementTree:
    root = ET.Element("score-partwise", version="4.0")
    work = _sub(root, "work")
    _sub(work, "work-title", score.title or "Untitled")
    ident = _sub(root, "identification")
    enc = _sub(ident, "encoding")
    _sub(enc, "software", "BandScribe")
    _sub(ident, "rights", "Automatic transcription. '?' marks low-confidence notes. "
                          "Non-commercial use only.")
    part_list = _sub(root, "part-list")
    for i, t in enumerate(score.tracks, start=1):
        sp = _sub(part_list, "score-part", id=f"P{i}")
        _sub(sp, "part-name", t.name)
        mi = _sub(sp, "midi-instrument", id=f"P{i}-I1")
        _sub(mi, "midi-channel", 10 if t.kind == "drums" else min(16, i if i < 10 else i + 1))
        _sub(mi, "midi-program", 1 if t.kind == "drums" else t.midi_program + 1)
    fifths = (score.key or {}).get("fifths")
    for i, t in enumerate(score.tracks, start=1):
        part = _sub(root, "part", id=f"P{i}")
        layout = layout_track(t, score.bars)
        sign, line, octave_change = _clef(t)
        # Guitar and bass are written an octave above sounding pitch (octave clef).
        octave_shift = 1 if octave_change == -1 else 0
        flat = [(bi, k) for bi, beats in enumerate(layout) for k in range(len(beats))]
        nxt_tied = {}
        for idx, (bi, k) in enumerate(flat):
            if idx + 1 < len(flat):
                nb, nk = flat[idx + 1]
                nxt_tied[(bi, k)] = layout[nb][nk].tied
        last_sig, last_bpm, last_sound_bpm = None, None, None
        for bar, beats in zip(score.bars, layout, strict=True):
            measure = _sub(part, "measure", number=str(bar.index + 1))
            sig = (bar.numerator, bar.denominator)
            if bar.index == 0 or sig != last_sig:
                attrs = _sub(measure, "attributes")
                if bar.index == 0:
                    _sub(attrs, "divisions", DIVISIONS)
                    if fifths is not None and t.kind != "drums":
                        key = _sub(attrs, "key")
                        _sub(key, "fifths", fifths)
                        _sub(key, "mode", (score.key or {}).get("mode", "major"))
                time = _sub(attrs, "time")
                _sub(time, "beats", bar.numerator)
                _sub(time, "beat-type", bar.denominator)
                if bar.index == 0:
                    clef = _sub(attrs, "clef")
                    _sub(clef, "sign", sign)
                    if sign != "percussion":
                        _sub(clef, "line", line)
                    if octave_change:
                        _sub(clef, "clef-octave-change", octave_change)
                last_sig = sig
            if i == 1:
                visible = last_bpm is None or abs(bar.bpm - last_bpm) >= 1
                if visible:
                    d = _sub(measure, "direction", placement="above")
                    met = _sub(_sub(d, "direction-type"), "metronome")
                    _sub(met, "beat-unit", "quarter")
                    _sub(met, "per-minute", round(bar.bpm))
                    _sub(d, "sound", tempo=str(round(bar.bpm, 2)))
                    last_bpm = bar.bpm
                    last_sound_bpm = bar.bpm
                elif last_sound_bpm is None or abs(bar.bpm - last_sound_bpm) > 0.005:
                    _sub(measure, "sound", tempo=str(round(bar.bpm, 2)))
                    last_sound_bpm = bar.bpm
            if bar.section and i == 1:
                d = _sub(measure, "direction", placement="above")
                _sub(_sub(d, "direction-type"), "rehearsal", bar.section)
            for k, beat in enumerate(beats):
                if beat.text:
                    d = _sub(measure, "direction", placement="above")
                    _sub(_sub(d, "direction-type"), "words", beat.text)
                if beat.rest:
                    el = _sub(measure, "note")
                    _sub(el, "rest")
                    _sub(el, "duration", int(beat.length_q * DIVISIONS))
                    _sub(el, "voice", 1)
                    _sub(el, "type", TYPES[beat.duration])
                    for _ in range(beat.dots):
                        _sub(el, "dot")
                    if beat.tuplet:
                        tm = _sub(el, "time-modification")
                        _sub(tm, "actual-notes", beat.tuplet[0])
                        _sub(tm, "normal-notes", beat.tuplet[1])
                    continue
                tie_start = bool(nxt_tied.get((bar.index, k)))
                for j, note in enumerate(beat.notes):
                    _note(measure, t, beat, note, j > 0, tie_start, octave_shift)
    return ET.ElementTree(root)


def export_musicxml(score: Score, path: Path) -> Path:
    tree = build_musicxml(score)
    ET.indent(tree)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        fh.write(b'<?xml version="1.0" encoding="UTF-8"?>\n')
        fh.write(b'<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 4.0 Partwise//EN" '
                 b'"http://www.musicxml.org/dtds/partwise.dtd">\n')
        tree.write(fh, encoding="utf-8", xml_declaration=False)
    return path


def measure_durations(path: Path) -> list[list[Fraction]]:
    """Per part, the summed note duration of each measure in quarter notes (for tests)."""
    root = ET.parse(path).getroot()
    out = []
    for part in root.findall("part"):
        rows = []
        for measure in part.findall("measure"):
            total = 0
            for note in measure.findall("note"):
                if note.find("chord") is None:
                    total += int(note.findtext("duration"))
            rows.append(Fraction(total, DIVISIONS))
        out.append(rows)
    return out
