"""Guitar Pro 5 `.gp5` export with PyGuitarPro: the fallback when `.gp` export fails (spec 4.2).

GP5 only knows fretted and percussion tracks. Staff tracks (vocals, piano, other) are written
on a synthetic 7-string tuning in octaves so every pitch is kept; they are for reference only.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import guitarpro
from guitarpro import models as g

from bandscribe.notation.layout import LBeat, layout_track
from bandscribe.notation.score_model import Score, Track

STAFF_TUNING = (96, 84, 72, 60, 48, 36, 24)  # string 1..7, an octave apart
QUARTER_TICKS = 960


@dataclass
class Gp5ExportReport:
    path: Path
    notes_written: int = 0
    notes_dropped: int = 0  # chord notes that did not fit the GP5 string model
    warnings: list[str] = field(default_factory=list)


def _ascii(text: str) -> str:
    return text.encode("cp1252", "replace").decode("cp1252")


def _positions(track: Track, beat: LBeat) -> list[tuple[int, int, int]]:
    """(string, value, velocity) for each note of a beat, strings distinct."""
    out: list[tuple[int, int, int]] = []
    used: set[int] = set()
    if track.kind == "tab":
        for n in beat.notes:
            if n.string is not None and n.string not in used:
                used.add(n.string)
                out.append((n.string, n.fret, n.velocity))
        return out
    if track.kind == "drums":
        for i, n in enumerate(beat.notes[:6]):
            out.append((i + 1, n.pitch, n.velocity))
        return out
    for n in sorted(beat.notes, key=lambda x: -x.pitch):
        for s, open_pitch in enumerate(STAFF_TUNING, start=1):
            fret = n.pitch - open_pitch
            if 0 <= fret <= 24 and s not in used:
                used.add(s)
                out.append((s, fret, n.velocity))
                break
    return out


def export_gp5(score: Score, path: Path) -> Gp5ExportReport:
    report = Gp5ExportReport(Path(path))
    song = g.Song()
    song.title = _ascii(score.title or "")
    song.artist = _ascii(str(score.meta.get("artist", "")))
    song.tab = "BandScribe (automatic transcription)"
    song.notice = ["'?' marks low-confidence notes.", "Non-commercial use only."]
    tempos = []
    target_time = written_time = 0.0
    for bar in score.bars:
        length = float(bar.length_q)
        duration = 60 * length / bar.bpm
        target_time += duration
        remaining = target_time - written_time
        bpm = max(1, round(60 * length / (remaining if remaining > 0 else duration)))
        tempos.append(bpm)
        written_time += 60 * length / bpm
    song.tempo = tempos[0] if tempos else 120
    song.tracks = []
    song.measureHeaders = []

    start = QUARTER_TICKS
    for b in score.bars:
        header = g.MeasureHeader(number=b.index + 1, start=start)
        header.timeSignature = g.TimeSignature(numerator=b.numerator,
                                               denominator=g.Duration(value=b.denominator))
        if b.section:
            header.marker = g.Marker(title=_ascii(b.section))
        song.measureHeaders.append(header)
        start += int(b.length_q * QUARTER_TICKS)

    channel = 0
    for k, t in enumerate(score.tracks):
        tuning = t.tuning if t.kind == "tab" else (
            (64, 59, 55, 50, 45, 40) if t.kind == "drums" else STAFF_TUNING)
        if len(tuning) > 7:
            report.warnings.append(f"{t.name}: GP5 supports at most 7 strings; track skipped "
                                   "(it is in the .gp file)")
            continue
        track = g.Track(song, number=len(song.tracks) + 1, name=_ascii(t.name),
                        strings=[g.GuitarString(i + 1, p) for i, p in enumerate(tuning)])
        track.measures = []
        if t.kind == "drums":
            track.isPercussionTrack = True
            track.channel = g.MidiChannel(channel=9, effectChannel=9, instrument=0)
        else:
            if channel in (9, 10):
                channel = 11
            track.channel = g.MidiChannel(channel=channel % 16, effectChannel=(channel + 1) % 16,
                                          instrument=t.midi_program)
            channel += 2
        layout = layout_track(t, score.bars)
        last_bpm = song.tempo
        for header, beats, bpm in zip(song.measureHeaders, layout, tempos, strict=True):
            measure = g.Measure(track, header)
            voice = measure.voices[0]
            for i, lb in enumerate(beats):
                dur = g.Duration(value=lb.duration, isDotted=bool(lb.dots))
                if lb.tuplet:
                    dur.tuplet = g.Tuplet(enters=lb.tuplet[0], times=lb.tuplet[1])
                beat = g.Beat(voice, duration=dur,
                              status=g.BeatStatus.rest if lb.rest else g.BeatStatus.normal)
                if lb.text:
                    beat.text = lb.text
                if i == 0 and k == 0 and abs(bpm - last_bpm) >= 1:
                    beat.effect.mixTableChange = g.MixTableChange(
                        tempo=g.MixTableItem(value=bpm, duration=0, allTracks=True))
                    last_bpm = bpm
                if not lb.rest:
                    placed = _positions(t, lb)
                    report.notes_dropped += len(lb.notes) - len(placed)
                    for string, value, velocity in placed:
                        beat.notes.append(g.Note(
                            beat, value=value, string=string, velocity=min(127, max(15, velocity)),
                            type=g.NoteType.tie if lb.tied else g.NoteType.normal))
                        report.notes_written += 1
                voice.beats.append(beat)
            track.measures.append(measure)
        song.tracks.append(track)
    report.path.parent.mkdir(parents=True, exist_ok=True)
    guitarpro.write(song, str(report.path), version=(5, 1, 0))
    return report
