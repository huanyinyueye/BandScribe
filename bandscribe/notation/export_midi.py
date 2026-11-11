"""MIDI export: one track per instrument, tempo per bar, time signatures and section markers."""

from __future__ import annotations

from fractions import Fraction
from pathlib import Path

import mido

from bandscribe.notation.score_model import Score

TICKS = 960


def _ticks(q: Fraction) -> int:
    return int(round(q * TICKS))


def export_midi(score: Score, path: Path) -> Path:
    mid = mido.MidiFile(type=1, ticks_per_beat=TICKS)
    meta = mido.MidiTrack()
    mid.tracks.append(meta)
    meta.append(mido.MetaMessage("track_name", name=score.title or "BandScribe", time=0))
    events: list[tuple[int, int, mido.MetaMessage]] = []
    last_sig, last_bpm = None, None
    for b in score.bars:
        t = _ticks(b.start_q)
        sig = (b.numerator, b.denominator)
        if sig != last_sig:
            events.append((t, 0, mido.MetaMessage("time_signature", numerator=sig[0],
                                                  denominator=sig[1])))
            last_sig = sig
        if last_bpm is None or abs(b.bpm - last_bpm) > 0.05:
            events.append((t, 1, mido.MetaMessage("set_tempo", tempo=mido.bpm2tempo(b.bpm))))
            last_bpm = b.bpm
        if b.section:
            events.append((t, 2, mido.MetaMessage("marker", text=b.section)))
    prev = 0
    for t, _order, msg in sorted(events, key=lambda e: (e[0], e[1])):
        msg.time = t - prev
        meta.append(msg)
        prev = t

    channel = 0
    for track in score.tracks:
        mt = mido.MidiTrack()
        mid.tracks.append(mt)
        mt.append(mido.MetaMessage("track_name", name=track.name, time=0))
        if track.kind == "drums":
            ch = 9
        else:
            if channel == 9:
                channel += 1
            ch = channel % 16
            channel += 1
            mt.append(mido.Message("program_change", channel=ch, program=track.midi_program,
                                   time=0))
        msgs: list[tuple[int, int, mido.Message]] = []
        for n in track.notes:
            start = _ticks(score.bars[n.bar].start_q + n.pos_q)
            end = max(start + 1, _ticks(score.bars[n.bar].start_q + n.pos_q + n.dur_q))
            vel = min(127, max(1, n.velocity))
            msgs.append((start, 1, mido.Message("note_on", channel=ch, note=n.pitch, velocity=vel)))
            msgs.append((end, 0, mido.Message("note_off", channel=ch, note=n.pitch, velocity=0)))
        prev = 0
        for t, _order, msg in sorted(msgs, key=lambda e: (e[0], e[1])):
            msg.time = t - prev
            mt.append(msg)
            prev = t
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mid.save(path)
    return path
