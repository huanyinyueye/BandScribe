import zipfile
from dataclasses import replace
from fractions import Fraction as F
from pathlib import Path
from xml.etree import ElementTree as ET

import guitarpro
import mido
import numpy as np
import pytest

from bandscribe.analysis.meter import analyze_meter
from bandscribe.analysis.sections import Section
from bandscribe.analysis.tempo import build_tempo_map
from bandscribe.notation.export_gp import build_payload, export_gp, read_gp, verify
from bandscribe.notation.export_gp5 import export_gp5
from bandscribe.notation.export_midi import export_midi
from bandscribe.notation.export_musicxml import export_musicxml, measure_durations
from bandscribe.notation.quantize import build_bars, quantize_notes
from bandscribe.notation.score_model import Score, Track
from bandscribe.tab.fret_dp import DPFretAssigner
from bandscribe.tab.tunings import TUNINGS
from bandscribe.transcription.base import Note


@pytest.fixture(scope="module")
def score():
    """4/4, 4/4, 7/8, 7/8 with a tempo change; guitar (drop D tab), bass, drums, vocals."""
    q = 0.5
    beats = list(1.0 + np.arange(8) * q)
    t = beats[-1] + q
    downs = [1.0, 1.0 + 4 * q]
    for _ in range(2):
        downs.append(t)
        for _ in range(7):
            beats.append(t)
            t += 0.2  # eighth notes at 150 BPM (quarter)
    beats = np.array(beats)
    tm = build_tempo_map(beats)
    meter = analyze_meter(beats, np.array(downs))
    bars = build_bars(tm, meter, 1.0, float(beats[-1]),
                      [Section("Section 1", 0, 1.0), Section("Section 2", 2, downs[2])])

    def mk(family, kind, raw, tuning=None, program=0, drums=False):
        notes = raw
        if tuning:
            res = DPFretAssigner().assign(raw, TUNINGS[tuning])
            assert not res.unplaced
            notes = res.notes
        sn, grid, _ = quantize_notes(notes, bars, tm, drums=drums)
        tr = Track(family, family.title(), family, kind, program, tuning,
                   TUNINGS[tuning].strings if tuning else (), sn, grid)
        return tr

    def N(onset, pitch, dur=0.25, conf=0.9, inst="guitar"):
        return Note(onset, onset + dur, pitch, conf=conf, instrument=inst)

    guitar = [N(1.0, 38, 0.5), N(1.0, 45, 0.5), N(1.0, 50, 0.5),          # D5 power chord
              N(1.5, 41, 0.25, conf=0.3),                                    # low confidence
              N(2.0, 62, 0.16), N(2.0 + 0.5 / 3, 64, 0.16), N(2.0 + 1.0 / 3, 65, 0.16),  # triplet
              N(2.5, 57, 2.0),                                               # tied over the barline
              N(downs[2], 38, 0.2), N(downs[2] + 0.2, 38, 0.2), N(downs[2] + 0.4, 50, 0.6)]
    bass = [N(1.0 + k * 0.5, 26 + (k % 2) * 12, 0.5, inst="bass") for k in range(8)]
    drums = [Note(1.0 + k * 0.25, 1.05 + k * 0.25, p, instrument="drums")
             for k, p in enumerate([36, 42, 38, 42, 36, 46, 38, 49])]
    drums += [Note(1.0, 1.05, 42, instrument="drums")]
    vocals = [N(1.0, 69, 1.0, inst="vocals"), N(3.0, 72, 0.5, inst="vocals")]
    tracks = [mk("guitar", "tab", guitar, "guitar_drop_d", 30),
              mk("bass", "tab", bass, "bass_drop_d", 33),
              mk("drums", "drums", drums, drums=True),
              mk("vocals", "staff", vocals, program=52)]
    return Score("Test Song", bars, tracks, key={"fifths": -1, "mode": "minor", "name": "D minor"})


def test_fixture_shape(score):
    assert [(b.numerator, b.denominator) for b in score.bars] == [(4, 4), (4, 4), (7, 8), (7, 8)]
    assert score.bars[0].bpm == pytest.approx(120, abs=0.5)
    assert score.bars[2].bpm == pytest.approx(150, abs=0.5)
    assert any(n.low_conf for n in score.tracks[0].notes)


def test_gp_export_reads_back_identically(score, tmp_path):
    report = export_gp(score, tmp_path / "song.gp")
    assert report.path.stat().st_size == report.bytes > 1000
    assert report.mismatches == [] and report.verified
    s = report.summary
    assert [(b["num"], b["den"]) for b in s["masterBars"]] == [(4, 4), (4, 4), (7, 8), (7, 8)]
    assert s["masterBars"][0]["section"] == "Section 1" and s["masterBars"][2]["section"] == "Section 2"
    assert s["masterBars"][0]["tempo"] == 120 and s["masterBars"][2]["tempo"] == 150
    g = s["tracks"][0]
    assert g["tablature"] and g["tuning"] == [64, 59, 55, 50, 45, 38]
    # Drop-D power chord: strings 6/5/4, all open -> pitches 38, 45, 50.
    bar0 = {(n[1], n[2], n[3]) for n in g["notes"] if n[0] == 0 and n[4] == 0}
    assert {(38, 6, 0), (45, 5, 0), (50, 4, 0)} <= bar0
    assert any(n[4] == 1 for n in g["notes"])  # the tied continuation survived
    assert s["tracks"][2]["percussion"] and {n[1] for n in s["tracks"][2]["notes"]} == {36, 38, 42, 46, 49}
    assert sorted(n[1] for n in s["tracks"][3]["notes"]) == [69, 72]
    again = read_gp(report.path)
    assert again["tracks"][0]["notes"] == g["notes"]


def test_gp_piano_large_tied_chord_retains_pitches(tmp_path):
    # Ringing Bloom exposed a piano chord larger than the six-string fallback:
    # alphaTab reused strings and changed pitches on the tied continuation.
    beats = 1.0 + np.arange(12) * 0.5
    tm = build_tempo_map(beats)
    bars = build_bars(tm, analyze_meter(beats, beats[::4]), 1.0, beats[-1])
    raw = [Note(2.5, 3.5, pitch, instrument="piano")
           for pitch in (54, 66, 69, 71, 76, 78, 83, 88)]
    notes, grid, _ = quantize_notes(raw, bars, tm)
    piano = Track("piano", "Piano", "piano", "staff", notes=notes, grid=grid)
    report = export_gp(Score("Piano", bars, [piano]), tmp_path / "piano.gp")
    assert report.verified, report.mismatches


def test_gp_integer_tempos_do_not_accumulate_timing_error(tmp_path):
    beats = np.round((0.4 + np.arange(401) * 60 / 195) / 0.02) * 0.02
    tm = build_tempo_map(beats)
    bars = build_bars(tm, analyze_meter(beats, beats[::4]), beats[0], beats[-1])
    score = Score("Tempo", bars, [Track("piano", "Piano", "piano", "staff")])
    report = export_gp(score, tmp_path / "tempo.gp")
    assert report.verified, report.mismatches
    elapsed, bpm = 0.0, None
    for bar, written in zip(bars, report.summary["masterBars"], strict=True):
        # Tempo can change inside a bar (per-beat corrections at ratio positions).
        pos = 0.0
        for ratio, value in [*written["tempos"], [1.0, None]]:
            if bpm is not None:
                elapsed += 60 * float(bar.length_q) * (ratio - pos) / bpm
            pos, bpm = ratio, value if value is not None else bpm
        assert abs(elapsed - (bar.end_time - bars[0].start_time)) < 0.05


def test_gp_keeps_a_real_small_tempo_change_visible(score, tmp_path):
    # Nominal tempo only: drop per-beat times so playback follows the edited bpm.
    bars = [replace(b, bpm=120 if i == 0 else 124, beat_times=())
            for i, b in enumerate(score.bars)]
    report = export_gp(Score("Tempo", bars, score.tracks), tmp_path / "tempo-change.gp")
    with zipfile.ZipFile(report.path) as archive:
        root = ET.fromstring(archive.read("Content/score.gpif"))
    tempo = [a for a in root.findall("./MasterTrack/Automations/Automation")
             if a.findtext("Type") == "Tempo"]
    assert any(a.findtext("Visible") == "true" and
               int(a.findtext("Value").split()[0]) in (123, 124, 125) for a in tempo[1:])


def test_gp_verify_detects_differences(score, tmp_path):
    payload = build_payload(score)
    report = export_gp(score, tmp_path / "song.gp")
    summary = report.summary
    assert verify(payload, summary) == []
    broken = {**summary, "tracks": [dict(t) for t in summary["tracks"]]}
    broken["tracks"][0]["notes"] = summary["tracks"][0]["notes"][:-1]
    assert any("notes differ" in m for m in verify(payload, broken))
    broken2 = {**summary, "masterBars": summary["masterBars"][:-1]}
    assert any("time signatures" in m for m in verify(payload, broken2))


def test_gp_payload_marks_low_confidence(score):
    payload = build_payload(score)
    texts = [b["text"] for beats in payload["tracks"][0]["bars"] for b in beats if b["text"]]
    assert texts == ["?"]


def test_gp5_export_round_trip(score, tmp_path):
    report = export_gp5(score, tmp_path / "song.gp5")
    assert report.notes_dropped == 0 and report.notes_written > 0
    song = guitarpro.parse(str(report.path))
    assert song.title == "Test Song" and len(song.tracks) == 4
    sigs = [(h.timeSignature.numerator, h.timeSignature.denominator.value) for h in song.measureHeaders]
    assert sigs == [(4, 4), (4, 4), (7, 8), (7, 8)]
    g = song.tracks[0]
    assert [s.value for s in g.strings] == [64, 59, 55, 50, 45, 38]
    opens = {s.number: s.value for s in g.strings}
    first = g.measures[0].voices[0].beats[0]
    assert sorted(opens[n.string] + n.value for n in first.notes) == [38, 45, 50]
    assert song.tracks[2].isPercussionTrack
    kinds = {n.type for m in g.measures for b in m.voices[0].beats for n in b.notes}
    assert guitarpro.NoteType.tie in kinds
    assert song.measureHeaders[2].marker.title == "Section 2"


def test_midi_export(score, tmp_path):
    path = export_midi(score, tmp_path / "song.mid")
    mid = mido.MidiFile(path)
    assert len(mid.tracks) == 5 and mid.ticks_per_beat == 960
    meta = [m for m in mid.tracks[0] if m.is_meta]
    sigs = [(m.numerator, m.denominator) for m in meta if m.type == "time_signature"]
    tempos = [round(mido.tempo2bpm(m.tempo)) for m in meta if m.type == "set_tempo"]
    assert sigs == [(4, 4), (7, 8)] and tempos == [120, 150]
    assert [m.text for m in meta if m.type == "marker"] == ["Section 1", "Section 2"]
    for track, src in zip(mid.tracks[1:], score.tracks, strict=True):
        ons = [m for m in track if m.type == "note_on"]
        assert len(ons) == len(src.notes)
        assert sorted(m.note for m in ons) == sorted(n.pitch for n in src.notes)
    assert all(m.channel == 9 for m in mid.tracks[3] if m.type == "note_on")


def test_musicxml_export(score, tmp_path):
    path = export_musicxml(score, tmp_path / "song.musicxml")
    durations = measure_durations(path)
    assert len(durations) == 4
    for part in durations:
        assert part == [F(4), F(4), F(7, 2), F(7, 2)]
    text = path.read_text(encoding="utf-8")
    assert "<fret>0</fret>" in text and "<string>6</string>" in text
    assert "<actual-notes>3</actual-notes>" in text and '<tie type="start"' in text
    assert "<words>?</words>" in text and "<rehearsal>Section 2</rehearsal>" in text
    assert "<beat-type>8</beat-type>" in text and "<unpitched>" in text


def test_gp_export_relative_path(score, tmp_path, monkeypatch):
    # Regression: the Node exporter runs with cwd=exporter/, so a relative output path used to
    # resolve there and fail (found in the first E2E run on a real song).
    monkeypatch.chdir(tmp_path)
    report = export_gp(score, Path("out") / "song.gp")
    assert report.verified and (tmp_path / "out" / "song.gp").is_file()
    assert read_gp(Path("out") / "song.gp")["tracks"]
