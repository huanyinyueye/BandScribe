from fractions import Fraction as F

import numpy as np
import pytest

from bandscribe.analysis.meter import analyze_meter
from bandscribe.analysis.sections import Section
from bandscribe.analysis.tempo import build_tempo_map
from bandscribe.notation.layout import layout_track
from bandscribe.notation.quantize import build_bars, quantize_notes
from bandscribe.notation.score_model import Score, Track
from bandscribe.transcription.base import Note


def grid(counts, period=0.5, start=1.0):
    n = sum(counts)
    beats = start + np.arange(n) * period
    idx = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return beats, beats[idx]


def setup(counts=(4, 4, 4, 4), period=0.5, start=1.0, first=None, last=None):
    beats, downs = grid(counts, period, start)
    tm = build_tempo_map(beats)
    meter = analyze_meter(beats, downs)
    bars = build_bars(tm, meter, first if first is not None else start,
                      last if last is not None else beats[-1])
    return tm, bars


def N(onset, pitch=60, dur=0.25, conf=0.9, **kw):
    return Note(onset, onset + dur, pitch, conf=conf, instrument="guitar", **kw)


def track_of(notes, bars, tm, kind="staff", **kw):
    sn, g, _ = quantize_notes(notes, bars, tm, drums=kind == "drums", **kw)
    return Track("t", "T", "guitar" if kind != "drums" else "drums", kind, notes=sn, grid=g)


# --- bars ----------------------------------------------------------------------------------

def test_bars_follow_the_meter_and_tempo():
    tm, bars = setup((4, 4, 3, 3, 4))
    assert [(b.numerator, b.denominator) for b in bars] == [(4, 4), (4, 4), (3, 4), (3, 4), (4, 4)]
    assert [b.start_q for b in bars] == [0, 4, 8, 11, 14]
    assert all(b.bpm == pytest.approx(120.0) for b in bars)
    assert bars[0].start_time == pytest.approx(1.0) and bars[2].start_beat == 8


def test_pickup_bar_before_first_downbeat():
    beats = 1.0 + np.arange(18) * 0.5
    tm = build_tempo_map(beats)
    bars = build_bars(tm, analyze_meter(beats, beats[2::4]), 1.0, beats[-1])
    assert (bars[0].numerator, bars[0].start_beat) == (2, 0)  # two-beat pickup
    assert bars[1].numerator == 4 and bars[1].start_beat == 2


def test_notes_before_first_beat_get_a_leading_bar():
    tm, bars = setup(first=0.2)  # audio starts 0.8 s before the first tracked beat
    assert bars[0].start_beat == -2 and bars[0].numerator == 2
    sn, _, _ = quantize_notes([N(0.5)], bars, tm)
    assert (sn[0].bar, sn[0].pos_q) == (0, F(1))


def test_bars_extend_to_cover_the_last_note_and_no_downbeats_fallback():
    tm, bars = setup(last=12.0)
    assert bars[-1].end_time >= 12.0 and all(b.numerator == 4 for b in bars)
    beats = np.arange(8) * 0.5
    tm2 = build_tempo_map(beats)
    bars2 = build_bars(tm2, analyze_meter(beats, np.array([])), 0.0, 3.5)
    assert [(b.numerator, b.denominator) for b in bars2] == [(4, 4), (4, 4)]


def test_sections_attach_to_nearest_bar():
    beats, downs = grid((4, 4, 4, 4))
    tm = build_tempo_map(beats)
    bars = build_bars(tm, analyze_meter(beats, downs), 1.0, beats[-1],
                      [Section("Section 1", 0, 1.0), Section("Section 2", 2, 5.0)])
    assert [b.section for b in bars] == ["Section 1", None, "Section 2", None]


# --- quantisation --------------------------------------------------------------------------

def test_sixteenths_snap_to_grid():
    tm, bars = setup()
    onsets = [1.0, 1.13, 1.26, 1.37, 1.5]  # roughly 16ths at 120 BPM, with jitter
    sn, g, stats = quantize_notes([N(t, 60 + i) for i, t in enumerate(onsets)], bars, tm)
    assert [n.pos_q for n in sn] == [F(0), F(1, 4), F(1, 2), F(3, 4), F(1)]
    assert g[(0, 0)] == 4 and stats.max_shift_ms < 15 and stats.notes_out == 5


def test_triplets_are_detected_per_beat():
    tm, bars = setup()
    trip = [1.0, 1.0 + 0.5 / 3, 1.0 + 1.0 / 3]  # triplet in beat 1
    straight = [1.5, 1.75]  # eighths in beat 2
    sn, g, _ = quantize_notes([N(t, 60 + i, dur=0.1) for i, t in enumerate(trip + straight)], bars, tm)
    assert [n.pos_q for n in sn] == [F(0), F(1, 3), F(2, 3), F(1), F(3, 2)]
    assert g[(0, 0)] == 3 and g[(0, 1)] == 4


def test_quantize_keeps_every_note_identity_and_never_adds():
    rng = np.random.default_rng(1)
    tm, bars = setup((4,) * 8)
    notes = [N(float(t), int(p), dur=0.2) for t, p in
             zip(rng.uniform(1.0, 16.0, 120), rng.integers(40, 80, 120), strict=True)]
    sn, _, stats = quantize_notes(notes, bars, tm)
    assert len(sn) + stats.duplicates_removed == len(notes)
    originals = {(round(n.onset, 4), n.pitch) for n in notes}
    assert all((round(n.onset, 4), n.pitch) in originals for n in sn)
    for n in sn:
        assert 0 <= n.pos_q < bars[n.bar].length_q and n.dur_q > 0
        assert bars[n.bar].start_q + n.pos_q + n.dur_q <= bars[-1].end_q


def test_low_confidence_marked_and_duplicates_keep_best():
    tm, bars = setup()
    sn, _, stats = quantize_notes([N(1.0, 60, conf=0.3), N(1.01, 60, conf=0.8), N(2.0, 62, conf=0.4)],
                                  bars, tm, low_confidence=0.5)
    assert stats.duplicates_removed == 1 and len(sn) == 2
    assert sn[0].conf == 0.8 and not sn[0].low_conf and sn[1].low_conf


def test_seven_eight_bar_positions():
    tm, bars = setup((7, 7), period=0.25)
    assert (bars[0].numerator, bars[0].denominator, bars[0].length_q) == (7, 8, F(7, 2))
    sn, _, _ = quantize_notes([N(1.0 + 0.25 * k, 60 + k, dur=0.1) for k in range(7)], bars, tm)
    assert [n.pos_q for n in sn] == [F(k, 2) for k in range(7)]


def test_quantize_empty():
    tm, bars = setup()
    assert quantize_notes([], bars, tm)[0] == []


# --- layout --------------------------------------------------------------------------------

def bar_lengths_ok(layout, bars):
    for beats, bar in zip(layout, bars, strict=True):
        assert sum((b.length_q for b in beats), F(0)) == bar.length_q


def desc(beats):
    return [(("r" if b.rest else "n") + ("~" if b.tied else ""), b.duration, b.dots,
             b.tuplet) for b in beats]


def test_empty_track_is_whole_bar_rests():
    tm, bars = setup((4, 3, 7), period=0.5)
    layout = layout_track(Track("t", "T", "guitar", "staff"), bars)
    bar_lengths_ok(layout, bars)
    assert desc(layout[0]) == [("r", 1, 0, None)]
    assert desc(layout[1]) == [("r", 2, 1, None)]  # dotted half in 3/4


def test_quarter_notes_and_rests():
    tm, bars = setup()
    t = track_of([N(1.0, dur=0.5), N(2.0, dur=0.5)], bars, tm)
    layout = layout_track(t, bars)
    bar_lengths_ok(layout, bars)
    assert desc(layout[0]) == [("n", 4, 0, None), ("r", 4, 0, None), ("n", 4, 0, None),
                               ("r", 4, 0, None)]


def test_long_note_is_tied_across_the_barline():
    tm, bars = setup()
    t = track_of([N(2.0, dur=2.5)], bars, tm)  # starts on beat 3, lasts 5 beats
    layout = layout_track(t, bars)
    bar_lengths_ok(layout, bars)
    assert desc(layout[0]) == [("r", 2, 0, None), ("n", 2, 0, None)]
    assert desc(layout[1])[0] == ("n~", 2, 1, None) or desc(layout[1])[:2] == [("n~", 2, 0, None), ("n~", 4, 0, None)]
    assert all(b.notes[0].pitch == 60 for b in layout[1] if not b.rest)


def test_sixteenth_patterns_and_dotted_values():
    tm, bars = setup()
    t = track_of([N(1.0, 60, dur=0.375), N(1.375, 62, dur=0.125)], bars, tm)  # dotted 8th + 16th
    layout = layout_track(t, bars)
    bar_lengths_ok(layout, bars)
    assert desc(layout[0])[:2] == [("n", 8, 1, None), ("n", 16, 0, None)]


def test_triplet_layout():
    tm, bars = setup()
    onsets = [1.0, 1.0 + 0.5 / 3, 1.0 + 1.0 / 3]
    t = track_of([N(x, 60 + i, dur=0.16) for i, x in enumerate(onsets)], bars, tm)
    layout = layout_track(t, bars)
    bar_lengths_ok(layout, bars)
    assert desc(layout[0])[:3] == [("n", 8, 0, (3, 2))] * 3


def test_chord_is_one_beat_and_low_conf_gets_text():
    tm, bars = setup()
    t = track_of([N(1.0, 60, conf=0.9), N(1.0, 64, conf=0.2), N(1.5, 67, conf=0.9)], bars, tm)
    layout = layout_track(t, bars)
    assert len(layout[0][0].notes) == 2 and layout[0][0].text == "?"
    assert layout[0][1].text is None


def test_drum_hits_do_not_ring_past_their_beat():
    tm, bars = setup()
    t = track_of([Note(1.0, 1.05, 36, instrument="drums")], bars, tm, kind="drums")
    layout = layout_track(t, bars)
    bar_lengths_ok(layout, bars)
    assert desc(layout[0])[0][0] == "n" and layout[0][0].length_q <= 1
    assert all(b.rest for b in layout[0][1:])


@pytest.mark.parametrize("counts,period", [((4,) * 6, 0.5), ((3,) * 6, 0.5), ((7,) * 4, 0.25),
                                            ((6,) * 4, 0.25), ((5,) * 4, 0.5), ((4, 7, 4, 3, 5), 0.5)])
def test_random_material_always_fills_bars_exactly(counts, period):
    rng = np.random.default_rng(sum(counts))
    tm, bars = setup(counts, period)
    span = sum(counts) * period
    notes = [N(float(t), int(p), dur=float(d)) for t, p, d in
             zip(rng.uniform(1.0, 1.0 + span, 150), rng.integers(40, 80, 150),
                 rng.uniform(0.05, 1.5, 150), strict=True)]
    t = track_of(notes, bars, tm)
    layout = layout_track(t, bars)
    bar_lengths_ok(layout, bars)
    # Every quantised note appears exactly once as an untied note.
    first = [n for beats in layout for b in beats if not b.tied for n in b.notes]
    assert len(first) == len(t.notes)
    for beats in layout:
        for b in beats:
            assert b.duration in (1, 2, 4, 8, 16, 32, 64) and b.dots in (0, 1)


# --- score JSON ----------------------------------------------------------------------------

def test_score_json_round_trip(tmp_path):
    tm, bars = setup()
    t = track_of([N(1.0, 60, string=2, fret=1), N(1.0 + 0.5 / 3, 62)], bars, tm, kind="tab")
    t.tuning, t.tuning_key = (64, 59, 55, 50, 45, 40), "guitar_standard"
    score = Score("Song", bars, [t], key={"name": "C major"}, meta={"x": 1})
    path = tmp_path / "score.json"
    score.save(path)
    back = Score.load(path)
    assert back.to_dict() == score.to_dict()
    assert back.tracks[0].notes[1].pos_q == F(1, 3) and back.tracks[0].grid == t.grid
    assert back.note_count() == 2
