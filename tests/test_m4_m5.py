import numpy as np
import pytest

from bandscribe.attribution.instrument_id import (
    AttributionLog,
    analyze_stems,
    apply_range,
    apply_stem_confidence,
    assert_no_unsupported_other_to_guitar,
    drop_inactive,
    promote_from_other,
    resolve_leakage,
    vote_with_mix,
)
from bandscribe.separation.base import Stem
from bandscribe.separation.ensemble import StemConfidence
from bandscribe.tab.fret_dp import DPFretAssigner, FretOptions, chord_fingerings, group_chords
from bandscribe.tab.tunings import TUNINGS, suggest_tuning
from bandscribe.transcription.adtof import peak_pick
from bandscribe.transcription.base import Note
from bandscribe.transcription.consensus import consensus, match_notes
from bandscribe.transcription.drums import flag_limb_overload, vote_drums

SR = 44100


def N(onset, pitch, conf=0.9, inst="guitar", src="a", dur=0.25, **kw):
    return Note(onset, onset + dur, pitch, conf=conf, sources=(src,), instrument=inst, **kw)


def tone_stem(name, seconds=6.0, amp=0.2, silent_until=0.0):
    t = np.arange(int(seconds * SR)) / SR
    x = (amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32)
    x[: int(silent_until * SR)] = 0
    return Stem(name, np.stack([x, x]), SR, "test")


# --- Note model ----------------------------------------------------------------------------

def test_note_validation():
    with pytest.raises(ValueError):
        Note(1.0, 0.5, 60)
    with pytest.raises(ValueError):
        Note(0, 1, 60, conf=1.5)
    with pytest.raises(ValueError):
        Note(0, 1, 60, instrument="banjo")
    n = N(0, 60).with_conf(7).flagged("x", "x", "y")
    assert n.conf == 1.0 and n.flags == ("x", "y")


# --- consensus -----------------------------------------------------------------------------

def test_match_notes_needs_same_pitch_and_close_onset():
    a = [N(1.00, 60), N(2.00, 62), N(3.00, 64)]
    b = [N(1.03, 60, src="b"), N(2.00, 63, src="b"), N(3.20, 64, src="b")]
    assert match_notes(a, b) == [(0, 0)]


def test_match_is_one_to_one_and_prefers_closest():
    a = [N(1.00, 60), N(1.04, 60)]
    b = [N(1.03, 60, src="b")]
    assert match_notes(a, b) == [(1, 0)]


def test_consensus_confidence_bands_and_sources():
    a = [N(1.0, 60, conf=1.0), N(2.0, 62, conf=1.0)]
    b = [N(1.01, 60, conf=0.5, src="b"), N(3.0, 65, conf=1.0, src="b")]
    out, stats = consensus(a, b)
    by_pitch = {n.pitch: n for n in out}
    assert by_pitch[60].conf == pytest.approx(0.6 + 0.4 * 0.75)
    assert by_pitch[60].sources == ("a", "b") and by_pitch[60].onset == 1.0
    assert by_pitch[62].conf == pytest.approx(0.45) and "single-source" in by_pitch[62].flags
    assert by_pitch[65].conf == pytest.approx(0.35)
    assert (stats.matched, stats.primary_only, stats.secondary_only) == (1, 1, 1)
    # Every single-source note is below the default 0.5 threshold.
    assert all(n.conf < 0.5 for n in out if "single-source" in n.flags)


def test_consensus_never_invents_notes():
    rng = np.random.default_rng(0)
    a = [N(float(t), int(p)) for t, p in zip(rng.uniform(0, 30, 80), rng.integers(40, 80, 80), strict=True)]
    b = [N(float(t), int(p), src="b") for t, p in zip(rng.uniform(0, 30, 80), rng.integers(40, 80, 80), strict=True)]
    out, _ = consensus(a, b)
    inputs = {(n.onset, n.pitch) for n in a + b}
    assert all((n.onset, n.pitch) in inputs for n in out)
    assert len(out) <= len(a) + len(b)


def test_consensus_strict_drops_unconfirmed_and_single_backend_mode():
    a = [N(1.0, 60), N(2.0, 62)]
    out, _ = consensus(a, [N(1.0, 60, src="b")], drop_unconfirmed=True)
    assert [n.pitch for n in out] == [60]
    solo, stats = consensus(a, None)
    assert all("no-second-opinion" in n.flags and n.conf == pytest.approx(0.72) for n in solo)
    assert stats.agreement == 0.0


# --- attribution ---------------------------------------------------------------------------

def test_silent_stem_is_not_present():
    mix = tone_stem("mix").audio
    act = analyze_stems({"guitar": tone_stem("guitar"),
                         "piano": Stem("piano", np.zeros((2, 6 * SR), np.float32), SR, "t"),
                         "bass": tone_stem("bass", amp=0.0005)}, mix)
    assert act["guitar"].present
    assert not act["piano"].present and act["piano"].active_seconds == 0
    assert not act["bass"].present  # -52 dB relative to the mix


def test_notes_in_silent_region_are_removed_and_logged():
    act = analyze_stems({"guitar": tone_stem("guitar", silent_until=3.0)}, tone_stem("m").audio)["guitar"]
    log = AttributionLog()
    kept = drop_inactive([N(1.0, 60), N(4.0, 62)], act, log)
    assert [n.pitch for n in kept] == [62]
    assert log.removed[0]["reason"] == "stem silent at onset"


def test_range_filter_logs_removals():
    log = AttributionLog()
    kept = apply_range([N(0, 10, inst="bass"), N(1, 40, inst="bass"), N(2, 90, inst="bass")], "bass", log)
    assert [n.pitch for n in kept] == [40] and len(log.removed) == 2


def test_separation_disagreement_lowers_confidence():
    conf = StemConfidence(1.0, np.array([1.0, 0.0, 0.6]), np.array([True, True, True]), "t")
    out = apply_stem_confidence([N(0.5, 60, conf=0.8), N(1.5, 60, conf=0.8), N(2.5, 60, conf=0.8)], conf)
    assert [round(n.conf, 2) for n in out] == [0.8, 0.4, 0.64]
    assert "separation-disagreement" in out[1].flags and not out[0].flags


def test_leakage_keeps_note_on_the_louder_stem():
    stems = {"guitar": tone_stem("guitar", amp=0.3), "piano": tone_stem("piano", amp=0.01)}
    log = AttributionLog()
    out = resolve_leakage({"guitar": [N(1.0, 60), N(2.0, 64)],
                           "piano": [N(1.01, 60, inst="piano"), N(3.0, 72, inst="piano")]}, stems, log)
    assert [n.pitch for n in out["guitar"]] == [60, 64]
    assert [n.pitch for n in out["piano"]] == [72]
    assert "leakage" in log.removed[0]["reason"]


def test_mix_vote():
    mix = [N(1.0, 60, inst="guitar", src="muscriptor_medium"),
           N(2.0, 62, inst="piano", src="muscriptor_medium")]
    out = vote_with_mix([N(1.0, 60, conf=0.8), N(2.0, 62, conf=0.8), N(3.0, 64, conf=0.8)],
                        "guitar", mix, AttributionLog())
    assert "mix-confirmed" in out[0].flags and out[0].conf == 0.8
    assert out[1].conf == pytest.approx(0.4) and "mix-disagrees:piano" in out[1].flags
    assert out[2].conf == 0.8 and not out[2].flags
    assert vote_with_mix(out, "guitar", None, AttributionLog()) == out


# --- hard rule 5.1: other is never guitar ---------------------------------------------------

def _other_notes():
    return [N(1.0, 60, inst="other", src="basic_pitch"), N(2.0, 64, inst="other", src="basic_pitch")]


def test_other_notes_stay_other_without_mix_transcription():
    act = analyze_stems({"other": tone_stem("other")}, tone_stem("m").audio)
    remaining, promoted = promote_from_other(_other_notes(), None, act, AttributionLog())
    assert promoted == {} and [n.instrument for n in remaining] == ["other", "other"]


def test_other_note_becomes_guitar_only_with_mix_support():
    act = analyze_stems({"other": tone_stem("other")}, tone_stem("m").audio)
    mix = [N(1.01, 60, inst="guitar", src="muscriptor_medium"),
           N(2.0, 64, inst="other", src="muscriptor_medium"),   # mix does not call it guitar
           N(5.0, 70, inst="guitar", src="muscriptor_medium")]  # not in the other stem at all
    log = AttributionLog()
    remaining, promoted = promote_from_other(_other_notes(), mix, act, log)
    assert [n.pitch for n in remaining] == [64]
    g = promoted["guitar"]
    assert [n.pitch for n in g] == [60]  # the pitch-70 mix note did not create a guitar note
    assert g[0].conf <= 0.5 and "from-other-stem" in g[0].flags
    assert "muscriptor_medium" in g[0].sources
    assert log.promoted == [{"to": "guitar", "onset": 1.0, "pitch": 60}]
    assert_no_unsupported_other_to_guitar(promoted)


def test_other_note_not_promoted_where_other_stem_is_silent():
    act = analyze_stems({"other": tone_stem("other", silent_until=3.0)}, tone_stem("m").audio)
    mix = [N(1.0, 60, inst="guitar", src="muscriptor_medium")]
    _, promoted = promote_from_other(_other_notes(), mix, act, AttributionLog())
    assert promoted == {}


def test_invariant_rejects_unsupported_promotion():
    bad = {"guitar": [N(1.0, 60, src="basic_pitch").flagged("from-other-stem")]}
    with pytest.raises(AssertionError):
        assert_no_unsupported_other_to_guitar(bad)
    assert_no_unsupported_other_to_guitar({"guitar": [N(1.0, 60)]})


# --- tunings -------------------------------------------------------------------------------

def _riff(low, n=6):
    return [N(i * 0.5, low + (i % 3) * 7) for i in range(n * 3)]


@pytest.mark.parametrize(("low", "expected"), [
    (40, "guitar_standard"), (45, "guitar_standard"), (39, "guitar_eb"), (38, "guitar_drop_d"),
    (37, "guitar_drop_db"), (36, "guitar_drop_c"), (35, "guitar_drop_b"), (33, "guitar7_drop_a"),
    (30, "guitar8_standard"), (28, "guitar8_drop_e"),
])
def test_guitar_tuning_from_lowest_note(low, expected):
    assert suggest_tuning(_riff(low), "guitar").tuning.key == expected


def test_tuning_ambiguity_is_reported():
    s = suggest_tuning(_riff(38), "guitar")
    assert s.tuning.key == "guitar_drop_d" and "guitar_d" in s.ambiguous_with


@pytest.mark.parametrize(("low", "expected"), [
    (28, "bass_standard"), (27, "bass_eb"), (26, "bass_drop_d"), (24, "bass_drop_c"),
    (23, "bass5_standard"),
])
def test_bass_tuning_from_lowest_note(low, expected):
    assert suggest_tuning([N(i * 0.5, low + (i % 2) * 5, inst="bass") for i in range(12)],
                          "bass").tuning.key == expected


def test_single_stray_low_note_does_not_retune():
    notes = _riff(40) + [N(9.0, 33)]
    assert suggest_tuning(notes, "guitar").tuning.key == "guitar_standard"


def test_tuning_default_when_no_notes():
    s = suggest_tuning([], "guitar")
    assert s.tuning.key == "guitar_standard" and s.lowest_pitch is None


def test_all_tunings_are_well_formed():
    for t in TUNINGS.values():
        assert list(t.strings) == sorted(t.strings, reverse=True) or t.key == "bass6_standard"
        assert t.n_strings in (4, 5, 6, 7, 8)


# --- TAB DP --------------------------------------------------------------------------------

STD = TUNINGS["guitar_standard"]


def _pitch_of(n, tuning=STD):
    return tuning.strings[n.string - 1] + n.fret


def test_tab_preserves_every_pitch_and_note():
    notes = [N(i * 0.25, p) for i, p in enumerate([40, 45, 50, 55, 59, 64, 67, 72, 76, 52, 47])]
    res = DPFretAssigner().assign(notes, STD)
    assert len(res.notes) == len(notes) and not res.unplaced
    for before, after in zip(notes, res.notes, strict=True):
        assert _pitch_of(after) == before.pitch and after.onset == before.onset
        assert 0 <= after.fret <= 24 and 1 <= after.string <= 6


def test_tab_open_e_major_chord():
    chord = [N(0.0, p) for p in (40, 47, 52, 56, 59, 64)]
    res = DPFretAssigner().assign(chord, STD)
    assert sorted((n.string, n.fret) for n in res.notes) == [(1, 0), (2, 0), (3, 1), (4, 2), (5, 2), (6, 0)]


def test_tab_chord_uses_distinct_strings_within_span():
    chord = [N(0.0, p) for p in (48, 55, 60, 64)]  # C major, several possible shapes
    res = DPFretAssigner(FretOptions(max_span=4)).assign(chord, STD)
    assert len({n.string for n in res.notes}) == 4
    fretted = [n.fret for n in res.notes if n.fret > 0]
    assert max(fretted) - min(fretted) <= 4


def test_tab_avoids_needless_position_jumps():
    # A scale run high on the neck should stay in one position rather than jump to open strings.
    notes = [N(i * 0.2, p) for i, p in enumerate([64, 66, 68, 69, 71, 73, 75, 76])]
    res = DPFretAssigner().assign(notes, STD)
    hands = [n.fret for n in res.notes if n.fret > 0]
    assert max(abs(a - b) for a, b in zip(hands, hands[1:], strict=False)) <= 5


def test_tab_drop_d_power_chord_is_one_fret():
    drop_d = TUNINGS["guitar_drop_d"]
    res = DPFretAssigner().assign([N(0.0, p) for p in (41, 48, 53)], drop_d)  # F5 power chord
    assert {n.fret for n in res.notes} == {3} and {n.string for n in res.notes} == {4, 5, 6}


def test_tab_unplayable_notes_are_flagged_not_transposed():
    notes = [N(0.0, 38), N(1.0, 52), N(2.0, 100)]  # D2 below standard tuning, C7 above fret 24
    res = DPFretAssigner().assign(notes, STD)
    assert [n.pitch for n in res.notes] == [52]
    assert sorted(n.pitch for n in res.unplaced) == [38, 100]
    assert all("unplayable" in n.flags and n.string is None for n in res.unplaced)


def test_tab_seven_note_chord_drops_least_confident():
    chord = [N(0.0, p, conf=0.9) for p in (40, 47, 52, 56, 59, 64)] + [N(0.0, 68, conf=0.2)]
    res = DPFretAssigner().assign(chord, STD)
    assert len(res.notes) == 6 and [n.pitch for n in res.unplaced] == [68]


def test_tab_bass_and_helpers():
    bass = TUNINGS["bass_standard"]
    res = DPFretAssigner().assign([N(i * 0.5, p, inst="bass") for i, p in enumerate([28, 33, 35, 40])], bass)
    assert [(n.string, n.fret) for n in res.notes][0] == (4, 0)
    assert chord_fingerings([28, 28], bass, FretOptions()) == []  # same pitch twice: one string
    assert [len(g) for g in group_chords([N(0, 60), N(0.01, 64), N(0.5, 67)])] == [2, 1]
    assert DPFretAssigner().assign([], bass).notes == []


# --- drums ---------------------------------------------------------------------------------

def D(onset, pitch, conf=0.9, src="mus"):
    return Note(onset, onset + 0.05, pitch, conf=conf, sources=(src,), instrument="drums")


def test_peak_picking_finds_isolated_hits():
    act = np.zeros(300)
    act[[50, 120, 121, 250]] = [0.9, 0.5, 0.8, 0.1]
    hits = peak_pick(act, 0.22)
    assert [round(t, 2) for t, _ in hits] == [0.5, 1.21]


def test_drum_vote_uses_detailed_kit_piece():
    detailed = [D(1.0, 46), D(2.0, 51), D(3.0, 36)]  # open hi-hat, ride, kick
    coarse = [D(1.01, 42, src="adtof"), D(2.0, 49, src="adtof"), D(4.0, 38, src="adtof")]
    out, stats, discarded = vote_drums(detailed, coarse)
    by_time = {round(n.onset): n for n in out}
    assert by_time[1].pitch == 46 and by_time[1].sources == ("mus", "adtof")  # stays open hi-hat
    assert by_time[2].pitch == 51 and by_time[2].conf >= 0.6
    assert "single-source" in by_time[3].flags and by_time[3].conf < 0.5
    assert by_time[4].pitch == 38 and by_time[4].conf < 0.5
    assert (stats.matched, stats.primary_only, stats.secondary_only, discarded) == (2, 1, 1, 0)


def test_drum_vote_single_backend_and_non_kit_sounds():
    out, stats, discarded = vote_drums(None, [D(1.0, 36, src="adtof"), D(2.0, 38, src="adtof")])
    assert len(out) == 2 and all("no-second-opinion" in n.flags for n in out)
    out, _, discarded = vote_drums([D(1.0, 36), D(1.5, 56)], None)  # 56 = cowbell: not in the kit map
    assert [n.pitch for n in out] == [36] and discarded == 1
    assert vote_drums(None, None)[0] == []


def test_drum_canonical_keys():
    out, _, _ = vote_drums([D(1.0, 35), D(2.0, 40), D(3.0, 41), D(4.0, 57)], None)
    assert [n.pitch for n in out] == [36, 38, 43, 49]


def test_limb_overload_flagged():
    ok = [D(1.0, 36), D(1.0, 42), D(1.0, 38)]  # kick + hat + snare
    assert not any(n.flags for n in flag_limb_overload(ok))
    five = [D(2.0, p) for p in (36, 38, 42, 49, 51)]
    assert all("limb-overload" in n.flags for n in flag_limb_overload(five))
    three_hands = [D(3.0, p) for p in (38, 42, 49)]
    assert all("limb-overload" in n.flags for n in flag_limb_overload(three_hands))
