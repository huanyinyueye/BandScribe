"""QA rules (spec section 5). Owner: QA session."""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace

import numpy as np

from bandscribe.attribution.instrument_id import AttributionLog
from bandscribe.notation.score_model import BarSpec, Score, ScoreNote, Track
from bandscribe.qa import run_checks
from bandscribe.separation.ensemble import StemConfidence
from bandscribe.transcription.base import Note

STD = (64, 59, 55, 50, 45, 40)
BASS = (43, 38, 33, 28)


def bars(n=4):
    return [BarSpec(i, 4, 4, i * 4, Fraction(4 * i), 120.0, 2.0 * i, 2.0 * (i + 1))
            for i in range(n)]


def sn(onset, pitch, conf=0.9, string=None, fret=None, flags=(), low=None, sources=("m",)):
    bar = int(onset // 2)
    return ScoreNote(bar=bar, pos_q=Fraction(0), dur_q=Fraction(1), pitch=pitch, conf=conf,
                     low_conf=(conf < 0.5) if low is None else low, sources=sources,
                     string=string, fret=fret, flags=tuple(flags), onset=onset,
                     offset=onset + 0.4)


def final_notes(score):
    out = {}
    for t in score.tracks:
        out.setdefault(t.family, []).extend(
            Note(n.onset, n.offset, n.pitch, conf=n.conf, sources=n.sources,
                 instrument=t.family) for n in t.notes)
    return out


def ctx(score, **kw):
    c = {"config": SimpleNamespace(low_confidence=0.5, strict=False, max_span=5),
         "notes": final_notes(score), "attribution": AttributionLog(), "manifest": {}}
    c.update(kw)
    return c


def guitar(notes):
    return Track("g", "Guitar (combined)", "guitar", "tab", tuning=STD, notes=notes)


def rule_counts(res, rid):
    return res["rules"][rid]


def test_clean_score_passes_hard_rules():
    s = Score("t", bars(), [guitar([sn(0.0, 45, string=5, fret=0), sn(0.5, 47, string=5,
                                                                        fret=2)])])
    res = run_checks(s, ctx(s, manifest={"models": {"m": {"version": "1", "sha256": "x"}}}))
    assert res["status"] == "pass"
    assert res["removed"] == 0 and "score" not in res


def test_other_stem_note_not_promoted_is_removed():
    """Regression for spec 5.1: other must never become guitar without mix confirmation."""
    bad = sn(0.0, 52, conf=0.4, string=4, fret=2, flags=("from-other-stem",))
    s = Score("t", bars(), [guitar([bad])])
    res = run_checks(s, ctx(s))
    assert rule_counts(res, "R1")["errors"] == 1
    assert res["status"] == "fail"
    assert res["score"].tracks[0].notes == []


def test_promoted_other_note_is_kept_but_conf_capped():
    ok = sn(0.0, 52, conf=0.4, string=4, fret=2, flags=("from-other-stem",))
    log = AttributionLog()
    log.promoted.append({"to": "guitar", "onset": 0.0, "pitch": 52})
    s = Score("t", bars(), [guitar([ok])])
    assert rule_counts(run_checks(s, ctx(s, attribution=log)), "R1")["errors"] == 0
    hot = sn(0.0, 52, conf=0.9, string=4, fret=2, flags=("from-other-stem",))
    s2 = Score("t", bars(), [guitar([hot])])
    assert rule_counts(run_checks(s2, ctx(s2, attribution=log)), "R1")["errors"] == 1


def test_untraceable_note_is_removed():
    real = sn(0.0, 45, string=5, fret=0)
    s = Score("t", bars(), [guitar([real])])
    c = ctx(s)
    invented = sn(1.0, 47, string=5, fret=2)
    s.tracks[0].notes.append(invented)
    res = run_checks(s, c)
    assert rule_counts(res, "R2")["errors"] == 1
    assert [n.pitch for n in res["score"].tracks[0].notes] == [45]


def test_strict_mode_removes_low_confidence():
    s = Score("t", bars(), [guitar([sn(0.0, 45, conf=0.3, string=5, fret=0)])])
    c = ctx(s)
    c["config"].strict = True
    res = run_checks(s, c)
    assert res["removed"] == 1


def test_low_conf_flag_mismatch_warns():
    s = Score("t", bars(), [guitar([sn(0.0, 45, conf=0.3, string=5, fret=0, low=False)])])
    assert rule_counts(run_checks(s, ctx(s)), "R3")["warnings"] == 1


def test_tab_arithmetic_and_range():
    s = Score("t", bars(), [guitar([sn(0.0, 46, string=5, fret=0),  # 45 + 0 != 46
                                    sn(0.5, 90, string=1, fret=26)])])
    res = run_checks(s, ctx(s))
    assert rule_counts(res, "R4")["errors"] >= 2


def test_playability_same_string_and_span():
    s = Score("t", bars(), [guitar([sn(0.0, 47, string=5, fret=2), sn(0.01, 48, string=5, fret=3),
                                    sn(1.0, 41, string=6, fret=1), sn(1.0, 59, string=3,
                                                                       fret=4 + 0)])])
    s.tracks[0].notes[-1] = sn(1.0, 64, string=3, fret=9)  # span 8 > 5
    res = run_checks(s, ctx(s))
    assert rule_counts(res, "R5")["warnings"] == 2


def test_bass_polyphony_flagged():
    t = Track("b", "Bass", "bass", "tab", tuning=BASS,
              notes=[sn(0.0, 28, string=4, fret=0), sn(0.0, 33, string=3, fret=0)])
    s = Score("t", bars(), [t])
    assert rule_counts(run_checks(s, ctx(s)), "R6")["warnings"] == 1


def test_limb_overflow_flagged():
    t = Track("d", "Drums", "drums", "drums",
              notes=[sn(0.0, p) for p in (36, 38, 42, 49, 47)])
    s = Score("t", bars(), [t])
    assert rule_counts(run_checks(s, ctx(s)), "R7")["warnings"] == 1


def test_separation_disagreement_flagged():
    s = Score("t", bars(), [guitar([sn(0.1, 45, conf=0.9, string=5, fret=0)])])
    conf = {"guitar": StemConfidence(1.0, np.array([0.2, 0.9]), np.array([True, True]), "x")}
    assert rule_counts(run_checks(s, ctx(s, confidence=conf)), "R8")["warnings"] == 1


def test_provenance_missing_hash_warns():
    s = Score("t", bars(), [guitar([])])
    res = run_checks(s, ctx(s, manifest={"models": {"m": {"version": "1"}}}))
    assert rule_counts(res, "R9")["warnings"] == 1


def test_report_html_written(tmp_path):
    s = Score("Song", bars(), [guitar([sn(0.0, 45, conf=0.3, string=5, fret=0)])])
    res = run_checks(s, ctx(s, out_dir=tmp_path))
    html = (tmp_path / "report.html").read_text(encoding="utf-8")
    assert "QA report" in html and "Guitar (combined)" in html
    assert res["tracks"]["g"]["low_conf_ratio"] == 1.0
