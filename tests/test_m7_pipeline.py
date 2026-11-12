import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from bandscribe import qa
from bandscribe.analysis.key import estimate_key
from bandscribe.analysis.meter import Bar
from bandscribe.analysis.sections import find_sections
from bandscribe.config import DEFAULTS_PATH, ConfigError, PipelineConfig, load_defaults
from bandscribe.notation.score_model import Score
from bandscribe.pipeline import notes_from_json, notes_to_json
from bandscribe.separation import get_separator
from bandscribe.transcription import TRANSCRIBER_IDS, get_transcriber
from bandscribe.transcription.base import Note

SR = 44100


# --- config --------------------------------------------------------------------------------

def test_defaults_file_matches_spec_assumptions():
    d = load_defaults("quality")
    assert d["separation"]["primary"] == "bs_roformer_sw"
    assert d["separation"]["secondary"] == "subroformer"  # measured, BENCHMARKS.md
    assert d["transcription"]["pitched"] == ["muscriptor_medium", "basic_pitch"]
    assert d["transcription"]["drums"] == ["muscriptor_medium", "adtof"]
    assert d["thresholds"]["low_confidence"] == 0.5


def test_profiles_override_defaults():
    q = PipelineConfig.from_defaults("quality")
    b = PipelineConfig.from_defaults("balanced")
    assert q.separator_secondary == "subroformer" and q.mix_vote == "muscriptor_medium"
    assert b.separator_secondary is None and b.mix_vote is None
    assert b.separator_primary == q.separator_primary


def test_config_reads_an_edited_defaults_file(tmp_path):
    raw = yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))
    raw["separation"]["primary"] = "htdemucs_6s"
    raw["thresholds"]["low_confidence"] = 0.3
    p = tmp_path / "defaults.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    cfg = PipelineConfig.from_defaults("quality", defaults_path=p)
    assert cfg.separator_primary == "htdemucs_6s" and cfg.low_confidence == 0.3


def test_config_overrides_and_validation():
    cfg = PipelineConfig.from_defaults(strict=True, guitar_tuning="guitar_drop_d", title=None)
    assert cfg.strict and cfg.guitar_tuning == "guitar_drop_d"
    with pytest.raises(ConfigError):
        PipelineConfig.from_defaults("turbo")
    with pytest.raises(ConfigError):
        PipelineConfig.from_defaults(nonsense=1)
    with pytest.raises(ConfigError):
        PipelineConfig.from_defaults(instruments=["banjo"])
    with pytest.raises(ConfigError):
        PipelineConfig.from_defaults(exports=["pdf"])
    assert json.dumps(cfg.to_dict())


def test_default_models_exist_in_registry():
    from bandscribe.models.registry import load_registry

    reg = load_registry()
    d = load_defaults()
    ids = [d["separation"]["primary"], d["separation"]["secondary"], d["beats"],
           *d["transcription"]["pitched"], *d["transcription"]["drums"]]
    assert all(i in reg.models for i in ids)
    # Spec 12.2: Mega 53 must not be a default.
    assert "mvsep_mega53" not in ids


# --- factories -----------------------------------------------------------------------------

def test_adapter_factories():
    for mid in ("bs_roformer_sw", "scnet_xl_ihf", "htdemucs_6s", "subroformer", "mvsep_mega53"):
        assert get_separator(mid).model_id == mid
    for mid in TRANSCRIBER_IDS:
        assert get_transcriber(mid).name == mid
    assert get_separator("mvsep_mega53").mixture_consistent is False
    with pytest.raises(KeyError):
        get_separator("nope")
    with pytest.raises(KeyError):
        get_transcriber("nope")


def test_backend_instrument_guards():
    with pytest.raises(ValueError):
        get_transcriber("basic_pitch").transcribe(np.zeros(100, np.float32), SR, "drums")
    with pytest.raises(ValueError):
        get_transcriber("adtof").transcribe(np.zeros(100, np.float32), SR, "guitar")


# --- key and sections ----------------------------------------------------------------------

def _scale(tonic, intervals, octaves=3):
    notes = []
    t = 0.0
    for _ in range(octaves):
        for k, iv in enumerate(intervals):
            dur = 1.0 if k in (0, 4) else 0.5  # weight tonic and dominant
            notes.append(Note(t, t + dur, 48 + tonic + iv, instrument="guitar"))
            t += dur
    return notes


def test_key_estimation():
    major = estimate_key(_scale(7, [0, 2, 4, 5, 7, 9, 11]))
    assert major.name == "G major" and major.fifths == 1
    minor = estimate_key(_scale(4, [0, 2, 3, 5, 7, 8, 10]))
    assert minor.name == "E minor" and minor.fifths == 1
    assert estimate_key([]) is None
    drums_only = [Note(0, 0.1, 36, instrument="drums")]
    assert estimate_key(drums_only) is None


def test_sections_found_at_a_clear_change():
    bars = [Bar(i, i * 2.0, i * 2.0 + 2.0, i * 4, 4, 4, True) for i in range(24)]
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=20), rng.normal(size=20)
    feats = np.array([a + rng.normal(scale=0.05, size=20) if i < 12 else
                      b + rng.normal(scale=0.05, size=20) for i in range(24)])
    sections = find_sections(feats, bars)
    assert [s.start_bar for s in sections] == [0, 12]
    assert [s.name for s in sections] == ["Section 1", "Section 2"]
    assert find_sections(np.zeros((0, 20)), []) == []


# --- QA hook and note JSON -----------------------------------------------------------------

def test_qa_hook_contract():
    """The pipeline only relies on this much of the QA result (docs/INTERFACES.md section 3)."""
    result = qa.run_checks(Score("x", [], []), {})
    assert {"status", "rules", "tracks"} <= set(result)
    result.pop("score", None)  # the only non-JSON key, interpreted by the pipeline
    assert json.dumps(result)


def test_notes_json_round_trip():
    notes = [Note(1.0, 1.5, 60, 90, 0.75, ("a", "b"), "guitar", 2, 1, ("single-source",))]
    data = notes_to_json(notes)
    assert set(data[0]) >= {"onset", "offset", "pitch", "velocity", "conf", "sources"}
    assert notes_from_json(json.loads(json.dumps(data))) == notes


# --- end to end on the real GPU ------------------------------------------------------------

def _band(seconds=16.0, bpm=120.0):
    """Kick/snare pattern + a bass line + a guitar-range melody, as stereo audio."""
    n = int(seconds * SR)
    t = np.arange(n) / SR
    out = np.zeros(n, dtype=np.float64)
    beat = 60.0 / bpm
    rng = np.random.default_rng(0)
    for k in range(int(seconds / beat)):
        s = int(k * beat * SR)
        m = min(n - s, int(0.12 * SR))
        tt = np.arange(m) / SR
        if k % 2 == 0:  # kick
            out[s:s + m] += 0.8 * np.sin(2 * np.pi * (120 * np.exp(-tt * 30) + 45) * tt) * np.exp(-tt * 25)
        else:  # snare
            out[s:s + m] += 0.5 * rng.standard_normal(m) * np.exp(-tt * 35)
        f_bass = 55.0 * 2 ** ([0, 0, 3, 5][k % 4] / 12)
        m2 = min(n - s, int(beat * 0.9 * SR))
        t2 = np.arange(m2) / SR
        out[s:s + m2] += 0.3 * np.sin(2 * np.pi * f_bass * t2) * np.exp(-t2 * 2)
        f_gtr = 220.0 * 2 ** ([0, 3, 7, 10][k % 4] / 12)
        saw = sum(np.sin(2 * np.pi * f_gtr * h * t2) / h for h in range(1, 6))
        out[s:s + m2] += 0.15 * saw * np.exp(-t2 * 3)
    del t
    out /= np.abs(out).max() * 1.1
    return np.stack([out, out], axis=1).astype(np.float32)


@pytest.mark.gpu
@pytest.mark.slow
def test_pipeline_end_to_end_output_structure(tmp_path):
    from bandscribe.notation.export_gp import read_gp
    from bandscribe.pipeline import STAGES, run_pipeline

    wav = tmp_path / "band.wav"
    sf.write(wav, _band(), SR, subtype="PCM_16")
    cfg = PipelineConfig.from_defaults(
        "balanced", pitched_transcribers=["basic_pitch"], drum_transcribers=["adtof"],
        title="Smoke")
    seen = []
    out = run_pipeline(wav, cfg, tmp_path / "out", lambda s, f, m: seen.append((s, f)))

    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "done"
    assert [s for s in STAGES if s in manifest["stages"]] == list(STAGES)
    assert {s for s, _ in seen} == set(STAGES) and all(0 <= f <= 1 for _, f in seen)
    sep = manifest["separation"]["bs_roformer_sw"]
    assert sep["device"] == "cuda" and 0 < sep["peak_vram_gb"] < 10 and sep["seconds"] > 0
    assert len(manifest["models"]["bs_roformer_sw"]["files_sha256"]["BS-Rofo-SW-Fixed.ckpt"]) == 64
    assert manifest["models"]["beat_this"]["sha256"] and manifest["models"]["basic_pitch"]["sha256"]
    assert all(r["leaked_bytes"] == 0 for r in manifest["vram"])
    assert manifest["qa"]["status"] in ("pass", "warn", "fail")
    assert json.loads((out / "qa.json").read_text(encoding="utf-8"))["rules"] == manifest["qa"]["rules"]

    stems = json.loads((out / "stems" / "stems.json").read_text(encoding="utf-8"))["stems"]
    assert set(stems) == {"vocals", "drums", "bass", "guitar", "piano", "other"}
    for name in stems:
        info = sf.info(out / "stems" / f"{name}.wav")
        assert info.samplerate == SR and info.channels == 2
    # The synthetic mix has no vocals / piano: those stems are silent and are not transcribed.
    assert stems["bass"]["activity"]["present"]
    assert not stems["vocals"]["activity"]["present"]
    assert not (out / "notes" / "final_vocals.json").exists()

    bass = json.loads((out / "notes" / "final_bass.json").read_text(encoding="utf-8"))
    assert len(bass) >= 16
    for n in bass[:5]:
        assert {"onset", "offset", "pitch", "velocity", "conf", "sources"} <= set(n)
        assert 0 <= n["conf"] <= 1 and n["sources"] == ["basic_pitch"]
        assert n["string"] in (1, 2, 3, 4) and 0 <= n["fret"] <= 24
    assert (out / "notes" / "raw_basic_pitch_bass.json").is_file()
    assert (out / "notes" / "attribution.json").is_file()
    # Spec 5.1: nothing from the `other` stem became a guitar note (no mix transcription ran).
    guitar_path = out / "notes" / "final_guitar.json"
    if guitar_path.is_file():
        assert not any("from-other-stem" in n["flags"]
                       for n in json.loads(guitar_path.read_text(encoding="utf-8")))

    score = Score.load(out / "score.json")
    assert score.title == "Smoke" and score.note_count() > 0
    analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["global_bpm"] == pytest.approx(120, abs=3)

    gp = manifest["exports"]["gp"]
    assert gp["verified"] and gp["mismatches"] == []
    summary = read_gp(out / "exports" / "Smoke.gp")
    assert len(summary["masterBars"]) == len(score.bars)
    assert sum(len(t["notes"]) for t in summary["tracks"]) == gp["notes_written"]
    for ext in ("gp5", "mid", "musicxml"):
        assert (out / "exports" / f"Smoke.{ext}").stat().st_size > 200

    # Second run reuses the cached stems.
    out2 = run_pipeline(wav, cfg, tmp_path / "out")
    m2 = json.loads((out2 / "manifest.json").read_text(encoding="utf-8"))
    assert m2["separation"].get("cached") is True and m2["status"] == "done"


GTZAN_CLIP = Path.home() / ".cache/bandscribe/datasets/gtzan/genres/metal/metal.00010.wav"


@pytest.mark.gpu
@pytest.mark.slow
@pytest.mark.skipif(not GTZAN_CLIP.is_file(), reason="GTZAN clip not downloaded")
def test_pipeline_on_real_band_audio(tmp_path):
    """30 s of real metal through the default drum / pitched paths (MuScriptor left out)."""
    from bandscribe.pipeline import run_pipeline

    cfg = PipelineConfig.from_defaults(
        "quality", pitched_transcribers=["basic_pitch"], drum_transcribers=["adtof"],
        mix_vote=None, title="Metal clip")
    out = run_pipeline(GTZAN_CLIP, cfg, tmp_path / "out")
    manifest = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "done"
    assert set(manifest["separation"]) == {"bs_roformer_sw", "scnet_xl_ihf"}
    for sep in manifest["separation"].values():
        assert sep["device"] == "cuda" and sep["peak_vram_gb"] < 10 and sep["oom_retries"] == 0
    tracks = manifest["summary"]["tracks"]
    assert tracks["guitar"] > 100 and tracks["bass"] > 50 and tracks["drums"] > 100
    stems = json.loads((out / "stems" / "stems.json").read_text(encoding="utf-8"))["stems"]
    assert stems["drums"]["confidence"]["source"] == "model agreement"
    assert stems["guitar"]["model_id"] == "bs_roformer_sw"  # never blended with a 4-stem `other`
    assert stems["drums"]["model_id"] == "ensemble:bs_roformer_sw+scnet_xl_ihf"
    assert manifest["exports"]["gp"]["verified"]
    score = Score.load(out / "score.json")
    guitar = next(t for t in score.tracks if t.family == "guitar")
    assert guitar.kind == "tab" and all(n.string and n.fret is not None for n in guitar.notes)


def test_guitar_dp_weights_come_from_defaults_and_can_be_overridden(tmp_path):
    from bandscribe.tab.fret_dp import FretOptions

    cfg = PipelineConfig.from_defaults()
    assert cfg.guitar_dp == load_defaults()["tab"]["guitar_dp"]
    assert cfg.guitar_dp["open_bonus"] < 0  # tuned value: open strings cost a little
    opts = FretOptions(max_span=cfg.max_span, **cfg.guitar_dp)
    assert opts.open_bonus == cfg.guitar_dp["open_bonus"] and opts.max_span == cfg.max_span
    over = PipelineConfig.from_defaults(guitar_dp={"open_bonus": 0.5})
    assert over.guitar_dp == {"open_bonus": 0.5}
    with pytest.raises(ConfigError, match="unknown guitar_dp"):
        PipelineConfig.from_defaults(guitar_dp={"max_fret": 12})
    with pytest.raises(ConfigError, match="must be a number"):
        PipelineConfig.from_defaults(guitar_dp={"open_bonus": "low"})
    # A defaults file without the key keeps the code defaults.
    raw = yaml.safe_load(DEFAULTS_PATH.read_text(encoding="utf-8"))
    raw["tab"].pop("guitar_dp")
    p = tmp_path / "defaults.yaml"
    p.write_text(yaml.safe_dump(raw), encoding="utf-8")
    assert PipelineConfig.from_defaults(defaults_path=p).guitar_dp == {}


def test_tab_stage_uses_tuned_weights_for_guitar_only(tmp_path):
    from bandscribe.attribution.instrument_id import AttributionLog
    from bandscribe.pipeline import _assign_tab, _Run

    def riff(inst, low):
        return [Note(i * 0.25, i * 0.25 + 0.2, low + 24, instrument=inst) for i in range(8)]

    cfg = PipelineConfig.from_defaults(guitar_dp={"open_bonus": -0.75, "high_fret_weight": 0.0})
    run = _Run(tmp_path / "x.wav", cfg, tmp_path, None, None)
    final = {"guitar": riff("guitar", 40), "bass": riff("bass", 19)}  # E4 and G2: open strings
    info = _assign_tab(run, final, AttributionLog())
    assert info["guitar"]["dp_weights"] == cfg.guitar_dp
    assert info["bass"]["dp_weights"] == "code defaults"
    assert all(n.fret > 0 for n in final["guitar"])   # penalty: avoids the open E string
    assert all(n.fret == 0 for n in final["bass"])    # bass keeps the open-string bonus
