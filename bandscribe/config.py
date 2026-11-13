"""Pipeline configuration. Defaults come from ``models/defaults.yaml`` (see that file)."""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

DEFAULTS_PATH = Path(__file__).parent / "models" / "defaults.yaml"
FAMILIES = ("vocals", "drums", "bass", "guitar", "piano", "other")
# FretOptions fields that tab.guitar_dp may set (max_span has its own key).
GUITAR_DP_KEYS = {"open_bonus", "high_fret_weight", "span_weight", "move_weight",
                  "string_jump_weight", "pause_reset"}


class ConfigError(ValueError):
    pass


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_defaults(profile: str = "quality", path: Path | None = None) -> dict[str, Any]:
    """Defaults for ``profile`` as a plain dict (profile overrides already applied)."""
    raw = yaml.safe_load((path or DEFAULTS_PATH).read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != 1:
        raise ConfigError("defaults.yaml: schema_version must be 1")
    profiles = raw.get("profiles") or {}
    if profile not in profiles:
        raise ConfigError(f"unknown profile {profile!r}; known: {sorted(profiles)}")
    base = {k: v for k, v in raw.items() if k not in ("profiles", "schema_version")}
    return _merge(base, profiles[profile] or {})


@dataclass
class PipelineConfig:
    profile: str = "quality"
    separator_primary: str = "bs_roformer_sw"
    separator_secondary: str | None = "scnet_xl_ihf"
    blend_weights: dict[str, float] = field(default_factory=dict)
    pitched_transcribers: list[str] = field(default_factory=lambda: ["basic_pitch"])
    drum_transcribers: list[str] = field(default_factory=lambda: ["adtof"])
    mix_vote: str | None = None
    # Pitch-shift test-time ensemble for MuScriptor: {shifts, min_votes, families}; {} = off.
    pitch_ensemble: dict[str, Any] = field(default_factory=dict)
    # Families whose MuScriptor power chords are written root + fifth (octave dropped).
    power_chord_families: list[str] = field(default_factory=list)
    beat_model: str = "beat_this"
    low_confidence: float = 0.5
    onset_tolerance: float = 0.05
    drum_onset_tolerance: float = 0.03
    strict: bool = False  # remove low-confidence notes instead of marking them
    max_span: int = 5
    # Cost weights of the guitar fret-assignment DP (FretOptions fields); empty = code defaults.
    # Bass always uses the code defaults.
    guitar_dp: dict[str, float] = field(default_factory=dict)
    guitar_tuning: str = "auto"
    bass_tuning: str = "auto"
    instruments: list[str] = field(default_factory=lambda: list(FAMILIES))
    exports: list[str] = field(default_factory=lambda: ["gp", "gp5", "midi", "musicxml"])
    title: str | None = None
    reuse_cache: bool = True  # reuse stems / notes already present in the output directory
    # Manual corrections from the GUI: stem name -> "absent" (do not transcribe) ...
    stem_overrides: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_defaults(cls, profile: str = "quality", defaults_path: Path | None = None,
                      **overrides: Any) -> PipelineConfig:
        d = load_defaults(profile, defaults_path)
        sep, tr, th, tab = d["separation"], d["transcription"], d["thresholds"], d["tab"]
        cfg = cls(
            profile=profile,
            separator_primary=sep["primary"], separator_secondary=sep.get("secondary"),
            blend_weights=dict(sep.get("blend_weights") or {}),
            pitched_transcribers=list(tr["pitched"]), drum_transcribers=list(tr["drums"]),
            mix_vote=tr.get("mix_vote"), pitch_ensemble=dict(tr.get("pitch_ensemble") or {}),
            power_chord_families=list(tr.get("power_chord_families") or []),
            beat_model=d["beats"],
            low_confidence=float(th["low_confidence"]),
            onset_tolerance=float(th["onset_tolerance"]),
            drum_onset_tolerance=float(th["drum_onset_tolerance"]),
            max_span=int(tab["max_span"]), guitar_dp=dict(tab.get("guitar_dp") or {}),
            guitar_tuning=tab["guitar_tuning"],
            bass_tuning=tab["bass_tuning"], instruments=list(d["instruments"]))
        for k, v in overrides.items():
            if not hasattr(cfg, k):
                raise ConfigError(f"unknown config field {k!r}")
            if v is not None:
                setattr(cfg, k, v)
        cfg.validate()
        return cfg

    def validate(self) -> None:
        bad = set(self.instruments) - set(FAMILIES)
        if bad:
            raise ConfigError(f"unknown instruments {sorted(bad)}")
        if not 0.0 <= self.low_confidence <= 1.0:
            raise ConfigError("low_confidence must be in [0, 1]")
        bad = set(self.exports) - {"gp", "gp5", "midi", "musicxml"}
        if bad:
            raise ConfigError(f"unknown export formats {sorted(bad)}")
        unknown = set(self.guitar_dp) - GUITAR_DP_KEYS
        if unknown:
            raise ConfigError(f"unknown guitar_dp weights {sorted(unknown)}; "
                              f"allowed: {sorted(GUITAR_DP_KEYS)}")
        for key, value in self.guitar_dp.items():
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ConfigError(f"guitar_dp.{key} must be a number")
        if self.pitch_ensemble:
            shifts = self.pitch_ensemble.get("shifts") or []
            votes = self.pitch_ensemble.get("min_votes")
            if (not shifts or any(isinstance(s, bool) or not isinstance(s, int) or s == 0
                                  for s in shifts)):
                raise ConfigError("pitch_ensemble.shifts must be non-zero semitone integers")
            if not isinstance(votes, int) or not 1 <= votes <= len(shifts) + 1:
                raise ConfigError("pitch_ensemble.min_votes must be between 1 and runs")
            bad = set(self.pitch_ensemble.get("families") or []) - set(FAMILIES)
            if bad:
                raise ConfigError(f"unknown pitch_ensemble families {sorted(bad)}")
        bad = set(self.power_chord_families) - set(FAMILIES)
        if bad:
            raise ConfigError(f"unknown power_chord_families {sorted(bad)}")
        if not self.pitched_transcribers:
            raise ConfigError("at least one pitched transcriber is required")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
