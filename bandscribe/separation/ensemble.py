"""Combine separators and derive a per-stem, per-time confidence (spec 2.2, 5.8).

Rules enforced here:
- A stem named ``guitar`` (or any instrument stem) only ever comes from a model that outputs
  that stem. The residual ``other`` stem of any model is never renamed or merged into an
  instrument stem (spec 5.1).
- Stems shared by two models are blended with fixed weights; where the two models disagree,
  the confidence of that time window drops.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bandscribe.separation.base import INSTRUMENT_STEMS, RESIDUAL_STEM, SeparationResult, Stem

WINDOW_SECONDS = 1.0
AGREE_DB_LOW = 0.0  # agreement at or below this -> confidence 0
AGREE_DB_HIGH = 15.0  # agreement at or above this -> confidence 1
SILENCE_DB = -60.0


@dataclass(frozen=True)
class StemConfidence:
    """Confidence per ``window`` seconds. ``active`` is False where the stem is silent."""

    window: float
    values: np.ndarray  # in [0, 1]
    active: np.ndarray  # bool
    source: str  # how it was derived

    def at(self, t: float) -> float:
        if len(self.values) == 0:
            return 1.0
        return float(self.values[min(len(self.values) - 1, max(0, int(t / self.window)))])


@dataclass(frozen=True)
class EnsembleResult:
    stems: dict[str, Stem]
    confidence: dict[str, StemConfidence]
    weights: dict[str, dict[str, float]]  # stem -> model id -> weight
    results: tuple[SeparationResult, ...]


def _window_power(x: np.ndarray, hop: int) -> np.ndarray:
    n = x.shape[-1]
    frames = int(np.ceil(n / hop)) if n else 0
    out = np.zeros(frames)
    sq = np.mean(np.square(x, dtype=np.float64), axis=0)
    for i in range(frames):
        out[i] = sq[i * hop:(i + 1) * hop].mean()
    return out


def agreement_confidence(a: np.ndarray, b: np.ndarray, sr: int,
                         window: float = WINDOW_SECONDS) -> StemConfidence:
    """Confidence from how closely two estimates of the same stem agree.

    Per window: agreement = 10 log10(P(mean(a, b)) / P(a - b)), mapped linearly from
    [AGREE_DB_LOW, AGREE_DB_HIGH] dB to [0, 1]. Windows where both estimates are silent are
    inactive and keep confidence 1 (there is nothing to disagree about).
    """
    n = min(a.shape[-1], b.shape[-1])
    a, b = a[..., :n], b[..., :n]
    hop = max(1, int(window * sr))
    p_avg = _window_power((a + b) / 2, hop)
    p_diff = _window_power(a - b, hop)
    floor = 10 ** (SILENCE_DB / 10)
    active = np.maximum(_window_power(a, hop), _window_power(b, hop)) > floor
    db = 10 * np.log10(np.maximum(p_avg, 1e-20) / np.maximum(p_diff, 1e-20))
    conf = np.clip((db - AGREE_DB_LOW) / (AGREE_DB_HIGH - AGREE_DB_LOW), 0.0, 1.0)
    conf[~active] = 1.0
    return StemConfidence(window, conf, active, "model agreement")


def ensemble(primary: SeparationResult, secondary: SeparationResult | None = None,
             weights: dict[str, float] | None = None) -> EnsembleResult:
    """Blend ``secondary`` into ``primary`` for the instrument stems both models output.

    ``weights[stem]`` is the weight of the *secondary* model for that stem (default 0.5).
    The set of output stems is exactly the primary model's stems: the secondary model can
    refine a stem but can never introduce one, and its ``other`` is only used as a
    consistency check, never as audio for an instrument.
    """
    weights = weights or {}
    stems: dict[str, Stem] = {}
    conf: dict[str, StemConfidence] = {}
    used: dict[str, dict[str, float]] = {}
    for name, stem in primary.stems.items():
        other = secondary.stems.get(name) if secondary is not None else None
        if other is None or name == RESIDUAL_STEM or name not in INSTRUMENT_STEMS:
            stems[name] = stem
            used[name] = {primary.model_id: 1.0}
            continue
        w = float(weights.get(name, 0.5))
        if not 0.0 <= w <= 1.0:
            raise ValueError(f"weight for {name!r} must be in [0, 1]")
        n = min(stem.audio.shape[-1], other.audio.shape[-1])
        mixed = (1 - w) * stem.audio[..., :n] + w * other.audio[..., :n]
        label = f"ensemble:{primary.model_id}+{secondary.model_id}"
        stems[name] = Stem(name, mixed.astype(np.float32), stem.sr, label)
        used[name] = {primary.model_id: 1 - w, secondary.model_id: w}
        conf[name] = agreement_confidence(stem.audio, other.audio, stem.sr)

    # Group check: a 4-stem model's `other` should equal everything the 6-stem model did not
    # put into vocals / drums / bass. The agreement becomes the confidence of every stem in
    # that group (guitar, piano, other): it cannot say which of them is wrong, only where the
    # two models disagree about the non-rhythm-section content.
    if secondary is not None and RESIDUAL_STEM in secondary.stems:
        group = [n for n in primary.stems if n not in secondary.stems or n == RESIDUAL_STEM]
        if len(group) > 1:
            n = min(min(primary.stems[g].audio.shape[-1] for g in group),
                    secondary.stems[RESIDUAL_STEM].audio.shape[-1])
            total = sum(primary.stems[g].audio[..., :n] for g in group)
            group_conf = agreement_confidence(total, secondary.stems[RESIDUAL_STEM].audio[..., :n],
                                              primary.stems[group[0]].sr)
            for g in group:
                hop = max(1, int(group_conf.window * primary.stems[g].sr))
                floor = 10 ** (SILENCE_DB / 10)
                active = _window_power(primary.stems[g].audio[..., :n], hop) > floor
                conf[g] = StemConfidence(group_conf.window, group_conf.values, active,
                                         "group agreement (guitar+piano+other vs 4-stem other)")
    results = (primary,) if secondary is None else (primary, secondary)
    return EnsembleResult(stems, conf, used, results)
