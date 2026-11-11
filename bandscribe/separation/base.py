"""Separator interface (spec section 4)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

ProgressFn = Callable[[float], None]

# Stems that are a specific instrument. `other` is a residual and is never an instrument
# (spec 5.1: other must never be treated as guitar).
INSTRUMENT_STEMS = ("vocals", "drums", "bass", "guitar", "piano")
RESIDUAL_STEM = "other"


@dataclass(frozen=True)
class Stem:
    name: str
    audio: np.ndarray  # float32, shape (2, n_samples)
    sr: int
    model_id: str  # the model (or "ensemble:<a>+<b>") that produced it

    def rms(self) -> float:
        return float(np.sqrt(np.mean(np.square(self.audio, dtype=np.float64)))) if self.audio.size else 0.0


@dataclass(frozen=True)
class SeparationResult:
    stems: dict[str, Stem]
    model_id: str
    files_sha256: dict[str, str]
    device: str
    batch_size: int
    chunk_size: int
    peak_vram_gb: float
    seconds: float
    oom_retries: int = 0
    notes: tuple[str, ...] = field(default_factory=tuple)


class Separator(ABC):
    model_id: str
    stem_names: tuple[str, ...]
    mixture_consistent: bool = True  # False when stems do not sum to the mix (Mega 53)

    @abstractmethod
    def separate(self, mix: np.ndarray, sr: int, progress: ProgressFn | None = None) -> SeparationResult:
        """Separate a (2, n) float32 mix sampled at 44.1 kHz."""
