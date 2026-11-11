"""Note model and Transcriber interface (spec section 4)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace

import numpy as np

# Instrument families used across the pipeline. `other` is a residual, never an instrument.
FAMILIES = ("vocals", "drums", "bass", "guitar", "piano", "other")

# Sounding MIDI range accepted per family before tuning-specific checks (spec 5.4).
FAMILY_RANGE = {
    "guitar": (26, 88),   # D1 (8-string, drop tunings) .. E6 (24th fret, high E)
    "bass": (19, 72),     # G0 (6-string, low tunings) .. C5
    "piano": (21, 108),
    "vocals": (36, 96),
    "other": (12, 120),
}


class TranscriberUnavailable(RuntimeError):
    """The backend cannot run on this machine (missing weights, missing login...).

    The message is meant for the user. The pipeline records it in the QA report and carries
    on with the remaining backends; it never substitutes invented notes.
    """


@dataclass(frozen=True)
class Note:
    onset: float  # seconds
    offset: float  # seconds
    pitch: int  # MIDI pitch; for drums a General MIDI percussion key
    velocity: int = 80  # 1..127
    conf: float = 1.0  # 0..1
    sources: tuple[str, ...] = ()
    instrument: str = "other"  # one of FAMILIES
    string: int | None = None  # 1 = highest string, set by TAB assignment
    fret: int | None = None
    flags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.offset < self.onset:
            raise ValueError("note offset before onset")
        if not 0.0 <= self.conf <= 1.0:
            raise ValueError(f"conf {self.conf} outside [0, 1]")
        if self.instrument not in FAMILIES:
            raise ValueError(f"unknown instrument family {self.instrument!r}")

    @property
    def duration(self) -> float:
        return self.offset - self.onset

    def with_conf(self, conf: float) -> Note:
        return replace(self, conf=float(min(1.0, max(0.0, conf))))

    def flagged(self, *flags: str) -> Note:
        return replace(self, flags=tuple(dict.fromkeys((*self.flags, *flags))))


class Transcriber(ABC):
    name: str

    @abstractmethod
    def transcribe(self, audio: np.ndarray, sr: int, instrument: str) -> list[Note]:
        """Transcribe ``audio`` (mono or (2, n)) assuming it contains ``instrument``.

        ``instrument`` is one of FAMILIES, or ``"mix"`` for an unconditioned transcription of
        the full mix, in which case each note carries the family the model detected.
        """


def to_mono(audio: np.ndarray) -> np.ndarray:
    audio = np.asarray(audio, dtype=np.float32)
    return audio.mean(axis=0) if audio.ndim == 2 else audio


def sort_notes(notes: list[Note]) -> list[Note]:
    return sorted(notes, key=lambda n: (n.onset, n.pitch, n.offset))
