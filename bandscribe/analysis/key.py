"""Key estimation from transcribed notes with Krumhansl-Kessler profiles (spec 3.4).

This is the first-version method named in the spec and is unbenchmarked: the result carries
the correlation margin so callers can show it as a suggestion.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bandscribe.transcription.base import Note

MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
NAMES = ("C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B")
# Circle-of-fifths position (sharps positive) of each major tonic.
FIFTHS = {0: 0, 7: 1, 2: 2, 9: 3, 4: 4, 11: 5, 6: 6, 1: -5, 8: -4, 3: -3, 10: -2, 5: -1}


@dataclass(frozen=True)
class KeyEstimate:
    tonic: int  # pitch class 0..11
    mode: str  # major | minor
    correlation: float
    margin: float  # distance to the second-best key; small = ambiguous
    histogram: tuple[float, ...]

    @property
    def name(self) -> str:
        return f"{NAMES[self.tonic]} {self.mode}"

    @property
    def fifths(self) -> int:
        """Key signature as a number of sharps (negative = flats)."""
        major_tonic = self.tonic if self.mode == "major" else (self.tonic + 3) % 12
        return FIFTHS[major_tonic]


def pitch_class_histogram(notes: list[Note]) -> np.ndarray:
    """Duration- and confidence-weighted pitch-class histogram; drums are excluded."""
    hist = np.zeros(12)
    for n in notes:
        if n.instrument != "drums":
            hist[n.pitch % 12] += max(n.duration, 0.05) * n.conf
    return hist


def estimate_key(notes: list[Note]) -> KeyEstimate | None:
    hist = pitch_class_histogram(notes)
    if hist.sum() <= 0:
        return None
    scores = []
    for tonic in range(12):
        for mode, profile in (("major", MAJOR), ("minor", MINOR)):
            r = float(np.corrcoef(hist, np.roll(profile, tonic))[0, 1])
            scores.append((0.0 if np.isnan(r) else r, tonic, mode))
    scores.sort(reverse=True)
    best = scores[0]
    return KeyEstimate(best[1], best[2], best[0], best[0] - scores[1][0],
                       tuple(float(x) for x in hist / hist.sum()))
