"""Guitar and bass tunings, and tuning suggestion from the lowest notes (spec 3.2)."""

from __future__ import annotations

from dataclasses import dataclass

from bandscribe.transcription.base import Note

MAX_FRET = 24


@dataclass(frozen=True)
class Tuning:
    key: str
    name: str
    instrument: str  # guitar | bass
    strings: tuple[int, ...]  # open-string MIDI pitches, string 1 (highest) first

    @property
    def n_strings(self) -> int:
        return len(self.strings)

    @property
    def lowest(self) -> int:
        return min(self.strings)

    @property
    def highest(self) -> int:
        return max(self.strings) + MAX_FRET

    def transposed(self, semitones: int, key: str, name: str) -> Tuning:
        return Tuning(key, name, self.instrument, tuple(s + semitones for s in self.strings))


_E_STD = (64, 59, 55, 50, 45, 40)
_BASS4 = (43, 38, 33, 28)

TUNINGS: dict[str, Tuning] = {t.key: t for t in (
    Tuning("guitar_standard", "Standard (E A D G B E)", "guitar", _E_STD),
    Tuning("guitar_eb", "Half step down (Eb)", "guitar", tuple(p - 1 for p in _E_STD)),
    Tuning("guitar_d", "D standard", "guitar", tuple(p - 2 for p in _E_STD)),
    Tuning("guitar_drop_d", "Drop D", "guitar", (64, 59, 55, 50, 45, 38)),
    Tuning("guitar_drop_db", "Drop C# / Db", "guitar", (63, 58, 54, 49, 44, 37)),
    Tuning("guitar_drop_c", "Drop C", "guitar", (62, 57, 53, 48, 43, 36)),
    Tuning("guitar_c", "C standard", "guitar", tuple(p - 4 for p in _E_STD)),
    Tuning("guitar_drop_b", "Drop B", "guitar", (61, 56, 52, 47, 42, 35)),
    Tuning("guitar7_standard", "7-string standard (B)", "guitar", (*_E_STD, 35)),
    Tuning("guitar7_drop_a", "7-string drop A", "guitar", (*_E_STD, 33)),
    Tuning("guitar8_standard", "8-string standard (F#)", "guitar", (*_E_STD, 35, 30)),
    Tuning("guitar8_drop_e", "8-string drop E", "guitar", (*_E_STD, 35, 28)),
    Tuning("bass_standard", "Bass standard (E A D G)", "bass", _BASS4),
    Tuning("bass_eb", "Bass half step down", "bass", tuple(p - 1 for p in _BASS4)),
    Tuning("bass_drop_d", "Bass drop D", "bass", (43, 38, 33, 26)),
    Tuning("bass_d", "Bass D standard", "bass", tuple(p - 2 for p in _BASS4)),
    Tuning("bass_drop_c", "Bass drop C", "bass", (41, 36, 31, 24)),
    Tuning("bass5_standard", "5-string bass (B E A D G)", "bass", (*_BASS4, 23)),
    Tuning("bass6_standard", "6-string bass (B E A D G C)", "bass", (48, *_BASS4, 23)),
)}

# Preference order when several tunings cover the material: most common first.
_PREFERENCE = {
    "guitar": ["guitar_standard", "guitar_eb", "guitar_drop_d", "guitar_d", "guitar_drop_db",
               "guitar_drop_c", "guitar_c", "guitar_drop_b", "guitar7_standard",
               "guitar7_drop_a", "guitar8_standard", "guitar8_drop_e"],
    "bass": ["bass_standard", "bass_eb", "bass_drop_d", "bass_d", "bass_drop_c",
             "bass5_standard", "bass6_standard"],
}


@dataclass(frozen=True)
class TuningSuggestion:
    tuning: Tuning
    lowest_pitch: int | None
    reason: str
    ambiguous_with: tuple[str, ...] = ()


def robust_lowest(notes: list[Note], min_count: int = 3, min_conf: float = 0.3) -> int | None:
    """Lowest pitch that occurs at least ``min_count`` times among reasonably confident notes.

    A single stray low note (octave error, bass leakage) must not retune the instrument.
    """
    counts: dict[int, int] = {}
    for n in notes:
        if n.conf >= min_conf:
            counts[n.pitch] = counts.get(n.pitch, 0) + 1
    frequent = [p for p, c in counts.items() if c >= min_count]
    if frequent:
        return min(frequent)
    return min(counts) if counts and len(notes) < min_count * 2 else None


def suggest_tuning(notes: list[Note], instrument: str) -> TuningSuggestion:
    """Suggest the most common tuning whose lowest string reaches the lowest played note.

    This is a suggestion, reported as such: several tunings share a lowest note (Drop D and
    D standard both reach D2), and pitch alone cannot tell them apart.
    """
    order = _PREFERENCE[instrument]
    default = TUNINGS[order[0]]
    low = robust_lowest(notes)
    if low is None:
        return TuningSuggestion(default, None, "no reliable low notes; assuming the default")
    if low >= default.lowest:
        return TuningSuggestion(default, low, f"lowest note {low} is within {default.name}")
    covering = [TUNINGS[k] for k in order if TUNINGS[k].lowest <= low]
    if not covering:
        deepest = min((TUNINGS[k] for k in order), key=lambda t: t.lowest)
        return TuningSuggestion(deepest, low, f"lowest note {low} is below every known tuning; "
                                "notes under the lowest string will be flagged")
    best_low = max(t.lowest for t in covering)
    exact = [t for t in covering if t.lowest == best_low]
    return TuningSuggestion(exact[0], low, f"lowest note {low} needs a string tuned to "
                            f"{best_low} or lower", tuple(t.key for t in exact[1:]))
