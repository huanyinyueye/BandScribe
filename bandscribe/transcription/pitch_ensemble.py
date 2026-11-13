"""Pitch-shift test-time ensemble: transcribe pitch-shifted copies and keep agreed notes.

The stem is also transcribed after shifting it by a few semitones; those notes are
shifted back, pooled with the unshifted run, and a note is kept when at least
``min_votes`` runs found the same pitch within ``tolerance`` seconds. Notes keep the
unshifted run's timing when it found them.

Roselia 30 s guitar-stem clips, MuScriptor medium, raw note F1 (2026-11-12,
docs/e2e/2026-11-12-roselia/result.md): unshifted / shifts +-1 with 2 of 3 votes =
Neo-Aspect 0.571 / 0.610, Ringing Bloom 0.771 / 0.773. The same rule was the best
choice on either song and did not lose on the other.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from bandscribe.transcription.base import Note, sort_notes


def shifted_audio(audio: np.ndarray, sr: int, semitones: int) -> np.ndarray:
    import librosa

    return np.ascontiguousarray(
        librosa.effects.pitch_shift(np.asarray(audio, dtype=np.float32), sr=sr,
                                    n_steps=semitones), dtype=np.float32)


def vote(runs: dict[int, list[Note]], min_votes: int, tolerance: float = 0.04) -> list[Note]:
    """``runs`` maps a shift in semitones (0 = original) to notes already shifted back."""
    pool = sorted(((n, s) for s, notes in runs.items() for n in notes),
                  key=lambda x: (x[0].onset, x[0].pitch))
    used = [False] * len(pool)
    out: list[Note] = []
    for i, (note, shift) in enumerate(pool):
        if used[i]:
            continue
        members, seen = [i], {shift}
        for j in range(i + 1, len(pool)):
            other, other_shift = pool[j]
            if other.onset - note.onset > tolerance:
                break
            if not used[j] and other.pitch == note.pitch and other_shift not in seen:
                members.append(j)
                seen.add(other_shift)
        for j in members:
            used[j] = True
        if len(members) < min_votes:
            continue
        base = next((pool[j][0] for j in members if pool[j][1] == 0), None)
        if base is None:
            notes = [pool[j][0] for j in members]
            onset = float(np.median([n.onset for n in notes]))
            offset = max(onset, float(np.median([n.offset for n in notes])))
            base = replace(notes[0], onset=onset, offset=offset)
        out.append(replace(base, flags=(*base.flags, f"pitch-votes:{len(members)}")))
    return sort_notes(out)


def unshift(notes: list[Note], semitones: int) -> list[Note]:
    return [replace(n, pitch=n.pitch - semitones) for n in notes if 0 <= n.pitch - semitones <= 127]
