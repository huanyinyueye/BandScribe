"""Write distorted power chords as root + fifth.

MuScriptor adds the octave of the root to almost every distorted power chord: the octave
is the root's strongest overtone, so root + fifth sounds like root + fifth + octave. When
the lowest note of a chord has both its fifth and its octave struck with it, the octave is
dropped. An octave dyad (no fifth) is left alone.

Roselia guitar stems, whole-song raw note F1 vs the reference GP after the pitch
ensemble (2026-11-13, docs/e2e/2026-11-12-roselia/result.md): Neo-Aspect 0.625 -> 0.681,
Ringing Bloom 0.768 -> 0.812. Of four rule variants this one was the best on both songs.
This is a notation convention: a three-string power chord is also written as two notes.
"""

from __future__ import annotations

from bandscribe.transcription.base import Note, sort_notes


def drop_root_octaves(notes: list[Note], tolerance: float = 0.035) -> tuple[list[Note], list[Note]]:
    """Return (kept, dropped). Notes within ``tolerance`` of a chord's first onset form the chord."""
    kept: list[Note] = []
    dropped: list[Note] = []
    chord: list[Note] = []

    def flush() -> None:
        pitches = {n.pitch for n in chord}
        root = min(pitches, default=0)
        for n in chord:
            is_octave = n.pitch == root + 12 and root + 7 in pitches
            (dropped if is_octave else kept).append(n)

    for note in sorted(notes, key=lambda n: n.onset):
        if chord and note.onset - chord[0].onset > tolerance:
            flush()
            chord = []
        chord.append(note)
    flush()
    return sort_notes(kept), sort_notes(dropped)
