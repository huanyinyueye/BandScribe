"""MuScriptor multi-instrument transcription. Weights: CC BY-NC 4.0 (gated on Hugging Face)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from bandscribe.models.registry import sha256_file
from bandscribe.runtime.vram import ModelSlot, gpu_slot
from bandscribe.transcription.base import (
    Note,
    Transcriber,
    TranscriberUnavailable,
    sort_notes,
    to_mono,
)

# MuScriptor instrument group -> BandScribe family.
GROUP_FAMILY = {
    "acoustic_guitar": "guitar", "clean_electric_guitar": "guitar",
    "distorted_electric_guitar": "guitar",
    "acoustic_bass": "bass", "electric_bass": "bass",
    "acoustic_piano": "piano", "electric_piano": "piano",
    "voice": "vocals", "drums": "drums",
}
# Family -> groups the model may emit when transcribing that family's stem (hard constraint).
FAMILY_GROUPS: dict[str, list[str]] = {}
for _group, _family in GROUP_FAMILY.items():
    FAMILY_GROUPS.setdefault(_family, []).append(_group)

DRUM_HIT_DURATION = 0.05


def family_of(group: str) -> str:
    return GROUP_FAMILY.get(group, "other")


class MuScriptorTranscriber(Transcriber):
    def __init__(self, size: str = "medium", slot: ModelSlot = gpu_slot) -> None:
        if size not in ("small", "medium", "large"):
            raise ValueError("size must be small, medium or large")
        self.size = size
        self.name = f"muscriptor_{size}"
        self.slot = slot
        self.model_sha256: str | None = None
        self.unavailable_reason: str | None = None

    def _loader(self):
        from muscriptor import TranscriptionModel
        from muscriptor.utils.download import ModelDownloadError

        device = "cuda" if self.slot.backend.available() else "cpu"
        try:
            return TranscriptionModel.load_model(self.size, device=device)
        except ModelDownloadError as exc:
            self.unavailable_reason = str(exc)
            raise TranscriberUnavailable(str(exc)) from exc

    def _record_hash(self) -> None:
        if self.model_sha256 is not None:
            return
        try:
            from huggingface_hub import try_to_load_from_cache

            path = try_to_load_from_cache(f"MuScriptor/muscriptor-{self.size}", "model.safetensors")
            if isinstance(path, str) and Path(path).is_file():
                self.model_sha256 = sha256_file(Path(path))
        except Exception:  # noqa: BLE001 - the hash is informational
            self.model_sha256 = None

    def transcribe_many(self, jobs: list[tuple[np.ndarray, int, str]]) -> list[list[Note]]:
        """Transcribe several (audio, sr, instrument) jobs with a single model load."""
        import torch
        from muscriptor import NoteEndEvent

        results: list[list[Note]] = []
        with self.slot.load(self.name, self._loader) as model:
            self._record_hash()
            for audio, sr, instrument in jobs:
                groups = None if instrument == "mix" else FAMILY_GROUPS.get(instrument)
                if instrument not in ("mix", *FAMILY_GROUPS):
                    groups = None  # e.g. the `other` stem: let the model name what it hears
                wav = torch.from_numpy(to_mono(audio))
                notes: list[Note] = []
                for ev in model.transcribe((wav, sr), instruments=groups):
                    if not isinstance(ev, NoteEndEvent):
                        continue
                    start = ev.start_event
                    family = family_of(start.instrument)
                    if instrument not in ("mix", "other") and family != instrument:
                        continue  # cannot happen with the hard constraint; kept as a guard
                    end = max(ev.end_time, start.start_time)
                    if family == "drums":
                        end = start.start_time + DRUM_HIT_DURATION
                    notes.append(Note(float(start.start_time), float(end), int(start.pitch),
                                      sources=(self.name,), instrument=family,
                                      flags=(f"mus:{start.instrument}",)))
                results.append(sort_notes(notes))
            del model
        return results

    def transcribe(self, audio: np.ndarray, sr: int, instrument: str) -> list[Note]:
        return self.transcribe_many([(audio, sr, instrument)])[0]
