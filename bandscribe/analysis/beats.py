"""Beat and downbeat tracking with Beat This! (no DBN post-processing)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bandscribe.io.audio_load import Audio
from bandscribe.models.download import ensure_model
from bandscribe.runtime.vram import ModelSlot, gpu_slot

MODEL_ID = "beat_this"


@dataclass(frozen=True)
class BeatResult:
    beats: np.ndarray  # seconds
    downbeats: np.ndarray  # seconds
    model_id: str
    model_sha256: str
    device: str


class BeatTracker:
    """Keeps the model loaded for a batch of files; use as a context manager."""

    def __init__(self, slot: ModelSlot = gpu_slot, device: str | None = None) -> None:
        self.slot = slot
        self.device = device or ("cuda" if slot.backend.available() else "cpu")
        (self._path, self.sha256), = ensure_model(MODEL_ID).values()
        self._cm = None
        self._model = None

    def __enter__(self) -> BeatTracker:
        from beat_this.inference import Audio2Beats

        self._cm = self.slot.load(MODEL_ID, lambda: Audio2Beats(
            str(self._path), device=self.device, float16=self.device == "cuda", dbn=False))
        self._model = self._cm.__enter__()
        return self

    def __exit__(self, *exc) -> None:
        self._model = None
        self._cm.__exit__(*exc)

    def track(self, audio: Audio | np.ndarray, sr: int | None = None) -> BeatResult:
        if isinstance(audio, Audio):
            signal, sr = audio.mono(), audio.sr
        else:
            signal = np.asarray(audio, dtype=np.float32)
            if sr is None:
                raise ValueError("sr is required when passing a raw array")
        beats, downbeats = self._model(signal, sr)
        return BeatResult(np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float),
                          MODEL_ID, self.sha256, self.device)


def track_beats(audio: Audio | np.ndarray, sr: int | None = None, *,
                slot: ModelSlot = gpu_slot, device: str | None = None) -> BeatResult:
    """Track beats on a mono mixdown, loading and releasing the model around the call."""
    with BeatTracker(slot, device) as tracker:
        return tracker.track(audio, sr)
