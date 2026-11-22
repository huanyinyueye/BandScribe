"""Beat and downbeat tracking with Beat This! (no DBN post-processing).

Runs on ONNX Runtime (DirectML GPU, else CPU) unless BANDSCRIBE_ENGINE=torch. The ONNX graph
gives the same beats, downbeats and bars as torch FP32 on Neo-Aspect
(docs/e2e/2026-11-22-audiocpp-gpu/result.md).
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np

from bandscribe.io.audio_load import Audio
from bandscribe.models.download import ensure_model
from bandscribe.runtime.audiocpp import ENV_ENGINE
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
        self.onnx = os.environ.get(ENV_ENGINE, "auto").strip().lower() != "torch"
        self.sha256 = ""
        self._device_override = device
        self.device = device or "cpu"
        self._cm = None
        self._model = None

    def _load(self):
        if self.onnx:
            from bandscribe.analysis.beat_onnx import OnnxBeatThis
            from bandscribe.models.prepare import beat_this_onnx

            graph, self.sha256 = beat_this_onnx()
            model = OnnxBeatThis(graph, None if self._device_override != "cpu"
                                 else ["CPUExecutionProvider"])
            self.device = f"{model.device} (onnx)"
            return model
        from beat_this.inference import Audio2Beats

        (path, self.sha256), = ensure_model(MODEL_ID).values()
        self.device = self._device_override or (
            "cuda" if self.slot.backend.available() else "cpu")
        return Audio2Beats(str(path), device=self.device, float16=self.device == "cuda",
                           dbn=False)

    def __enter__(self) -> BeatTracker:
        self._cm = self.slot.load(MODEL_ID, self._load)
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
        if self.onnx:
            self.device = f"{self._model.device} (onnx)"  # CPU if the GPU run failed
        return BeatResult(np.asarray(beats, dtype=float), np.asarray(downbeats, dtype=float),
                          MODEL_ID, self.sha256, self.device)


def track_beats(audio: Audio | np.ndarray, sr: int | None = None, *,
                slot: ModelSlot = gpu_slot, device: str | None = None) -> BeatResult:
    """Track beats on a mono mixdown, loading and releasing the model around the call."""
    with BeatTracker(slot, device) as tracker:
        return tracker.track(audio, sr)
