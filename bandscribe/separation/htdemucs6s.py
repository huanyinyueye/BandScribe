"""HTDemucs 6s baseline separator (demucs v4)."""

from __future__ import annotations

import time

import numpy as np
import torch

from bandscribe.models.download import ensure_model
from bandscribe.runtime.vram import ModelSlot, gpu_slot, run_with_oom_backoff
from bandscribe.separation.base import ProgressFn, SeparationResult, Separator, Stem

SR = 44100
MODEL_ID = "htdemucs_6s"


class HTDemucs6sSeparator(Separator):
    model_id = MODEL_ID
    stem_names = ("drums", "bass", "other", "vocals", "guitar", "piano")

    def __init__(self, slot: ModelSlot = gpu_slot, overlap: float = 0.25, shifts: int = 1) -> None:
        self.slot = slot
        self.overlap = overlap
        self.shifts = shifts

    def separate(self, mix: np.ndarray, sr: int = SR,
                 progress: ProgressFn | None = None) -> SeparationResult:
        from demucs.apply import apply_model
        from demucs.states import load_model

        if sr != SR:
            raise ValueError(f"separators expect {SR} Hz audio, got {sr}")
        files = ensure_model(MODEL_ID)
        (path, sha), = files.values()
        backend = self.slot.backend
        started = time.perf_counter()
        peak = 0

        def loader():
            # demucs checkpoints pickle their own classes, so weights_only loading is not
            # possible; the file is trusted only because its sha256 is pinned in the registry.
            return load_model(torch.load(path, map_location="cpu", weights_only=False)).eval()

        wav = torch.from_numpy(np.ascontiguousarray(mix, dtype=np.float32))
        ref = wav.mean(0)
        mean, std = ref.mean(), ref.std().clamp_min(1e-8)
        with self.slot.load(MODEL_ID, loader) as model:
            names = tuple(model.sources)

            def attempt(_size: int, device: str) -> np.ndarray:
                nonlocal peak
                backend.reset_peak()
                with torch.no_grad():
                    out = apply_model(model, ((wav - mean) / std)[None],  # noqa: F821
                                      device=device,
                                      shifts=self.shifts, split=True, overlap=self.overlap,
                                      progress=False)[0]
                peak = backend.peak_bytes()
                return (out * std + mean).cpu().numpy()

            res = run_with_oom_backoff(attempt, [1], backend=backend)
            del model
        if progress is not None:
            progress(1.0)
        stems = {n: Stem(n, res.value[i], SR, MODEL_ID) for i, n in enumerate(names)}
        return SeparationResult(stems=stems, model_id=MODEL_ID, files_sha256={path.name: sha},
                                device=res.device, batch_size=1, chunk_size=0,
                                peak_vram_gb=peak / 1024**3,
                                seconds=time.perf_counter() - started, oom_retries=res.attempts)
