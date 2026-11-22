"""BS-RoFormer separation through audio.cpp (no torch).

The mix goes to audio.cpp as a float32 wav and the stems come back the same way. Numbers on
the whole Neo-Aspect song vs the torch separator: SI-SDR 52-60 dB on every stem, 2.6x faster
on an RTX 4070 (docs/e2e/2026-11-22-audiocpp-gpu/result.md).
"""

from __future__ import annotations

import json
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from bandscribe.models.prepare import PKG, audiocpp_bs_roformer
from bandscribe.runtime import audiocpp
from bandscribe.runtime.vram import ModelSlot, gpu_slot
from bandscribe.separation.base import ProgressFn, SeparationResult, Separator, Stem

SR = 44100
MODELS = ("bs_roformer_sw",)  # converted configs live in models/audiocpp/<id>.json


class AudiocppSeparator(Separator):
    def __init__(self, model_id: str, slot: ModelSlot = gpu_slot) -> None:
        if model_id not in MODELS:
            raise KeyError(f"no audio.cpp separator for {model_id!r}")
        self.model_id = model_id
        self.slot = slot
        cfg = json.loads((PKG / "audiocpp" / f"{model_id}.json").read_text(encoding="utf-8"))
        self.stem_names = tuple(cfg["instruments"])
        self.chunk_size = int(cfg["chunk_size"])

    def separate(self, mix: np.ndarray, sr: int = SR,
                 progress: ProgressFn | None = None) -> SeparationResult:
        if sr != SR:
            raise ValueError(f"separators expect {SR} Hz audio, got {sr}")
        if mix.ndim != 2 or mix.shape[0] != 2:
            raise ValueError("mix must have shape (2, n_samples)")
        model_dir, shas = audiocpp_bs_roformer(self.model_id)
        if progress is not None:
            progress(0.0)
        with tempfile.TemporaryDirectory(prefix="bandscribe-sep-") as tmp:
            tmp_dir = Path(tmp)
            sf.write(tmp_dir / "mix.wav", mix.T, SR, subtype="FLOAT")
            started = time.perf_counter()
            with self.slot.load(self.model_id, lambda: None):
                ran = audiocpp.run("sep", "bs_roformer", model_dir,
                             ["--audio", str(tmp_dir / "mix.wav"), "--out-dir", str(tmp_dir),
                              "--out-format", "float32"], tmp_dir / "log.txt")
            seconds = time.perf_counter() - started
            stems = {}
            for name in self.stem_names:
                audio, file_sr = sf.read(tmp_dir / f"{name}.wav", dtype="float32",
                                         always_2d=True)
                if file_sr != SR or audio.shape[0] != mix.shape[1]:
                    raise RuntimeError(f"audio.cpp returned a malformed {name} stem")
                stems[name] = Stem(name, np.ascontiguousarray(audio.T), SR, self.model_id)
        if progress is not None:
            progress(1.0)
        return SeparationResult(
            stems=stems, model_id=self.model_id, files_sha256=shas,
            device=f"{ran['backend']} (audio.cpp)", batch_size=1,
            chunk_size=self.chunk_size, peak_vram_gb=0.0, seconds=seconds,
            notes=("engine: audio.cpp",
                   *([f"GPU run failed, used CPU ({ran['fallback']})"] if "fallback" in ran
                     else [])))
