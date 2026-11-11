"""ADTOF Frame_RNN drum transcription (ONNX). Weights: CC BY-NC-SA 4.0, non-commercial."""

from __future__ import annotations

import time

import numpy as np

from bandscribe.models.download import ensure_model
from bandscribe.transcription.base import Note, Transcriber, sort_notes, to_mono

MODEL_ID = "adtof"
SR = 44100
FPS = 100
# Output classes in order, the General MIDI key each maps to, and the authors' thresholds.
CLASSES = (("kick", 36, 0.22), ("snare", 38, 0.24), ("tom", 47, 0.32),
           ("hihat", 42, 0.22), ("cymbal", 49, 0.30))
HIT_DURATION = 0.05


def _moving(x: np.ndarray, pre: int, post: int, fn) -> np.ndarray:
    n = len(x)
    out = np.empty(n)
    for i in range(n):
        out[i] = fn(x[max(0, i - pre):min(n, i + post + 1)])
    return out


def peak_pick(act: np.ndarray, threshold: float, *, fps: int = FPS, pre_avg: float = 0.1,
              post_avg: float = 0.01, pre_max: float = 0.02, post_max: float = 0.01,
              combine: float = 0.02) -> list[tuple[float, float]]:
    """madmom-style peak picking. Returns (time, activation) pairs."""
    if len(act) == 0:
        return []
    mov_max = _moving(act, round(pre_max * fps), round(post_max * fps), np.max)
    mov_avg = _moving(act, round(pre_avg * fps), round(post_avg * fps), np.mean)
    peaks = np.flatnonzero((act == mov_max) & (act >= mov_avg + threshold))
    picked: list[tuple[float, float]] = []
    for i in peaks:
        t = i / fps
        if picked and t - picked[-1][0] <= combine:
            if act[i] > picked[-1][1]:
                picked[-1] = (t, float(act[i]))
            continue
        picked.append((t, float(act[i])))
    return picked


class AdtofTranscriber(Transcriber):
    name = MODEL_ID

    def __init__(self) -> None:
        self._session = None
        self.model_sha256: str | None = None

    def _load(self):
        if self._session is None:
            import onnxruntime as ort

            (path, sha), = ensure_model(MODEL_ID).values()
            self.model_sha256 = sha
            opts = ort.SessionOptions()
            opts.log_severity_level = 3
            self._session = ort.InferenceSession(str(path), opts,
                                                 providers=["CPUExecutionProvider"])
        return self._session

    def activations(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if sr != SR:
            raise ValueError(f"ADTOF expects {SR} Hz audio")
        wave = np.clip(to_mono(audio), -1.0, 1.0)[None].astype(np.float32)
        return self._load().run(["activations"], {"waveform": wave})[0][0]

    def transcribe(self, audio: np.ndarray, sr: int, instrument: str = "drums") -> list[Note]:
        if instrument != "drums":
            raise ValueError("ADTOF only transcribes drums")
        started = time.perf_counter()
        act = self.activations(audio, sr)
        notes = []
        for k, (_label, pitch, threshold) in enumerate(CLASSES):
            for t, a in peak_pick(act[:, k], threshold):
                conf = float(min(1.0, max(0.0, a)))
                notes.append(Note(t, t + HIT_DURATION, pitch,
                                  velocity=int(min(127, max(30, round(40 + 87 * conf)))),
                                  conf=conf, sources=(MODEL_ID,), instrument="drums"))
        self.last_run = {"device": "cpu (onnxruntime)", "peak_vram_gb": 0.0, "oom_retries": 0,
                         "seconds": round(time.perf_counter() - started, 2), "jobs": 1}
        return sort_notes(notes)
