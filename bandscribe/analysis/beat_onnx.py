"""Torch-free Beat This! inference: numpy log-mel + ONNX Runtime + upstream "minimal" peaks.

Mirrors beat_this.inference.Audio2Beats(dbn=False): the same spectrogram, 1500-frame chunks
with 6-frame borders (keep_first), max-pool peak picking at 50 fps and downbeats snapped to
the nearest beat. The ONNX graph comes from scripts/export_beat_this_onnx.py.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path

import numpy as np

SR = 22050
N_FFT = 1024
HOP = 441
FPS = 50
CHUNK = 1500
BORDER = 6


def _hz_to_mel(f: np.ndarray) -> np.ndarray:  # slaney
    f = np.asarray(f, dtype=np.float64)
    lin = f / (200.0 / 3)
    logstep = np.log(6.4) / 27.0
    return np.where(f >= 1000.0, 15.0 + np.log(np.maximum(f, 1e-10) / 1000.0) / logstep, lin)


def _mel_to_hz(m: np.ndarray) -> np.ndarray:  # slaney
    logstep = np.log(6.4) / 27.0
    return np.where(m >= 15.0, 1000.0 * np.exp(logstep * (m - 15.0)), m * (200.0 / 3))


@lru_cache(maxsize=1)
def _mel_fbank() -> np.ndarray:
    """torchaudio.functional.melscale_fbanks(513, 30, 11000, 128, 22050, None, "slaney")."""
    freqs = np.linspace(0, SR // 2, N_FFT // 2 + 1)
    m = np.linspace(_hz_to_mel(30.0), _hz_to_mel(11000.0), 128 + 2)
    f_pts = _mel_to_hz(m)
    f_diff = np.diff(f_pts)
    slopes = f_pts[None, :] - freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return np.maximum(0.0, np.minimum(down, up)).astype(np.float32)  # (513, 128)


def log_mel(signal: np.ndarray) -> np.ndarray:
    """(frames, 128) log-mel of a 22.05 kHz mono signal, as beat_this.preprocessing.LogMelSpect."""
    x = np.pad(np.asarray(signal, dtype=np.float32), N_FFT // 2, mode="reflect")
    window = (0.5 - 0.5 * np.cos(2 * np.pi * np.arange(N_FFT) / N_FFT)).astype(np.float32)
    frames = 1 + (len(x) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(frames)[:, None]
    mag = np.abs(np.fft.rfft(x[idx] * window, axis=1)).astype(np.float32)
    mag /= np.float32(np.sqrt(N_FFT))  # torchaudio normalized="frame_length"
    return np.log1p(1000.0 * (mag @ _mel_fbank())).astype(np.float32)


def _chunks(spect: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    n = len(spect)
    starts = np.arange(-BORDER, n - BORDER, CHUNK - 2 * BORDER)
    if n > CHUNK - 2 * BORDER:
        starts[-1] = n - (CHUNK - BORDER)
    out = []
    for s in starts:
        piece = spect[max(s, 0):min(s + CHUNK, n)]
        left, right = max(0, -s), max(0, min(BORDER, s + CHUNK - n))
        out.append(np.pad(piece, ((left, right), (0, 0))))
    return out, starts


def _dedup(frames: np.ndarray) -> np.ndarray:
    result: list[float] = []
    it = iter(int(f) for f in frames)
    try:
        p: float = next(it)
    except StopIteration:
        return np.array([])
    c = 1
    for p2 in it:
        if p2 - p <= 1:
            c += 1
            p += (p2 - p) / c
        else:
            result.append(p)
            p, c = p2, 1
    result.append(p)
    return np.array(result)


def _peaks(logits: np.ndarray) -> np.ndarray:
    padded = np.pad(logits, 3, constant_values=-np.inf)
    pooled = np.lib.stride_tricks.sliding_window_view(padded, 7).max(axis=1)
    return _dedup(np.flatnonzero((logits == pooled) & (logits > 0)))


def frames_to_beats(beat: np.ndarray, downbeat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    beat_t = _peaks(beat) / FPS
    down_t = _peaks(downbeat) / FPS
    if len(beat_t):
        down_t = np.array([beat_t[np.argmin(np.abs(beat_t - d))] for d in down_t])
    return beat_t, np.unique(down_t)


def build_weights(ckpt: str | Path, manifest_path: str | Path,
                  out_dir: str | Path | None = None) -> Path:
    """Write the external weights file of the ONNX graph from final0.ckpt, without torch.

    The manifest (from scripts/export_beat_this_onnx.py) gives each tensor's checkpoint key and
    byte range; the result must hash to the manifest's sha256. It is written to ``out_dir``
    (default: next to the manifest), which must also hold the graph.
    """
    from bandscribe.models.ckpt_reader import state_dict

    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    state = state_dict(ckpt)
    prefix = manifest.get("key_prefix", "")
    blob = bytearray(manifest["size_bytes"])
    for t in manifest["tensors"]:
        arr = state.get(prefix + t["key"], state.get(t["key"]))
        if arr is None or list(arr.shape) != t["shape"]:
            raise ValueError(f"checkpoint is missing {t['key']} {t['shape']}")
        raw = np.ascontiguousarray(arr.astype(t["dtype"], copy=False)).tobytes()
        blob[t["offset"]:t["offset"] + t["length"]] = raw
    if hashlib.sha256(blob).hexdigest() != manifest["sha256"]:
        raise ValueError("rebuilt Beat This! weights do not match the exported graph")
    out = Path(out_dir or manifest_path.parent) / manifest["weights"]
    tmp = out.with_suffix(".part")
    tmp.write_bytes(blob)
    tmp.replace(out)
    return out


def gpu_providers() -> list[str]:
    """DirectML (any DirectX 12 GPU) first when the onnxruntime-directml build is installed."""
    import onnxruntime as ort

    available = ort.get_available_providers()
    return [p for p in ("DmlExecutionProvider", "CPUExecutionProvider") if p in available]


class OnnxBeatThis:
    def __init__(self, model_path: str | Path, providers: list[str] | None = None) -> None:
        self.model_path = str(model_path)
        self.fallback_reason: str | None = None
        self._open(gpu_providers() if providers is None else providers)

    def _open(self, providers: list[str]) -> None:
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.log_severity_level = 4  # DirectML errors surface as exceptions; keep stderr quiet
        if "DmlExecutionProvider" in providers:  # DirectML requires these two settings
            opts.enable_mem_pattern = False
            opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        self.session = ort.InferenceSession(self.model_path, opts, providers=providers)
        self.device = "gpu" if self.session.get_providers()[0] != "CPUExecutionProvider" else "cpu"

    def logits(self, spect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        try:
            return self._logits(spect)
        except Exception as exc:  # noqa: BLE001 - e.g. GPU hung (DXGI 887A0006) under load
            if self.device == "cpu":
                raise
            self.fallback_reason = f"GPU inference failed ({type(exc).__name__}); used CPU"
            self._open(["CPUExecutionProvider"])
            return self._logits(spect)

    def _logits(self, spect: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        chunks, starts = _chunks(spect)
        n = len(spect)
        beat = np.full(n, -1000.0, dtype=np.float32)
        down = np.full(n, -1000.0, dtype=np.float32)
        # keep_first: write later chunks first so earlier ones overwrite the overlap
        for start, chunk in reversed(list(zip(starts, chunks, strict=True))):
            b, d = self.session.run(None, {"spect": chunk[None]})
            sl = slice(start + BORDER, start + len(chunk) - BORDER)
            beat[sl] = b[0, BORDER:-BORDER]
            down[sl] = d[0, BORDER:-BORDER]
        return beat, down

    def __call__(self, signal: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
        signal = np.asarray(signal, dtype=np.float32)
        if signal.ndim == 2:
            signal = signal.mean(1)
        if sr != SR:
            import soxr

            signal = soxr.resample(signal, in_rate=sr, out_rate=SR)
        return frames_to_beats(*self.logits(log_mel(signal)))
