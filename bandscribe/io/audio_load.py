"""Decode MP3/WAV/FLAC to 44.1 kHz float32 stereo, keeping the original format info."""

from __future__ import annotations

import hashlib
import re
import shutil
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

TARGET_SR = 44100
SUPPORTED_SUFFIXES = {".mp3", ".wav", ".flac"}
NORMALIZE_PEAK = 10 ** (-1.0 / 20)  # -1 dBFS


class AudioLoadError(RuntimeError):
    pass


@dataclass(frozen=True)
class Audio:
    samples: np.ndarray  # float32, shape (2, n_samples)
    sr: int
    path: Path | None = None
    orig_sr: int | None = None
    orig_channels: int | None = None
    file_sha256: str | None = None
    gain: float = 1.0  # linear gain applied by normalisation

    @property
    def duration(self) -> float:
        return self.samples.shape[1] / self.sr

    def mono(self) -> np.ndarray:
        return self.samples.mean(axis=0)

    def peak(self) -> float:
        return float(np.abs(self.samples).max()) if self.samples.size else 0.0


def ffmpeg_exe() -> str:
    if exe := shutil.which("ffmpeg"):
        return exe
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:
        raise AudioLoadError("ffmpeg not found; run `bandscribe doctor`") from exc


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while block := fh.read(1 << 20):
            h.update(block)
    return h.hexdigest()


def probe(path: Path) -> tuple[int, int]:
    """Return (sample rate, channels) of the first audio stream."""
    proc = subprocess.run([ffmpeg_exe(), "-hide_banner", "-i", str(path)], check=False,
                          capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"Stream #\d+:\d+.*?: Audio: .*?, (\d+) Hz, ([^,\n]+)", proc.stderr)
    if not m:
        raise AudioLoadError(f"{path.name}: no audio stream found")
    layout = m.group(2).strip()
    ch = {"mono": 1, "stereo": 2}.get(layout)
    if ch is None:
        cm = re.match(r"(\d+)(?:\.(\d+))?", layout)  # e.g. "5.1" or "2 channels"
        ch = int(cm.group(1)) + int(cm.group(2) or 0) if cm else 2
    return int(m.group(1)), ch


def normalize(audio: Audio, peak: float = NORMALIZE_PEAK) -> Audio:
    """Peak-normalise to ``peak``. Silent input is returned unchanged."""
    current = audio.peak()
    if current <= 0:
        return audio
    g = peak / current
    return replace(audio, samples=(audio.samples * g).astype(np.float32), gain=audio.gain * g)


def load_audio(path: str | Path, *, normalize_peak: bool = True) -> Audio:
    path = Path(path)
    if not path.is_file():
        raise AudioLoadError(f"file not found: {path}")
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        raise AudioLoadError(f"unsupported format {path.suffix!r}; use MP3, WAV or FLAC")
    orig_sr, orig_ch = probe(path)
    # Decode at the native rate, then resample with soxr VHQ.
    proc = subprocess.run(
        [ffmpeg_exe(), "-v", "error", "-nostdin", "-i", str(path), "-map", "0:a:0",
         "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-"],
        check=False, capture_output=True)
    if proc.returncode != 0:
        msg = proc.stderr.decode("utf-8", "replace").strip()[:400]
        raise AudioLoadError(f"{path.name}: ffmpeg failed: {msg}")
    data = np.frombuffer(proc.stdout, dtype="<f4")
    if data.size < 2:
        raise AudioLoadError(f"{path.name}: decoded to zero samples")
    data = data[: data.size // 2 * 2].reshape(-1, 2)
    if orig_sr != TARGET_SR:
        import soxr

        data = soxr.resample(data, orig_sr, TARGET_SR, quality="VHQ")
    samples = np.ascontiguousarray(data.T, dtype=np.float32)
    audio = Audio(samples, TARGET_SR, path, orig_sr, orig_ch, _file_sha256(path))
    return normalize(audio) if normalize_peak else audio
