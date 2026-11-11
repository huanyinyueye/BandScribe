"""Section boundaries from a bar-level self-similarity novelty curve (spec 3.4 fallback).

Sections get neutral names ("Section 1", "Section 2", ...): the method finds where the music
changes, it does not know what a verse or chorus is, so it does not claim to.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bandscribe.analysis.meter import Bar

MIN_SECTION_BARS = 4


@dataclass(frozen=True)
class Section:
    name: str
    start_bar: int
    start_time: float


def bar_features(mono: np.ndarray, sr: int, bars: tuple[Bar, ...] | list[Bar]) -> np.ndarray:
    """One feature vector per bar: mean chroma (12) + normalised log band energies (8)."""
    import librosa

    hop = 2048
    chroma = librosa.feature.chroma_stft(y=mono, sr=sr, n_fft=4096, hop_length=hop)
    mel = librosa.feature.melspectrogram(y=mono, sr=sr, n_fft=4096, hop_length=hop, n_mels=8)
    logmel = np.log1p(mel * 100)
    feats = []
    for b in bars:
        a = int(b.start_time * sr / hop)
        z = max(a + 1, int(b.end_time * sr / hop))
        c = chroma[:, a:z].mean(axis=1) if a < chroma.shape[1] else np.zeros(12)
        m = logmel[:, a:z].mean(axis=1) if a < logmel.shape[1] else np.zeros(8)
        feats.append(np.concatenate([c, m]))
    f = np.array(feats)
    if len(f):
        f = (f - f.mean(axis=0)) / (f.std(axis=0) + 1e-8)
    return f


def novelty(features: np.ndarray, kernel_bars: int = 4) -> np.ndarray:
    """Checkerboard-kernel novelty along the diagonal of the bar self-similarity matrix."""
    n = len(features)
    if n == 0:
        return np.zeros(0)
    norm = features / (np.linalg.norm(features, axis=1, keepdims=True) + 1e-8)
    ssm = norm @ norm.T
    k = kernel_bars
    out = np.zeros(n)
    for i in range(n):
        a, b = max(0, i - k), min(n, i + k)
        if i - a < 2 or b - i < 2:
            continue
        before, after = ssm[a:i, a:i].mean(), ssm[i:b, i:b].mean()
        cross = ssm[a:i, i:b].mean()
        out[i] = (before + after) / 2 - cross
    return out


def find_sections(features: np.ndarray, bars: tuple[Bar, ...] | list[Bar],
                  min_bars: int = MIN_SECTION_BARS) -> list[Section]:
    if not len(bars):
        return []
    nov = novelty(features)
    starts = [0]
    if len(nov):
        threshold = nov.mean() + nov.std()
        order = np.argsort(nov)[::-1]
        for i in order:
            if nov[i] <= threshold or nov[i] <= 0:
                break
            if all(abs(int(i) - s) >= min_bars for s in starts) and len(bars) - int(i) >= min_bars:
                starts.append(int(i))
    starts.sort()
    return [Section(f"Section {n + 1}", s, bars[s].start_time) for n, s in enumerate(starts)]


def detect_sections(mono: np.ndarray, sr: int, bars: tuple[Bar, ...] | list[Bar]) -> list[Section]:
    return find_sections(bar_features(mono, sr, bars), bars)
