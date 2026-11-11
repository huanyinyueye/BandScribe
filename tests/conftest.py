import numpy as np
import pytest
import soundfile as sf


def click_track(beat_times, sr=44100, duration=None, accent_every=0):
    """Stereo click track: short decaying 1 kHz bursts at the given times."""
    duration = duration or (max(beat_times) + 1.0)
    out = np.zeros(int(duration * sr), dtype=np.float32)
    n = int(0.03 * sr)
    t = np.arange(n) / sr
    for i, bt in enumerate(beat_times):
        freq = 1500.0 if accent_every and i % accent_every == 0 else 1000.0
        burst = np.sin(2 * np.pi * freq * t) * np.exp(-t * 120)
        start = int(bt * sr)
        out[start:start + n] += burst[: len(out) - start].astype(np.float32)
    return np.stack([out, out], axis=1)


@pytest.fixture
def make_wav(tmp_path):
    def _make(samples, sr=44100, name="x.wav", subtype="PCM_16"):
        path = tmp_path / name
        sf.write(path, samples, sr, subtype=subtype)
        return path

    return _make
