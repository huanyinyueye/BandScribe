import numpy as np
import pytest

from bandscribe.analysis.meter import analyze_meter
from bandscribe.analysis.tempo import build_tempo_map
from bandscribe.io.audio_load import (
    NORMALIZE_PEAK,
    AudioLoadError,
    load_audio,
)
from tests.conftest import click_track

# --- audio loading -------------------------------------------------------------------------

def _sine(sr, seconds=1.0, freq=440.0, amp=0.5, channels=2):
    t = np.arange(int(sr * seconds)) / sr
    x = (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)
    return np.stack([x] * channels, axis=1) if channels > 1 else x


def test_load_wav_stereo_44k(make_wav):
    a = load_audio(make_wav(_sine(44100)), normalize_peak=False)
    assert a.samples.shape == (2, 44100) and a.samples.dtype == np.float32
    assert (a.sr, a.orig_sr, a.orig_channels) == (44100, 44100, 2)
    assert a.duration == pytest.approx(1.0)
    assert a.peak() == pytest.approx(0.5, abs=1e-3)
    assert len(a.file_sha256) == 64


def test_load_resamples_and_upmixes_mono(make_wav):
    a = load_audio(make_wav(_sine(48000, channels=1), sr=48000), normalize_peak=False)
    assert (a.orig_sr, a.orig_channels, a.sr) == (48000, 1, 44100)
    assert a.samples.shape[0] == 2 and abs(a.samples.shape[1] - 44100) <= 2
    # Pitch must survive resampling: the spectral peak stays at 440 Hz.
    spec = np.abs(np.fft.rfft(a.samples[0]))
    assert np.argmax(spec) * 44100 / a.samples.shape[1] == pytest.approx(440.0, abs=2.0)


def test_load_flac(make_wav):
    a = load_audio(make_wav(_sine(44100), name="x.flac"), normalize_peak=False)
    assert a.samples.shape == (2, 44100)


def test_normalize_records_gain(make_wav):
    a = load_audio(make_wav(_sine(44100, amp=0.25)))
    assert a.peak() == pytest.approx(NORMALIZE_PEAK, rel=1e-3)
    assert a.gain == pytest.approx(NORMALIZE_PEAK / 0.25, rel=1e-2)


def test_silence_is_not_amplified(make_wav):
    a = load_audio(make_wav(np.zeros((44100, 2), dtype=np.float32)))
    assert a.peak() == 0.0 and a.gain == 1.0


def test_load_errors(tmp_path):
    with pytest.raises(AudioLoadError, match="not found"):
        load_audio(tmp_path / "nope.wav")
    bad = tmp_path / "x.ogg"
    bad.write_bytes(b"x")
    with pytest.raises(AudioLoadError, match="unsupported"):
        load_audio(bad)
    junk = tmp_path / "junk.wav"
    junk.write_bytes(b"this is not audio")
    with pytest.raises(AudioLoadError):
        load_audio(junk)


# --- tempo map -----------------------------------------------------------------------------

def test_constant_tempo():
    tm = build_tempo_map(np.arange(64) * 0.5)
    assert tm.global_bpm == pytest.approx(120.0)
    assert len(tm.segments) == 1 and tm.segments[0].bpm == pytest.approx(120.0)


def test_tempo_change_detected():
    first = np.arange(40) * 0.5  # 120 BPM
    second = first[-1] + np.arange(1, 41) * 0.4  # 150 BPM
    tm = build_tempo_map(np.concatenate([first, second]))
    assert [round(s.bpm) for s in tm.segments] == [120, 150]
    assert tm.segments[1].start_time == pytest.approx(first[-1], abs=1.0)
    assert tm.bpm_at(5.0) == pytest.approx(120, abs=1) and tm.bpm_at(30.0) == pytest.approx(150, abs=1)


def test_jitter_and_single_outlier_do_not_create_tempo_changes():
    rng = np.random.default_rng(0)
    beats = np.arange(100) * 0.5 + rng.normal(0, 0.008, 100)
    beats[50] += 0.06  # one badly placed beat
    assert len(build_tempo_map(np.sort(beats)).segments) == 1


def test_time_beat_round_trip_follows_rubato():
    beats = np.cumsum(np.concatenate([[1.0], np.linspace(0.5, 0.4, 30)]))  # accelerando
    tm = build_tempo_map(beats)
    assert tm.time_to_beat(beats[7]) == pytest.approx(7.0)
    assert tm.time_to_beat((beats[7] + beats[8]) / 2) == pytest.approx(7.5)
    assert tm.time_to_beat(beats[0] - 0.25) == pytest.approx(-0.5)  # pickup, extrapolated
    q = np.array([-1.0, 0.0, 3.25, 29.0, 31.5])
    assert tm.time_to_beat(tm.beat_to_time(q)) == pytest.approx(q)


def test_tempo_map_degenerate_inputs():
    assert build_tempo_map(np.array([])).segments == ()
    assert build_tempo_map(np.array([1.0])).global_bpm == 0.0
    with pytest.raises(ValueError):
        build_tempo_map(np.array([0.0, 1.0, 1.0]))


# --- meter ---------------------------------------------------------------------------------

def _grid(counts, period=0.5, start=0.0):
    """Beats and downbeats for a sequence of bar lengths."""
    n = sum(counts)
    beats = start + np.arange(n) * period
    idx = np.concatenate([[0], np.cumsum(counts)[:-1]])
    return beats, beats[idx]


def test_four_four():
    m = analyze_meter(*_grid([4] * 8))
    assert m.dominant == (4, 4) and len(m.bars) == 8 and len(m.changes) == 1
    assert m.pickup_beats == 0 and all(b.confident for b in m.bars)


def test_three_four_and_pickup():
    beats = np.arange(25) * 0.5
    m = analyze_meter(beats, beats[1::3])
    assert m.dominant == (3, 4) and m.pickup_beats == 1


def test_seven_eight_fast_pulse():
    m = analyze_meter(*_grid([7] * 6, period=0.25))  # 240 BPM pulse -> eighth notes
    assert m.dominant == (7, 8)


def test_five_four_is_not_forced_to_four_four():
    assert analyze_meter(*_grid([5] * 6)).dominant == (5, 4)


def test_meter_changes_are_kept():
    m = analyze_meter(*_grid([4, 4, 4, 7, 7, 7, 4, 4]))
    assert [(c.bar_index, c.numerator) for c in m.changes] == [(0, 4), (3, 7), (6, 4)]
    assert all(b.confident for b in m.bars)


def test_lone_odd_bar_is_kept_but_flagged():
    m = analyze_meter(*_grid([4, 4, 5, 4, 4]))
    assert [b.numerator for b in m.bars] == [4, 4, 5, 4, 4]  # not "corrected"
    assert [b.confident for b in m.bars] == [True, True, False, True, True]


def test_bar_times_are_contiguous():
    m = analyze_meter(*_grid([4, 3, 4]))
    for a, b in zip(m.bars, m.bars[1:], strict=False):
        assert a.end_time == pytest.approx(b.start_time)
    assert m.bars[-1].end_time == pytest.approx(11 * 0.5)


def test_meter_empty_inputs():
    assert analyze_meter(np.array([]), np.array([])).bars == ()
    assert analyze_meter(np.arange(4.0), np.array([])).pickup_beats == 4


# --- Beat This! on the real GPU -------------------------------------------------------------

@pytest.mark.gpu
def test_beat_tracker_on_click_track(make_wav):
    from bandscribe.analysis.beats import track_beats
    from bandscribe.models.registry import load_registry
    from bandscribe.runtime.vram import ModelSlot

    ref = 0.5 + np.arange(40) * 0.5  # 120 BPM
    audio = load_audio(make_wav(click_track(ref, accent_every=4)))
    slot = ModelSlot()
    res = track_beats(audio, slot=slot)
    assert res.model_sha256 == load_registry().get("beat_this").files[0].sha256
    import mir_eval

    assert mir_eval.beat.f_measure(ref, res.beats, f_measure_threshold=0.07) > 0.95
    assert build_tempo_map(res.beats).global_bpm == pytest.approx(120.0, abs=1.0)
    assert slot.current is None and slot.records[0].model_id == "beat_this"
