import numpy as np
import pytest
import torch

from bandscribe.separation.base import SeparationResult, Stem
from bandscribe.separation.chunking import chunked_separate, fade_window
from bandscribe.separation.ensemble import agreement_confidence, ensemble

SR = 44100


def _noise(n, seed=0, scale=0.3):
    return (np.random.default_rng(seed).standard_normal((2, n)) * scale).astype(np.float32)


# --- chunking ------------------------------------------------------------------------------

def _two_stem_forward(calls=None):
    def forward(x):  # stem 0 = input, stem 1 = -0.5 * input
        if calls is not None:
            calls.append(tuple(x.shape))
        return torch.stack([x, -0.5 * x], dim=1)

    return forward


@pytest.mark.parametrize("n", [100, 999, 4096, 10_000, 44_100])
@pytest.mark.parametrize("overlap", [2, 4])
def test_chunked_identity_reconstructs_exactly(n, overlap):
    mix = _noise(n)
    out = chunked_separate(_two_stem_forward(), mix, 2, chunk_size=1000, num_overlap=overlap,
                           batch_size=3)
    assert out.shape == (2, 2, n)
    np.testing.assert_allclose(out[0], mix, atol=1e-5)
    np.testing.assert_allclose(out[1], -0.5 * mix, atol=1e-5)


def test_chunked_respects_batch_and_chunk_size():
    calls = []
    chunked_separate(_two_stem_forward(calls), _noise(20_000), 2, chunk_size=2000,
                     num_overlap=2, batch_size=4)
    assert all(shape[1:] == (2, 2000) for shape in calls)
    assert max(shape[0] for shape in calls) == 4


def test_chunked_reports_progress():
    seen = []
    chunked_separate(_two_stem_forward(), _noise(20_000), 2, chunk_size=2000, num_overlap=2,
                     batch_size=2, progress=seen.append)
    assert seen == sorted(seen) and seen[-1] == 1.0


def test_chunked_rejects_bad_arguments_and_shapes():
    with pytest.raises(ValueError):
        chunked_separate(_two_stem_forward(), _noise(100), 2, chunk_size=1000, num_overlap=1,
                         batch_size=1)
    with pytest.raises(RuntimeError, match="expected"):
        chunked_separate(lambda x: x[:, None], _noise(5000), 2, chunk_size=1000, num_overlap=2,
                         batch_size=1)


def test_fade_window():
    w = fade_window(100, 10)
    assert w[0] == 0 and w[-1] == 0 and torch.all(w[10:90] == 1)


# --- ensemble ------------------------------------------------------------------------------

def _result(model_id, stems):
    return SeparationResult({k: Stem(k, v, SR, model_id) for k, v in stems.items()}, model_id,
                            {}, "cpu", 1, 0, 0.0, 0.0)


def _six(seed=0, n=SR * 3):
    names = ["vocals", "drums", "bass", "guitar", "piano", "other"]
    return {name: _noise(n, seed + i) for i, name in enumerate(names)}


def test_agreement_confidence_levels():
    a = _noise(SR * 3, 1)
    same = agreement_confidence(a, a.copy(), SR)
    assert np.all(same.values == 1.0) and same.active.all()
    different = agreement_confidence(a, _noise(SR * 3, 2), SR)
    assert np.all(different.values < 0.3)
    slight = agreement_confidence(a, a + _noise(SR * 3, 3, scale=0.03), SR)
    assert np.all(slight.values > 0.9)


def test_agreement_is_local_in_time():
    a = _noise(SR * 3, 1)
    b = a.copy()
    b[:, SR:2 * SR] = _noise(SR, 9)  # models disagree only in the second second
    c = agreement_confidence(a, b, SR)
    assert c.values[0] == 1.0 and c.values[2] == 1.0 and c.values[1] < 0.3
    assert c.at(0.5) == 1.0 and c.at(1.5) < 0.3


def test_silence_is_inactive_not_low_confidence():
    z = np.zeros((2, SR * 2), dtype=np.float32)
    c = agreement_confidence(z, z, SR)
    assert not c.active.any() and np.all(c.values == 1.0)


def test_ensemble_blends_shared_instrument_stems():
    six = _six()
    four = {"vocals": _noise(SR * 3, 20), "drums": six["drums"].copy(),
            "bass": six["bass"].copy(),
            "other": six["guitar"] + six["piano"] + six["other"]}
    out = ensemble(_result("six", six), _result("four", four))
    np.testing.assert_allclose(out.stems["vocals"].audio, (six["vocals"] + four["vocals"]) / 2,
                               atol=1e-6)
    assert out.stems["vocals"].model_id == "ensemble:six+four"
    assert out.weights["drums"] == {"six": 0.5, "four": 0.5}
    assert np.all(out.confidence["drums"].values == 1.0)
    assert np.all(out.confidence["vocals"].values < 0.3)
    # The guitar+piano+other group matches the 4-stem other exactly here.
    assert np.all(out.confidence["guitar"].values == 1.0)


def test_other_is_never_turned_into_guitar():
    """Spec 5.1: the guitar stem must come only from a model that outputs a guitar stem."""
    six = _six()
    loud_other = _noise(SR * 3, 50, scale=0.9)
    four = {"vocals": six["vocals"], "drums": six["drums"], "bass": six["bass"],
            "other": loud_other}
    out = ensemble(_result("six", six), _result("four", four), weights={"guitar": 1.0})
    np.testing.assert_array_equal(out.stems["guitar"].audio, six["guitar"])
    assert out.stems["guitar"].model_id == "six"
    assert out.weights["guitar"] == {"six": 1.0}
    np.testing.assert_array_equal(out.stems["other"].audio, six["other"])
    # ...but the disagreement lowers the confidence of the whole guitar/piano/other group.
    assert np.all(out.confidence["guitar"].values < 0.5)


def test_four_stem_primary_never_gains_a_guitar_stem():
    six = _six()
    four = {k: six[k] for k in ("vocals", "drums", "bass", "other")}
    out = ensemble(_result("four", four), _result("six", six))
    assert set(out.stems) == {"vocals", "drums", "bass", "other"}
    assert "guitar" not in out.stems


def test_ensemble_weights():
    six = _six()
    four = {"drums": _noise(SR * 3, 30)}
    out = ensemble(_result("six", six), _result("four", four), weights={"drums": 0.0})
    np.testing.assert_allclose(out.stems["drums"].audio, six["drums"], atol=1e-7)
    with pytest.raises(ValueError):
        ensemble(_result("six", six), _result("four", four), weights={"drums": 1.5})


def test_ensemble_single_model_passthrough():
    six = _six()
    out = ensemble(_result("six", six))
    assert set(out.stems) == set(six) and out.confidence == {}
