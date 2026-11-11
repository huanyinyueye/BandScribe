import gc
import threading
import weakref

import pytest

from bandscribe.runtime.vram import ModelSlot, SlotBusyError, run_with_oom_backoff


class FakeOOM(RuntimeError):
    pass


class FakeBackend:
    """Simulated GPU: `allocated` is driven by live FakeModel objects."""

    def __init__(self, available=True):
        self._available = available
        self.allocated = 0
        self.peak = 0
        self.empty_cache_calls = 0

    def alloc(self, n):
        self.allocated += n
        self.peak = max(self.peak, self.allocated)

    def free(self, n):
        self.allocated -= n

    def available(self):
        return self._available

    def reset_peak(self):
        self.peak = self.allocated

    def peak_bytes(self):
        return self.peak

    def allocated_bytes(self):
        return self.allocated

    def total_bytes(self):
        return 10 * 1024**3

    def empty_cache(self):
        self.empty_cache_calls += 1

    def is_oom(self, exc):
        return isinstance(exc, FakeOOM)


class FakeModel:
    def __init__(self, backend, size):
        self.backend, self.size = backend, size
        backend.alloc(size)

    def __del__(self):
        self.backend.free(self.size)


def test_model_released_and_peak_recorded():
    be = FakeBackend()
    slot = ModelSlot(be)
    with slot.load("a", lambda: FakeModel(be, 4_000)) as m:
        ref = weakref.ref(m)
        assert slot.current == "a"
        be.alloc(1_500)  # inference activations
        be.free(1_500)
        del m  # caller drops its reference before the block ends
    gc.collect()
    assert ref() is None, "slot must not keep the model alive after the block"
    assert be.allocated == 0
    assert slot.current is None
    (rec,) = slot.records
    assert (rec.model_id, rec.device, rec.peak_bytes, rec.leaked_bytes) == ("a", "cuda", 5_500, 0)
    assert be.empty_cache_calls >= 2  # before load and after release


def test_only_one_model_at_a_time():
    be = FakeBackend()
    slot = ModelSlot(be)
    with slot.load("a", lambda: FakeModel(be, 10)):
        with pytest.raises(SlotBusyError, match="'a' is still loaded"),                 slot.load("b", lambda: FakeModel(be, 10)):
            pass
        assert slot.current == "a"  # failed request must not disturb the holder
    assert slot.current is None


def test_sequential_models_never_coexist():
    be = FakeBackend()
    slot = ModelSlot(be)
    for mid, size in (("sep", 6_000), ("amt", 7_000), ("drums", 500)):
        with slot.load(mid, lambda s=size: FakeModel(be, s)):
            assert be.allocated == size  # previous model fully gone
        gc.collect()
    assert slot.peak_by_model() == {"sep": 6_000, "amt": 7_000, "drums": 500}


def test_slot_freed_when_inference_raises():
    be = FakeBackend()
    slot = ModelSlot(be)
    with pytest.raises(ValueError), slot.load("a", lambda: FakeModel(be, 10)):
        raise ValueError("boom")
    gc.collect()
    assert slot.current is None and be.allocated == 0
    with slot.load("b", lambda: FakeModel(be, 10)):
        pass


def test_slot_freed_when_loader_raises():
    slot = ModelSlot(FakeBackend())

    def bad():
        raise OSError("missing weights")

    with pytest.raises(OSError), slot.load("a", bad):
        pass
    assert slot.current is None
    assert slot.records[0].model_id == "a"


def test_leak_is_reported_not_hidden():
    be = FakeBackend()
    slot = ModelSlot(be)
    with slot.load("a", lambda: FakeModel(be, 100)) as m:
        kept = m  # caller wrongly keeps the model
    assert slot.records[0].leaked_bytes == 100
    del kept, m


def test_slot_is_exclusive_across_threads():
    be = FakeBackend()
    slot = ModelSlot(be)
    inside, release, errors = threading.Event(), threading.Event(), []

    def holder():
        with slot.load("a", lambda: FakeModel(be, 1)):
            inside.set()
            release.wait(5)

    t = threading.Thread(target=holder)
    t.start()
    assert inside.wait(5)
    try:
        with slot.load("b", lambda: FakeModel(be, 1)):
            pass
    except SlotBusyError as e:
        errors.append(e)
    release.set()
    t.join(5)
    assert len(errors) == 1


def test_cpu_device_recorded_without_cuda():
    slot = ModelSlot(FakeBackend(available=False))
    with slot.load("a", object):
        pass
    assert slot.records[0].device == "cpu"


def test_backoff_first_size_fits():
    calls = []
    r = run_with_oom_backoff(lambda c, d: calls.append((c, d)) or "out", [8, 4, 2],
                             backend=FakeBackend())
    assert (r.value, r.device, r.chunk_size, r.attempts) == ("out", "cuda", 8, 0)
    assert calls == [(8, "cuda")]


def test_backoff_shrinks_chunk_on_oom():
    be, calls, retries = FakeBackend(), [], []

    def fn(chunk, device):
        calls.append((chunk, device))
        if chunk > 2:
            raise FakeOOM("CUDA out of memory")
        return chunk

    r = run_with_oom_backoff(fn, [8, 4, 2], backend=be, on_retry=lambda c, d: retries.append(c))
    assert calls == [(8, "cuda"), (4, "cuda"), (2, "cuda")]
    assert (r.device, r.chunk_size, r.attempts) == ("cuda", 2, 2)
    assert retries == [8, 4]
    assert be.empty_cache_calls == 2


def test_backoff_falls_back_to_cpu():
    calls = []

    def fn(chunk, device):
        calls.append((chunk, device))
        if device == "cuda":
            raise FakeOOM("oom")
        return "cpu-result"

    r = run_with_oom_backoff(fn, [4, 2], backend=FakeBackend())
    assert calls == [(4, "cuda"), (2, "cuda"), (2, "cpu")]
    assert (r.value, r.device, r.attempts) == ("cpu-result", "cpu", 2)


def test_backoff_without_cpu_fallback_raises_oom():
    def fn(chunk, device):
        raise FakeOOM("oom")

    with pytest.raises(FakeOOM):
        run_with_oom_backoff(fn, [4, 2], backend=FakeBackend(), cpu_fallback=False)


def test_backoff_does_not_swallow_other_errors():
    calls = []

    def fn(chunk, device):
        calls.append(chunk)
        raise KeyError("bug")

    with pytest.raises(KeyError):
        run_with_oom_backoff(fn, [4, 2], backend=FakeBackend())
    assert calls == [4]


def test_backoff_no_cuda_goes_straight_to_cpu():
    r = run_with_oom_backoff(lambda c, d: (c, d), [8, 4], backend=FakeBackend(available=False))
    assert r.value == (8, "cpu") and r.attempts == 0


@pytest.mark.parametrize("sizes", [[], [2, 4], [4, 4]])
def test_backoff_rejects_bad_chunk_sizes(sizes):
    with pytest.raises(ValueError):
        run_with_oom_backoff(lambda c, d: None, sizes, backend=FakeBackend())


@pytest.mark.gpu
def test_real_gpu_slot_releases_memory():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    slot = ModelSlot()
    before = torch.cuda.memory_allocated()
    with slot.load("dummy", lambda: torch.nn.Linear(4096, 4096).cuda().half()) as model:
        with torch.no_grad():
            model(torch.ones(8, 4096, device="cuda", dtype=torch.float16))
        del model
    rec = slot.records[0]
    assert rec.device == "cuda"
    assert rec.peak_bytes >= 4096 * 4096 * 2  # FP16 weight matrix
    assert rec.leaked_bytes == 0
    # <= because releasing also frees the cuBLAS workspace an earlier test may have created.
    assert torch.cuda.memory_allocated() <= before


@pytest.mark.gpu
def test_real_gpu_oom_is_detected_and_backed_off():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    total = torch.cuda.get_device_properties(0).total_memory

    def fn(n_bytes, device):
        return torch.empty(n_bytes, dtype=torch.uint8, device=device).numel()

    r = run_with_oom_backoff(fn, [total * 4, 64 * 1024**2], cpu_fallback=False)
    assert (r.device, r.chunk_size, r.attempts) == ("cuda", 64 * 1024**2, 1)
