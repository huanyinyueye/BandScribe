"""VRAM management for a single 10 GB GPU (spec section 12.2).

Rules enforced here:
- Only one model is loaded at a time (:class:`ModelSlot`). Loading a second one while the
  first is still held is an error, not a silent eviction, so pipeline bugs surface early.
- A model is released as soon as its ``with`` block ends (drop references, gc, empty_cache).
- Peak VRAM per model is recorded (``torch.cuda.max_memory_allocated``) for BENCHMARKS.md.
- :func:`run_with_oom_backoff` retries a chunked inference call with smaller chunks on CUDA
  OOM and finally falls back to CPU.

``torch`` is imported lazily so the rest of the package (and the unit tests) work without it.
"""

from __future__ import annotations

import gc
import itertools
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from typing import Any, Protocol, TypeVar

T = TypeVar("T")


class SlotBusyError(RuntimeError):
    """A second model was requested while another one is still loaded."""


class GpuBackend(Protocol):
    def available(self) -> bool: ...
    def reset_peak(self) -> None: ...
    def peak_bytes(self) -> int: ...
    def allocated_bytes(self) -> int: ...
    def total_bytes(self) -> int: ...
    def empty_cache(self) -> None: ...
    def is_oom(self, exc: BaseException) -> bool: ...


class TorchBackend:
    """Real backend. Every method degrades to a no-op / 0 when CUDA is unavailable."""

    def __init__(self, device_index: int = 0) -> None:
        self.device_index = device_index

    @staticmethod
    def _torch() -> Any | None:
        try:
            import torch
        except ImportError:
            return None
        return torch

    def available(self) -> bool:
        torch = self._torch()
        return bool(torch is not None and torch.cuda.is_available())

    def reset_peak(self) -> None:
        if self.available():
            self._torch().cuda.reset_peak_memory_stats(self.device_index)

    def peak_bytes(self) -> int:
        return self._torch().cuda.max_memory_allocated(self.device_index) if self.available() else 0

    def allocated_bytes(self) -> int:
        return self._torch().cuda.memory_allocated(self.device_index) if self.available() else 0

    def total_bytes(self) -> int:
        if not self.available():
            return 0
        return self._torch().cuda.get_device_properties(self.device_index).total_memory

    def empty_cache(self) -> None:
        if self.available():
            torch = self._torch()
            # cuBLAS keeps a workspace inside torch's allocator after the first matmul; it is
            # not model memory but would otherwise stay resident between models.
            clear_ws = getattr(torch._C, "_cuda_clearCublasWorkspaces", None)
            if clear_ws is not None:
                clear_ws()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    def is_oom(self, exc: BaseException) -> bool:
        torch = self._torch()
        if torch is not None and isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
        return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


@dataclass(frozen=True)
class VramRecord:
    model_id: str
    device: str
    peak_bytes: int
    leaked_bytes: int  # still allocated after release, relative to before load
    seconds: float

    @property
    def peak_gb(self) -> float:
        return self.peak_bytes / 1024**3

    def as_dict(self) -> dict[str, Any]:
        return {**asdict(self), "peak_gb": round(self.peak_gb, 3)}


class _Handle:
    """Holds the only strong reference the slot keeps to the loaded model."""

    def __init__(self, model: Any) -> None:
        self.model = model


class ModelSlot:
    """The single GPU slot. Use one instance per process (see :data:`gpu_slot`)."""

    def __init__(self, backend: GpuBackend | None = None) -> None:
        self._backend: GpuBackend = backend or TorchBackend()
        self._lock = threading.Lock()
        self._current: str | None = None
        self.records: list[VramRecord] = []

    @property
    def current(self) -> str | None:
        return self._current

    @property
    def backend(self) -> GpuBackend:
        return self._backend

    @contextmanager
    def load(self, model_id: str, loader: Callable[[], T]) -> Iterator[T]:
        """Load ``loader()`` into the slot, yield it, and release it on exit.

        The caller must not keep the yielded object (or tensors on the GPU) alive after the
        block; anything that outlives it shows up as ``leaked_bytes`` in the record.
        """
        with self._lock:
            if self._current is not None:
                raise SlotBusyError(
                    f"cannot load {model_id!r}: {self._current!r} is still loaded "
                    "(one model at a time)"
                )
            self._current = model_id

        backend = self._backend
        device = "cuda" if backend.available() else "cpu"
        gc.collect()
        backend.empty_cache()
        baseline = backend.allocated_bytes()
        backend.reset_peak()
        started = time.perf_counter()
        handle: _Handle | None = None
        try:
            handle = _Handle(loader())
            yield handle.model
        finally:
            peak = backend.peak_bytes()
            if handle is not None:
                handle.model = None
            del handle
            gc.collect()
            backend.empty_cache()
            leaked = max(0, backend.allocated_bytes() - baseline)
            self.records.append(
                VramRecord(model_id, device, peak, leaked, time.perf_counter() - started)
            )
            with self._lock:
                self._current = None

    def peak_by_model(self) -> dict[str, int]:
        """Highest recorded peak per model id, in bytes."""
        out: dict[str, int] = {}
        for r in self.records:
            out[r.model_id] = max(out.get(r.model_id, 0), r.peak_bytes)
        return out


gpu_slot = ModelSlot()


@dataclass(frozen=True)
class BackoffResult:
    value: Any
    device: str
    chunk_size: int
    attempts: int  # number of OOM failures before the successful call


def run_with_oom_backoff(
    fn: Callable[[int, str], T],
    chunk_sizes: Sequence[int],
    *,
    backend: GpuBackend | None = None,
    cpu_fallback: bool = True,
    on_retry: Callable[[int, str], None] | None = None,
) -> BackoffResult:
    """Call ``fn(chunk_size, device)`` with decreasing chunk sizes until it fits.

    ``chunk_sizes`` is tried in order on CUDA (it must be strictly decreasing). Only OOM errors
    trigger a retry; every other exception propagates. If all sizes OOM and ``cpu_fallback`` is
    set, ``fn(chunk_sizes[-1], "cpu")`` is the last attempt. Without CUDA the call goes straight
    to CPU with the first chunk size.
    """
    if not chunk_sizes:
        raise ValueError("chunk_sizes must not be empty")
    if any(b >= a for a, b in itertools.pairwise(chunk_sizes)):
        raise ValueError("chunk_sizes must be strictly decreasing")
    backend = backend or TorchBackend()

    if not backend.available():
        return BackoffResult(fn(chunk_sizes[0], "cpu"), "cpu", chunk_sizes[0], 0)

    failures = 0
    for size in chunk_sizes:
        try:
            return BackoffResult(fn(size, "cuda"), "cuda", size, failures)
        except Exception as exc:
            if not backend.is_oom(exc):
                raise
            failures += 1
            last_oom = exc
        # Outside the except block so the traceback (which pins GPU tensors) is released.
        gc.collect()
        backend.empty_cache()
        if on_retry is not None:
            on_retry(size, "cuda")

    if not cpu_fallback:
        raise last_oom
    size = chunk_sizes[-1]
    return BackoffResult(fn(size, "cpu"), "cpu", size, failures)
