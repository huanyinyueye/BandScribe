"""Read tensors from a PyTorch zip checkpoint (``torch.save``) without importing torch.

The pickle is loaded with an allowlist: tensors are rebuilt as numpy arrays from the zip's
storage records, plain containers pass through, and every other global (Lightning hyper
parameters, argparse namespaces, ...) becomes an inert placeholder, so no checkpoint code
runs. Only the legacy ``.tar`` format (pre-1.6 torch) is unsupported.
"""

from __future__ import annotations

import collections
import pickle
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

_DTYPES = {
    "FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64,
    "BFloat16Storage": "bfloat16", "LongStorage": np.int64, "IntStorage": np.int32,
    "ShortStorage": np.int16, "CharStorage": np.int8, "ByteStorage": np.uint8,
    "BoolStorage": np.bool_,
}


class _Placeholder:
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.args = args

    def __setstate__(self, state: Any) -> None:
        self.state = state


class _StorageType:
    def __init__(self, name: str) -> None:
        self.name = name


def _bf16_to_f32(raw: bytes) -> np.ndarray:
    return (np.frombuffer(raw, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def _rebuild_tensor(storage: np.ndarray, offset: int, size: tuple, stride: tuple, *_: Any) -> np.ndarray:
    if not size:
        return np.array(storage[offset])
    item = storage.itemsize
    return np.lib.stride_tricks.as_strided(
        storage[offset:], shape=tuple(size), strides=tuple(s * item for s in stride)).copy()


def _rebuild_parameter(data: np.ndarray, *_: Any) -> np.ndarray:
    return data


class _Unpickler(pickle.Unpickler):
    def __init__(self, fh: Any, archive: zipfile.ZipFile, prefix: str) -> None:
        super().__init__(fh)
        self._archive = archive
        self._prefix = prefix
        self._cache: dict[str, np.ndarray] = {}

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils" and name in ("_rebuild_tensor_v2", "_rebuild_tensor"):
            return _rebuild_tensor
        if module == "torch._utils" and name == "_rebuild_parameter":
            return _rebuild_parameter
        if module == "torch" and name in _DTYPES:
            return _StorageType(name)
        if module == "collections" and name == "OrderedDict":
            return collections.OrderedDict
        return _Placeholder

    def persistent_load(self, pid: Any) -> np.ndarray:
        kind, storage_type, key, _location, _numel = pid
        if kind != "storage":
            raise ValueError(f"unsupported persistent id {kind!r}")
        if key not in self._cache:
            raw = self._archive.read(f"{self._prefix}/data/{key}")
            dtype = _DTYPES[storage_type.name]
            self._cache[key] = _bf16_to_f32(raw) if dtype == "bfloat16" else np.frombuffer(raw, dtype=dtype)
        return self._cache[key]


def load_checkpoint(path: str | Path) -> Any:
    """The unpickled checkpoint with every tensor as a numpy array (bfloat16 widened to f32)."""
    with zipfile.ZipFile(path) as archive:
        pkl = next((n for n in archive.namelist() if n.endswith("/data.pkl")), None)
        if pkl is None:
            raise ValueError(f"{path} is not a zip-format torch checkpoint")
        prefix = pkl[: -len("/data.pkl")]
        with archive.open(pkl) as fh:
            return _Unpickler(fh, archive, prefix).load()


def state_dict(path: str | Path) -> dict[str, np.ndarray]:
    """The tensor mapping of a checkpoint: a bare state dict or one nested under a common key."""
    payload = load_checkpoint(path)
    if isinstance(payload, dict):
        for key in ("state_dict", "model", "model_state_dict"):
            if isinstance(payload.get(key), dict) and payload[key]:
                payload = payload[key]
                break
    tensors = {k: v for k, v in payload.items() if isinstance(v, np.ndarray)} \
        if isinstance(payload, dict) else {}
    if not tensors:
        raise ValueError(f"{path} contains no tensors")
    return tensors
