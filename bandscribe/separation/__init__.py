"""Source separation. Use :func:`get_separator` to build a separator by registry id."""

from __future__ import annotations

from bandscribe.runtime.vram import ModelSlot, gpu_slot
from bandscribe.separation.base import Separator


def get_separator(model_id: str, slot: ModelSlot = gpu_slot, **kwargs) -> Separator:
    from bandscribe.separation.htdemucs6s import HTDemucs6sSeparator
    from bandscribe.separation.torch_separator import SPECS, TorchSeparator

    if model_id == "htdemucs_6s":
        return HTDemucs6sSeparator(slot, **kwargs)
    if model_id in SPECS:
        return TorchSeparator(SPECS[model_id], slot, **kwargs)
    raise KeyError(f"no separator for model id {model_id!r}")
