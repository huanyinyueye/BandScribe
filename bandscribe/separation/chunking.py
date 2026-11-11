"""Overlap-add chunked inference (same scheme as MSST's generic `demix`).

The mix stays on the CPU and results are accumulated on the CPU, so VRAM use depends only on
the chunk and batch size, not on track length or stem count.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch

from bandscribe.separation.base import ProgressFn

ForwardFn = Callable[[torch.Tensor], torch.Tensor]  # (b, 2, chunk) -> (b, stems, 2, chunk)


def fade_window(chunk_size: int, fade_size: int) -> torch.Tensor:
    window = torch.ones(chunk_size)
    if fade_size > 0:
        window[:fade_size] = torch.linspace(0, 1, fade_size)
        window[-fade_size:] = torch.linspace(1, 0, fade_size)
    return window


def chunked_separate(forward: ForwardFn, mix: np.ndarray, n_stems: int, *, chunk_size: int,
                     num_overlap: int, batch_size: int,
                     progress: ProgressFn | None = None) -> np.ndarray:
    """Run ``forward`` over overlapping chunks of ``mix`` (2, n); return (n_stems, 2, n)."""
    if num_overlap < 2:
        raise ValueError("num_overlap must be >= 2 so every sample is covered by a non-zero window")
    if batch_size < 1 or chunk_size < 16:
        raise ValueError("invalid batch_size or chunk_size")
    x = torch.from_numpy(np.ascontiguousarray(mix, dtype=np.float32))
    length = x.shape[-1]
    fade = chunk_size // 10
    step = chunk_size // num_overlap
    border = chunk_size - step
    padded = length > 2 * border
    if padded:
        x = torch.nn.functional.pad(x[None], (border, border), mode="reflect")[0]
    total = x.shape[-1]
    window = fade_window(chunk_size, fade)

    result = torch.zeros((n_stems, 2, total), dtype=torch.float32)
    counter = torch.zeros(total, dtype=torch.float32)
    batch: list[torch.Tensor] = []
    locations: list[tuple[int, int]] = []
    i = 0
    while i < total:
        part = x[:, i:i + chunk_size]
        seg = part.shape[-1]
        if seg < chunk_size:
            mode = "reflect" if seg > chunk_size // 2 else "constant"
            part = torch.nn.functional.pad(part[None], (0, chunk_size - seg), mode=mode)[0]
        batch.append(part)
        locations.append((i, seg))
        i += step
        if len(batch) >= batch_size or i >= total:
            out = forward(torch.stack(batch)).to("cpu", torch.float32)
            if out.shape[1:] != (n_stems, 2, chunk_size):
                raise RuntimeError(f"model returned shape {tuple(out.shape)}, expected "
                                   f"(b, {n_stems}, 2, {chunk_size})")
            for j, (start, seg_len) in enumerate(locations):
                w = window.clone()
                if start == 0:
                    w[:fade] = 1
                if start + step >= total:
                    w[-fade:] = 1
                result[..., start:start + seg_len] += out[j, ..., :seg_len] * w[:seg_len]
                counter[start:start + seg_len] += w[:seg_len]
            batch.clear()
            locations.clear()
            if progress is not None:
                progress(min(1.0, i / total))
    result /= counter.clamp_min(1e-8)
    if padded:
        result = result[..., border:-border]
    return result.numpy()
