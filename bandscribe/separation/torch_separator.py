"""Separators for checkpoint + YAML-config models (BS-RoFormer SW, SCNet, SubRoformer, Mega 53).

One model is loaded at a time through the GPU slot; inference is FP16 autocast with overlap-add
chunking, and on CUDA OOM the batch and then the chunk shrink before falling back to CPU
(spec 12.2).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from bandscribe.models.download import ensure_model
from bandscribe.runtime.vram import ModelSlot, gpu_slot, run_with_oom_backoff
from bandscribe.separation.base import ProgressFn, SeparationResult, Separator, Stem
from bandscribe.separation.chunking import chunked_separate
from bandscribe.separation.vendor import import_msst_model

SR = 44100


class _TupleLoader(yaml.SafeLoader):
    """MSST configs use ``!!python/tuple``; load those as plain tuples without full unsafe load."""


_TupleLoader.add_constructor("tag:yaml.org,2002:python/tuple",
                             lambda loader, node: tuple(loader.construct_sequence(node)))


def load_yaml_config(path: Path) -> dict:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_TupleLoader)  # noqa: S506


def _state_dict(path: Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    for key in ("state_dict", "state", "model_state_dict"):
        if isinstance(ckpt, dict) and key in ckpt and isinstance(ckpt[key], dict):
            return ckpt[key]
    return ckpt


@dataclass(frozen=True)
class TorchModelSpec:
    model_id: str
    kind: str  # bs_roformer_infer | scnet | sub_roformer | msst_bs_roformer
    checkpoint: str
    config: str | None  # registry file name; None = config shipped inside a Python package
    batch_size: int = 2
    mixture_consistent: bool = True
    chunk_size: int | None = None  # overrides the config's chunk (to fit 10 GB of VRAM)


def _build_model(spec: TorchModelSpec, config: dict) -> torch.nn.Module:
    # The model classes are beartype-checked and expect tuples where YAML gives lists.
    params = {k: tuple(v) if isinstance(v, list) else v for k, v in config["model"].items()}
    if spec.kind == "bs_roformer_infer":
        from bs_roformer.bs_roformer import BSRoformer

        return BSRoformer(**params)
    if spec.kind == "msst_bs_roformer":
        return import_msst_model("models.bs_roformer", "BSRoformer")(**params)
    if spec.kind == "scnet":
        return import_msst_model("models.scnet", "SCNet")(**params)
    if spec.kind == "sub_roformer":
        return import_msst_model("models.sub_roformer", "SBRoformer")(**params)
    raise ValueError(f"unknown model kind {spec.kind!r}")


class TorchSeparator(Separator):
    def __init__(self, spec: TorchModelSpec, slot: ModelSlot = gpu_slot,
                 batch_size: int | None = None, chunk_size: int | None = None) -> None:
        self.spec = spec
        self.model_id = spec.model_id
        self.mixture_consistent = spec.mixture_consistent
        self.slot = slot
        self._batch_override = batch_size
        self._chunk_override = chunk_size
        self._files: dict[str, tuple[Path, str]] | None = None
        self._config: dict | None = None

    # -- configuration ---------------------------------------------------------------------
    def _ensure(self) -> None:
        if self._config is not None:
            return
        self._files = ensure_model(self.model_id)
        if self.spec.config is not None:
            cfg_path = self._files[self.spec.config][0]
        else:  # BS-RoFormer SW: the config ships inside bs-roformer-infer
            import bs_roformer

            cfg_path = Path(bs_roformer.__file__).parent / "configs" / "BS-Rofo-SW-Fixed.yaml"
        self._config = load_yaml_config(cfg_path)

    @property
    def stem_names(self) -> tuple[str, ...]:
        self._ensure()
        return tuple(self._config["training"]["instruments"])

    def _chunk_and_overlap(self) -> tuple[int, int]:
        self._ensure()
        inf = self._config.get("inference", {})
        chunk = inf.get("chunk_size") or self._config.get("audio", {}).get("chunk_size")
        if not chunk:
            raise ValueError(f"{self.model_id}: config has no chunk_size")
        return (int(self._chunk_override or self.spec.chunk_size or chunk),
                max(2, int(inf.get("num_overlap", 2))))

    # -- inference -------------------------------------------------------------------------
    def _load(self) -> torch.nn.Module:
        model = _build_model(self.spec, self._config)
        model.load_state_dict(_state_dict(self._files[self.spec.checkpoint][0]))
        return model.eval()

    def separate(self, mix: np.ndarray, sr: int = SR,
                 progress: ProgressFn | None = None) -> SeparationResult:
        if sr != SR:
            raise ValueError(f"separators expect {SR} Hz audio, got {sr}")
        if mix.ndim != 2 or mix.shape[0] != 2:
            raise ValueError("mix must have shape (2, n_samples)")
        self._ensure()
        names = self.stem_names
        chunk, overlap = self._chunk_and_overlap()
        batch = self._batch_override or self.spec.batch_size
        backend = self.slot.backend

        # Candidate (batch, chunk) pairs, largest first: shrink the batch, then the chunk.
        cands: list[tuple[int, int]] = []
        b = batch
        while b >= 1:
            cands.append((b, chunk))
            b //= 2
        cands += [(1, chunk // 2), (1, chunk // 4)]
        by_size = {bb * cc: (bb, cc) for bb, cc in cands}
        sizes = sorted(by_size, reverse=True)
        peak = 0

        with self.slot.load(self.model_id, self._load) as model:
            started = time.perf_counter()  # inference time only; model loading excluded
            def attempt(size: int, device: str) -> np.ndarray:
                nonlocal peak
                bb, cc = by_size[size]
                model.to(device)  # noqa: F821 (deleted after use to free VRAM)
                backend.reset_peak()
                use_amp = device == "cuda"

                def forward(x: torch.Tensor) -> torch.Tensor:
                    with torch.no_grad(), torch.autocast("cuda", enabled=use_amp):
                        return model(x.to(device))  # noqa: F821

                out = chunked_separate(forward, mix, len(names), chunk_size=cc,
                                       num_overlap=overlap, batch_size=bb, progress=progress)
                peak = backend.peak_bytes()
                return out

            res = run_with_oom_backoff(attempt, sizes, backend=backend)
            del model
        bb, cc = by_size[res.chunk_size]
        notes = []
        if cc != chunk:
            notes.append(f"chunk reduced from {chunk} to {cc} after CUDA OOM (quality may drop)")
        if res.device == "cpu" and backend.available():
            notes.append("fell back to CPU after CUDA OOM")
        stems = {n: Stem(n, res.value[i], SR, self.model_id) for i, n in enumerate(names)}
        return SeparationResult(
            stems=stems, model_id=self.model_id,
            files_sha256={name: sha for name, (_, sha) in self._files.items()},
            device=res.device, batch_size=bb, chunk_size=cc, peak_vram_gb=peak / 1024**3,
            seconds=time.perf_counter() - started, oom_retries=res.attempts, notes=tuple(notes))


SPECS: dict[str, TorchModelSpec] = {
    "bs_roformer_sw": TorchModelSpec("bs_roformer_sw", "bs_roformer_infer",
                                     "BS-Rofo-SW-Fixed.ckpt", None, batch_size=2),
    "scnet_xl_ihf": TorchModelSpec("scnet_xl_ihf", "scnet", "model_scnet_ep_36_sdr_10.0891.ckpt",
                                   "config_musdb18_scnet_xl_more_wide_v5.yaml", batch_size=2),
    "subroformer": TorchModelSpec("subroformer", "sub_roformer", "sub_roformer_base_4stem.ckpt",
                                  "subroformer_base_4stem.yaml", batch_size=2),
    "mvsep_mega53": TorchModelSpec("mvsep_mega53", "msst_bs_roformer",
                                   "mvsep_mega_model_bs_roformer_53_stems_v1.ckpt",
                                   "mvsep_mega_model_bs_roformer_53_stems.yaml", batch_size=1,
                                   mixture_consistent=False,
                                   # The config's 882000-sample chunk peaks at 10.25 GB on the
                                   # RTX 3080 and spills into shared memory (489 s per audio
                                   # minute); 441000 peaks at 7.04 GB (31 s per audio minute).
                                   chunk_size=441000),
}
