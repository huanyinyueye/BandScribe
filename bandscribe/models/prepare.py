"""First-run model preparation for the torch-free engines (audio.cpp and ONNX Runtime).

Weights are never bundled: the registry files are downloaded, then converted locally.
- BS-RoFormer SW: torch checkpoint -> model.safetensors + config.json for audio.cpp.
- Beat This!: final0.ckpt -> external weights of the ONNX graph shipped in models/onnx.
- MuScriptor: the Hugging Face snapshot (config.json + model.safetensors) loads as is.
Each result is stamped with the source sha256 so a changed download is converted again.
"""

from __future__ import annotations

import json
import shutil
from functools import lru_cache
from pathlib import Path

import numpy as np

from bandscribe.models.download import ensure_model
from bandscribe.models.registry import HashMismatchError, load_registry, sha256_file

PKG = Path(__file__).parent
STAMP = "source.sha256"


def _stamped(out: Path, sha: str, names: tuple[str, ...]) -> bool:
    stamp = out / STAMP
    return (stamp.is_file() and stamp.read_text().strip() == sha
            and all((out / n).is_file() for n in names))


def audiocpp_bs_roformer(model_id: str = "bs_roformer_sw") -> tuple[Path, dict[str, str]]:
    """Folder audio.cpp loads for ``model_id`` and the sha256 of its source files."""
    from safetensors.numpy import save_file

    from bandscribe.models.ckpt_reader import state_dict

    files = ensure_model(model_id)
    (ckpt, sha), = files.values()
    shas = {name: s for name, (_, s) in files.items()}
    out = ckpt.parent / "audiocpp"
    if _stamped(out, sha, ("model.safetensors", "config.json")):
        return out, shas
    out.mkdir(parents=True, exist_ok=True)
    state = {k: np.ascontiguousarray(v) for k, v in state_dict(ckpt).items()
             if not k.endswith(".rotary_embed.freqs")}  # recomputed by audio.cpp
    save_file(state, str(out / "model.safetensors.part"))
    (out / "model.safetensors.part").replace(out / "model.safetensors")
    shutil.copyfile(PKG / "audiocpp" / f"{model_id}.json", out / "config.json")
    (out / STAMP).write_text(sha)
    return out, shas


def beat_this_onnx() -> tuple[Path, str]:
    """The Beat This! ONNX graph with its weights rebuilt next to it, and the ckpt sha256."""
    from bandscribe.analysis.beat_onnx import build_weights

    (ckpt, sha), = ensure_model("beat_this").values()
    src = PKG / "onnx" / "beat_this_final0.manifest.json"
    manifest = json.loads(src.read_text(encoding="utf-8"))
    out = ckpt.parent / "onnx"
    graph = out / "beat_this_final0.onnx"
    if not _stamped(out, sha, (graph.name, manifest["weights"])):
        out.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PKG / "onnx" / graph.name, graph)
        build_weights(ckpt, src, out)
        (out / STAMP).write_text(sha)
    return graph, sha


@lru_cache(maxsize=1)
def muscriptor_snapshot(size: str = "medium") -> tuple[Path, str]:
    """Local Hugging Face snapshot of MuScriptor (gated: needs an accepted licence + token)."""
    from huggingface_hub import snapshot_download

    path = Path(snapshot_download(f"MuScriptor/muscriptor-{size}",
                                  allow_patterns=["config.json", "model.safetensors"]))
    sha = sha256_file(path / "model.safetensors")
    pinned = next((f.sha256 for f in load_registry().get(f"muscriptor_{size}").files
                   if f.name == "model.safetensors"), None)
    if pinned and pinned != sha:
        raise HashMismatchError(f"muscriptor_{size}: model.safetensors sha256 {sha} "
                                f"does not match pinned {pinned}")
    return path, sha
