"""Export the Beat This! checkpoint to an ONNX graph whose weights are filled in at install time.

    python scripts/export_beat_this_onnx.py <final0.ckpt> <out_dir>

Dev tool (needs torch + onnx). Writes `beat_this_final0.onnx`, a graph whose parameters point at
external data `beat_this_final0.weights`, and `beat_this_final0.manifest.json`, which maps each
external tensor to its checkpoint key and byte range. The app downloads `final0.ckpt` and
rebuilds the weights file torch-free (bandscribe.analysis.beat_onnx.build_weights).

Input `spect` (1, frames, 128) log-mel; outputs `beat` and `downbeat` logits (1, frames).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import onnx
import torch
from beat_this.inference import load_model
from onnx import numpy_helper

STEM = "beat_this_final0"


class _Logits(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, spect: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.model(spect)
        return out["beat"], out["downbeat"]


def main(ckpt: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / f"{STEM}.full.onnx"
    model = _Logits(load_model(ckpt, "cpu")).eval()
    # The dynamo exporter fails on rotary_embedding_torch; the tracer handles it. No constant
    # folding, so parameters keep their state-dict names (ORT fuses BN/transposes at load).
    torch.onnx.export(model, (torch.randn(1, 1500, 128),), str(tmp), input_names=["spect"],
                      output_names=["beat", "downbeat"], dynamo=False, opset_version=17,
                      do_constant_folding=False,
                      dynamic_axes={"spect": {1: "frames"}, "beat": {1: "frames"},
                                    "downbeat": {1: "frames"}})
    graph = onnx.load(str(tmp))
    state = {f"model.{k.removeprefix('model.')}": v
             for k, v in torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"].items()}
    entries, offset, blob = [], 0, bytearray()
    for init in graph.graph.initializer:
        if init.name not in state:
            continue  # small traced constants stay inside the graph
        arr = numpy_helper.to_array(init)
        src = state[init.name].numpy()
        if arr.shape != src.shape or arr.dtype != src.dtype or not np.array_equal(arr, src):
            raise SystemExit(f"{init.name}: graph tensor differs from the checkpoint")
        raw = np.ascontiguousarray(arr).tobytes()
        entries.append({"name": init.name, "key": init.name.removeprefix("model."),
                        "dtype": str(arr.dtype), "shape": list(arr.shape), "offset": offset,
                        "length": len(raw)})
        init.ClearField("raw_data")
        init.data_location = onnx.TensorProto.EXTERNAL
        for k, v in (("location", f"{STEM}.weights"), ("offset", str(offset)), ("length", str(len(raw)))):
            init.external_data.add(key=k, value=v)
        blob += raw
        offset += len(raw)
    onnx.save(graph, str(out / f"{STEM}.onnx"))
    (out / f"{STEM}.weights").write_bytes(blob)
    manifest = {"weights": f"{STEM}.weights", "sha256": hashlib.sha256(blob).hexdigest(),
                "size_bytes": len(blob), "key_prefix": "model.", "tensors": entries}
    (out / f"{STEM}.manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    tmp.unlink()
    print(f"{len(entries)} external tensors, {len(blob) / 2**20:.1f} MiB")


if __name__ == "__main__":
    main(*sys.argv[1:3])