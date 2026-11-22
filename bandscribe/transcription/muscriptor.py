"""MuScriptor multi-instrument transcription. Weights: CC BY-NC 4.0 (gated on Hugging Face)."""

from __future__ import annotations

import contextlib
import io
from pathlib import Path

import numpy as np

from bandscribe.models.registry import HashMismatchError, load_registry, sha256_file
from bandscribe.runtime.vram import ModelSlot, gpu_slot
from bandscribe.transcription.base import (
    Note,
    Transcriber,
    TranscriberUnavailable,
    sort_notes,
    to_mono,
)

# MuScriptor instrument group -> BandScribe family.
GROUP_FAMILY = {
    "acoustic_guitar": "guitar", "clean_electric_guitar": "guitar",
    "distorted_electric_guitar": "guitar",
    "acoustic_bass": "bass", "electric_bass": "bass",
    "acoustic_piano": "piano", "electric_piano": "piano",
    "voice": "vocals", "drums": "drums",
}
# Family -> groups the model may emit when transcribing that family's stem (hard constraint).
FAMILY_GROUPS: dict[str, list[str]] = {}
for _group, _family in GROUP_FAMILY.items():
    FAMILY_GROUPS.setdefault(_family, []).append(_group)

DRUM_HIT_DURATION = 0.05
# Pitched onsets come out late vs human-aligned GP scores. Whole-song raw note F1 by
# correction (Neo-Aspect / Ringing Bloom): 0 ms 0.386/0.691, 20 ms 0.567/0.749,
# 30 ms 0.588/0.747, 50 ms 0.596/0.704. Each song's best value also improved the other
# song; 30 ms is the best mean. See docs/e2e/2026-11-12-roselia/result.md. Drums are not
# corrected (no evidence yet).
PITCHED_ONSET_LATENCY = 0.03


def family_of(group: str) -> str:
    return GROUP_FAMILY.get(group, "other")


def groups_for(instrument: str) -> list[str] | None:
    """Hard instrument constraint for a stem; None lets the model name what it hears."""
    return FAMILY_GROUPS.get(instrument) if instrument not in ("mix", "other") else None


def to_note(name: str, instrument: str, group: str, pitch: int, start: float,
            end: float) -> Note | None:
    """One MuScriptor note as a BandScribe Note (latency-corrected; drums get a fixed length)."""
    family = family_of(group)
    if instrument not in ("mix", "other") and family != instrument:
        return None  # cannot happen with the hard constraint; kept as a guard
    onset = float(start)
    end = max(float(end), onset)
    if family == "drums":
        end = onset + DRUM_HIT_DURATION
    else:
        onset = max(0.0, onset - PITCHED_ONSET_LATENCY)
        end = max(onset, end - PITCHED_ONSET_LATENCY)
    return Note(onset, end, int(pitch), sources=(name,), instrument=family,
                flags=(f"mus:{group}",))


class MuScriptorTranscriber(Transcriber):
    def __init__(self, size: str = "medium", slot: ModelSlot = gpu_slot) -> None:
        if size != "medium":  # user decision 2026-11-12: medium is the only size used
            raise ValueError("only the medium MuScriptor model is supported")
        self.size = size
        self.name = f"muscriptor_{size}"
        self.slot = slot
        self.model_sha256: str | None = None
        self.unavailable_reason: str | None = None
        self.last_run = None

    def _loader(self):
        from muscriptor import TranscriptionModel
        from muscriptor.utils.download import ModelDownloadError

        device = "cuda" if self.slot.backend.available() else "cpu"
        try:
            return TranscriptionModel.load_model(self.size, device=device)
        except ModelDownloadError as exc:
            self.unavailable_reason = str(exc)
            raise TranscriberUnavailable(str(exc)) from exc

    def _record_hash(self) -> None:
        if self.model_sha256 is not None:
            return
        try:
            from huggingface_hub import try_to_load_from_cache

            path = try_to_load_from_cache(f"MuScriptor/muscriptor-{self.size}", "model.safetensors")
            if isinstance(path, str) and Path(path).is_file():
                self.model_sha256 = sha256_file(Path(path))
                self._check_pin()
        except HashMismatchError:
            raise
        except Exception:  # noqa: BLE001 - failing to locate the cached file is not fatal
            self.model_sha256 = None

    def _check_pin(self) -> None:
        """Refuse weights whose hash differs from the registry pin (spec section 7)."""
        try:
            files = load_registry().get(self.name).files
        except KeyError:
            return
        pinned = next((f.sha256 for f in files if f.name == "model.safetensors"), None)
        if pinned and pinned != self.model_sha256:
            raise HashMismatchError(f"{self.name}: model.safetensors sha256 {self.model_sha256} "
                                    f"does not match pinned {pinned}")

    def transcribe_many(self, jobs: list[tuple[np.ndarray, int, str]],
                        progress=None) -> list[list[Note]]:
        """Transcribe several (audio, sr, instrument) jobs with a single model load.

        ``progress(fraction, message)`` is called as chunks finish (fraction over all jobs).
        """
        import torch
        from muscriptor import NoteEndEvent
        from muscriptor.events import ProgressEvent

        results: list[list[Note]] = []
        with self.slot.load(self.name, self._loader) as model:
            self._record_hash()
            for job_index, (audio, sr, instrument) in enumerate(jobs):
                groups = groups_for(instrument)
                wav = torch.from_numpy(to_mono(audio))
                notes: list[Note] = []
                # MuScriptor prints per-chunk timings to stderr; keep the console readable.
                events = []
                with contextlib.redirect_stderr(io.StringIO()):
                    for ev in model.transcribe((wav, sr), instruments=groups):
                        if isinstance(ev, ProgressEvent):
                            if progress is not None and ev.total:
                                progress((job_index + ev.completed / ev.total) / len(jobs),
                                         f"{self.name}: {instrument}")
                        else:
                            events.append(ev)
                for ev in events:
                    if isinstance(ev, NoteEndEvent):
                        st = ev.start_event
                        note = to_note(self.name, instrument, st.instrument, st.pitch,
                                       st.start_time, ev.end_time)
                        if note is not None:
                            notes.append(note)
                results.append(sort_notes(notes))
            del model
        rec = self.slot.records[-1]
        self.last_run = {"device": rec.device, "seconds": round(rec.seconds, 2),
                         "peak_vram_gb": round(rec.peak_gb, 3), "oom_retries": 0,
                         "jobs": len(jobs), "includes_model_load": True}
        return results

    def transcribe(self, audio: np.ndarray, sr: int, instrument: str) -> list[Note]:
        return self.transcribe_many([(audio, sr, instrument)])[0]


class AudiocppMuScriptorTranscriber(MuScriptorTranscriber):
    """MuScriptor through audio.cpp: all jobs run as one request sequence (one model load)."""

    def _snapshot(self) -> Path:
        from bandscribe.models.prepare import muscriptor_snapshot

        try:
            path, self.model_sha256 = muscriptor_snapshot(self.size)
        except HashMismatchError:
            raise
        except Exception as exc:  # noqa: BLE001 - gated, offline or missing: skip the backend
            self.unavailable_reason = (f"MuScriptor weights unavailable ({exc}); accept the "
                                       "licence on Hugging Face and run `uvx hf auth login`")
            raise TranscriberUnavailable(self.unavailable_reason) from exc
        return path

    def transcribe_many(self, jobs: list[tuple[np.ndarray, int, str]],
                        progress=None) -> list[list[Note]]:
        import json
        import tempfile

        import soundfile as sf

        from bandscribe.runtime import audiocpp

        model_dir = self._snapshot()
        with tempfile.TemporaryDirectory(prefix="bandscribe-mus-") as tmp:
            tmp_dir = Path(tmp)
            requests = []
            for i, (audio, sr, instrument) in enumerate(jobs):
                sf.write(tmp_dir / f"job{i}.wav", to_mono(audio), sr, subtype="FLOAT")
                options = {"output_format": "json"}
                groups = groups_for(instrument)
                if groups:
                    options["instruments"] = ",".join(groups)
                requests.append({"id": f"job{i}", "audio": f"job{i}.wav", "options": options})
            (tmp_dir / "requests.json").write_text(json.dumps({"requests": requests}))
            outputs = [tmp_dir / f"notes_job{i}.json" for i in range(len(jobs))]

            def done(n: int) -> None:
                if progress is not None:
                    progress(n / len(jobs), f"{self.name}: {jobs[min(n, len(jobs) - 1)][2]}")

            with self.slot.load(self.name, lambda: None):
                ran = audiocpp.run("midi", "muscriptor", model_dir,
                             ["--request-sequence", str(tmp_dir / "requests.json"),
                              "--text-out", str(tmp_dir / "notes.json")],
                             tmp_dir / "log.txt", watch=outputs, on_done=done)
            results = []
            for (_, _, instrument), out in zip(jobs, outputs, strict=True):
                events = json.loads(out.read_text(encoding="utf-8"))
                starts = {e["index"]: e for e in events if e["type"] == "start"}
                notes = []
                for e in events:
                    st = starts.get(e.get("start_event_index")) if e["type"] == "end" else None
                    if st is not None:
                        note = to_note(self.name, instrument, st["instrument"], st["pitch"],
                                       st["start_time"], e["end_time"])
                        if note is not None:
                            notes.append(note)
                results.append(sort_notes(notes))
        rec = self.slot.records[-1]
        self.last_run = {"device": f"{ran['backend']} (audio.cpp)",
                         "seconds": round(rec.seconds, 2), "peak_vram_gb": 0.0,
                         "oom_retries": 0, "jobs": len(jobs), "includes_model_load": True}
        return results