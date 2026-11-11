"""Basic Pitch (Spotify) through its bundled ONNX model: the instrument-agnostic second opinion."""

from __future__ import annotations

import contextlib
import io
import logging
import tempfile
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

from bandscribe.models.registry import sha256_file
from bandscribe.transcription.base import FAMILY_RANGE, Note, Transcriber, sort_notes, to_mono

MODEL_ID = "basic_pitch"


def _midi_to_hz(m: float) -> float:
    return 440.0 * 2 ** ((m - 69) / 12)


class BasicPitchTranscriber(Transcriber):
    name = MODEL_ID

    def __init__(self, onset_threshold: float = 0.5, frame_threshold: float = 0.3,
                 minimum_note_ms: float = 58.0) -> None:
        self.onset_threshold = onset_threshold
        self.frame_threshold = frame_threshold
        self.minimum_note_ms = minimum_note_ms
        self._model = None
        self.model_sha256: str | None = None

    def _load(self):
        if self._model is None:
            logging.getLogger().setLevel(logging.ERROR)  # basic_pitch warns about absent TF
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                from basic_pitch import ICASSP_2022_MODEL_PATH
                from basic_pitch.inference import Model

                path = Path(ICASSP_2022_MODEL_PATH)
                if path.suffix != ".onnx":
                    path = path.parent / "nmp.onnx"
                self.model_sha256 = sha256_file(path)
                self._model = Model(path)
        return self._model

    def transcribe(self, audio: np.ndarray, sr: int, instrument: str) -> list[Note]:
        if instrument == "drums":
            raise ValueError("Basic Pitch is a pitched-note model; it cannot transcribe drums")
        from basic_pitch.inference import predict

        model = self._load()
        family = instrument if instrument in FAMILY_RANGE else "other"
        lo, hi = FAMILY_RANGE[family]
        with tempfile.TemporaryDirectory() as tmp:
            wav = Path(tmp) / "stem.wav"
            sf.write(wav, to_mono(audio), sr, subtype="FLOAT")
            with contextlib.redirect_stdout(io.StringIO()), warnings.catch_warnings():
                warnings.simplefilter("ignore")
                _, _, events = predict(
                    wav, model, onset_threshold=self.onset_threshold,
                    frame_threshold=self.frame_threshold,
                    minimum_note_length=self.minimum_note_ms,
                    # No minimum_frequency: values below Basic Pitch's 27.5 Hz base wrap to a
                    # negative bin index upstream and silently erase every note.
                    maximum_frequency=_midi_to_hz(hi + 0.5),
                    multiple_pitch_bends=False, melodia_trick=True)
        notes = []
        for start, end, pitch, amplitude, _bends in events:
            if not lo <= int(pitch) <= hi:
                continue
            amp = float(min(1.0, max(0.0, amplitude)))
            notes.append(Note(float(start), float(end), int(pitch),
                              velocity=int(min(127, max(1, round(amp * 127)))), conf=amp,
                              sources=(MODEL_ID,), instrument=family))
        return sort_notes(notes)
