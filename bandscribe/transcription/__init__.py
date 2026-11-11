"""Transcription backends. Use :func:`get_transcriber` to build one by registry id."""

from __future__ import annotations

from bandscribe.transcription.base import Transcriber

TRANSCRIBER_IDS = ("muscriptor_small", "muscriptor_medium", "muscriptor_large", "basic_pitch",
                   "adtof")


def get_transcriber(model_id: str, **opts) -> Transcriber:
    """Build a transcriber.

    ``muscriptor_small|medium|large`` (pitched, drums and full mix), ``basic_pitch`` (pitched
    only), ``adtof`` (drums only). Options are passed to the backend constructor.
    """
    if model_id.startswith("muscriptor_"):
        from bandscribe.transcription.muscriptor import MuScriptorTranscriber

        return MuScriptorTranscriber(model_id.split("_", 1)[1], **opts)
    if model_id == "basic_pitch":
        from bandscribe.transcription.basic_pitch import BasicPitchTranscriber

        return BasicPitchTranscriber(**opts)
    if model_id == "adtof":
        from bandscribe.transcription.adtof import AdtofTranscriber

        return AdtofTranscriber(**opts)
    raise KeyError(f"no transcriber for model id {model_id!r}; known: {TRANSCRIBER_IDS}")
