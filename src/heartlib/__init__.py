"""HeartMuLa: A family of open-sourced music foundation models.

This package provides high-level pipelines for music generation and lyrics
transcription built on top of the HeartMuLa language model, the HeartCodec
audio codec, and the HeartTranscriptor whisper-based lyrics transcriber.

Typical usage::

    from heartlib import HeartMuLaGenPipeline

    pipeline = HeartMuLaGenPipeline.from_pretrained(
        "./ckpt", device=torch.device("cuda"), dtype=torch.bfloat16, version="3B",
    )
    pipeline({"tags": "piano,happy", "lyrics": "Hello world"})
"""

from .pipelines.music_generation import HeartMuLaGenPipeline
from .pipelines.lyrics_transcription import HeartTranscriptorPipeline

__all__ = [
    "HeartMuLaGenPipeline",
    "HeartTranscriptorPipeline"
]