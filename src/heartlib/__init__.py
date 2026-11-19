try:
    from .pipelines.music_generation import HeartMuLaGenPipeline
except ModuleNotFoundError:
    HeartMuLaGenPipeline = None
from .pipelines.minimax_music_generation import (
    MUSIC_GENERATION_PROVIDERS,
    MiniMaxMusicConfig,
    MiniMaxMusicGenerationPipeline,
    build_music_request,
    load_music_generation_pipeline,
)
try:
    from .pipelines.lyrics_transcription import HeartTranscriptorPipeline
except ModuleNotFoundError:
    HeartTranscriptorPipeline = None

__all__ = [
    "HeartMuLaGenPipeline",
    "MUSIC_GENERATION_PROVIDERS",
    "MiniMaxMusicConfig",
    "MiniMaxMusicGenerationPipeline",
    "build_music_request",
    "load_music_generation_pipeline",
    "HeartTranscriptorPipeline"
]
