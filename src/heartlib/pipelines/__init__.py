try:
    from .music_generation import HeartMuLaGenPipeline
except ModuleNotFoundError:
    HeartMuLaGenPipeline = None
from .minimax_music_generation import (
    MUSIC_GENERATION_PROVIDERS,
    MiniMaxMusicConfig,
    MiniMaxMusicGenerationPipeline,
    build_music_request,
    load_music_generation_pipeline,
)

__all__ = [
    "HeartMuLaGenPipeline",
    "MUSIC_GENERATION_PROVIDERS",
    "MiniMaxMusicConfig",
    "MiniMaxMusicGenerationPipeline",
    "build_music_request",
    "load_music_generation_pipeline",
]
