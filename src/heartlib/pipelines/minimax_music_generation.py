from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional
from urllib import error, request


MUSIC_REGIONAL_ENDPOINTS = {
    "global_en": {
        "openai_base_url": "https://api.minimax.io/v1",
        "anthropic_base_url": "https://api.minimax.io/anthropic",
        "docs_root": "https://platform.minimax.io/docs",
    },
    "cn_zh": {
        "openai_base_url": "https://api.minimaxi.com/v1",
        "anthropic_base_url": "https://api.minimaxi.com/anthropic",
        "docs_root": "https://platform.minimaxi.com/docs",
    },
}

MUSIC_MODELS = ("music-3.0", "music-2.6", "music-3.0-free", "music-2.6-free")
MUSIC_COVER_MODELS = ("music-cover", "music-cover-free")
MUSIC_OUTPUT_FORMATS = ("url", "hex")
MUSIC_AUDIO_FORMATS = ("mp3", "wav", "pcm")

MUSIC_REFERENCE = {
    "operation_id": "generateMusic",
    "method": "POST",
    "authorization": "Bearer",
    "content_type": "application/json",
    "required_fields": ["model"],
    "request_fields": [
        "model",
        "prompt",
        "lyrics",
        "stream",
        "output_format",
        "audio_setting",
        "lyrics_optimizer",
        "is_instrumental",
        "audio_url",
        "audio_base64",
        "cover_feature_id",
    ],
    "output_formats": list(MUSIC_OUTPUT_FORMATS),
    "stream_output_formats": ["hex"],
    "audio_formats": list(MUSIC_AUDIO_FORMATS),
    "url_ttl_hours": 24,
    "regional_fields": {"global_en": [], "cn_zh": ["aigc_watermark"]},
    "cover": {
        "input_one_of": ["audio_url", "audio_base64"],
        "input_min_seconds": 6,
        "input_max_seconds": 360,
        "input_max_mb": 50,
    },
    "response": {
        "status_field": "data.status",
        "status_values": {"in_progress": 1, "completed": 2},
        "audio_field": "data.audio",
        "success_code_field": "base_resp.status_code",
        "success_code": 0,
        "task_id_field": None,
        "query_endpoint": None,
    },
}


@dataclass(frozen=True)
class MiniMaxMusicRegionConfig:
    region: str
    openai_base_url: str
    anthropic_base_url: str
    docs_root: str


@dataclass(frozen=True)
class MiniMaxMusicConfig:
    provider_name: str = "MiniMax"
    default_model: str = "music-3.0"
    model_ids: tuple[str, ...] = MUSIC_MODELS
    cover_model_ids: tuple[str, ...] = MUSIC_COVER_MODELS
    output_formats: tuple[str, ...] = MUSIC_OUTPUT_FORMATS
    audio_formats: tuple[str, ...] = MUSIC_AUDIO_FORMATS
    regions: tuple[MiniMaxMusicRegionConfig, ...] = field(
        default_factory=lambda: tuple(
            MiniMaxMusicRegionConfig(region=name, **config)
            for name, config in MUSIC_REGIONAL_ENDPOINTS.items()
        )
    )
    reference: Mapping[str, Any] = field(default_factory=lambda: dict(MUSIC_REFERENCE))

    def region_config(self, region: str) -> MiniMaxMusicRegionConfig:
        for region_config in self.regions:
            if region_config.region == region:
                return region_config
        raise ValueError(f"Unsupported MiniMax region: {region}")


def build_music_request(
    *,
    model: str,
    prompt: Optional[str] = None,
    lyrics: Optional[str] = None,
    stream: Optional[bool] = None,
    output_format: Optional[str] = None,
    audio_setting: Optional[Dict[str, Any]] = None,
    lyrics_optimizer: Optional[bool] = None,
    is_instrumental: Optional[bool] = None,
    audio_url: Optional[str] = None,
    audio_base64: Optional[str] = None,
    cover_feature_id: Optional[str] = None,
    region: str = "global_en",
    **kwargs: Any,
) -> Dict[str, Any]:
    config = MiniMaxMusicConfig()
    config.region_config(region)
    if model not in config.model_ids:
        raise ValueError(f"Unsupported MiniMax music model: {model}")
    if output_format is not None and output_format not in config.output_formats:
        raise ValueError(f"Unsupported output format: {output_format}")

    payload: Dict[str, Any] = {"model": model}
    for key, value in (
        ("prompt", prompt),
        ("lyrics", lyrics),
        ("stream", stream),
        ("output_format", output_format),
        ("audio_setting", audio_setting),
        ("lyrics_optimizer", lyrics_optimizer),
        ("is_instrumental", is_instrumental),
        ("audio_url", audio_url),
        ("audio_base64", audio_base64),
        ("cover_feature_id", cover_feature_id),
    ):
        if value is not None:
            payload[key] = value

    if region == "cn_zh" and "aigc_watermark" in kwargs:
        payload["aigc_watermark"] = kwargs["aigc_watermark"]

    for key, value in kwargs.items():
        if key != "aigc_watermark" and value is not None:
            payload[key] = value

    if "audio_setting" in payload and isinstance(payload["audio_setting"], dict):
        audio_setting_value = payload["audio_setting"]
        if "format" in audio_setting_value and audio_setting_value["format"] not in config.audio_formats:
            raise ValueError(f"Unsupported audio format: {audio_setting_value['format']}")

    return payload


class MiniMaxMusicGenerationPipeline:
    def __init__(
        self,
        api_key: Optional[str] = None,
        *,
        model: str = "music-3.0",
        region: str = "global_en",
        base_url: Optional[str] = None,
        output_format: str = "url",
        timeout: float = 60.0,
    ) -> None:
        self.config = MiniMaxMusicConfig()
        self.region_config = self.config.region_config(region)
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY", "")
        if not self.api_key:
            raise ValueError("MiniMax API key is required.")
        if model not in self.config.model_ids:
            raise ValueError(f"Unsupported MiniMax music model: {model}")
        if output_format not in self.config.output_formats:
            raise ValueError(f"Unsupported output format: {output_format}")

        self.model = model
        self.output_format = output_format
        self.timeout = timeout
        self.base_url = (base_url or self.region_config.openai_base_url).rstrip("/")

    @classmethod
    def from_environment(
        cls,
        *,
        model: str = "music-3.0",
        region: str = "global_en",
        base_url: Optional[str] = None,
        output_format: str = "url",
        timeout: float = 60.0,
    ) -> "MiniMaxMusicGenerationPipeline":
        return cls(
            api_key=os.getenv("MINIMAX_API_KEY"),
            model=model,
            region=region,
            base_url=base_url,
            output_format=output_format,
            timeout=timeout,
        )

    def _endpoint(self) -> str:
        return f"{self.base_url}/music_generation"

    def _request(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        req = request.Request(
            self._endpoint(),
            data=body,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"MiniMax music generation failed: {exc.code} {detail}") from exc

    def generate(self, **kwargs: Any) -> Dict[str, Any]:
        payload = build_music_request(
            model=kwargs.pop("model", self.model),
            output_format=kwargs.pop("output_format", self.output_format),
            region=self.region_config.region,
            **kwargs,
        )
        response = self._request(payload)
        data = response.get("data", {})
        return {
            "status": data.get("status"),
            "audio": data.get("audio"),
            "base_resp": response.get("base_resp", {}),
            "raw": response,
        }

    def __call__(self, inputs: Dict[str, Any], **kwargs: Any) -> Dict[str, Any]:
        payload = dict(inputs)
        payload.update(kwargs)
        return self.generate(**payload)


def load_music_generation_pipeline(provider: str, **kwargs: Any) -> Any:
    normalized_provider = provider.strip().lower()
    if normalized_provider in {"heartmula", "local"}:
        from .music_generation import HeartMuLaGenPipeline

        return HeartMuLaGenPipeline.from_pretrained(**kwargs)
    if normalized_provider == "minimax":
        return MiniMaxMusicGenerationPipeline(**kwargs)
    raise ValueError(f"Unsupported music generation provider: {provider}")


MUSIC_GENERATION_PROVIDERS = {
    "minimax": MiniMaxMusicGenerationPipeline,
}
