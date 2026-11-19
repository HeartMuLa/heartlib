import unittest

from heartlib.pipelines.minimax_music_generation import (
    MiniMaxMusicConfig,
    MiniMaxMusicGenerationPipeline,
    build_music_request,
)


class MiniMaxMusicGenerationTests(unittest.TestCase):
    def test_build_music_request_includes_supported_fields(self):
        payload = build_music_request(
            model="music-3.0",
            prompt="create a synthwave track",
            lyrics="verse one",
            output_format="url",
            audio_setting={"format": "mp3"},
            region="global_en",
        )
        self.assertEqual(payload["model"], "music-3.0")
        self.assertEqual(payload["prompt"], "create a synthwave track")
        self.assertEqual(payload["lyrics"], "verse one")
        self.assertEqual(payload["audio_setting"]["format"], "mp3")

    def test_pipeline_uses_region_base_url(self):
        pipeline = MiniMaxMusicGenerationPipeline(
            api_key="test-key",
            region="cn_zh",
            base_url=None,
        )
        self.assertEqual(pipeline.base_url, "https://api.minimaxi.com/v1")
        self.assertEqual(pipeline.region_config.region, "cn_zh")
        self.assertEqual(MiniMaxMusicConfig().default_model, "music-3.0")


if __name__ == "__main__":
    unittest.main()
