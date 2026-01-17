from transformers.pipelines.base import Pipeline
from tokenizers import Tokenizer
from ..heartmula.modeling_heartmula import HeartMuLa
from ..heartcodec.modeling_heartcodec import HeartCodec
import torch
from typing import Dict, Any, Optional
import os
from dataclasses import dataclass
from tqdm import tqdm
import torchaudio
import json
from transformers import BitsAndBytesConfig


@dataclass
class HeartMuLaGenConfig:
    text_bos_id: int = 128000
    text_eos_id: int = 128001
    audio_eos_id: int = 8193
    empty_id: int = 0

    @classmethod
    def from_file(cls, path: str):
        with open(path, encoding="utf-8") as fp:
            data = json.load(fp)
        return cls(**data)


class HeartMuLaGenPipeline(Pipeline):
    def __init__(
        self,
        model: HeartMuLa,
        audio_codec: HeartCodec,
        muq_mulan: Optional[Any],
        text_tokenizer: Tokenizer,
        config: HeartMuLaGenConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__(model, dtype=dtype)
        self.model = model
        self.audio_codec = audio_codec
        self.muq_mulan = muq_mulan
        self.text_tokenizer = text_tokenizer
        self.config = config

        self._parallel_number = audio_codec.config.num_quantizers + 1
        self._muq_dim = model.config.muq_dim

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {"cfg_scale": kwargs.get("cfg_scale", 1.5)}
        forward_kwargs = {
            "max_audio_length_ms": kwargs.get("max_audio_length_ms", 120_000),
            "temperature": kwargs.get("temperature", 1.0),
            "topk": kwargs.get("topk", 50),
            "cfg_scale": kwargs.get("cfg_scale", 1.5),
        }
        postprocess_kwargs = {
            "save_path": kwargs.get("save_path", "output.mp3"),
        }
        return preprocess_kwargs, forward_kwargs, postprocess_kwargs

    def preprocess(self, inputs: Dict[str, Any], cfg_scale: float):

        # process tags
        tags = inputs["tags"]
        if os.path.isfile(tags):
            with open(tags, encoding="utf-8") as fp:
                tags = fp.read()
        assert isinstance(tags, str), f"tags must be a string, but got {type(tags)}"

        tags = tags.lower()
        # encapsulate with special <tag> and </tag> tokens
        if not tags.startswith("<tag>"):
            tags = f"<tag>{tags}"
        if not tags.endswith("</tag>"):
            tags = f"{tags}</tag>"

        tags_ids = self.text_tokenizer.encode(tags).ids
        if tags_ids[0] != self.config.text_bos_id:
            tags_ids = [self.config.text_bos_id] + tags_ids
        if tags_ids[-1] != self.config.text_eos_id:
            tags_ids = tags_ids + [self.config.text_eos_id]

        # process reference audio
        ref_audio = inputs.get("ref_audio", None)
        if ref_audio is not None:
            raise NotImplementedError("ref_audio is not supported yet.")
        muq_embed = torch.zeros([self._muq_dim], dtype=self.dtype)
        muq_idx = len(tags)

        # process lyrics
        lyrics = inputs["lyrics"]
        if os.path.isfile(lyrics):
            with open(lyrics, encoding="utf-8") as fp:
                lyrics = fp.read()
        assert isinstance(
            lyrics, str
        ), f"lyrics must be a string, but got {type(lyrics)}"
        lyrics = lyrics.lower()

        lyrics_ids = self.text_tokenizer.encode(lyrics).ids
        if lyrics_ids[0] != self.config.text_bos_id:
            lyrics_ids = [self.config.text_bos_id] + lyrics_ids
        if lyrics_ids[-1] != self.config.text_eos_id:
            lyrics_ids = lyrics_ids + [self.config.text_eos_id]

        # cat them together. tags, ref_audio, lyrics
        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)

        tokens = torch.zeros([prompt_len, self._parallel_number], dtype=torch.long)
        tokens[: len(tags_ids), -1] = torch.tensor(tags_ids)
        tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids)

        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens_mask[:, -1] = True

        bs_size = 2 if cfg_scale != 1.0 else 1

        def _cfg_cat(tensor: torch.Tensor, cfg_scale: float):
            tensor = tensor.unsqueeze(0)
            if cfg_scale != 1.0:
                tensor = torch.cat([tensor, tensor], dim=0)
            return tensor

        return {
            "tokens": _cfg_cat(tokens, cfg_scale),
            "tokens_mask": _cfg_cat(tokens_mask, cfg_scale),
            "muq_embed": _cfg_cat(muq_embed, cfg_scale),
            "muq_idx": [muq_idx] * bs_size,
            "pos": _cfg_cat(torch.arange(prompt_len, dtype=torch.long), cfg_scale),
        }

    def _forward(
        self,
        model_inputs: Dict[str, Any],
        max_audio_length_ms: int,
        temperature: float,
        topk: int,
        cfg_scale: float,
    ):
        prompt_tokens = model_inputs["tokens"]
        prompt_tokens_mask = model_inputs["tokens_mask"]
        continuous_segment = model_inputs["muq_embed"]
        starts = model_inputs["muq_idx"]
        prompt_pos = model_inputs["pos"]

        frames = []

        bs_size = 2 if cfg_scale != 1.0 else 1
        self.model.setup_caches(bs_size)
        with torch.autocast(device_type=self.device.type, dtype=self.dtype):
            curr_token = self.model.generate_frame(
                tokens=prompt_tokens,
                tokens_mask=prompt_tokens_mask,
                input_pos=prompt_pos,
                temperature=temperature,
                topk=topk,
                cfg_scale=cfg_scale,
                continuous_segments=continuous_segment,
                starts=starts,
            )
        frames.append(curr_token[0:1,])

        def _pad_audio_token(token: torch.Tensor):
            padded_token = (
                torch.ones(
                    (token.shape[0], self._parallel_number),
                    device=token.device,
                    dtype=torch.long,
                )
                * self.config.empty_id
            )
            padded_token[:, :-1] = token
            padded_token = padded_token.unsqueeze(1)
            padded_token_mask = torch.ones_like(
                padded_token, device=token.device, dtype=torch.bool
            )
            padded_token_mask[..., -1] = False
            return padded_token, padded_token_mask

        max_audio_frames = max_audio_length_ms // 80

        for i in tqdm(range(max_audio_frames)):
            curr_token, curr_token_mask = _pad_audio_token(curr_token)
            with torch.autocast(device_type=self.device.type, dtype=self.dtype):
                curr_token = self.model.generate_frame(
                    tokens=curr_token,
                    tokens_mask=curr_token_mask,
                    input_pos=prompt_pos[..., -1:] + i + 1,
                    temperature=temperature,
                    topk=topk,
                    cfg_scale=cfg_scale,
                    continuous_segments=None,
                    starts=None,
                )
            if torch.any(curr_token[0:1, :] >= self.config.audio_eos_id):
                break
            frames.append(curr_token[0:1,])
        frames = torch.stack(frames).permute(1, 2, 0).squeeze(0)
        wav = self.audio_codec.detokenize(frames)
        return {"wav": wav}

    def postprocess(self, model_outputs: Dict[str, Any], save_path: str):
        wav = model_outputs["wav"]
        torchaudio.save(save_path, wav, 48000)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        device: torch.device,
        dtype: torch.dtype,
        version: str,
        bnb_config: Optional[BitsAndBytesConfig] = None,
    ):

        if os.path.exists(
            heartcodec_path := os.path.join(pretrained_path, "HeartCodec-oss")
        ):
            heartcodec = HeartCodec.from_pretrained(heartcodec_path, device_map=device)
        else:
            raise FileNotFoundError(
                f"Expected to find checkpoint for HeartCodec at {heartcodec_path} but not found. Please check your folder {pretrained_path}."
            )

        if os.path.exists(
            heartmula_path := os.path.join(pretrained_path, f"HeartMuLa-oss-{version}")
        ):
            heartmula = HeartMuLa.from_pretrained(
                heartmula_path, dtype=dtype, quantization_config=bnb_config, device_map=device
            )
        else:
            raise FileNotFoundError(
                f"Expected to find checkpoint for HeartMuLa at {heartmula_path} but not found. Please check your folder {pretrained_path}."
            )

        if os.path.isfile(
            vocab_path := os.path.join(pretrained_path, "tokenizer.json")
        ):
            tokenizer = Tokenizer.from_file(vocab_path)
        else:
            raise FileNotFoundError(
                f"Expected to find tokenizer.json for HeartMuLa at {vocab_path} but not found. Please check your folder {pretrained_path}."
            )

        if os.path.isfile(
            gen_config_path := os.path.join(pretrained_path, "gen_config.json")
        ):
            gen_config = HeartMuLaGenConfig.from_file(gen_config_path)
        else:
            raise FileNotFoundError(
                f"Expected to find gen_config.json for HeartMuLa at {gen_config_path} but not found. Please check your folder {pretrained_path}."
            )

        return cls(heartmula, heartcodec, None, tokenizer, gen_config, device, dtype)
