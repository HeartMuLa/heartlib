import json
import os
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch
import torchaudio
from tokenizers import Tokenizer
from tqdm import tqdm
from transformers import BitsAndBytesConfig
from transformers.pipelines.base import Pipeline
from transformers.utils.generic import ModelOutput

from ..heartcodec.modeling_heartcodec import HeartCodec
from ..heartmula.modeling_heartmula import HeartMuLa
from ..accelerators.torchtune_metal import try_enable_torchtune_metal


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
        super().__init__(model, device=device, dtype=dtype)
        self.model = model
        self.audio_codec = audio_codec
        self.muq_mulan = muq_mulan
        self.text_tokenizer = text_tokenizer
        self.config = config
        self._device = device

        # Optional, opt-in MPS fast path (custom Metal kernels) for torchtune Llama blocks.
        # Enable with: HEARTLIB_ENABLE_MPS_METAL=1
        try:
            try_enable_torchtune_metal(
                self.model,
                enabled=(os.getenv("HEARTLIB_ENABLE_MPS_METAL", "0") == "1"),
                verbose=(os.getenv("HEARTLIB_MPS_METAL_VERBOSE", "0") == "1"),
            )
        except Exception:
            # Never fail inference if optional kernels are unavailable.
            pass

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

    def preprocess(self, input_: Dict[str, Any], **preprocess_parameters: Any):
        cfg_scale: float = preprocess_parameters.get("cfg_scale", 1.5)

        tags = input_["tags"]
        if os.path.isfile(tags):
            with open(tags, encoding="utf-8") as fp:
                tags = fp.read()
        assert isinstance(tags, str), f"tags must be a string, but got {type(tags)}"

        tags = tags.lower()
        if not tags.startswith("<tag>"):
            tags = f"<tag>{tags}"
        if not tags.endswith("</tag>"):
            tags = f"{tags}</tag>"

        tags_ids = self.text_tokenizer.encode(tags).ids
        if tags_ids[0] != self.config.text_bos_id:
            tags_ids = [self.config.text_bos_id] + tags_ids
        if tags_ids[-1] != self.config.text_eos_id:
            tags_ids = tags_ids + [self.config.text_eos_id]

        ref_audio = input_.get("ref_audio", None)
        if ref_audio is not None:
            raise NotImplementedError("ref_audio is not supported yet.")
        muq_embed = torch.zeros([self._muq_dim], dtype=self.dtype)
        muq_idx = len(tags)

        lyrics = input_["lyrics"]
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

        prompt_len = len(tags_ids) + 1 + len(lyrics_ids)

        tokens = torch.zeros([prompt_len, self._parallel_number], dtype=torch.long)
        tokens[: len(tags_ids), -1] = torch.tensor(tags_ids)
        tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids)

        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
        tokens_mask[:, -1] = True

        bs_size = 2 if cfg_scale != 1.0 else 1

        def _cfg_cat(tensor: torch.Tensor, scale: float) -> torch.Tensor:
            tensor = tensor.unsqueeze(0)
            if scale != 1.0:
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
        input_tensors: Dict[str, Any],
        **forward_parameters: Any,
    ) -> ModelOutput:
        max_audio_length_ms: int = forward_parameters.get(
            "max_audio_length_ms", 120_000
        )
        temperature: float = forward_parameters.get("temperature", 1.0)
        topk: int = forward_parameters.get("topk", 50)
        cfg_scale: float = forward_parameters.get("cfg_scale", 1.5)

        prompt_tokens = input_tensors["tokens"]
        prompt_tokens_mask = input_tensors["tokens_mask"]
        continuous_segment = input_tensors["muq_embed"]
        starts = input_tensors["muq_idx"]
        prompt_pos = input_tensors["pos"]

        frames = []

        bs_size = 2 if cfg_scale != 1.0 else 1
        self.model.setup_caches(bs_size)

        device_type = (
            self._device.type if isinstance(self._device, torch.device) else "cpu"
        )
        # Autocast support varies by PyTorch build/version (not all support "mps").
        # Prefer autocast when available, but never fail if unsupported.
        def _autocast_ctx():
            try:
                return torch.autocast(device_type=device_type, dtype=self.dtype)
            except (RuntimeError, TypeError, ValueError):
                return nullcontext()

        autocast_ctx = _autocast_ctx()

        # Keep a stable view of the base position tensor to avoid re-slicing every step.
        base_pos = prompt_pos[..., -1:]

        with torch.inference_mode(), autocast_ctx:
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

            # Preallocate the padded audio token + mask and reuse them every step.
            padded_token = torch.full(
                (curr_token.shape[0], 1, self._parallel_number),
                fill_value=self.config.empty_id,
                device=curr_token.device,
                dtype=torch.long,
            )
            padded_token_mask = torch.ones(
                (curr_token.shape[0], 1, self._parallel_number),
                device=curr_token.device,
                dtype=torch.bool,
            )
            padded_token_mask[..., -1] = False

            max_audio_frames = max_audio_length_ms // 80
            # Preallocate a frame buffer for the *un-padded* audio tokens (first sample only).
            frame_buf = torch.empty(
                (max_audio_frames + 1, curr_token.shape[1]),
                device=curr_token.device,
                dtype=curr_token.dtype,
            )
            frame_buf[0] = curr_token[0]
            frame_len = 1

            for i in tqdm(range(max_audio_frames)):
                padded_token[:, 0, :-1] = curr_token
                curr_token = self.model.generate_frame(
                    tokens=padded_token,
                    tokens_mask=padded_token_mask,
                    input_pos=base_pos + i + 1,
                    temperature=temperature,
                    topk=topk,
                    cfg_scale=cfg_scale,
                    continuous_segments=None,
                    starts=None,
                )
                if torch.any(curr_token[0:1, :] >= self.config.audio_eos_id):
                    break
                frame_buf[frame_len] = curr_token[0]
                frame_len += 1

        frames = frame_buf[:frame_len].transpose(0, 1).contiguous()
        wav = self.audio_codec.detokenize(frames)
        return ModelOutput(wav=wav)

    def postprocess(
        self, model_outputs: ModelOutput, **postprocess_parameters: Any
    ) -> None:
        save_path: str = postprocess_parameters.get("save_path", "output.mp3")
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
                heartmula_path, dtype=dtype, quantization_config=bnb_config
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
