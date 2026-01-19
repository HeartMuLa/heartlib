import json
import os
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Any, Dict, Optional

import torch
import torchaudio
import torch.nn.functional as F
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
            "codes_path": kwargs.get("codes_path", None),
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

        def _load_ref_audio(ref: Any) -> tuple[torch.Tensor, int]:
            """
            Returns (mono_waveform, sample_rate) where mono_waveform is 1D [T].
            """
            if isinstance(ref, str):
                wav, sr = torchaudio.load(ref)
            elif isinstance(ref, torch.Tensor):
                wav = ref
                sr = int(input_.get("ref_audio_sr", 0) or 0)
                if sr <= 0:
                    raise ValueError(
                        "ref_audio was provided as a Tensor but `ref_audio_sr` was missing/invalid."
                    )
            else:
                raise TypeError(
                    f"ref_audio must be a file path or torch.Tensor, got {type(ref)}"
                )

            # Accept [T], [C,T], or [B,C,T] (take the first batch).
            if wav.ndim == 3:
                wav = wav[0]
            if wav.ndim == 2:
                wav = wav.mean(dim=0)
            elif wav.ndim != 1:
                raise ValueError(f"Unsupported ref_audio tensor shape: {tuple(wav.shape)}")

            wav = wav.to(dtype=torch.float32)
            return wav, int(sr)

        def _prepare_muq_audio(wav: torch.Tensor, sr: int) -> torch.Tensor:
            """
            Resample to MuQ sample rate (default 24k) and take/pad a ~10s segment.
            Returns waveform shaped [1, T] on self._device.
            """
            muq_sr = int(input_.get("muq_sample_rate", 24_000))
            seg_s = float(input_.get("muq_segment_sec", 10.0))
            seg_len = max(1, int(round(muq_sr * seg_s)))

            if sr != muq_sr:
                wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=muq_sr)

            if wav.numel() >= seg_len:
                start = (wav.numel() - seg_len) // 2
                wav = wav[start : start + seg_len]
            else:
                wav = F.pad(wav, (0, seg_len - wav.numel()))

            # Common MuQ-style encoders expect [B, T].
            return wav.unsqueeze(0).to(device=self._device)

        def _run_muq_mulan(audio_bt: torch.Tensor, sample_rate: int) -> torch.Tensor:
            """
            Runs the provided MuQ-MuLan model and returns a 1D [muq_dim] embedding.
            Tries a few common APIs / output layouts.
            """
            if self.muq_mulan is None:
                raise ValueError(
                    "ref_audio was provided but `muq_mulan` is None. "
                    "Pass a pretrained MuQ-MuLan model to HeartMuLaGenPipeline."
                )

            model = self.muq_mulan
            was_training = getattr(model, "training", False)
            if hasattr(model, "eval"):
                model.eval()

            with torch.inference_mode():
                out = None
                # Common: model.encode_audio(audio, sample_rate=...)
                if hasattr(model, "encode_audio") and callable(getattr(model, "encode_audio")):
                    try:
                        out = model.encode_audio(audio_bt, sample_rate=sample_rate)
                    except TypeError:
                        out = model.encode_audio(audio_bt)
                # Fallback: callable model(audio, sample_rate=...)
                if out is None and callable(model):
                    try:
                        out = model(audio_bt, sample_rate=sample_rate)
                    except TypeError:
                        out = model(audio_bt)

            if was_training and hasattr(model, "train"):
                model.train()

            def _to_tensor(x: Any) -> Optional[torch.Tensor]:
                if x is None:
                    return None
                if isinstance(x, torch.Tensor):
                    return x
                if isinstance(x, (tuple, list)) and x:
                    return _to_tensor(x[0])
                if isinstance(x, (dict, ModelOutput)):
                    for k in (
                        "joint_embedding",
                        "joint_embeds",
                        "embedding",
                        "embeddings",
                        "audio_embedding",
                        "audio_embeds",
                        "audio_embed",
                        "audio_features",
                        "audio_feature",
                    ):
                        if k in x:
                            return _to_tensor(x[k])
                for attr in (
                    "joint_embedding",
                    "embedding",
                    "embeddings",
                    "audio_embedding",
                    "audio_embeds",
                    "audio_features",
                ):
                    if hasattr(x, attr):
                        return _to_tensor(getattr(x, attr))
                return None

            emb = _to_tensor(out)
            if emb is None:
                raise ValueError(
                    "Could not extract an embedding from `muq_mulan` output. "
                    "Expected a Tensor or a dict/ModelOutput with an embedding field."
                )

            # Accept [D], [1,D], or [B,D] (take first).
            emb = emb.detach()
            if emb.ndim == 2:
                emb = emb[0]
            elif emb.ndim != 1:
                raise ValueError(f"Unsupported muq embedding shape: {tuple(emb.shape)}")

            if emb.numel() != self._muq_dim:
                raise ValueError(
                    f"MuQ-MuLan embedding dim mismatch: expected {self._muq_dim}, got {emb.numel()}."
                )

            # Normalize is common for joint embeddings; safe and improves conditioning stability.
            emb = emb / (emb.norm(p=2) + 1e-12)
            return emb.to(device="cpu", dtype=self.dtype)

        ref_audio = input_.get("ref_audio", None)
        if ref_audio is not None:
            wav, sr = _load_ref_audio(ref_audio)
            muq_sr = int(input_.get("muq_sample_rate", 24_000))
            audio_bt = _prepare_muq_audio(wav, sr)
            muq_embed = _run_muq_mulan(audio_bt, sample_rate=muq_sr)
        else:
            muq_embed = torch.zeros([self._muq_dim], dtype=self.dtype)

        # The reserved slot is the blank "+1" token after tags_ids.
        muq_idx = len(tags_ids)

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
        # Include tokens in the output so postprocess can optionally persist them.
        # This is opt-in (see postprocess `codes_path`) and does not change default behavior.
        return ModelOutput(wav=wav, codes=frames.detach().cpu())

    def postprocess(
        self, model_outputs: ModelOutput, **postprocess_parameters: Any
    ) -> None:
        save_path: str = postprocess_parameters.get("save_path", "output.mp3")
        codes_path: Optional[str] = postprocess_parameters.get("codes_path", None)
        wav = model_outputs["wav"]
        torchaudio.save(save_path, wav, 48000)
        if codes_path:
            codes = model_outputs.get("codes", None)
            if codes is None:
                raise ValueError(
                    "codes_path was provided but no `codes` were found in model outputs."
                )
            torch.save(codes, codes_path)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        device: torch.device,
        dtype: torch.dtype,
        version: str,
        bnb_config: Optional[BitsAndBytesConfig] = None,
        *,
        load_muq_mulan: bool = False,
        muq_model_id: Optional[str] = None,
        muq_cache_dir: Optional[str] = None,
        muq_revision: Optional[str] = None,
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

        # Optional: load MuQ-MuLan from Hugging Face (auto-download + cache).
        # Enable via argument or env HEARTLIB_LOAD_MUQ_MULAN=1.
        if not load_muq_mulan:
            load_muq_mulan = os.getenv("HEARTLIB_LOAD_MUQ_MULAN", "0") == "1"

        muq_mulan = None
        if load_muq_mulan:
            model_id = (
                muq_model_id
                or os.getenv("HEARTLIB_MUQ_MULAN_ID", "").strip()
                or "OpenMuQ/MuQ-MuLan-large"
            )
            try:
                # MuQ's own library wraps Hugging Face download via .from_pretrained().
                # Install: pip install muq
                from muq import MuQMuLan  # type: ignore
            except Exception as e:  # pragma: no cover
                raise ImportError(
                    "MuQ-MuLan auto-download requested, but the `muq` package is not installed. "
                    "Install it with: pip install muq"
                ) from e

            kwargs: Dict[str, Any] = {}
            if muq_cache_dir is not None:
                kwargs["cache_dir"] = muq_cache_dir
            if muq_revision is not None:
                kwargs["revision"] = muq_revision

            muq_mulan = MuQMuLan.from_pretrained(model_id, **kwargs)
            if hasattr(muq_mulan, "to"):
                muq_mulan = muq_mulan.to(device)
            if hasattr(muq_mulan, "eval"):
                muq_mulan.eval()

        return cls(heartmula, heartcodec, muq_mulan, tokenizer, gen_config, device, dtype)
