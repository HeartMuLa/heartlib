from tokenizers import Tokenizer
from ..heartmula.modeling_heartmula import HeartMuLa
from ..heartcodec.modeling_heartcodec import HeartCodec
import torch
from typing import Dict, Any, Optional, Union
import os
from dataclasses import dataclass
from tqdm import tqdm
import torchaudio
import json
from contextlib import contextmanager
import gc


def _resolve_paths(pretrained_path: str, version: str):

    heartmula_path = os.path.join(pretrained_path, f"HeartMuLa-oss-{version}")
    heartcodec_path = os.path.join(pretrained_path, "HeartCodec-oss")
    tokenizer_path = os.path.join(pretrained_path, "tokenizer.json")
    gen_config_path = os.path.join(pretrained_path, "gen_config.json")

    if not os.path.exists(heartmula_path):
        raise FileNotFoundError(
            f"Expected to find checkpoint for HeartMuLa at {heartmula_path} but not found. Please check your folder {pretrained_path}."
        )
    if not os.path.exists(heartcodec_path):
        raise FileNotFoundError(
            f"Expected to find checkpoint for HeartCodec at {heartcodec_path} but not found. Please check your folder {pretrained_path}."
        )
    if not os.path.isfile(tokenizer_path):
        raise FileNotFoundError(
            f"Expected to find tokenizer.json for HeartMuLa at {tokenizer_path} but not found. Please check your folder {pretrained_path}."
        )
    if not os.path.isfile(gen_config_path):
        raise FileNotFoundError(
            f"Expected to find gen_config.json for HeartMuLa at {gen_config_path} but not found. Please check your folder {pretrained_path}."
        )

    return heartmula_path, heartcodec_path, tokenizer_path, gen_config_path


def _resolve_devices(
    device: Union[torch.device, Dict[str, torch.device]], lazy_load: bool
):
    if isinstance(device, torch.device):
        print(f"All model components will be loaded to device: {device}.")
        mula_device = device
        codec_device = device
    elif isinstance(device, dict):
        print("Model components will be loaded to devices as specified:")
        for k, v in device.items():
            print(f"  {k}: {v}")
        mula_device = device["mula"]
        codec_device = device["codec"]
    else:
        raise ValueError(
            "device must be either torch.device or Dict[str, torch.device]"
        )

    single_device = mula_device == codec_device
    if not single_device:
        print(
            f"HeartMuLa and HeartCodec will be loaded to different devices. In this case, lazy_load is turned off."
        )
        lazy_load = False

    return mula_device, codec_device, lazy_load


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


class HeartMuLaGenPipeline:
    def __init__(
        self,
        heartmula_path: str,
        heartcodec_path: str,
        heartmula_device: torch.device,
        heartcodec_device: torch.device,
        heartmula_dtype: torch.dtype,
        heartcodec_dtype: torch.dtype,
        lazy_load: bool,
        muq_mulan: Optional[Any],
        text_tokenizer: Tokenizer,
        config: HeartMuLaGenConfig,
    ):

        self.muq_mulan = muq_mulan
        self.text_tokenizer = text_tokenizer
        self.config = config

        # Remain fixed here for simplicity.
        self._parallel_number = 8 + 1
        self._muq_dim = 512

        self.mula_dtype = heartmula_dtype
        self.mula_path = heartmula_path
        self.mula_device = heartmula_device
        self.codec_dtype = heartcodec_dtype
        self.codec_path = heartcodec_path
        self.codec_device = heartcodec_device

        self._mula: Optional[HeartMuLa] = None
        self._codec: Optional[HeartCodec] = None
        if not lazy_load:
            print(
                f"You have set lazy_load = False. Loading HeartMuLa and HeartCodec onto device..."
            )
            self._mula = HeartMuLa.from_pretrained(
                self.mula_path,
                device_map=self.mula_device,
                dtype=self.mula_dtype,
            )
            self._codec = HeartCodec.from_pretrained(
                self.codec_path,
                device_map=self.codec_device,
                dtype=self.codec_dtype,
            )
        self.lazy_load = lazy_load

    @property
    def mula(self) -> HeartMuLa:
        if isinstance(self._mula, HeartMuLa):
            return self._mula
        self._mula = HeartMuLa.from_pretrained(
            self.mula_path,
            device_map=self.mula_device,
            dtype=self.mula_dtype,
        )
        return self._mula

    @property
    def codec(self) -> HeartCodec:
        if isinstance(self._codec, HeartCodec):
            return self._codec
        self._codec = HeartCodec.from_pretrained(
            self.codec_path,
            device_map=self.codec_device,
            dtype=self.codec_dtype,
        )
        return self._codec

    def _unload(self):
        if not self.lazy_load:
            return
        if isinstance(self._mula, HeartMuLa):
            print(f"You have set lazy_load=True. Unloading HeartMuLa from device.")
            print(
                f"CUDA memory before unloading: {torch.cuda.memory_allocated(self.mula_device) / 1024**3:.2f} GB"
            )
            del self._mula
            gc.collect()
            torch.cuda.empty_cache()
            print(
                f"CUDA memory after unloading: {torch.cuda.memory_allocated(self.mula_device) / 1024**3:.2f} GB"
            )
            self._mula = None
        if isinstance(self._codec, HeartCodec):
            print(f"You have set lazy_load=True. Unloading HeartCodec from device.")
            print(
                f"CUDA memory before unloading: {torch.cuda.memory_allocated(self.codec_device) / 1024**3:.2f} GB"
            )
            del self._codec
            gc.collect()
            torch.cuda.empty_cache()
            print(
                f"CUDA memory after unloading: {torch.cuda.memory_allocated(self.codec_device) / 1024**3:.2f} GB"
            )
            self._codec = None
        return

    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {"cfg_scale": kwargs.get("cfg_scale", 1.5)}
        forward_kwargs = {
            "max_audio_length_ms": kwargs.get("max_audio_length_ms", 120_000),
            "temperature": kwargs.get("temperature", 1.0),
            "topk": kwargs.get("topk", 50),
            "cfg_scale": kwargs.get("cfg_scale", 1.5),
            "seed": kwargs.get("seed", None),
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
        muq_embed = torch.zeros([self._muq_dim], dtype=self.mula_dtype)
        muq_idx = len(tags_ids)

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

    @torch.inference_mode()
    def _forward(
        self,
        model_inputs: Dict[str, Any],
        max_audio_length_ms: int,
        temperature: float,
        topk: int,
        cfg_scale: float,
        seed: Optional[int] = None,
    ):
        if seed is None:
            seed = torch.randint(0, 2 ** 32, (1,)).item()
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

        prompt_tokens = model_inputs["tokens"].to(self.mula_device)
        prompt_tokens_mask = model_inputs["tokens_mask"].to(self.mula_device)
        continuous_segment = model_inputs["muq_embed"].to(self.mula_device)
        starts = model_inputs["muq_idx"]
        prompt_pos = model_inputs["pos"].to(self.mula_device)
        frames = []

        bs_size = 2 if cfg_scale != 1.0 else 1
        self.mula.setup_caches(bs_size)
        with torch.autocast(device_type=self.mula_device.type, dtype=self.mula_dtype):
            curr_token = self.mula.generate_frame(
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
            with torch.autocast(
                device_type=self.mula_device.type, dtype=self.mula_dtype
            ):
                curr_token = self.mula.generate_frame(
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
        self._unload()
        return {"frames": frames, "seed": seed}

    def postprocess(self, model_outputs: Dict[str, Any], save_path: str):
        seed = model_outputs.get("seed")
        frames = model_outputs["frames"].to(self.codec_device)
        wav = self.codec.detokenize(frames)
        self._unload()
        save_path = save_path.replace("_SEED_", str(seed))
        torchaudio.save(save_path, wav.to(torch.float32).cpu(), 48000)
        return {"save_path": save_path, "seed": seed}

    def __call__(self, inputs: Dict[str, Any], **kwargs):
        preprocess_kwargs, forward_kwargs, postprocess_kwargs = (
            self._sanitize_parameters(**kwargs)
        )
        model_inputs = self.preprocess(inputs, **preprocess_kwargs)
        model_outputs = self._forward(model_inputs, **forward_kwargs)
        return self.postprocess(model_outputs, **postprocess_kwargs)

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        device: Union[torch.device, Dict[str, torch.device]],
        dtype: Union[torch.dtype, Dict[str, torch.dtype]],
        version: str,
        lazy_load: bool = False,
    ):

        mula_path, codec_path, tokenizer_path, gen_config_path = _resolve_paths(
            pretrained_path, version
        )
        mula_device, codec_device, lazy_load = _resolve_devices(device, lazy_load)
        tokenizer = Tokenizer.from_file(tokenizer_path)
        gen_config = HeartMuLaGenConfig.from_file(gen_config_path)

        mula_dtype = dtype["mula"] if isinstance(dtype, dict) else dtype
        codec_dtype = dtype["codec"] if isinstance(dtype, dict) else dtype

        return cls(
            heartmula_path=mula_path,
            heartcodec_path=codec_path,
            heartmula_device=mula_device,
            heartcodec_device=codec_device,
            lazy_load=lazy_load,
            muq_mulan=None,
            text_tokenizer=tokenizer,
            config=gen_config,
            heartmula_dtype=mula_dtype,
            heartcodec_dtype=codec_dtype,
        )
