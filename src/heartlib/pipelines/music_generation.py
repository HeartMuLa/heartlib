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
import gc
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


class HeartMuLaGenPipeline:  # Removed inheritance from Pipeline
    def __init__(
        self,
        model: Optional[HeartMuLa],
        audio_codec: Optional[HeartCodec],
        muq_mulan: Optional[Any],
        text_tokenizer: Tokenizer,
        config: HeartMuLaGenConfig,
        device: torch.device,
        dtype: torch.dtype,
        # New: store model paths and configs for on-demand loading
        heartmula_path: Optional[str] = None,
        heartcodec_path: Optional[str] = None,
        bnb_config: Optional[BitsAndBytesConfig] = None,
        num_quantizers: Optional[int] = None,  # New parameter
    ):
        self.model = model
        self.audio_codec = audio_codec
        self.muq_mulan = muq_mulan
        self.text_tokenizer = text_tokenizer
        self.config = config
        self.device = device
        self.dtype = dtype

        # Store configurations required for lazy loading
        self.heartmula_path = heartmula_path
        self.heartcodec_path = heartcodec_path
        self.bnb_config = bnb_config

        # Defer initialization of these attributes (set after model loading)
        self._parallel_number = None
        self._muq_dim = None

        if audio_codec is not None:
            self._parallel_number = audio_codec.config.num_quantizers + 1
        elif num_quantizers is not None:
            # If audio_codec is not loaded but num_quantizers is provided, use it
            self._parallel_number = num_quantizers + 1
        else:
            # Default value (SQ-Codec typically uses 8 quantizers)
            self._parallel_number = 8 + 1
            print(f"Warning: num_quantizers not provided; using default value 8")

        if model is not None:
            self._muq_dim = model.config.muq_dim

    def load_heartmula(self) -> None:
        """Load HeartMuLa model on demand"""
        if self.model is not None:
            print("HeartMuLa already loaded, skipping")
            return
        
        print("Loading HeartMuLa...")
        if self.heartmula_path is None:
            raise ValueError("HeartMuLa model path not provided")
        
        self.model = HeartMuLa.from_pretrained(
            self.heartmula_path, 
            dtype=self.dtype, 
            quantization_config=self.bnb_config
        )
        self.model.to(self.device)
        self.model.eval()
        self._muq_dim = self.model.config.muq_dim
        print(f"HeartMuLa loaded successfully. GPU memory usage: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")

    def unload_heartmula(self) -> None:
        """Unload HeartMuLa model to free GPU memory"""
        if self.model is None:
            return
        
        print("Unloading HeartMuLa...")
        del self.model
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()
        print(f"HeartMuLa unloaded. Current GPU memory usage: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")

    def load_heartcodec(self) -> None:
        """Load HeartCodec model on demand"""
        if self.audio_codec is not None:
            print("HeartCodec already loaded, skipping")
            return
        
        print("Loading HeartCodec...")
        if self.heartcodec_path is None:
            raise ValueError("HeartCodec model path not provided")
        
        self.audio_codec = HeartCodec.from_pretrained(self.heartcodec_path, device_map=self.device)
        self._parallel_number = self.audio_codec.config.num_quantizers + 1
        print(f"HeartCodec loaded successfully. GPU memory usage: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")

    def unload_heartcodec(self) -> None:
        """Unload HeartCodec model to free GPU memory"""
        if self.audio_codec is None:
            return
        
        print("Unloading HeartCodec...")
        del self.audio_codec
        self.audio_codec = None
        gc.collect()
        torch.cuda.empty_cache()
        print(f"HeartCodec unloaded. Current GPU memory usage: {torch.cuda.memory_allocated(self.device) / 1024**3:.2f} GB")

    def preprocess(self, inputs: Dict[str, Any], cfg_scale: float):
        # Ensure HeartMuLa is loaded to access configuration
        if self._muq_dim is None and self.model is None:
            self.load_heartmula()

        # process tags
        tags = inputs["tags"]
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

        # process reference audio
        ref_audio = inputs.get("ref_audio", None)
        if ref_audio is not None:
            raise NotImplementedError("ref_audio is not supported yet.")
        muq_embed = torch.zeros([self._muq_dim], dtype=self.dtype, device=self.device)  # [Modified] Added device
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

        # [Modified] Move all tensors to the correct device
        tokens = torch.zeros([prompt_len, self._parallel_number], dtype=torch.long, device=self.device)
        tokens[: len(tags_ids), -1] = torch.tensor(tags_ids, device=self.device)
        tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids, device=self.device)

        tokens_mask = torch.zeros_like(tokens, dtype=torch.bool, device=self.device)
        tokens_mask[:, -1] = True

        bs_size = 2 if cfg_scale != 1.0 else 1

        def _cfg_cat(tensor: torch.Tensor, cfg_scale: float):
            tensor = tensor.unsqueeze(0)
            if cfg_scale != 1.0:
                tensor = torch.cat([tensor, tensor], dim=0)
            return tensor

        # [Modified] Move pos tensor to device
        pos = _cfg_cat(torch.arange(prompt_len, dtype=torch.long, device=self.device), cfg_scale)

        return {
            "tokens": _cfg_cat(tokens, cfg_scale),
            "tokens_mask": _cfg_cat(tokens_mask, cfg_scale),
            "muq_embed": _cfg_cat(muq_embed, cfg_scale),
            "muq_idx": [muq_idx] * bs_size,
            "pos": pos,
        }

    def _forward(
        self,
        model_inputs: Dict[str, Any],
        max_audio_length_ms: int,
        temperature: float,
        topk: int,
        cfg_scale: float,
    ):
        # 1. Load HeartMuLa
        self.load_heartmula()

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
        
        # 2. Unload HeartMuLa to free GPU memory
        print("\n===== Audio token generation complete. Preparing for decoding =====")
        self.unload_heartmula()
        
        # 3. Move frames to CPU (if not already) to avoid occupying GPU memory
        frames = frames.cpu()
        torch.cuda.empty_cache()
        
        return {"frames": frames}  # Return frames instead of wav

    def postprocess(self, model_outputs: Dict[str, Any], save_path: str):
        # 4. Load HeartCodec
        self.load_heartcodec()
        
        frames = model_outputs["frames"]
        
        # Move frames back to GPU
        frames = frames.to(self.device)
        
        # 5. Decode audio
        print("\n===== Starting audio decoding =====")
        wav = self.audio_codec.detokenize(frames)
        
        # 6. Save audio
        torchaudio.save(save_path, wav, 48000)
        
        # 7. Unload HeartCodec
        print("\n===== Audio decoding completed =====")
        self.unload_heartcodec()

    def __call__(self, inputs: Dict[str, Any], **kwargs):
        """Implement the pipeline call interface"""
        # Parse arguments
        preprocess_kwargs, forward_kwargs, postprocess_kwargs = self._sanitize_parameters(**kwargs)
        
        # Execute pipeline stages
        model_inputs = self.preprocess(inputs, cfg_scale=preprocess_kwargs["cfg_scale"])
        model_outputs = self._forward(model_inputs, **forward_kwargs)
        self.postprocess(model_outputs, **postprocess_kwargs)

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

    @classmethod
    def from_pretrained(
        cls,
        pretrained_path: str,
        device: torch.device,
        dtype: torch.dtype,
        version: str,
        bnb_config: Optional[BitsAndBytesConfig] = None,
        lazy_load: bool = True,
    ):
        # Get HeartCodec path
        heartcodec_path = os.path.join(pretrained_path, "HeartCodec-oss")
        if not os.path.exists(heartcodec_path):
            raise FileNotFoundError(
                f"Expected to find checkpoint for HeartCodec at {heartcodec_path} but not found. Please check your folder {pretrained_path}."
            )

        # Get HeartMuLa path
        heartmula_path = os.path.join(pretrained_path, f"HeartMuLa-oss-{version}")
        if not os.path.exists(heartmula_path):
            raise FileNotFoundError(
                f"Expected to find checkpoint for HeartMuLa at {heartmula_path} but not found. Please check your folder {pretrained_path}."
            )

        # Load tokenizer and config
        vocab_path = os.path.join(pretrained_path, "tokenizer.json")
        if not os.path.isfile(vocab_path):
            raise FileNotFoundError(
                f"Expected to find tokenizer.json for HeartMuLa at {vocab_path} but not found. Please check your folder {pretrained_path}."
            )
        tokenizer = Tokenizer.from_file(vocab_path)

        gen_config_path = os.path.join(pretrained_path, "gen_config.json")
        if not os.path.isfile(gen_config_path):
            raise FileNotFoundError(
                f"Expected to find gen_config.json for HeartMuLa at {gen_config_path} but not found. Please check your folder {pretrained_path}."
            )
        gen_config = HeartMuLaGenConfig.from_file(gen_config_path)

        # [New] Read HeartCodec config to get num_quantizers
        codec_config_path = os.path.join(heartcodec_path, "config.json")
        if os.path.isfile(codec_config_path):
            with open(codec_config_path, encoding="utf-8") as f:
                codec_config = json.load(f)
            num_quantizers = codec_config.get("num_quantizers", 8)  # Default to 8
            print(f"Read num_quantizers = {num_quantizers} from config")
        else:
            # If no config file exists, use default value
            num_quantizers = 8
            print(f"HeartCodec config not found; using default num_quantizers = {num_quantizers}")

        if lazy_load:
            print("Using lazy loading mode: models will be loaded on demand")
            return cls(
                None,  # model
                None,  # audio_codec
                None,  # muq_mulan
                tokenizer,
                gen_config,
                device,
                dtype,
                heartmula_path=heartmula_path,
                heartcodec_path=heartcodec_path,
                bnb_config=bnb_config,
                num_quantizers=num_quantizers,  # Pass num_quantizers
            )
        else:
            print("Using eager loading mode: loading all models immediately")
            heartcodec = HeartCodec.from_pretrained(heartcodec_path, device_map=device)
            heartmula = HeartMuLa.from_pretrained(
                heartmula_path, dtype=dtype, quantization_config=bnb_config
            )
            return cls(
                heartmula,
                heartcodec,
                None,
                tokenizer,
                gen_config,
                device,
                dtype,
                heartmula_path=heartmula_path,
                heartcodec_path=heartcodec_path,
                bnb_config=bnb_config,
                num_quantizers=num_quantizers,  # Pass num_quantizers
            )