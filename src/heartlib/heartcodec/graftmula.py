#!/usr/bin/env python3
"""
GraftMuLa: HeartCodec RVQ Token Surgery
=======================================

Content/texture-level audio style transfer using HeartCodec residual
vector quantization (RVQ) tokens.

GraftMuLa combines two audio tracks into a hybrid representation:

- Source track: supplies the content and determines output duration.
- Style track: supplies the texture for selected RVQ levels.

Features
--------
- Exact token-level swapping.
- Optional embedding-space blending without mandatory re-quantization.
- Configurable content cut points, transition ramps, and per-level weights.
- Optional grid search over cut points and ramp settings.
- Deterministic decode comparisons using a fixed seed.
- Optional embedding-decoder parity checks.
- Optional mixed-precision decoding and torch.compile acceleration.

API
---
TokenSurgeryConfig
    Configuration for token surgery and decoding.
TokenSurgery
    Tokenization, hybrid construction, decoding, and audio processing.

Example
-------
    from graftmula import TokenSurgery, TokenSurgeryConfig

    config = TokenSurgeryConfig(
        mode="blend",
        content_levels=3,
        ramp=2,
        guidance_scale=3.0,
    )
    surgery = TokenSurgery(config)

    source = surgery.tokenize_file("content.wav")
    style = surgery.tokenize_file("texture.wav")
    hybrid = surgery.build_hybrid(source, style)
    audio = surgery.decode(hybrid, seed=42)

CLI
---
Run ``python graftmula.py --help`` for command-line options.

Compatibility
-------------
The embedding-decoding path depends on HeartCodec decoder internals.
Parity, numerical behavior, and performance may vary across supported
PyTorch, HeartCodec, vector-quantization, and hardware versions.
See the project documentation for tested configurations.

License
-------
Apache-2.0, subject to the repository's contribution and licensing terms.

Project Los Angeles
Tegridy Code 2026
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from heartlib.heartcodec.modeling_heartcodec import HeartCodec

NUM_QUANTIZERS = 8        # verified: config.num_quantizers == 8 (12.5 Hz streams)
TOKEN_FPS = 12.5          # token frame rate
LATENT_FPS = 25           # SQ-Codec latent frame rate (2x tokens)
WINDOW_SECONDS = 29.76    # internal decoder window (do not change)


# --------------------------------------------------------------------------- #
# Audio I/O (mirrors the official run_music_reconstruction.py example)
# --------------------------------------------------------------------------- #

def load_waveform(path: str) -> tuple[torch.Tensor, int]:
    """Read any soundfile-supported audio as float32 [channels, samples]."""
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.shape[1] > 2:  # >2ch: fold to stereo by averaging pairs
        audio = audio[:, :2]
    waveform = torch.from_numpy(audio.T.copy())  # [C, S]
    if not torch.isfinite(waveform).all():
        raise ValueError(f"{path}: audio contains non-finite samples")
    return waveform, int(sample_rate)


def save_audio(path: Path, stereo_np: np.ndarray, sample_rate: int) -> None:
    """Save stereo [T, C] float audio; 320 kbps MP3 via ffmpeg, or 24-bit WAV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".mp3":
        subprocess.run(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "f32le", "-ar", str(sample_rate),
                "-ac", str(stereo_np.shape[1]), "-i", "pipe:0",
                "-c:a", "libmp3lame", "-b:a", "320k", str(path),
            ],
            input=stereo_np.astype(np.dtype("<f4"), copy=False).tobytes(),
            check=True,
        )
    else:
        sf.write(path, stereo_np, sample_rate, subtype="PCM_24")


def trim_and_export(waveform: torch.Tensor, out_frames: int,
                    path: Path, sample_rate: int) -> None:
    """Validate, trim to target length, and write a decoded stereo tensor."""
    if waveform.ndim != 2 or waveform.shape[0] != 2:
        raise RuntimeError(f"Expected stereo [2, T]; got {tuple(waveform.shape)}")
    if waveform.shape[1] < out_frames:
        raise RuntimeError(
            f"Decoded {waveform.shape[1]} frames < expected {out_frames}"
        )
    output = waveform[:, :out_frames].float().cpu().T.contiguous().numpy()
    if not np.isfinite(output).all():
        raise RuntimeError("Decoder returned non-finite samples")
    save_audio(path, output, sample_rate)
    print(f"  saved -> {path}")


def set_seed(seed: int) -> None:
    """Reseed global RNG right before each decode for controlled comparisons."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# --------------------------------------------------------------------------- #
# Embedding-space decoding: faithful transcription of HeartCodecDecoder.
#   (heartlib src/heartlib/heartcodec/models/decoder.py, commit 5858ca8)
# The ONLY change vs. stock: the per-window conditioning comes from a supplied
# [T, 512] embedding sequence instead of ResidualVQ.get_output_from_indices.
# RNG draw order, dtypes, window geometry, context stitching, overlap-add and
# trimming are kept byte-for-byte identical so verify_embed_decode can prove
# equivalence on the user's checkpoint.
# --------------------------------------------------------------------------- #

def _inference_embeddings(fm, emb, true_latents, latent_length,
                          incontext_length, guidance_scale, num_steps,
                          scenario="other_seg"):
    """Transcription of FlowMatching.inference_codes with emb [B, T, 512]."""
    device = true_latents.device
    dtype = true_latents.dtype
    batch_size = emb.shape[0]

    quantized_feature_emb = fm.cond_feature_emb(emb)              # [B, T, 512]
    quantized_feature_emb = F.interpolate(
        quantized_feature_emb.permute(0, 2, 1), scale_factor=2, mode="nearest"
    ).permute(0, 2, 1)                                            # [B, 2T, 512]

    num_frames = quantized_feature_emb.shape[1]
    latents = torch.randn(
        (batch_size, num_frames, fm.latent_dim), device=device, dtype=dtype
    )
    latent_masks = torch.zeros(
        latents.shape[0], latents.shape[1], dtype=torch.int64, device=latents.device
    )
    latent_masks[:, 0:latent_length] = 2
    if scenario == "other_seg":
        latent_masks[:, 0:incontext_length] = 1

    quantized_feature_emb = (latent_masks > 0.5).unsqueeze(
        -1
    ) * quantized_feature_emb + (latent_masks < 0.5).unsqueeze(
        -1
    ) * fm.zero_cond_embedding1.unsqueeze(0)

    incontext_latents = (
        true_latents
        * ((latent_masks > 0.5) * (latent_masks < 1.5)).unsqueeze(-1).float()
    )
    incontext_length = ((latent_masks > 0.5) * (latent_masks < 1.5)).sum(-1)[0]

    additional_model_input = torch.cat([quantized_feature_emb], 1)
    t_span = torch.linspace(
        0, 1, num_steps + 1, device=quantized_feature_emb.device
    )
    latents = fm.solve_euler(
        latents * 1.0,
        incontext_latents.to(dtype),
        incontext_length,
        t_span,
        additional_model_input,
        guidance_scale,
    )
    latents[:, 0:incontext_length, :] = incontext_latents[
        :, 0:incontext_length, :
    ]
    return latents


@torch.inference_mode()
def detokenize_embeddings(decoder, emb512, duration=WINDOW_SECONDS,
                          num_steps=10, guidance_scale=1.25,
                          disable_progress=False):
    """Decode a [T, 512] conditioning embedding via the stock windowed loop.

    ``decoder`` is a HeartCodecDecoder (i.e. ``model.decoder``). Returns a CPU
    stereo waveform [2, ~T/12.5 * sample_rate], exactly like
    HeartCodecDecoder.detokenize. ``disable_progress`` is accepted for
    signature parity; heartlib's inner solver tqdm is always shown.
    """
    if emb512.ndim == 2:
        emb512 = emb512.unsqueeze(0)                 # [1, T, 512]
    emb512 = emb512.to(next(decoder.parameters()).device)
    frames = emb512.shape[1]

    # -- identical preamble to HeartCodecDecoder.detokenize --
    first_latent = torch.randn(
        emb512.shape[0], int(duration * LATENT_FPS), decoder.flow_matching.latent_dim,
        dtype=decoder.dtype,
    ).to(next(decoder.parameters()).device)
    first_latent_length = 0
    first_latent_codes_length = 0
    min_samples = int(duration * TOKEN_FPS)          # token frames per window
    hop_samples = min_samples // 93 * 80             # 320 at 29.76 s
    ovlp_samples = min_samples - hop_samples         # 52 token frames
    ovlp_frames = ovlp_samples * 2                   # 104 latent frames
    target_len = int(
        (frames - first_latent_codes_length) / TOKEN_FPS * decoder.sample_rate
    )

    # repeat (tiled) when shorter than one window -- identical to stock
    if frames < min_samples:
        while emb512.shape[1] < min_samples:
            emb512 = torch.cat([emb512, emb512], 1)
        emb512 = emb512[:, 0:min_samples, :]
    frames = emb512.shape[1]
    if (frames - ovlp_samples) % hop_samples > 0:
        len_codes = (
            math.ceil((frames - ovlp_samples) / float(hop_samples)) * hop_samples
            + ovlp_samples
        )
        while emb512.shape[1] < len_codes:
            emb512 = torch.cat([emb512, emb512], 1)
        emb512 = emb512[:, 0:len_codes, :]
    latent_length = int(duration * LATENT_FPS)
    latent_list = []

    # -- windowed flow matching with latent context stitching --
    for sinx in range(0, emb512.shape[1] - hop_samples + 1, hop_samples):
        emb_window = emb512[:, sinx : sinx + min_samples, :]
        if sinx == 0 or ovlp_frames == 0:
            incontext_length = first_latent_length
            latents = _inference_embeddings(
                decoder.flow_matching,
                emb_window,
                first_latent,
                latent_length,
                incontext_length,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                scenario="other_seg",
            )
            latent_list.append(latents)
        else:
            true_latent = latent_list[-1][:, -ovlp_frames:, :]
            len_add_to_latent = latent_length - true_latent.shape[1]
            incontext_length = true_latent.shape[1]
            true_latent = torch.cat(
                [
                    true_latent,
                    torch.randn(
                        true_latent.shape[0],
                        len_add_to_latent,
                        true_latent.shape[-1],
                        dtype=decoder.dtype,
                    ).to(next(decoder.parameters()).device),
                ],
                1,
            )
            latents = _inference_embeddings(
                decoder.flow_matching,
                emb_window,
                true_latent,
                latent_length,
                incontext_length,
                guidance_scale=guidance_scale,
                num_steps=num_steps,
                scenario="other_seg",
            )
            latent_list.append(latents)

    latent_list[0] = latent_list[0][:, first_latent_length:, :]

    # -- identical SQ-Codec overlap-add assembly --
    min_samples = int(duration * decoder.sample_rate)  # waveform samples now
    hop_samples = min_samples // 93 * 80
    ovlp_samples = min_samples - hop_samples

    output = None
    for i in range(len(latent_list)):
        latent = latent_list[i]
        latent = latent.reshape(
            latent.shape[0], latent.shape[1], 2, latent.shape[2] // 2
        ).permute(0, 2, 1, 3)
        latent = latent.reshape(
            latent.shape[0] * 2, latent.shape[2], latent.shape[3]
        )
        cur_output = (
            decoder.scalar_model.decode(latent.transpose(1, 2))
            .squeeze(0).squeeze(1)
        )
        cur_output = cur_output[:, 0:min_samples].detach().cpu()
        if cur_output.dim() == 3:
            cur_output = cur_output[0]

        if output is None:
            output = cur_output
        else:
            if ovlp_samples == 0:
                output = torch.cat([output, cur_output], -1)
            else:
                ov_win = torch.from_numpy(
                    np.linspace(0, 1, ovlp_samples)[None, :]
                )
                ov_win = torch.cat([ov_win, 1 - ov_win], -1)
                output[:, -ovlp_samples:] = (
                    output[:, -ovlp_samples:] * ov_win[:, -ovlp_samples:]
                    + cur_output[:, 0:ovlp_samples] * ov_win[:, 0:ovlp_samples]
                )
                output = torch.cat([output, cur_output[:, ovlp_samples:]], -1)
    output = output[:, 0:target_len]
    return output


@torch.inference_mode()
def codes_to_conditioning(decoder, codes) -> torch.Tensor:
    """[8, T] codes -> the exact [T, 512] conditioning the stock path computes.

    Equivalent to what HeartCodecDecoder.detokenize derives internally via
    ``flow_matching.vq_embed.get_output_from_indices``; used for parity
    checks and for constructing blends in the same space.
    """
    quantizer = decoder.flow_matching.vq_embed
    device = next(quantizer.parameters()).device
    idx = codes.to(device).t().unsqueeze(0)          # [1, T, Q]
    return quantizer.get_output_from_indices(idx)[0].float().cpu()


# --------------------------------------------------------------------------- #
# Token surgery helpers
# --------------------------------------------------------------------------- #

def crop_style_segment(style_tokens: torch.Tensor, offset_sec: float,
                       duration_sec: float) -> torch.Tensor:
    """Crop [8, T] style tokens to the user-selected segment."""
    start = int(offset_sec * TOKEN_FPS)
    if start >= style_tokens.shape[1]:
        raise ValueError(
            f"style_offset {offset_sec}s is beyond the style track "
            f"({style_tokens.shape[1] / TOKEN_FPS:.2f}s)."
        )
    end = style_tokens.shape[1]
    if duration_sec > 0:
        end = min(end, start + int(duration_sec * TOKEN_FPS))
    return style_tokens[:, start:end].clone()


def tile_to_length(segment: torch.Tensor, target_len: int) -> torch.Tensor:
    """Tile (and crop) a [8, t] token segment up to target_len frames.

    Deterministic wraparound tiling. Audible loop seams are possible at tile
    boundaries every t/12.5 seconds.
    """
    seg_len = segment.shape[1]
    if seg_len >= target_len:
        return segment[:, :target_len].clone()
    repeats = math.ceil(target_len / seg_len)
    tiled = segment.repeat(1, repeats)[:, :target_len]
    return tiled.contiguous()


def build_hybrid_swap(source_tokens: torch.Tensor,
                      style_segment: torch.Tensor,
                      content_levels: int) -> tuple[torch.Tensor, dict]:
    """Hard token-ID transplant: keep levels [0, C) from source, rest from style."""
    src = source_tokens.clone()
    T = src.shape[1]
    texture = tile_to_length(style_segment, T)
    src[content_levels:, :] = texture[content_levels:, :]
    info = {"mode": "swap", "content_levels": content_levels}
    return src, info


def parse_level_weights(spec: str) -> tuple[float, ...]:
    """Parse '1,1,1,0.5,0,0,0,0' -> tuple of source weights per level."""
    parts = [p.strip() for p in spec.split(",") if p.strip() != ""]
    if len(parts) != NUM_QUANTIZERS:
        raise ValueError(
            f"level_weights needs exactly {NUM_QUANTIZERS} values, got {len(parts)}"
        )
    weights = tuple(float(p) for p in parts)
    if any(not math.isfinite(w) or w < 0 or w > 1 for w in weights):
        raise ValueError("level_weights values must be finite and within [0, 1]")
    return weights


def parse_int_list(spec: str, option: str = "--grid_levels") -> tuple[int, ...]:
    """Parse '1, 2,3' -> (1, 2, 3); dedupes while keeping order."""
    parts = [p.strip() for p in spec.split(",") if p.strip() != ""]
    if not parts:
        raise ValueError(f"{option} needs at least one integer")
    values: list[int] = []
    for part in parts:
        try:
            values.append(int(part))
        except ValueError:
            raise ValueError(f"{option}: {part!r} is not an integer") from None
    return tuple(dict.fromkeys(values))


def ramp_weights(content_levels: int, ramp: int) -> tuple[float, ...]:
    """Source weight per level: 1.0 below the cut, then uniform 1->0 steps.

    With ``ramp=R`` the weights at levels C..C+R descend by 1/(R+1) per level
    (e.g. C=5, R=2 -> [1,1,1,1,1, 2/3, 1/3, 0]). Because there are only 8
    levels, large R truncates: the last level may stay above 0.
    """
    w = [0.0] * NUM_QUANTIZERS
    for k in range(min(content_levels, NUM_QUANTIZERS)):
        w[k] = 1.0
    step = 1.0 / (ramp + 1)
    for i in range(ramp):
        k = content_levels + i
        if k < NUM_QUANTIZERS:
            w[k] = 1.0 - (i + 1) * step
    return tuple(max(0.0, min(1.0, x)) for x in w)


@torch.inference_mode()
def build_hybrid_blend(source_tokens: torch.Tensor,
                       style_segment: torch.Tensor,
                       level_weights: Sequence[float],
                       quantizer,
                       requantize: bool = False):
    """Per-level weighted embedding mix -> [T, 512] conditioning (+ optional codes).

    level_weights[k] is the SOURCE weight at level k (style gets 1 - w).
    The mix is assembled in codebook space [T, 32] and projected ONCE through
    ``project_out`` (Linear 32->512), so the bias term matches the stock
    conditioning path exactly. Uses only verified ResidualVQ internals.
    """
    device = next(quantizer.parameters()).device
    T = source_tokens.shape[1]
    style_seg = tile_to_length(style_segment, T)

    src_idx = source_tokens.to(device).t().unsqueeze(0)   # [1, T, Q]
    sty_idx = style_seg.to(device).t().unsqueeze(0)       # [1, T, Q]

    src_codes = quantizer.get_codes_from_indices(src_idx)  # [Q, 1, T, 32]
    sty_codes = quantizer.get_codes_from_indices(sty_idx)  # [Q, 1, T, 32]

    w = torch.tensor(level_weights, dtype=torch.float32,
                     device=device).view(-1, 1, 1, 1)      # [Q, 1, 1, 1]
    mixed_codebook = (w * src_codes + (1.0 - w) * sty_codes).sum(dim=0)  # [1, T, 32]
    emb512 = quantizer.project_out(mixed_codebook)[0].float().cpu()      # [T, 512]

    requant_codes = None
    if requantize:
        _, new_indices, _ = quantizer(quantizer.project_out(mixed_codebook))
        requant_codes = new_indices[0].t().long().cpu()      # [Q, T]

    info = {"mode": "blend", "level_weights": [float(x) for x in level_weights]}
    return emb512, requant_codes, info

# --------------------------------------------------------------------------- #
# Diagnostics
# --------------------------------------------------------------------------- #

@torch.inference_mode()
def print_level_report(name: str, tokens: torch.Tensor, quantizer) -> None:
    """Per-level codebook usage + 512-space embedding contribution norms."""
    device = next(quantizer.parameters()).device
    idx = tokens.to(device).t().unsqueeze(0)               # [1, T, Q]
    codes = quantizer.get_codes_from_indices(idx)          # [Q, 1, T, 32]
    T = tokens.shape[1]
    print(f"\n  [{name}]  {T} frames ({T / TOKEN_FPS:.2f}s)")
    print("  level | usage(unique/8192) | mean|emb|512 | top-code share")
    for k in range(tokens.shape[0]):
        level_codes = codes[k]
        emb512 = quantizer.project_out(level_codes)[0]     # [T, 512]
        norm = emb512.norm(dim=-1).mean().item()
        uniq = tokens[k].unique().numel()
        counts = torch.bincount(tokens[k], minlength=8192).float()
        top_share = (counts.max() / T).item()
        print(f"    {k}   |       {uniq:5d}       |    {norm:7.3f}   |     {top_share:.3f}")


# --------------------------------------------------------------------------- #
# Programmatic API
# --------------------------------------------------------------------------- #

_AMP_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16}
_COMPILE_MODES = {None, "default", "reduce-overhead",
                  "max-autotune-no-cudagraphs", "max-autotune"}


@dataclass
class TokenSurgeryConfig:
    """Every option of the CLI, available programmatically.

    Fields and defaults mirror the argparse interface one-to-one.
    """
    # paths & runtime
    decoder_path: str = "./ckpt/HeartCodec-oss-20260123"
    encoder_path: str = "./ckpt/HeartCodec-oss-encoder"
    device: str = "cuda"
    batch_size: int = 1
    # surgery
    mode: str = "swap"                       # "swap" | "blend"
    content_levels: int = 3                  # levels 0..C-1 kept from source
    ramp: int = 0                            # blend: uniform weight ramp width
    level_weights: Optional[Sequence[float]] = None  # blend: explicit weights
    style_offset: float = 0.0                # seconds into the style track
    style_duration: float = 0.0              # 0 = until end of track
    requantize: bool = False                 # blend: also produce token tuple
    verify_embed_decode: bool = True         # blend: stock-vs-embedding parity
    strict_parity: bool = False              # raise at 5e-2 instead of warning
    # sampling
    seed: int = 42
    num_steps: int = 10
    guidance_scale: float = 1.25
    # performance
    torch_compile: bool = False
    compile_mode: Optional[str] = None       # None = torch default
    amp_autocast: bool = False
    amp_dtype: str = "bfloat16"              # "bfloat16" | "float16"
    # output
    baselines: bool = True                   # render source-only / style-only
    dump_tokens: bool = False                # save .pt token/embedding dumps
    quiet: bool = False                      # suppress our own logging
    verbose: bool = True                     # print reports (programmatic use)
    # compare grid (consumed by compare_grid(); --compare_grid dispatches to it)
    compare_grid: bool = False               # CLI only: run() -> compare_grid()
    grid_levels: Sequence[int] = (1, 2, 3, 4, 5)   # content_levels values
    grid_ramps: Sequence[int] = (0,)         # ramp values (blend mode)

    def __post_init__(self):
        if self.mode not in ("swap", "blend"):
            raise ValueError(f"mode must be 'swap' or 'blend', got {self.mode!r}")
        if not (0 <= self.content_levels <= NUM_QUANTIZERS):
            raise ValueError(f"content_levels must be within 0..{NUM_QUANTIZERS}")
        if self.ramp < 0:
            raise ValueError("ramp must be >= 0")
        if self.level_weights is not None:
            if len(self.level_weights) != NUM_QUANTIZERS:
                raise ValueError(
                    f"level_weights needs exactly {NUM_QUANTIZERS} values"
                )
            if any(not math.isfinite(w) or w < 0 or w > 1
                   for w in self.level_weights):
                raise ValueError("level_weights values must be in [0, 1]")
        if self.num_steps < 1:
            raise ValueError("num_steps must be positive")
        if not math.isfinite(self.guidance_scale):
            raise ValueError("guidance_scale must be finite")
        if self.amp_dtype not in _AMP_DTYPES:
            raise ValueError(f"amp_dtype must be one of {sorted(_AMP_DTYPES)}")
        if self.compile_mode not in _COMPILE_MODES:
            raise ValueError(f"compile_mode must be one of {sorted(_COMPILE_MODES)}")
        for name, seq in (("grid_levels", self.grid_levels),
                          ("grid_ramps", self.grid_ramps)):
            if isinstance(seq, str):
                raise TypeError(
                    f"{name} must be a sequence of ints; parse CLI strings "
                    "with parse_int_list() first")
            if not seq:
                raise ValueError(f"{name} must not be empty")
            if any(not isinstance(v, int) or isinstance(v, bool) or v < 0
                   for v in seq):
                raise ValueError(f"{name} must contain non-negative ints")
        if any(v > NUM_QUANTIZERS for v in self.grid_levels):
            raise ValueError(
                f"grid_levels values must be in 0..{NUM_QUANTIZERS}")


@dataclass
class HybridTokens:
    """Result of build_hybrid: exactly one of codes / embedding is primary."""
    kind: str                                # "codes" | "embedding"
    data: torch.Tensor                       # [8, T] int64 | [T, 512] float32
    requant_codes: Optional[torch.Tensor] = None   # blend: best-effort [8, T]
    info: dict = field(default_factory=dict)


@dataclass
class SurgeryResult:
    """Everything run() produced, for programmatic consumption."""
    output_path: Optional[Path]
    source_recon_path: Optional[Path]
    style_recon_path: Optional[Path]
    source_tokens: torch.Tensor
    style_tokens: torch.Tensor
    style_segment: torch.Tensor
    hybrid: HybridTokens
    waveforms: dict                          # name -> [2, S] cpu float32
    out_frames: int
    sample_rate: int
    embed_parity_diff: Optional[float] = None


@dataclass
class GridVariant:
    """One (content_levels, ramp) point rendered by compare_grid."""
    content_levels: int
    ramp: int
    level_weights: tuple
    tag: str                                 # file tag, e.g. "cl3_r2"
    hybrid: HybridTokens
    waveform: torch.Tensor                   # [2, S] cpu float32
    path: Optional[Path]                     # None when save=False


@dataclass
class GridResult:
    """Everything compare_grid produced, for programmatic consumption."""
    variants: list
    baselines: dict                          # "source"/"style" -> [2, S] cpu
    baseline_paths: dict
    source_tokens: torch.Tensor
    style_tokens: torch.Tensor
    out_frames: int
    sample_rate: int
    embed_parity_diff: Optional[float] = None


class TokenSurgery:
    """Lazy-loading façade over HeartCodec for token-level style surgery.

    Construct cheaply; the checkpoint loads on first use (or call
    load_model() explicitly). All decodes reseed the global RNG so results
    are reproducible and A/B comparable.
    """

    def __init__(self, config: Optional[TokenSurgeryConfig] = None, /,
                 **overrides):
        if config is None:
            config = TokenSurgeryConfig()
        if overrides:
            valid = set(TokenSurgeryConfig.__dataclass_fields__)
            unknown = set(overrides) - valid
            if unknown:
                raise TypeError(f"unknown options: {sorted(unknown)}")
            config = replace(config, **overrides)
        self.config = config
        self._model: Optional[HeartCodec] = None
        self._decoder = None
        self._orig_estimator = None          # pre-compile callables (fallback)
        self._orig_scalar_decode = None

    # -- lifecycle ---------------------------------------------------------- #

    def _log(self, msg: str) -> None:
        if self.config.verbose and not self.config.quiet:
            print(msg)

    def load_model(self) -> HeartCodec:
        """Load HeartCodec (FP32, strict split checkpoints); idempotent."""
        if self._model is not None:
            return self._model
        device = torch.device(self.config.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA unavailable; install a CUDA PyTorch build or use device='cpu'"
            )
        if device.type == "cpu" and self.config.amp_dtype == "float16":
            raise ValueError("float16 autocast is unsupported on CPU; use bfloat16")
        self._log("Loading HeartCodec (FP32, strict split checkpoints) ...")
        model = HeartCodec.from_encoder_decoder_pretrained(
            self.config.decoder_path, self.config.encoder_path,
            dtype=torch.float32,
        ).to(device).eval()
        self._maybe_compile(model)
        self._model = model
        self._decoder = model.decoder
        return model

    def _maybe_compile(self, model: HeartCodec) -> None:
        if not self.config.torch_compile:
            return
        if not hasattr(torch, "compile"):
            self._log("torch.compile unavailable in this torch build; running eager.")
            return
        mode = self.config.compile_mode or "default"
        uses_cudagraphs = mode in ("reduce-overhead", "max-autotune")
        if uses_cudagraphs:
            self._prepare_cudagraphs(mode)
        fm = model.decoder.flow_matching
        try:
            wrapped = torch.compile(fm.estimator, mode=mode)
            if uses_cudagraphs:
                wrapped = self._with_step_mark(wrapped)
            self._orig_estimator = fm.estimator
            fm.estimator = wrapped
            self._log(f"torch.compile: flow_matching.estimator (mode={mode})")
        except Exception as error:  # pragma: no cover - environment dependent
            self._log(f"torch.compile(estimator) failed ({error}); running eager.")
        try:
            decode = model.decoder.scalar_model.decode
            wrapped = torch.compile(decode, mode=mode)
            if uses_cudagraphs:
                wrapped = self._with_step_mark(wrapped)
            self._orig_scalar_decode = decode
            model.decoder.scalar_model.decode = wrapped
            self._log(f"torch.compile: scalar_model.decode (mode={mode})")
        except Exception as error:  # pragma: no cover
            self._log(f"torch.compile(scalar decode) failed ({error}); skipped.")

    @staticmethod
    def _with_step_mark(compiled):
        """Mark a CUDA-graphs step before each compiled invocation.

        heartlib's euler loop and the RoPE sin/cos cache keep tensors alive
        across compiled invocations; without a step mark, cudagraph_trees
        raises "accessing tensor output of CUDAGraphs that has been
        overwritten by a subsequent run".
        """
        marker = getattr(getattr(torch, "compiler", None),
                         "cudagraph_mark_step_begin", None)
        if marker is None:
            return compiled

        def run(*args, **kwargs):
            marker()
            return compiled(*args, **kwargs)

        return run

    def _prepare_cudagraphs(self, mode: str) -> None:
        """Best-effort inductor mitigation for CUDA-graph tensor reuse."""
        try:
            from torch._inductor import config as inductor_config

            triton = getattr(inductor_config, "triton", None)
            attr = "cudagraph_trees_generation_cloning"
            if triton is not None and hasattr(triton, attr):
                setattr(triton, attr, "user_visible")
                self._log(f"inductor: {attr}='user_visible' "
                          f"(CUDA-graph reuse guard, mode={mode})")
                return
        except Exception:
            pass
        self._log("NOTE: could not set the inductor cudagraph cloning config; "
                  "if you hit 'CUDAGraph ... overwritten' errors use "
                  "--compile_mode max-autotune-no-cudagraphs or default.")

    def _uncompile(self) -> None:
        """Restore the original eager callables (CUDA-graph fallback path)."""
        try:
            decoder = self.load_model().decoder
            if self._orig_estimator is not None:
                decoder.flow_matching.estimator = self._orig_estimator
                self._orig_estimator = None
            if self._orig_scalar_decode is not None:
                decoder.scalar_model.decode = self._orig_scalar_decode
                self._orig_scalar_decode = None
            if hasattr(torch, "_dynamo"):
                torch._dynamo.reset()
        except Exception as error:  # pragma: no cover
            self._log(f"un-compile cleanup issue ({error}); continuing eager.")
        self.config.torch_compile = False

    def _amp_ctx(self):
        device_type = torch.device(self.config.device).type
        return torch.autocast(
            device_type=device_type,
            dtype=_AMP_DTYPES[self.config.amp_dtype],
            enabled=self.config.amp_autocast,
        )

    # -- tokenization -------------------------------------------------------- #

    def tokenize(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """[C, S] float waveform -> int64 [8, T] tokens at 12.5 Hz (FP32 path)."""
        model = self.load_model()
        return model.tokenize(waveform, sample_rate,
                              batch_size=self.config.batch_size)

    def tokenize_file(self, path: str) -> torch.Tensor:
        waveform, sample_rate = load_waveform(path)
        return self.tokenize(waveform, sample_rate)

    # -- hybrid construction -------------------------------------------------- #

    def resolve_level_weights(self, content_levels=None, ramp=None,
                              level_weights=None) -> Optional[tuple]:
        cfg = self.config
        content_levels = cfg.content_levels if content_levels is None else content_levels
        ramp = cfg.ramp if ramp is None else ramp
        level_weights = cfg.level_weights if level_weights is None else level_weights
        if level_weights is not None:
            return tuple(float(w) for w in level_weights)
        return ramp_weights(content_levels, ramp)

    def build_hybrid(self, source_tokens: torch.Tensor,
                     style_tokens: torch.Tensor, *, mode: Optional[str] = None,
                     content_levels: Optional[int] = None,
                     ramp: Optional[int] = None,
                     level_weights: Optional[Sequence[float]] = None,
                     style_offset: Optional[float] = None,
                     style_duration: Optional[float] = None,
                     requantize: Optional[bool] = None) -> HybridTokens:
        """Build the hybrid from two token tensors; see TokenSurgeryConfig."""
        cfg = self.config
        mode = cfg.mode if mode is None else mode
        content_levels = (cfg.content_levels if content_levels is None
                          else content_levels)
        style_offset = cfg.style_offset if style_offset is None else style_offset
        style_duration = (cfg.style_duration if style_duration is None
                          else style_duration)
        requantize = cfg.requantize if requantize is None else requantize

        style_segment = crop_style_segment(style_tokens, style_offset,
                                           style_duration)
        if mode == "swap":
            codes, info = build_hybrid_swap(source_tokens, style_segment,
                                            content_levels)
            return HybridTokens("codes", codes, None, info)

        # blend: assemble the exact conditioning embedding, no requant snap
        model = self.load_model()
        weights = self.resolve_level_weights(content_levels, ramp, level_weights)
        quantizer = model.decoder.flow_matching.vq_embed
        emb512, requant_codes, info = build_hybrid_blend(
            source_tokens, style_segment, weights, quantizer,
            requantize=bool(requantize),
        )
        return HybridTokens("embedding", emb512, requant_codes, info)

    # -- decoding -------------------------------------------------------------- #

    def decode(self, hybrid, *, seed: Optional[int] = None,
               num_steps: Optional[int] = None,
               guidance_scale: Optional[float] = None) -> torch.Tensor:
        """Decode a HybridTokens, [8, T] codes, or [T, 512] embeddings.

        Returns a CPU stereo waveform [2, ~T/12.5 * 48000]. Reseeded per call.
        If a compiled decode hits a CUDA-graph error, falls back to eager
        (torch.compile disabled for the rest of the session) and retries once.
        """
        if isinstance(hybrid, HybridTokens):
            payload, kind = hybrid.data, hybrid.kind
        else:
            payload = hybrid
            # robust routing: floating tensors are embeddings, integer tensors
            # are codebooks ([8, T] codes vs [T, 512] embeddings can collide on
            # shapes, never on dtype)
            kind = "embedding" if payload.is_floating_point() else "codes"
        seed = self.config.seed if seed is None else seed
        num_steps = self.config.num_steps if num_steps is None else num_steps
        guidance_scale = (self.config.guidance_scale if guidance_scale is None
                          else guidance_scale)
        try:
            return self._decode_raw(payload, kind, seed, num_steps,
                                    guidance_scale)
        except RuntimeError as error:
            message = str(error)
            if not (self.config.torch_compile
                    and "cudagraph" in message.lower()):
                raise
            self._log(
                "\nWARNING: compiled decode hit a CUDA-graph error:\n  "
                + message.strip().splitlines()[-1][:300]
                + "\nFalling back to eager (torch.compile off) and retrying ...")
            self._uncompile()
            return self._decode_raw(payload, kind, seed, num_steps,
                                    guidance_scale)

    def _decode_raw(self, payload, kind, seed, num_steps, guidance_scale):
        set_seed(seed)
        with self._amp_ctx():
            if kind == "codes":
                return self.load_model().detokenize(
                    payload.contiguous(), num_steps=num_steps,
                    guidance_scale=guidance_scale,
                )
            return detokenize_embeddings(
                self.load_model().decoder, payload,
                num_steps=num_steps, guidance_scale=guidance_scale,
            )

    def verify_embed_decode(self, codes: torch.Tensor,
                            seed: Optional[int] = None,
                            num_steps: Optional[int] = None,
                            guidance_scale: Optional[float] = None) -> float:
        """Cross-check the embedding path against the stock decoder on one window.

        Decodes the first (up to one window of) ``codes`` twice with identical
        RNG: once through HeartCodec.detokenize, once through
        detokenize_embeddings. Two diagnostics are printed:

          * ``conditioning diff`` -- max|stock lookup - codes_to_conditioning|
            in the [T, 512] conditioning space (isolates quantizer-path
            drift, e.g. a vq-pytorch version difference); and
          * ``decode diff`` -- max|stock waveform - embedding-path waveform|
            (isolates flow-decode drift, e.g. a heartlib version difference).

        On a matched environment both are 0.0 (bit-exact on our test
        geometry). A small nonzero decode diff points to version or CUDA
        numerics drift and only WARNS -- blend results stay usable. The run
        aborts on non-finite values or a decode diff above 0.5 (results
        would be meaningless), and ``strict_parity=True`` raises at 5e-2.
        Returns the max absolute sample difference.
        """
        decoder = self.load_model().decoder
        quantizer = decoder.flow_matching.vq_embed
        window = int(WINDOW_SECONDS * TOKEN_FPS)
        codes = codes[:, :min(codes.shape[1], window)].contiguous()
        seed = self.config.seed if seed is None else seed
        num_steps = self.config.num_steps if num_steps is None else num_steps
        guidance_scale = (self.config.guidance_scale if guidance_scale is None
                          else guidance_scale)

        with torch.inference_mode():
            qdev = next(quantizer.parameters()).device
            stock_cond = quantizer.get_output_from_indices(
                codes.to(qdev).t().unsqueeze(0))[0].float().cpu()
        our_cond = codes_to_conditioning(decoder, codes)
        cond_diff = (stock_cond - our_cond).abs().max().item()
        self._log(f"\n  conditioning-space parity: "
                  f"max|stock - embedding-path| = {cond_diff:.3e}")

        set_seed(seed)
        with self._amp_ctx():
            reference = self.load_model().detokenize(
                codes, num_steps=num_steps, guidance_scale=guidance_scale
            )
        set_seed(seed)
        with self._amp_ctx():
            candidate = detokenize_embeddings(
                decoder, our_cond, num_steps=num_steps,
                guidance_scale=guidance_scale
            )
        diff = (reference - candidate).abs().max().item()
        self._log(f"\n  embed-decode parity check: max|stock - embedding-path| "
                  f"= {diff:.3e}")

        if not math.isfinite(cond_diff) or not math.isfinite(diff) or diff > 0.5:
            raise RuntimeError(
                f"Embedding decode path diverges from the stock decoder "
                f"(conditioning diff={cond_diff:.3e}, decode diff={diff:.3e}). "
                "This is too large to be benign numerics; blend results would "
                "be meaningless. Likely a heartlib/vq-pytorch version mismatch "
                "-- please diff your checkout against heartlib commit 5858ca8 "
                "and vq-pytorch==1.27.15."
            )
        if diff > 5e-2 or cond_diff > 1e-4:
            message = (
                f"Embedding path differs slightly from the stock decoder "
                f"(conditioning diff={cond_diff:.3e}, decode diff={diff:.3e}). "
                "On a matched environment this check is bit-exact (0.0), so a "
                "small nonzero value points to heartlib/vq-pytorch version or "
                "CUDA-numerics drift. Blend results remain usable -- trust "
                "your ears; diff your heartlib against commit 5858ca8 if the "
                "audio sounds off. Pass strict_parity=True to raise instead.")
            if self.config.strict_parity:
                raise RuntimeError(message)
            self._log("  WARNING: " + message)
        return diff

    # -- one-shot orchestration ------------------------------------------------ #

    def _tokenize_pair(self, source_path: str, style_path: str):
        """Load the model and tokenize both donors once (run + compare_grid)."""
        cfg = self.config
        model = self.load_model()
        self._log("Tokenizing source (content donor) ...")
        src_wave, src_sr = load_waveform(source_path)
        source_tokens = self.tokenize(src_wave, src_sr)
        self._log("Tokenizing style (texture donor) ...")
        sty_wave, sty_sr = load_waveform(style_path)
        style_tokens = self.tokenize(sty_wave, sty_sr)

        if cfg.verbose and not cfg.quiet:
            print(f"\nsource: {tuple(source_tokens.shape)}  ({source_path})")
            print(f"style : {tuple(style_tokens.shape)}  ({style_path})")
            quantizer = model.decoder.flow_matching.vq_embed
            print_level_report("source", source_tokens, quantizer)
            print_level_report("style", style_tokens, quantizer)
            print(
                "\n  Low levels = coarse/semantic content, high levels = fine"
                "\n  acoustic texture. Pick content_levels where the source's"
                "\n  structural information ends and the style's character begins."
            )
        return model, source_tokens, style_tokens, src_wave, src_sr

    def run(self, source_path: str, style_path: str, output_path: str,
            *, save: bool = True) -> SurgeryResult:
        """Tokenize both donors, build the hybrid, decode, optionally save."""
        cfg = self.config
        output = Path(output_path)
        if output.suffix.lower() not in {".mp3", ".wav"}:
            raise ValueError("output_path must end in .mp3 or .wav")
        if output.suffix.lower() == ".mp3" and shutil.which("ffmpeg") is None:
            raise RuntimeError("MP3 output requires ffmpeg on PATH")

        model, source_tokens, style_tokens, src_wave, src_sr = (
            self._tokenize_pair(source_path, style_path))

        hybrid = self.build_hybrid(source_tokens, style_tokens)

        parity = None
        if cfg.mode == "blend" and cfg.verify_embed_decode:
            parity = self.verify_embed_decode(source_tokens, seed=cfg.seed)

        if hybrid.kind == "embedding":
            wtxt = ", ".join(f"{w:g}" for w in hybrid.info["level_weights"])
            self._log(f"\nHybrid built: mode=blend (direct embedding decode), "
                      f"source weights = [{wtxt}]"
                      + (", requant tuple attached" if hybrid.requant_codes is not None
                         else ""))
        else:
            C = hybrid.info["content_levels"]
            self._log(f"\nHybrid built: mode=swap, levels 0..{C - 1} <- source, "
                      f"levels {C}..7 <- style")

        out_frames = round(src_wave.shape[1] * model.sample_rate / src_sr)

        jobs = [("hybrid", hybrid)]
        if cfg.baselines:
            segment = crop_style_segment(style_tokens, cfg.style_offset,
                                         cfg.style_duration)
            # tile to source length so every rendered A/B file has equal length
            segment = tile_to_length(segment, source_tokens.shape[1])
            jobs.insert(0, ("style-only",
                            HybridTokens("codes", segment, None, {})))
            jobs.insert(0, ("source-only (sanity)",
                            HybridTokens("codes", source_tokens, None, {})))

        self._log(f"\nDecoding with flow matching (steps={cfg.num_steps}, "
                  f"guidance={cfg.guidance_scale}, seed={cfg.seed}"
                  f", compile={'on' if cfg.torch_compile else 'off'}"
                  f", autocast={'on' if cfg.amp_autocast else 'off'}) ...")
        waveforms: dict[str, torch.Tensor] = {}
        paths = {"hybrid": None, "source": None, "style": None}
        for name, payload in jobs:
            key = "hybrid" if name == "hybrid" else name.split()[0]
            self._log(f"\n- decoding {name} ...")
            wav = self.decode(payload, seed=cfg.seed)
            waveforms[key] = wav
            if save:
                path = Path(f"{output.with_suffix('')}.{key}{output.suffix}")
                trim_and_export(wav, out_frames, path, model.sample_rate)
                paths[key] = path

        if cfg.dump_tokens:
            dumps = [("source", source_tokens),
                     ("style", crop_style_segment(style_tokens, cfg.style_offset,
                                                  cfg.style_duration))]
            dumps.append(("hybrid", hybrid.requant_codes
                          if hybrid.kind == "embedding" and hybrid.requant_codes
                          is not None else
                          (hybrid.data if hybrid.kind == "codes" else None)))
            if hybrid.kind == "embedding":
                dumps.append(("hybrid_emb", hybrid.data))
            for suffix, tensor in dumps:
                if tensor is None:
                    continue
                dump = Path(f"{output.with_suffix('')}.{suffix}.pt")
                torch.save(tensor.cpu(), dump)
                self._log(f"  tokens -> {dump}")

        if torch.device(cfg.device).type == "cuda":
            peak = torch.cuda.max_memory_allocated() / 2**30
            self._log(f"\nDone. Peak GPU memory: {peak:.2f} GiB")
        self._log("Tip: A/B the .source / .style / .hybrid files with the same seed.")

        return SurgeryResult(
            output_path=paths["hybrid"] if save else None,
            source_recon_path=paths["source"] if save else None,
            style_recon_path=paths["style"] if save else None,
            source_tokens=source_tokens,
            style_tokens=style_tokens,
            style_segment=crop_style_segment(style_tokens, cfg.style_offset,
                                             cfg.style_duration),
            hybrid=hybrid,
            waveforms=waveforms,
            out_frames=out_frames,
            sample_rate=model.sample_rate,
            embed_parity_diff=parity,
        )

    def compare_grid(self, source_path: str, style_path: str, output_path: str,
                     *, levels: Optional[Sequence[int]] = None,
                     ramps: Optional[Sequence[int]] = None,
                     save: bool = True) -> GridResult:
        """Render every (content_levels x ramp) variant from ONE tokenization.

        Tokenizing both donors -- the expensive part -- happens exactly once;
        only build_hybrid + decode repeat per variant. mode, seed, steps,
        guidance and all performance options come from the config. In blend
        mode the embed-decode parity check runs once (it does not depend on
        the weights), and the source/style baselines -- if enabled -- are
        decoded once as well, so each extra grid cell costs exactly one
        decode. Variant files are written next to ``output_path`` as
        ``<stem>.cl{C}<ext>`` (or ``<stem>.cl{C}_r{R}<ext>`` for variants
        whose ramp value is nonzero). Returns a GridResult holding every
        waveform in memory.
        """
        cfg = self.config
        output = Path(output_path)
        if output.suffix.lower() not in {".mp3", ".wav"}:
            raise ValueError("output_path must end in .mp3 or .wav")
        if output.suffix.lower() == ".mp3" and shutil.which("ffmpeg") is None:
            raise RuntimeError("MP3 output requires ffmpeg on PATH")

        levels = tuple(cfg.grid_levels) if levels is None else tuple(levels)
        ramps = tuple(dict.fromkeys(
            cfg.grid_ramps if ramps is None else ramps))   # dedupe, keep order
        if not levels:
            raise ValueError("compare_grid needs at least one content_levels value")
        if any(not isinstance(v, int) or isinstance(v, bool)
               or not 0 <= v <= NUM_QUANTIZERS for v in levels):
            raise ValueError(
                f"grid levels must be ints in 0..{NUM_QUANTIZERS}")
        if not ramps or any(not isinstance(v, int) or isinstance(v, bool)
                            or v < 0 for v in ramps):
            raise ValueError("grid ramps must be non-negative ints")
        if cfg.mode == "swap" and any(r != 0 for r in ramps):
            self._log("NOTE: mode='swap' ignores ramp (hard split); "
                      "collapsing grid_ramps to 0.")
            ramps = (0,)
        if cfg.level_weights is not None:
            raise ValueError(
                "config.level_weights is set; compare_grid sweeps "
                "content_levels x ramp and would ignore it. Set "
                "level_weights=None first.")

        model, source_tokens, style_tokens, src_wave, src_sr = (
            self._tokenize_pair(source_path, style_path))

        parity = None
        if cfg.mode == "blend" and cfg.verify_embed_decode:
            parity = self.verify_embed_decode(source_tokens, seed=cfg.seed)

        out_frames = round(src_wave.shape[1] * model.sample_rate / src_sr)
        stem = output.with_suffix("")

        baselines: dict[str, torch.Tensor] = {}
        baseline_paths: dict[str, Path] = {}
        if cfg.baselines:
            segment = tile_to_length(
                crop_style_segment(style_tokens, cfg.style_offset,
                                   cfg.style_duration),
                source_tokens.shape[1],
            )
            for name, payload in (
                ("source", HybridTokens("codes", source_tokens, None, {})),
                ("style", HybridTokens("codes", segment, None, {})),
            ):
                self._log(f"\n- decoding baseline '{name}' ...")
                wav = self.decode(payload, seed=cfg.seed)
                baselines[name] = wav
                if save:
                    path = Path(f"{stem}.{name}{output.suffix}")
                    trim_and_export(wav, out_frames, path, model.sample_rate)
                    baseline_paths[name] = path

        self._log(f"\nGrid: {len(levels)} level(s) x {len(ramps)} ramp "
                  f"value(s) = {len(levels) * len(ramps)} decode(s) "
                  f"(mode={cfg.mode}, steps={cfg.num_steps}, "
                  f"guidance={cfg.guidance_scale}, seed={cfg.seed})")
        variants: list[GridVariant] = []
        for content in levels:
            for ramp in ramps:
                tag = f"cl{content}" + (f"_r{ramp}" if ramp else "")
                weights = ramp_weights(content, ramp)
                hybrid = self.build_hybrid(source_tokens, style_tokens,
                                           content_levels=content, ramp=ramp,
                                           level_weights=weights)
                self._log(f"\n- decoding {tag} (source weights = "
                          f"{[round(w, 3) for w in weights]}) ...")
                wav = self.decode(hybrid, seed=cfg.seed)
                path = None
                if save:
                    path = Path(f"{stem}.{tag}{output.suffix}")
                    trim_and_export(wav, out_frames, path, model.sample_rate)
                if cfg.dump_tokens:
                    if hybrid.kind == "codes":
                        torch.save(hybrid.data.cpu(), Path(f"{stem}.{tag}.pt"))
                    else:
                        torch.save(hybrid.data.cpu(),
                                   Path(f"{stem}.{tag}_emb.pt"))
                        if hybrid.requant_codes is not None:
                            torch.save(hybrid.requant_codes.cpu(),
                                       Path(f"{stem}.{tag}_requant.pt"))
                variants.append(GridVariant(
                    content_levels=content, ramp=ramp, level_weights=weights,
                    tag=tag, hybrid=hybrid, waveform=wav, path=path,
                ))

        if cfg.verbose and not cfg.quiet:
            print("\n  grid summary:")
            for v in variants:
                wtxt = ", ".join(f"{w:g}" for w in v.level_weights)
                dest = str(v.path) if v.path is not None else "(kept in memory)"
                print(f"    {v.tag:<10} source-weights=[{wtxt}]  -> {dest}")

        if cfg.dump_tokens:
            torch.save(source_tokens.cpu(), Path(f"{stem}.source.pt"))
            torch.save(crop_style_segment(style_tokens, cfg.style_offset,
                                          cfg.style_duration).cpu(),
                       Path(f"{stem}.style.pt"))

        return GridResult(
            variants=variants,
            baselines=baselines,
            baseline_paths=baseline_paths,
            source_tokens=source_tokens,
            style_tokens=style_tokens,
            out_frames=out_frames,
            sample_rate=model.sample_rate,
            embed_parity_diff=parity,
        )


# --------------------------------------------------------------------------- #
# CLI (same options as v1, plus torch.compile / autocast / blend / grid)
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--source_path", required=True,
                   help="CONTENT donor audio (defines output length/structure).")
    p.add_argument("--style_path", required=True,
                   help="TEXTURE donor audio (defines the sonic surface).")
    p.add_argument("--output_path", default="./assets/hybrid.wav",
                   help="Output MP3 or WAV (extension decides format).")
    p.add_argument("--decoder_path", default="./ckpt/HeartCodec-oss-20260123")
    p.add_argument("--encoder_path", default="./ckpt/HeartCodec-oss-encoder")
    p.add_argument("--mode", choices=("swap", "blend"), default="swap",
                   help="swap = exact token transplant through the stock decoder; "
                        "blend = per-level embedding mix decoded directly "
                        "(no re-quantization snap).")
    p.add_argument("--content_levels", type=int, default=3,
                   help="Levels 0..C-1 kept from source (0..8). Default 3.")
    p.add_argument("--ramp", type=int, default=0,
                   help="[blend] uniform weight ramp over R levels after the cut "
                        "(C=5,R=2 -> 1,1,1,1,1,.67,.33,0). 0 = hard weights.")
    p.add_argument("--level_weights", default=None,
                   help="[blend] explicit 8 source weights, e.g. 1,1,1,0.5,0,0,0,0. "
                        "Overrides --content_levels/--ramp.")
    p.add_argument("--requantize", action=argparse.BooleanOptionalAction, default=False,
                   help="[blend] ALSO re-quantize the mix into token IDs (v1 "
                        "behavior, may garble) and use those for decoding.")
    p.add_argument("--verify_embed_decode",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="[blend] cross-check the embedding decode path against "
                        "the stock decoder on one window (max sample diff).")
    p.add_argument("--strict_parity", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="[blend] Raise instead of warning when the parity check "
                        "exceeds 5e-2 (default: warn and continue; the run "
                        "always aborts on non-finite or > 0.5 diffs).")
    p.add_argument("--style_offset", type=float, default=0.0,
                   help="Seconds into the style track to start sampling texture.")
    p.add_argument("--style_duration", type=float, default=0.0,
                   help="Seconds of style texture to use (0 = until end of track).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_steps", type=int, default=10)
    p.add_argument("--guidance_scale", type=float, default=1.25)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--device", default="cuda", help="e.g. cuda:0 or cpu")
    p.add_argument("--torch_compile", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="torch.compile the flow-matching DiT and SQ decoder.")
    p.add_argument("--compile_mode", default=None,
                   choices=sorted(_COMPILE_MODES - {None}),
                   help="torch.compile mode (default: torch's 'default').")
    p.add_argument("--amp_autocast", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Wrap decoding in torch.autocast (checkpoint stays FP32).")
    p.add_argument("--amp_dtype", default="bfloat16",
                   choices=sorted(_AMP_DTYPES),
                   help="autocast dtype (bfloat16 recommended; float16 can "
                        "overflow). Ignored unless --amp_autocast.")
    p.add_argument("--baselines", action=argparse.BooleanOptionalAction, default=True,
                   help="Also decode pure-source and pure-style tokens with the "
                        "same seed for controlled A/B listening.")
    p.add_argument("--compare_grid", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Render a grid of (content_levels x ramp) variants from a "
                        "single tokenization pass instead of one hybrid.")
    p.add_argument("--grid_levels", default="1,2,3,4,5",
                   help="[compare_grid] comma list of content_levels values, "
                        "e.g. 1,2,3,4,5.")
    p.add_argument("--grid_ramps", default="0",
                   help="[compare_grid] comma list of ramp values (blend mode), "
                        "e.g. 0,2 renders cl{C} and cl{C}_r{R} per level.")
    p.add_argument("--dump_tokens", action="store_true",
                   help="Save source/style/hybrid token (and embedding) tensors "
                        "next to the output.")
    p.add_argument("--quiet", action="store_true",
                   help="Suppress our logging (the inner solver bar always shows; "
                        "heartlib hardcodes it).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    level_weights = None
    if args.level_weights:
        level_weights = parse_level_weights(args.level_weights)

    config = TokenSurgeryConfig(
        decoder_path=args.decoder_path,
        encoder_path=args.encoder_path,
        device=args.device,
        batch_size=args.batch_size,
        mode=args.mode,
        content_levels=args.content_levels,
        ramp=args.ramp,
        level_weights=level_weights,
        style_offset=args.style_offset,
        style_duration=args.style_duration,
        requantize=args.requantize,
        verify_embed_decode=args.verify_embed_decode,
        strict_parity=args.strict_parity,
        seed=args.seed,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        torch_compile=args.torch_compile,
        compile_mode=args.compile_mode,
        amp_autocast=args.amp_autocast,
        amp_dtype=args.amp_dtype,
        baselines=args.baselines,
        dump_tokens=args.dump_tokens,
        quiet=args.quiet,
        verbose=True,
        compare_grid=args.compare_grid,
        grid_levels=parse_int_list(args.grid_levels, "--grid_levels"),
        grid_ramps=parse_int_list(args.grid_ramps, "--grid_ramps"),
    )
    surgery = TokenSurgery(config)
    if config.compare_grid:
        surgery.compare_grid(args.source_path, args.style_path,
                             args.output_path)
    else:
        surgery.run(args.source_path, args.style_path, args.output_path)


if __name__ == "__main__":
    main()