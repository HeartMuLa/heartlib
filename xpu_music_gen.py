import argparse
import torch
import gc
import logging
import sys
import os
import time
import warnings
import json
import random
import numpy as np
from datetime import datetime
import torchaudio
from tqdm import tqdm

# Suppress PyTorch warnings
warnings.filterwarnings("ignore")

# Import installed library modules
from heartlib.heartcodec.modeling_heartcodec import HeartCodec
from heartlib.heartmula.modeling_heartmula import HeartMuLa
from heartlib.heartmula.configuration_heartmula import HeartMuLaConfig
from tokenizers import Tokenizer 
from heartlib.pipelines.music_generation import HeartMuLaGenConfig

# ==========================================
# UI & LOGGING
# ==========================================
class ColoredFormatter(logging.Formatter):
    grey = "\x1b[38;20m"
    cyan = "\x1b[36;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format_str = "%(asctime)s - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: grey + format_str + reset,
        logging.INFO: cyan + "%(asctime)s" + reset + " [" + green + "%(levelname)s" + reset + "] %(message)s",
        logging.WARNING: yellow + format_str + reset,
        logging.ERROR: red + format_str + reset,
        logging.CRITICAL: bold_red + format_str + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt="%H:%M:%S")
        return formatter.format(record)

logger = logging.getLogger("HeartMuLa")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(ColoredFormatter())
logger.addHandler(ch)

def print_banner():
    print(f"\033[1;36m")
    print(r"""
  _    _                 _   __  __       _        
 | |  | |               | | |  \/  |     | |       
 | |__| | ___  __ _ _ __| |_| \  / |_   _| |     __ _ 
 |  __  |/ _ \/ _` | '__| __| |\/| | | | | |    / _` |
 | |  | |  __/ (_| | |  | |_| |  | | |_| | |___| (_| |
 |_|  |_|\___|\__,_|_|   \__|_|  |_|\__,_|______\__,_|
                                     INTEL XPU EDITION ∞
    """)
    print(f"\033[0m")

# ==========================================
# SYSTEM LOGIC
# ==========================================
def clean_memory():
    """Aggressive memory cleaning."""
    gc.collect()
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        torch.xpu.empty_cache()

def optimize_xpu_inference(module):
    compile_fn = getattr(torch, "compile", None)
    if callable(compile_fn):
        try:
            compiled = compile_fn(module, backend="inductor", mode="reduce-overhead", fullgraph=False)
            logger.info("Applied torch.compile(..., backend='inductor', mode='reduce-overhead').")
            return compiled
        except Exception as ex:
            logger.info(f"torch.compile unavailable/failed ({ex}); using standard eval/autocast path.")
            return module

    logger.info("No advanced XPU inference optimizer available; using standard eval/autocast path.")
    return module

def set_seed(seed):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, 'xpu'):
        torch.xpu.manual_seed_all(seed)

def prepare_output_path(user_path):
    if os.path.splitext(user_path)[1]:
        base_dir = os.path.dirname(user_path)
        if base_dir and not os.path.exists(base_dir):
            os.makedirs(base_dir, exist_ok=True)
        if os.path.exists(user_path):
            filename, ext = os.path.splitext(user_path)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            return f"{filename}_{timestamp}{ext}"
        return user_path
    
    if not os.path.exists(user_path):
        os.makedirs(user_path, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(user_path, f"song_{timestamp}.mp3")

def save_metadata(audio_path, args, seed, lyrics_used, tags_used):
    """Saves generation settings alongside audio."""
    json_path = os.path.splitext(audio_path)[0] + ".json"
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "seed": seed,
        "model_version": args.version,
        "parameters": {
            "temperature": args.temperature,
            "topk": args.topk,
            "cfg_scale": args.cfg_scale,
            "max_length_ms": args.max_audio_length_ms
        },
        "prompt": {
            "tags": tags_used,
            "lyrics": lyrics_used
        }
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)

def setup_defaults_and_validate(args):
    if not os.path.exists(args.model_path):
        logger.critical(f"Model directory not found at: {args.model_path}")
        sys.exit(1)

    if not os.path.exists(args.lyrics):
        lyrics_dir = os.path.dirname(args.lyrics)
        if lyrics_dir and not os.path.exists(lyrics_dir):
            os.makedirs(lyrics_dir, exist_ok=True)
        
        logger.warning(f"Lyrics file missing. Creating sample at {args.lyrics}")
        sample_lyrics = """[Verse]
Processors humming, fans in flight
We code the dawn, we rule the night
[Chorus]
Infinity loop, never ending
The future signals we are sending"""
        with open(args.lyrics, "w", encoding="utf-8") as f:
            f.write(sample_lyrics)
    
    if args.tags is None:
        default_tag_path = "./inputs/tags/default.txt"
        if not os.path.exists(default_tag_path):
            tag_dir = os.path.dirname(default_tag_path)
            if tag_dir and not os.path.exists(tag_dir):
                os.makedirs(tag_dir, exist_ok=True)
            with open(default_tag_path, "w", encoding="utf-8") as f:
                f.write("electronic, synth, futuristic, driving")
        args.tags = default_tag_path

# ==========================================
# CORE PIPELINE
# ==========================================
def preprocess_inputs(args, tokenizer, config, muq_dim):
    # Tags
    tags = args.tags
    tags_used = tags
    if os.path.isfile(tags):
        logger.info(f"Reading Tags from: {tags}")
        with open(tags, encoding="utf-8") as fp:
            tags = fp.read()
        tags_used = tags 
    else:
        logger.info(f"Using Custom Tags")

    tags = tags.lower().strip()
    if not tags.startswith("<tag>"): tags = f"<tag>{tags}"
    if not tags.endswith("</tag>"): tags = f"{tags}</tag>"
    
    tags_ids = tokenizer.encode(tags).ids
    if tags_ids[0] != config.text_bos_id: tags_ids = [config.text_bos_id] + tags_ids
    if tags_ids[-1] != config.text_eos_id: tags_ids = tags_ids + [config.text_eos_id]

    # Lyrics
    logger.info(f"Reading Lyrics from: {args.lyrics}")
    with open(args.lyrics, encoding="utf-8") as fp:
        lyrics_content = fp.read()
    
    lyrics_used = lyrics_content
    lyrics = lyrics_content.lower().strip()
    lyrics_ids = tokenizer.encode(lyrics).ids
    if lyrics_ids[0] != config.text_bos_id: lyrics_ids = [config.text_bos_id] + lyrics_ids
    if lyrics_ids[-1] != config.text_eos_id: lyrics_ids = lyrics_ids + [config.text_eos_id]

    # Tensors
    muq_embed = torch.zeros([muq_dim], dtype=torch.bfloat16) 
    muq_idx = len(tags_ids)
    
    prompt_len = len(tags_ids) + 1 + len(lyrics_ids)
    parallel_number = 9 
    
    tokens = torch.zeros([prompt_len, parallel_number], dtype=torch.long)
    tokens[: len(tags_ids), -1] = torch.tensor(tags_ids)
    tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids)
    
    tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
    tokens_mask[:, -1] = True
    
    bs_size = 2 if args.cfg_scale != 1.0 else 1
    
    def _cfg_cat(tensor, scale):
        tensor = tensor.unsqueeze(0)
        if scale != 1.0:
            tensor = torch.cat([tensor, tensor], dim=0)
        return tensor

    return {
        "tokens": _cfg_cat(tokens, args.cfg_scale),
        "tokens_mask": _cfg_cat(tokens_mask, args.cfg_scale),
        "muq_embed": _cfg_cat(muq_embed, args.cfg_scale),
        "muq_idx": [muq_idx] * bs_size,
        "pos": _cfg_cat(torch.arange(prompt_len, dtype=torch.long), args.cfg_scale),
    }, lyrics_used, tags_used

def generate_tokens(args, base_model_path, inputs, config):
    logger.info(">>> [PHASE 1] Loading HeartMuLa 3B (LLM)...")
    
    heartmula_path = os.path.join(base_model_path, f"HeartMuLa-oss-{args.version}")
    
    # Load with memory mapping for faster loading
    model = HeartMuLa.from_pretrained(
        heartmula_path, 
        dtype=torch.bfloat16, 
        local_files_only=True,
        low_cpu_mem_usage=True,  # Enable memory-mapped loading
    )
    model.to("xpu")
    model.eval()  # Set to eval mode
    
    # Apply native PyTorch XPU optimizations
    logger.info("Applying native PyTorch XPU optimizations...")
    # Enable oneDNN for XPU (native in PyTorch 2.10+)
    model = optimize_xpu_inference(model)
    
    # Move to XPU
    prompt_tokens = inputs["tokens"].to("xpu")
    prompt_tokens_mask = inputs["tokens_mask"].to("xpu")
    prompt_pos = inputs["pos"].to("xpu")
    continuous_segment = inputs["muq_embed"].to("xpu").to(torch.bfloat16)
    starts = inputs["muq_idx"] 
    
    model.setup_caches(2 if args.cfg_scale != 1.0 else 1)
    
    frames = []
    max_audio_frames = args.max_audio_length_ms // 80
    audio_eos_id = config.audio_eos_id
    empty_id = config.empty_id
    
    logger.info(f"Generating up to \033[1;33m{max_audio_frames}\033[0m frames...")
    start_time = time.time()
    
    # Pre-allocate tensor for better performance
    pad_shape = (prompt_tokens.shape[0], 9)

    with torch.no_grad():
        with torch.autocast(device_type="xpu", dtype=torch.bfloat16, enabled=True):
            curr_token = model.generate_frame(
                tokens=prompt_tokens,
                tokens_mask=prompt_tokens_mask,
                input_pos=prompt_pos,
                temperature=args.temperature,
                topk=args.topk,
                cfg_scale=args.cfg_scale,
                continuous_segments=continuous_segment,
                starts=starts,
            )
            frames.append(curr_token[0:1,].cpu()) 
            
            pbar = tqdm(range(max_audio_frames), 
                       desc="Composing", 
                       unit="frame", 
                       colour="cyan",
                       bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]")

            try:
                for i in pbar:
                    padded_token = torch.full(pad_shape, empty_id, device="xpu", dtype=torch.long)
                    padded_token[:, :-1] = curr_token
                    padded_token = padded_token.unsqueeze(1)
                    padded_token_mask = torch.ones_like(padded_token, dtype=torch.bool)
                    padded_token_mask[..., -1] = False
                    
                    curr_token = model.generate_frame(
                        tokens=padded_token,
                        tokens_mask=padded_token_mask,
                        input_pos=prompt_pos[..., -1:] + i + 1,
                        temperature=args.temperature,
                        topk=args.topk,
                        cfg_scale=args.cfg_scale,
                        continuous_segments=None,
                        starts=None,
                    )
                    
                    if torch.any(curr_token[0:1, :] >= audio_eos_id):
                        logger.info("Song finished naturally.")
                        break
                    
                    frames.append(curr_token[0:1,].cpu())

            except KeyboardInterrupt:
                logger.warning("\nInterrupted! Finalizing...")
    
    # Final sync to ensure all operations complete
    torch.xpu.synchronize()

    duration = time.time() - start_time
    logger.info(f"Generation took {duration:.2f}s (Speed: {len(frames)/duration:.2f} frames/s)")

    logger.info("Cleaning VRAM...")
    if hasattr(model, "backbone") and hasattr(model.backbone, "caches"): model.backbone.caches = None
    if hasattr(model, "decoder") and hasattr(model.decoder, "caches"): model.decoder.caches = None
    del model
    clean_memory()
    
    return torch.stack(frames).permute(1, 2, 0).squeeze(0)

def decode_audio(base_model_path, frames):
    logger.info(">>> [PHASE 2] Loading HeartCodec...")
    codec = HeartCodec.from_pretrained(
        os.path.join(base_model_path, "HeartCodec-oss"), 
        local_files_only=True,
        low_cpu_mem_usage=True,  # Enable memory-mapped loading
    )
    codec.to("xpu")
    codec.eval()  # Set to eval mode
    
    # Apply native PyTorch XPU optimizations
    codec = optimize_xpu_inference(codec)
    
    logger.info("Rendering Audio...")
    with torch.no_grad():
        with torch.autocast(device_type="xpu", dtype=torch.float32, enabled=True):
            # FIX: The installed library version does not accept 'device="xpu"' as an argument.
            # We manually move frames to XPU and call detokenize without the device argument.
            frames = frames.to("xpu")
            wav = codec.detokenize(frames)
    
    # Sync before cleanup
    torch.xpu.synchronize()
    
    del codec
    clean_memory()
    return wav.cpu()

def main():
    print_banner()
    clean_memory()
    
    parser = argparse.ArgumentParser(description="HeartMuLa Music Generation (Intel XPU Infinity)")
    
    parser.add_argument("--model_path", type=str, default="./ckpt")
    parser.add_argument("--version", type=str, default="3B")
    parser.add_argument("--lyrics", type=str, default="./inputs/lyrics/default.txt")
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--save_path", type=str, default="./output")
    parser.add_argument("--max_audio_length_ms", type=int, default=360_000)
    parser.add_argument("--topk", type=int, default=50)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--cfg_scale", type=float, default=1.5)
    parser.add_argument("--seed", type=int, default=None)
    
    args = parser.parse_args()

    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device_name = torch.xpu.get_device_name(0)
        total_mem = torch.xpu.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Hardware: \033[1;33m{device_name}\033[0m ({total_mem:.2f} GB VRAM)")
        logger.info(f"PyTorch: {torch.__version__} (Native XPU Support)")
        
        # Set XPU memory allocation strategy for better performance
        os.environ['SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS'] = '1'
        os.environ['SYCL_CACHE_PERSISTENT'] = '1'
        
        # Enable oneDNN optimizations (native in PyTorch 2.10+)
        torch.backends.mkldnn.enabled = True
    else:
        logger.error("No Intel XPU detected!")

    if args.seed is None: args.seed = random.randint(0, 2**32 - 1)
    logger.info(f"Global Seed: \033[1;36m{args.seed}\033[0m")
    set_seed(args.seed)

    args.model_path = os.path.abspath(args.model_path)
    setup_defaults_and_validate(args)
    final_save_path = prepare_output_path(args.save_path)

    gen_config = HeartMuLaGenConfig.from_file(os.path.join(args.model_path, "gen_config.json"))
    heartmula_path = os.path.join(args.model_path, f"HeartMuLa-oss-{args.version}")
    model_config = HeartMuLaConfig.from_pretrained(heartmula_path, local_files_only=True)
    tokenizer = Tokenizer.from_file(os.path.join(args.model_path, "tokenizer.json"))
    
    inputs, lyrics_used, tags_used = preprocess_inputs(args, tokenizer, gen_config, model_config.muq_dim)
    frames = generate_tokens(args, args.model_path, inputs, gen_config)
    wav = decode_audio(args.model_path, frames)
    
    logger.info(f"Saving audio to: \033[1;35m{final_save_path}\033[0m")
    torchaudio.save(final_save_path, wav, 48000)
    save_metadata(final_save_path, args, args.seed, lyrics_used, tags_used)
    
    logger.info("\033[1;32mProcess Complete!\033[0m")

if __name__ == "__main__":
    main()
