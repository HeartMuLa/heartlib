import argparse
import torch
import gc
import logging
import sys
import os
import warnings
import json
import time
import re
import textwrap
from datetime import datetime

# Suppress warnings
warnings.filterwarnings("ignore")

# Import HeartLib
from heartlib import HeartTranscriptorPipeline

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
        logging.INFO: cyan + "%(asctime)s" + reset + " [" + green + "%(levelname)s" + reset + "] %(message)s",
        logging.ERROR: red + format_str + reset,
    }
    def format(self, record):
        return logging.Formatter(self.FORMATS.get(record.levelno), datefmt="%H:%M:%S").format(record)

logger = logging.getLogger("Transcriptor")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setFormatter(ColoredFormatter())
logger.addHandler(ch)

def print_banner():
    print(f"\033[1;34m")
    print(r"""
  _    _                 _   _____                
 | |  | |               | | |_   _|               
 | |__| | ___  __ _ _ __| |_  | |_ __ __ _ _ __  
 |  __  |/ _ \/ _` | '__| __| | | '__/ _` | '_ \ 
 | |  | |  __/ (_| | |  | |_  | | | | (_| | | | |
 |_|  |_|\___|\__,_|_|   \__| |_|_|  \__,_|_| |_|
                                  INTEL XPU EDITION ∞
    """)
    print(f"\033[0m")

def clean_memory():
    gc.collect()
    if hasattr(torch, 'xpu'): 
        torch.xpu.empty_cache()

def optimize_xpu_inference(module, module_name="model"):
    compile_fn = getattr(torch, "compile", None)
    if callable(compile_fn):
        try:
            compiled = compile_fn(module, backend="inductor", mode="reduce-overhead", fullgraph=False)
            logger.info(f"Applied torch.compile(..., backend='inductor') to {module_name}.")
            return compiled
        except Exception as ex:
            logger.info(f"torch.compile unavailable/failed for {module_name} ({ex}); using standard eval/autocast path.")
            return module

    logger.info(f"No advanced XPU inference optimizer available for {module_name}; using standard eval/autocast path.")
    return module

def format_timestamp_value(value):
    if value is None:
        return "?"
    try:
        return f"{float(value):.2f}s"
    except (TypeError, ValueError):
        return "?"

def clean_transcribed_text(value):
    if not value:
        return ""
    return "".join(ch for ch in value if (ch in "\n\t" or ch.isprintable())).strip()

def format_transcribed_block(value, width=88):
    cleaned = clean_transcribed_text(value)
    if not cleaned:
        return ""

    normalized = re.sub(r"\s+", " ", cleaned).strip()
    sentences = re.split(r"(?<=[.!?])\s+", normalized)

    wrapped_lines = []
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        wrapped_lines.append(
            textwrap.fill(
                sentence,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            )
        )

    return "\n".join(wrapped_lines).strip()

def configure_third_party_logging():
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    logging.getLogger("transformers").setLevel(logging.ERROR)
    logging.getLogger("transformers.generation").setLevel(logging.ERROR)
    logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

    try:
        from transformers.utils import logging as hf_logging
        hf_logging.set_verbosity_error()
    except Exception:
        pass

def main():
    print_banner()
    clean_memory()
    configure_third_party_logging()

    parser = argparse.ArgumentParser(description="HeartTranscriptor - Lyrics Transcription (Intel XPU Optimized)")
    parser.add_argument("--music_path", type=str, required=True, help="Path to mp3/wav file")
    parser.add_argument("--model_path", type=str, default="./ckpt", help="Path to checkpoint folder")
    parser.add_argument("--save_path", type=str, default=None, help="Output txt file path")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--chunk_length_s", type=int, default=30, help="Audio chunk length in seconds")
    parser.add_argument(
        "--compile_mode",
        type=str,
        choices=["auto", "on", "off"],
        default="auto",
        help="Control torch.compile usage on model: auto (use on XPU), on (force), off (disable)",
    )
    args = parser.parse_args()

    # Hardware Check
    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device_name = torch.xpu.get_device_name(0)
        total_mem = torch.xpu.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"Hardware: \033[1;33m{device_name}\033[0m ({total_mem:.2f} GB VRAM)")
        logger.info(f"PyTorch: {torch.__version__} (Native XPU Support)")
        device = torch.device("xpu")
        
        # Set XPU optimizations
        os.environ['SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS'] = '1'
        os.environ['SYCL_CACHE_PERSISTENT'] = '1'
        torch.backends.mkldnn.enabled = True
    else:
        logger.error("No Intel XPU detected! Falling back to CPU.")
        device = torch.device("cpu")

    # Validate Paths
    if not os.path.exists(args.music_path):
        logger.error(f"File not found: {args.music_path}")
        return
    
    # Load Pipeline with optimizations
    logger.info("Loading HeartTranscriptor...")
    load_start = time.time()
    
    try:
        # Use FP16 for Whisper on XPU (good balance of speed/quality)
        pipe = HeartTranscriptorPipeline.from_pretrained(
            args.model_path,
            device=device,
            dtype=torch.float16,
        )
        
        # Apply native XPU optimizations
        if device.type == "xpu":
            logger.info("Applying native PyTorch XPU optimizations...")
            pipe.model.eval()
            should_compile = args.compile_mode == "on" or (
                args.compile_mode == "auto" and device.type == "xpu"
            )
            if should_compile:
                pipe.model = optimize_xpu_inference(pipe.model, module_name="HeartTranscriptor")
            else:
                logger.info("torch.compile disabled via --compile_mode=off.")
        
        load_time = time.time() - load_start
        logger.info(f"Model loaded in {load_time:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return

    logger.info(f"Transcribing: \033[1;36m{os.path.basename(args.music_path)}\033[0m")
    
    # Run Inference with optimizations
    transcribe_start = time.time()
    
    with torch.no_grad():
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=(device.type == "xpu")):
            # Using parameters optimized for singing voice
            result = pipe(
                args.music_path,
                max_new_tokens=256,
                num_beams=2,
                task="transcribe",
                condition_on_prev_tokens=False,
                compression_ratio_threshold=1.8,
                temperature=(0.0, 0.1, 0.2, 0.4),
                logprob_threshold=-1.0,
                no_speech_threshold=0.4,
                return_timestamps=True,
                batch_size=args.batch_size,
            )
    
    # Sync before measuring time
    if device.type == "xpu":
        torch.xpu.synchronize()
    
    transcribe_time = time.time() - transcribe_start
    
    # Output Handling
    text = format_transcribed_block(result.get("text", ""))
    chunks = result.get("chunks", [])

    print("\n" + "="*60)
    print("TRANSCRIPTION RESULT")
    print("="*60)
    print(text)
    print("="*60)
    
    # Calculate audio duration and RTF
    try:
        import torchaudio
        audio_info = torchaudio.info(args.music_path)
        audio_duration = audio_info.num_frames / audio_info.sample_rate
        rtf = transcribe_time / audio_duration
        logger.info(f"Transcription time: {transcribe_time:.2f}s (RTF: {rtf:.3f}x)")
    except:
        logger.info(f"Transcription time: {transcribe_time:.2f}s")
    
    print()

    # Save to file
    if args.save_path:
        out_file = args.save_path
    else:
        out_file = os.path.splitext(args.music_path)[0] + "_lyrics.txt"
    
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(f"Source: {os.path.basename(args.music_path)}\n")
        f.write(f"Date: {datetime.now().isoformat()}\n")
        f.write(f"Transcription Time: {transcribe_time:.2f}s\n")
        f.write("-" * 40 + "\n\n")
        f.write(text + "\n\n")
        
        if chunks:
            f.write("-" * 40 + "\n")
            f.write("TIMESTAMPS\n")
            f.write("-" * 40 + "\n")
            for chunk in chunks:
                ts = chunk.get('timestamp', (0, 0))
                txt = format_transcribed_block(chunk.get('text', ''))
                if ts and len(ts) >= 2 and txt:
                    start_ts = format_timestamp_value(ts[0])
                    end_ts = format_timestamp_value(ts[1])
                    if start_ts == "?" and end_ts == "?":
                        wrapped_txt = textwrap.indent(txt, "    ")
                        f.write(f"[timestamp unavailable]:\n{wrapped_txt}\n")
                    else:
                        wrapped_txt = textwrap.indent(txt, "    ")
                        f.write(f"[{start_ts} -> {end_ts}]:\n{wrapped_txt}\n")

    logger.info(f"Saved lyrics to: \033[1;35m{out_file}\033[0m")
    
    del pipe
    clean_memory()
    
    logger.info("\033[1;32mTranscription complete!\033[0m")

if __name__ == "__main__":
    main()
