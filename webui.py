#!/usr/bin/env python3
"""
HeartMuLa Gradio WebUI
Complete interface for music generation and lyrics transcription on Intel XPU
"""

import gradio as gr
import torch
import os
import json
import time
import gc
import sys
import warnings
from datetime import datetime
from pathlib import Path
import torchaudio
from typing import Optional, Tuple

# Ensure local `src` package is imported before any installed `site-packages` copy.
_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# Suppress known non-actionable warnings from Whisper pipeline usage.
warnings.filterwarnings("ignore", message=r"Using `chunk_length_s` is very experimental.*")
warnings.filterwarnings("ignore", message=r"`torch_dtype` is deprecated! Use `dtype` instead!.*")
warnings.filterwarnings("ignore", message=r"`generation_config` default values have been modified.*")
warnings.filterwarnings("ignore", message=r"A custom logits processor of type <class 'transformers\.generation\.logits_process\.SuppressTokensLogitsProcessor'> has been passed.*")
warnings.filterwarnings("ignore", message=r"A custom logits processor of type <class 'transformers\.generation\.logits_process\.SuppressTokensAtBeginLogitsProcessor'> has been passed.*")
warnings.filterwarnings("ignore", message=r"Whisper did not predict an ending timestamp.*")

# Import HeartLib components
from heartlib.heartcodec.modeling_heartcodec import HeartCodec
from heartlib.heartmula.modeling_heartmula import HeartMuLa
from heartlib.heartmula.configuration_heartmula import HeartMuLaConfig
from heartlib import HeartTranscriptorPipeline
from heartlib.pipelines.music_generation import HeartMuLaGenConfig
from tokenizers import Tokenizer

# ==========================================
# GLOBAL STATE
# ==========================================
class AppState:
    def __init__(self):
        self.mula_model = None
        self.codec_model = None
        self.transcriptor = None
        self.tokenizer = None
        self.gen_config = None
        self.model_config = None
        self.model_path = "./ckpt"
        self.device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
        
    def clear_memory(self):
        gc.collect()
        if torch.xpu.is_available():
            torch.xpu.empty_cache()

state = AppState()

APP_THEME = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="cyan",
    neutral_hue="slate",
    font=["Inter", "ui-sans-serif", "system-ui"],
).set(
    body_background_fill="*neutral_50",
    block_title_text_weight="600",
    block_border_width="1px",
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
)

APP_CSS = """
        .header {text-align: center; padding: 20px;}
        .status-box {padding: 15px; border-radius: 8px; margin: 10px 0;}
        .metric {font-weight: 600; color: #2563eb;}
"""

# ==========================================
# UTILITY FUNCTIONS
# ==========================================
def get_gpu_info():
    """Get GPU information"""
    if not torch.xpu.is_available():
        return "❌ No Intel XPU detected"
    
    try:
        device_name = torch.xpu.get_device_name(0)
        props = torch.xpu.get_device_properties(0)
        total_mem = props.total_memory / (1024**3)
        used_mem = (props.total_memory - torch.xpu.memory_reserved(0)) / (1024**3)
        
        return f"""
🎮 **GPU Info**
- Device: {device_name}
- Total VRAM: {total_mem:.2f} GB
- Available: {used_mem:.2f} GB
- PyTorch: {torch.__version__}
- XPU: Native Support ✓
"""
    except Exception as e:
        return f"⚠️ Error getting GPU info: {e}"

def optimize_xpu_inference(module, module_name: str = "model"):
    compile_fn = getattr(torch, "compile", None)
    if callable(compile_fn):
        try:
            compiled = compile_fn(module, backend="inductor", mode="reduce-overhead", fullgraph=False)
            print(f"✅ Applied torch.compile(..., backend='inductor') to {module_name}")
            return compiled
        except Exception as ex:
            print(f"ℹ️ torch.compile unavailable/failed for {module_name} ({ex}); using standard eval/autocast path.")
            return module

    print(f"ℹ️ No advanced XPU inference optimizer available for {module_name}; using standard eval/autocast path.")
    return module

def load_models(model_path: str, version: str = "3B") -> str:
    """Load HeartMuLa and HeartCodec models"""
    try:
        state.model_path = model_path
        
        # Load configs
        state.gen_config = HeartMuLaGenConfig.from_file(
            os.path.join(model_path, "gen_config.json")
        )
        heartmula_path = os.path.join(model_path, f"HeartMuLa-oss-{version}")
        state.model_config = HeartMuLaConfig.from_pretrained(
            heartmula_path, local_files_only=True
        )
        state.tokenizer = Tokenizer.from_file(
            os.path.join(model_path, "tokenizer.json")
        )
        
        # Load HeartMuLa
        state.mula_model = HeartMuLa.from_pretrained(
            heartmula_path,
            dtype=torch.bfloat16,
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        state.mula_model.to(state.device)
        state.mula_model.eval()
        
        # Apply XPU optimizations
        if state.device.type == "xpu":
            state.mula_model = optimize_xpu_inference(state.mula_model, module_name="HeartMuLa")
        
        # Load HeartCodec
        state.codec_model = HeartCodec.from_pretrained(
            os.path.join(model_path, "HeartCodec-oss"),
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        state.codec_model.to(state.device)
        state.codec_model.eval()
        
        if state.device.type == "xpu":
            state.codec_model = optimize_xpu_inference(state.codec_model, module_name="HeartCodec")
        
        return "✅ Models loaded successfully!"
        
    except Exception as e:
        return f"❌ Error loading models: {str(e)}"

def load_transcriptor(model_path: str) -> str:
    """Load HeartTranscriptor model"""
    try:
        state.transcriptor = HeartTranscriptorPipeline.from_pretrained(
            model_path,
            device=state.device,
            dtype=torch.float16,
            chunk_length_s=30,
            ignore_warning=True,
        )
        
        if state.device.type == "xpu":
            state.transcriptor.model.eval()
            state.transcriptor.model = optimize_xpu_inference(
                state.transcriptor.model, module_name="HeartTranscriptor"
            )
        
        return "✅ Transcriptor loaded successfully!"
        
    except Exception as e:
        return f"❌ Error loading transcriptor: {str(e)}"

# ==========================================
# MUSIC GENERATION
# ==========================================
def preprocess_inputs(lyrics: str, tags: str, cfg_scale: float):
    """Preprocess lyrics and tags"""
    # Process tags
    tags = tags.lower().strip()
    if not tags.startswith("<tag>"): tags = f"<tag>{tags}"
    if not tags.endswith("</tag>"): tags = f"{tags}</tag>"
    
    tags_ids = state.tokenizer.encode(tags).ids
    if tags_ids[0] != state.gen_config.text_bos_id:
        tags_ids = [state.gen_config.text_bos_id] + tags_ids
    if tags_ids[-1] != state.gen_config.text_eos_id:
        tags_ids = tags_ids + [state.gen_config.text_eos_id]
    
    # Process lyrics
    lyrics = lyrics.lower().strip()
    lyrics_ids = state.tokenizer.encode(lyrics).ids
    if lyrics_ids[0] != state.gen_config.text_bos_id:
        lyrics_ids = [state.gen_config.text_bos_id] + lyrics_ids
    if lyrics_ids[-1] != state.gen_config.text_eos_id:
        lyrics_ids = lyrics_ids + [state.gen_config.text_eos_id]
    
    # Create tensors
    muq_embed = torch.zeros([state.model_config.muq_dim], dtype=torch.bfloat16)
    muq_idx = len(tags_ids)
    
    prompt_len = len(tags_ids) + 1 + len(lyrics_ids)
    parallel_number = 9
    
    tokens = torch.zeros([prompt_len, parallel_number], dtype=torch.long)
    tokens[: len(tags_ids), -1] = torch.tensor(tags_ids)
    tokens[len(tags_ids) + 1 :, -1] = torch.tensor(lyrics_ids)
    
    tokens_mask = torch.zeros_like(tokens, dtype=torch.bool)
    tokens_mask[:, -1] = True
    
    bs_size = 2 if cfg_scale != 1.0 else 1
    
    def _cfg_cat(tensor, scale):
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

def generate_music(
    lyrics: str,
    tags: str,
    max_length_ms: int,
    temperature: float,
    topk: int,
    cfg_scale: float,
    seed: Optional[int],
    progress=gr.Progress()
) -> Tuple[Optional[str], str, dict]:
    """Generate music from lyrics and tags"""
    
    if not lyrics.strip():
        return None, "❌ Please provide lyrics", {}
    
    if state.mula_model is None or state.codec_model is None:
        return None, "❌ Models not loaded. Please load models first.", {}
    
    try:
        # Set seed
        if seed is None or seed < 0:
            seed = torch.randint(0, 2**32, (1,)).item()
        
        torch.manual_seed(seed)
        if state.device.type == "xpu":
            torch.xpu.manual_seed_all(seed)
        
        progress(0, desc="Preprocessing inputs...")
        
        # Preprocess
        inputs = preprocess_inputs(lyrics, tags, cfg_scale)
        
        # Generate tokens
        progress(0.1, desc="Generating audio tokens...")
        
        prompt_tokens = inputs["tokens"].to(state.device)
        prompt_tokens_mask = inputs["tokens_mask"].to(state.device)
        prompt_pos = inputs["pos"].to(state.device)
        continuous_segment = inputs["muq_embed"].to(state.device).to(torch.bfloat16)
        starts = inputs["muq_idx"]
        
        state.mula_model.setup_caches(2 if cfg_scale != 1.0 else 1)
        
        frames = []
        max_audio_frames = max_length_ms // 80
        audio_eos_id = state.gen_config.audio_eos_id
        empty_id = state.gen_config.empty_id
        
        start_time = time.time()
        
        with torch.no_grad():
            with torch.autocast(device_type=state.device.type, dtype=torch.bfloat16, enabled=True):
                # Initial frame
                curr_token = state.mula_model.generate_frame(
                    tokens=prompt_tokens,
                    tokens_mask=prompt_tokens_mask,
                    input_pos=prompt_pos,
                    temperature=temperature,
                    topk=topk,
                    cfg_scale=cfg_scale,
                    continuous_segments=continuous_segment,
                    starts=starts,
                )
                frames.append(curr_token[0:1,].cpu())
                
                pad_shape = (curr_token.shape[0], 9)
                
                # Generate remaining frames
                for i in range(max_audio_frames):
                    progress((i + 1) / max_audio_frames * 0.8 + 0.1, 
                            desc=f"Composing: {i+1}/{max_audio_frames} frames")
                    
                    padded_token = torch.full(pad_shape, empty_id, device=state.device, dtype=torch.long)
                    padded_token[:, :-1] = curr_token
                    padded_token = padded_token.unsqueeze(1)
                    padded_token_mask = torch.ones_like(padded_token, dtype=torch.bool)
                    padded_token_mask[..., -1] = False
                    
                    curr_token = state.mula_model.generate_frame(
                        tokens=padded_token,
                        tokens_mask=padded_token_mask,
                        input_pos=prompt_pos[..., -1:] + i + 1,
                        temperature=temperature,
                        topk=topk,
                        cfg_scale=cfg_scale,
                        continuous_segments=None,
                        starts=None,
                    )
                    
                    if torch.any(curr_token[0:1, :] >= audio_eos_id):
                        break
                    
                    frames.append(curr_token[0:1,].cpu())
        
        if state.device.type == "xpu":
            torch.xpu.synchronize()
        
        gen_time = time.time() - start_time
        fps = len(frames) / gen_time
        
        progress(0.9, desc="Decoding audio...")
        
        # Decode audio
        frames_tensor = torch.stack(frames).permute(1, 2, 0).squeeze(0)
        frames_tensor = frames_tensor.to(state.device)
        
        with torch.no_grad():
            with torch.autocast(device_type=state.device.type, dtype=torch.float32, enabled=True):
                wav = state.codec_model.detokenize(frames_tensor)
        
        if state.device.type == "xpu":
            torch.xpu.synchronize()
        
        # Save audio
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"song_{timestamp}.mp3")
        
        torchaudio.save(output_path, wav.cpu(), 48000)
        
        # Save metadata
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "seed": seed,
            "parameters": {
                "temperature": temperature,
                "topk": topk,
                "cfg_scale": cfg_scale,
                "max_length_ms": max_length_ms,
            },
            "prompt": {
                "tags": tags,
                "lyrics": lyrics,
            },
            "performance": {
                "generation_time": gen_time,
                "frames_per_second": fps,
                "total_frames": len(frames),
            }
        }
        
        metadata_path = output_path.replace(".mp3", ".json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        status = f"""
✅ **Generation Complete!**

📊 **Performance**
- Generation Time: {gen_time:.2f}s
- Speed: {fps:.2f} frames/s
- Total Frames: {len(frames)}
- Seed: {seed}

💾 **Output**
- Audio: {output_path}
- Metadata: {metadata_path}
"""
        
        state.clear_memory()
        
        return output_path, status, metadata
        
    except Exception as e:
        state.clear_memory()
        return None, f"❌ Error: {str(e)}", {"error": str(e)}

# ==========================================
# TRANSCRIPTION
# ==========================================
def transcribe_audio(
    audio_path: str,
    batch_size: int,
    chunk_length_s: int,
    progress=gr.Progress()
) -> Tuple[str, str, str]:
    """Transcribe audio to lyrics"""
    
    if not audio_path:
        return "", "❌ Please provide an audio file", ""
    
    if state.transcriptor is None:
        return "", "❌ Transcriptor not loaded. Please load transcriptor first.", ""
    
    try:
        progress(0, desc="Loading audio...")
        
        start_time = time.time()
        
        progress(0.2, desc="Transcribing...")
        
        with torch.no_grad():
            with torch.autocast(device_type=state.device.type, dtype=torch.float16, 
                              enabled=(state.device.type == "xpu")):
                result = state.transcriptor(
                    audio_path,
                    max_new_tokens=256,
                    num_beams=2,
                    task="transcribe",
                    condition_on_prev_tokens=False,
                    compression_ratio_threshold=1.8,
                    temperature=(0.0, 0.1, 0.2, 0.4),
                    logprob_threshold=-1.0,
                    no_speech_threshold=0.4,
                    return_timestamps=True,
                    batch_size=batch_size,
                )
        
        if state.device.type == "xpu":
            torch.xpu.synchronize()
        
        transcribe_time = time.time() - start_time
        
        text = result.get("text", "").strip()
        chunks = result.get("chunks", [])
        
        # Calculate RTF
        try:
            audio_info = torchaudio.info(audio_path)
            audio_duration = audio_info.num_frames / audio_info.sample_rate
            rtf = transcribe_time / audio_duration
            rtf_text = f"RTF: {rtf:.3f}x ({1/rtf:.1f}x faster than real-time)"
        except:
            rtf_text = ""
        
        # Save output
        output_dir = "./output"
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(output_dir, f"transcription_{timestamp}.txt")
        
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(f"Source: {os.path.basename(audio_path)}\n")
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
                    txt = chunk.get('text', '').strip()
                    if ts and len(ts) >= 2 and txt:
                        f.write(f"[{ts[0]:.2f}s -> {ts[1]:.2f}s]: {txt}\n")
        
        # Create timestamped output
        timestamp_text = ""
        if chunks:
            timestamp_text = "\n".join([
                f"[{chunk.get('timestamp', (0,0))[0]:.2f}s]: {chunk.get('text', '').strip()}"
                for chunk in chunks if chunk.get('text', '').strip()
            ])
        
        status = f"""
✅ **Transcription Complete!**

📊 **Performance**
- Transcription Time: {transcribe_time:.2f}s
- {rtf_text}

💾 **Output**
- Saved to: {output_path}
"""
        
        state.clear_memory()
        
        return text, status, timestamp_text
        
    except Exception as e:
        state.clear_memory()
        return "", f"❌ Error: {str(e)}", ""

# ==========================================
# GRADIO INTERFACE
# ==========================================
def create_ui():
    """Create Gradio interface"""

    with gr.Blocks(
        title="HeartMuLa - Intel XPU Edition",
    ) as app:
        
        # Header
        gr.Markdown(
            """
            # 🎵 HeartMuLa - Intel XPU Edition
            ### AI Music Generation & Lyrics Transcription
            
            **Optimized for Intel ARC GPUs** • PyTorch 2.10+ Native XPU Support
            """,
            elem_classes="header"
        )
        
        with gr.Tabs() as tabs:
            
            # ==========================================
            # TAB 1: MUSIC GENERATION
            # ==========================================
            with gr.Tab("🎼 Music Generation", id=0):
                with gr.Row():
                    with gr.Column(scale=2):
                        gr.Markdown("### 📝 Input")
                        
                        lyrics_input = gr.Textbox(
                            label="Lyrics",
                            placeholder="""[Verse]
Walking down the street
Feeling the beat

[Chorus]
This is my song
Singing along""",
                            lines=10,
                            info="Enter your lyrics with structure tags like [Verse], [Chorus], etc."
                        )
                        
                        tags_input = gr.Textbox(
                            label="Tags",
                            placeholder="electronic,synth,upbeat,energetic",
                            info="Comma-separated tags (no spaces)"
                        )
                        
                        with gr.Accordion("⚙️ Advanced Settings", open=False):
                            with gr.Row():
                                max_length = gr.Slider(
                                    minimum=30000,
                                    maximum=600000,
                                    value=240000,
                                    step=30000,
                                    label="Max Length (ms)",
                                    info="Audio duration (4 minutes default)"
                                )
                                temperature = gr.Slider(
                                    minimum=0.5,
                                    maximum=1.5,
                                    value=1.0,
                                    step=0.1,
                                    label="Temperature",
                                    info="Higher = more creative"
                                )
                            
                            with gr.Row():
                                topk = gr.Slider(
                                    minimum=10,
                                    maximum=100,
                                    value=50,
                                    step=5,
                                    label="Top-K",
                                    info="Sampling pool size"
                                )
                                cfg_scale = gr.Slider(
                                    minimum=1.0,
                                    maximum=3.0,
                                    value=1.5,
                                    step=0.1,
                                    label="CFG Scale",
                                    info="1.0=fast, 2.0+=quality"
                                )
                            
                            seed_input = gr.Number(
                                label="Seed",
                                value=-1,
                                precision=0,
                                info="-1 for random seed"
                            )
                        
                        with gr.Row():
                            generate_btn = gr.Button(
                                "🎵 Generate Music",
                                variant="primary",
                                size="lg"
                            )
                            clear_gen_btn = gr.Button("🗑️ Clear", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎧 Output")
                        
                        audio_output = gr.Audio(
                            label="Generated Music",
                            type="filepath"
                        )
                        
                        gen_status = gr.Markdown(
                            "Ready to generate music!",
                            elem_classes="status-box"
                        )
                        
                        with gr.Accordion("📄 Metadata", open=False):
                            metadata_output = gr.JSON(label="Generation Details")
                
                # Examples
                gr.Markdown("### 💡 Examples")
                gr.Examples(
                    examples=[
                        [
                            "[Verse]\nWalking through the city lights\nEverything feels so right\n\n[Chorus]\nDancing in the moonlight\nUntil the morning light",
                            "electronic,synth,dance,upbeat",
                            180000, 1.0, 50, 1.5, -1
                        ],
                        [
                            "[Intro]\n\n[Verse]\nSitting by the window\nWatching rain fall slow\n\n[Chorus]\nMemories come and go\nLike rivers flow",
                            "piano,melancholic,slow,emotional",
                            240000, 1.0, 50, 2.0, -1
                        ],
                    ],
                    inputs=[lyrics_input, tags_input, max_length, temperature, topk, cfg_scale, seed_input],
                    outputs=[audio_output, gen_status, metadata_output],
                    fn=generate_music,
                    cache_examples=False,
                )
            
            # ==========================================
            # TAB 2: LYRICS TRANSCRIPTION
            # ==========================================
            with gr.Tab("🎤 Lyrics Transcription", id=1):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown("### 🎵 Input Audio")
                        
                        audio_input = gr.Audio(
                            label="Upload Audio",
                            type="filepath",
                            sources=["upload", "microphone"]
                        )
                        
                        with gr.Accordion("⚙️ Settings", open=False):
                            batch_size_trans = gr.Slider(
                                minimum=4,
                                maximum=32,
                                value=16,
                                step=4,
                                label="Batch Size",
                                info="Higher = faster (more VRAM)"
                            )
                            
                            chunk_length = gr.Slider(
                                minimum=15,
                                maximum=60,
                                value=30,
                                step=5,
                                label="Chunk Length (s)",
                                info="Audio processing chunk size"
                            )
                        
                        with gr.Row():
                            transcribe_btn = gr.Button(
                                "🎤 Transcribe",
                                variant="primary",
                                size="lg"
                            )
                            clear_trans_btn = gr.Button("🗑️ Clear", size="lg")
                    
                    with gr.Column(scale=1):
                        gr.Markdown("### 📝 Transcription")
                        
                        transcription_output = gr.Textbox(
                            label="Lyrics",
                            lines=10
                        )
                        
                        trans_status = gr.Markdown(
                            "Ready to transcribe!",
                            elem_classes="status-box"
                        )
                        
                        with gr.Accordion("⏱️ Timestamps", open=False):
                            timestamps_output = gr.Textbox(
                                label="Timestamped Lyrics",
                                lines=10
                            )
            
            # ==========================================
            # TAB 3: SETTINGS & MODEL MANAGEMENT
            # ==========================================
            with gr.Tab("⚙️ Settings", id=2):
                gr.Markdown("### 🔧 Model Management")
                
                with gr.Row():
                    model_path_input = gr.Textbox(
                        label="Model Path",
                        value="./ckpt",
                        info="Path to checkpoint directory"
                    )
                    model_version = gr.Dropdown(
                        label="Model Version",
                        choices=["3B"],
                        value="3B",
                        info="HeartMuLa model size"
                    )
                
                with gr.Row():
                    load_gen_models_btn = gr.Button(
                        "📥 Load Generation Models",
                        variant="primary"
                    )
                    load_trans_model_btn = gr.Button(
                        "📥 Load Transcription Model",
                        variant="primary"
                    )
                    clear_models_btn = gr.Button(
                        "🗑️ Clear Memory"
                    )
                
                model_status = gr.Markdown("Models not loaded")
                
                gr.Markdown("---")
                gr.Markdown("### 📋 Presets")
                
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Speed Mode**")
                        gr.Markdown("- CFG Scale: 1.0\n- Temperature: 0.9\n- Top-K: 30")
                        speed_preset_btn = gr.Button("Apply Speed Preset")
                    
                    with gr.Column():
                        gr.Markdown("**Quality Mode**")
                        gr.Markdown("- CFG Scale: 2.0\n- Temperature: 1.0\n- Top-K: 50")
                        quality_preset_btn = gr.Button("Apply Quality Preset")
                    
                    with gr.Column():
                        gr.Markdown("**Balanced Mode**")
                        gr.Markdown("- CFG Scale: 1.5\n- Temperature: 1.0\n- Top-K: 50")
                        balanced_preset_btn = gr.Button("Apply Balanced Preset")
            
            # ==========================================
            # TAB 4: SYSTEM INFO
            # ==========================================
            with gr.Tab("📊 System Info", id=3):
                gr.Markdown("### 🎮 GPU Information")
                
                gpu_info_output = gr.Markdown(get_gpu_info())
                refresh_gpu_btn = gr.Button("🔄 Refresh GPU Info")
                
                gr.Markdown("---")
                gr.Markdown("### 📚 Quick Reference")
                
                gr.Markdown("""
                #### Music Generation Tips
                - **Speed**: Use `cfg_scale=1.0` for ~30-40% faster generation
                - **Quality**: Use `cfg_scale=2.0` for better adherence to prompts
                - **GPU Usage**: Longer sequences (6-8 min) utilize GPU better
                
                #### Transcription Tips
                - **Speed**: Increase `batch_size` to 24-32 for shorter files
                - **Accuracy**: Increase `chunk_length` to 35-40 for better context
                - **VRAM**: Reduce `batch_size` if you run out of memory
                
                #### Performance Metrics
                - **Generation**: 4.5-6.5 fps (optimized)
                - **Transcription**: RTF 0.15-0.3 (5-7x real-time)
                - **GPU Utilization**: 75-90% (generation), 65-85% (transcription)
                """)
                
                gr.Markdown("---")
                gr.Markdown("""
                ### 🚀 Keyboard Shortcuts
                - `Ctrl/Cmd + Enter` in text fields to trigger actions
                - `Tab` to navigate between fields
                """)
        
        # ==========================================
        # EVENT HANDLERS
        # ==========================================
        
        # Generation
        generate_btn.click(
            fn=generate_music,
            inputs=[lyrics_input, tags_input, max_length, temperature, topk, cfg_scale, seed_input],
            outputs=[audio_output, gen_status, metadata_output],
        )
        
        clear_gen_btn.click(
            lambda: ("", "", 240000, 1.0, 50, 1.5, -1, None, "Cleared!", {}),
            outputs=[lyrics_input, tags_input, max_length, temperature, topk, cfg_scale, 
                    seed_input, audio_output, gen_status, metadata_output]
        )
        
        # Transcription
        transcribe_btn.click(
            fn=transcribe_audio,
            inputs=[audio_input, batch_size_trans, chunk_length],
            outputs=[transcription_output, trans_status, timestamps_output]
        )
        
        clear_trans_btn.click(
            lambda: (None, "", "Cleared!", ""),
            outputs=[audio_input, transcription_output, trans_status, timestamps_output]
        )
        
        # Model management
        load_gen_models_btn.click(
            fn=load_models,
            inputs=[model_path_input, model_version],
            outputs=[model_status]
        )
        
        load_trans_model_btn.click(
            fn=load_transcriptor,
            inputs=[model_path_input],
            outputs=[model_status]
        )
        
        clear_models_btn.click(
            fn=lambda: (state.clear_memory(), "✅ Memory cleared")[1],
            outputs=[model_status]
        )
        
        # Presets
        speed_preset_btn.click(
            lambda: (1.0, 0.9, 30),
            outputs=[cfg_scale, temperature, topk]
        )
        
        quality_preset_btn.click(
            lambda: (2.0, 1.0, 50),
            outputs=[cfg_scale, temperature, topk]
        )
        
        balanced_preset_btn.click(
            lambda: (1.5, 1.0, 50),
            outputs=[cfg_scale, temperature, topk]
        )
        
        # GPU info refresh
        refresh_gpu_btn.click(
            fn=get_gpu_info,
            outputs=[gpu_info_output]
        )
    
    return app

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    # Set XPU optimizations
    if torch.xpu.is_available():
        os.environ['SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS'] = '1'
        os.environ['SYCL_CACHE_PERSISTENT'] = '1'
        torch.backends.mkldnn.enabled = True
    
    print("🎵 Starting HeartMuLa WebUI...")
    print(f"📍 Device: {state.device}")
    print(f"🎮 GPU: {torch.xpu.get_device_name(0) if torch.xpu.is_available() else 'N/A'}")
    
    app = create_ui()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        favicon_path=None,
        theme=APP_THEME,
        css=APP_CSS,
    )
