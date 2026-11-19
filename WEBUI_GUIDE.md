# HeartMuLa WebUI Documentation

## Overview

A comprehensive Gradio-based web interface for HeartMuLa, providing:
- **Music Generation** from lyrics and tags
- **Lyrics Transcription** from audio files
- **Model Management** and settings
- **Performance Monitoring** and GPU stats
- **XPU-Optimized** for Intel ARC GPUs

## 🚀 Quick Start

### Installation

1. **Install Gradio** (if not already installed):
```bash
pip install gradio
```

Or install all requirements:
```bash
pip install -r requirements-xpu.txt
```

### Launch WebUI

**Option 1: Use the launcher script (recommended)**
```bash
./launch_webui.sh
```

**Option 2: Direct launch**
```bash
python webui.py
```

The WebUI will be available at:
- **Local**: http://localhost:7860
- **Network**: http://YOUR_IP:7860

## 📱 Features

### 1. Music Generation Tab 🎼

Generate music from lyrics and tags with full control over generation parameters.

**Features:**
- Text input for lyrics with structure support ([Verse], [Chorus], etc.)
- Tag input for music style and mood
- Advanced parameter controls:
  - Max Length: 30s to 10 minutes
  - Temperature: Creativity control (0.5-1.5)
  - Top-K: Sampling diversity (10-100)
  - CFG Scale: Quality vs speed (1.0-3.0)
  - Seed: For reproducible results
- Real-time progress tracking
- Audio playback in browser
- Metadata export (JSON)
- Built-in examples

**Usage:**
1. Enter your lyrics in the text box
2. Add comma-separated tags (e.g., "electronic,upbeat,dance")
3. Adjust parameters if needed (or use presets)
4. Click "Generate Music"
5. Wait for generation (progress bar shows status)
6. Play audio directly in browser or download

**Tips:**
- Use structure tags like [Intro], [Verse], [Chorus], [Bridge], [Outro]
- Keep tags simple and descriptive
- Lower CFG Scale (1.0) for faster generation
- Higher CFG Scale (2.0+) for better quality

### 2. Lyrics Transcription Tab 🎤

Transcribe audio files to extract lyrics with timestamps.

**Features:**
- Upload audio files (mp3, wav, flac, etc.)
- Microphone recording support
- Batch size control (4-32)
- Chunk length adjustment (15-60s)
- Real-time transcription progress
- Timestamped output
- Copy-to-clipboard support
- Performance metrics (RTF)

**Usage:**
1. Upload an audio file or record from microphone
2. Adjust batch size for speed vs memory tradeoff
3. Click "Transcribe"
4. View transcription results
5. Check timestamps in accordion
6. Copy or download results

**Tips:**
- Increase batch size (24-32) for shorter files
- Decrease batch size (8-12) for longer files or to save VRAM
- Longer chunk length (40s+) provides better context
- RTF < 0.3 means faster than real-time

### 3. Settings Tab ⚙️

Manage models, presets, and configuration.

**Model Management:**
- Load Generation Models (HeartMuLa + HeartCodec)
- Load Transcription Model (HeartTranscriptor)
- Clear GPU memory
- Change model path
- Select model version

**Presets:**
- **Speed Mode**: Fast generation, good quality
  - CFG Scale: 1.0
  - Temperature: 0.9
  - Top-K: 30
  
- **Quality Mode**: Best quality, slower
  - CFG Scale: 2.0
  - Temperature: 1.0
  - Top-K: 50
  
- **Balanced Mode**: Good balance (default)
  - CFG Scale: 1.5
  - Temperature: 1.0
  - Top-K: 50

**Usage:**
1. Set model path (default: `./ckpt`)
2. Click "Load Generation Models" before generating music
3. Click "Load Transcription Model" before transcribing
4. Use presets to quickly apply optimized settings
5. Clear memory when switching between tasks

### 4. System Info Tab 📊

View GPU information and documentation.

**Features:**
- Real-time GPU stats
- VRAM usage monitoring
- PyTorch version info
- Quick reference guide
- Keyboard shortcuts
- Performance tips

**GPU Info Displays:**
- Device name (e.g., Intel ARC A770)
- Total VRAM
- Available VRAM
- PyTorch version
- XPU support status

## 🎨 Interface Features

### Theme
- Modern "Soft" theme with blue/cyan accents
- Clean, professional design
- Responsive layout
- High contrast for readability

### Layout
- **Tab-based navigation** for organized workflow
- **Responsive columns** adapt to screen size
- **Accordion panels** for advanced options
- **Progress indicators** for long operations
- **Status messages** with emoji indicators

### User Experience
- Real-time progress tracking
- Clear status messages
- Error handling with descriptive messages
- Copy-to-clipboard support
- Download options for all outputs
- Keyboard navigation support

## 🔧 Advanced Usage

### Model Loading Strategy

**For Music Generation Only:**
```
Settings Tab → Load Generation Models
```
This loads HeartMuLa (3B) and HeartCodec (~8GB VRAM)

**For Transcription Only:**
```
Settings Tab → Load Transcription Model
```
This loads HeartTranscriptor (~2GB VRAM)

**For Both:**
Load both sets of models, but be aware of VRAM usage (~10-12GB total)

**Memory Management:**
- Click "Clear Memory" between tasks to free VRAM
- Models stay loaded until you clear or restart
- WebUI remembers loaded models across tabs

### Parameter Optimization

**Music Generation:**

| Goal | CFG Scale | Temperature | Top-K | Length |
|------|-----------|-------------|-------|--------|
| Fast | 1.0 | 0.9 | 30 | 3-4 min |
| Quality | 2.0 | 1.0 | 50 | 4-6 min |
| Creative | 1.5 | 1.2 | 60 | 4-6 min |
| Consistent | 1.5 | 0.8 | 40 | 4-6 min |

**Transcription:**

| Goal | Batch Size | Chunk Length |
|------|------------|--------------|
| Speed | 32 | 20s |
| Accuracy | 12 | 40s |
| Balanced | 16 | 30s |
| Low VRAM | 8 | 25s |

### Workflow Examples

**Example 1: Generate and Verify**
```
1. Go to Settings → Load Generation Models
2. Go to Music Generation tab
3. Enter lyrics and tags
4. Generate music
5. Download the audio file
6. Go to Settings → Clear Memory → Load Transcription Model
7. Go to Transcription tab
8. Upload the generated audio
9. Transcribe to verify lyrics
```

**Example 2: Batch Transcription**
```
1. Go to Settings → Load Transcription Model
2. Go to Transcription tab
3. Upload first audio file
4. Transcribe
5. Copy/download results
6. Upload next audio file
7. Repeat (model stays loaded)
```

**Example 3: Experimentation**
```
1. Load Generation Models
2. Generate with Speed preset
3. Generate with Quality preset
4. Generate with Balanced preset
5. Compare outputs
```

## 🖥️ System Requirements

### Minimum
- Intel ARC GPU (A380 or higher)
- 8GB VRAM
- 16GB System RAM
- PyTorch 2.10+ with XPU support

### Recommended
- Intel ARC A770 (16GB VRAM)
- 32GB System RAM
- SSD storage
- Updated Intel GPU drivers

### VRAM Usage

| Task | Models Loaded | VRAM Usage |
|------|--------------|------------|
| Music Gen (CFG=1.0) | MuLa + Codec | 6-8 GB |
| Music Gen (CFG=1.5) | MuLa + Codec | 8-10 GB |
| Music Gen (CFG=2.0) | MuLa + Codec | 10-12 GB |
| Transcription (B=16) | Transcriptor | 4-5 GB |
| Transcription (B=32) | Transcriptor | 8-10 GB |
| Both Loaded | All models | 12-16 GB |

## 🔐 Security & Privacy

### Network Access
- Default: Local only (localhost:7860)
- To enable network access: Edit `webui.py` and set `share=True`
- To change port: Edit `server_port=7860`

### Data Privacy
- All processing happens locally on your machine
- No data is sent to external servers
- Generated files saved to `./output` directory
- Metadata saved alongside audio files

### File Access
- WebUI can only access files you explicitly upload
- Microphone requires browser permission
- Output directory is sandboxed to `./output`

## 🐛 Troubleshooting

### WebUI Won't Start

**Problem:** `ModuleNotFoundError: No module named 'gradio'`
**Solution:**
```bash
pip install gradio
```

**Problem:** Port 7860 already in use
**Solution:** Edit `webui.py` and change `server_port` to another port (e.g., 7861)

### Models Won't Load

**Problem:** "Models not loaded" error
**Solution:**
1. Check that `./ckpt` directory exists and contains models
2. Verify path in Settings tab
3. Check terminal for detailed error messages
4. Ensure models are properly downloaded

**Problem:** Out of memory when loading models
**Solution:**
1. Close other GPU applications
2. Load only needed models (generation OR transcription)
3. Use "Clear Memory" button between tasks
4. Reduce batch size for transcription

### Generation/Transcription Issues

**Problem:** Very slow generation
**Solution:**
1. Check GPU utilization in System Info tab
2. Ensure XPU optimizations are enabled
3. Try Speed preset
4. Reduce max length

**Problem:** Poor quality output
**Solution:**
1. Use Quality preset
2. Increase CFG Scale to 2.0
3. Adjust temperature (try 1.0)
4. Check input lyrics quality

**Problem:** Transcription errors
**Solution:**
1. Ensure audio file is valid
2. Try reducing batch size
3. Increase chunk length
4. Check audio is not corrupted

### GPU Issues

**Problem:** "No Intel XPU detected"
**Solution:**
1. Verify PyTorch XPU installation: `python -c "import torch; print(torch.xpu.is_available())"`
2. Check Intel GPU drivers are installed
3. Restart system if drivers were just updated

**Problem:** Low GPU utilization
**Solution:**
1. Use longer audio lengths (6-8 minutes)
2. Increase batch size for transcription
3. Check no CPU bottleneck (use `htop`)

## 📊 Performance Metrics

### Expected Performance (Intel ARC A770)

**Music Generation:**
- Loading time: 8-12s first time, <1s cached
- Generation speed: 4.5-6.5 fps
- 3-minute song: ~45-60 seconds
- 6-minute song: ~90-120 seconds

**Transcription:**
- Loading time: 4-6s
- RTF: 0.15-0.3 (5-7x real-time)
- 5-minute audio: ~60-90 seconds
- 10-minute audio: ~2.5-3 minutes

### GPU Utilization
- Music generation: 75-90%
- Transcription: 65-85%
- Idle: <5%

## 🎯 Best Practices

### For Best Results

1. **Use proper lyric structure**: Include [Verse], [Chorus], etc.
2. **Be specific with tags**: "electronic,synthwave,80s" vs just "electronic"
3. **Start with presets**: Modify after seeing baseline results
4. **Monitor VRAM**: Keep an eye on System Info tab
5. **Save good seeds**: Note seed numbers for reproducible results

### For Best Performance

1. **Use the launcher script**: Sets optimal environment variables
2. **Load models once**: Keep loaded for multiple generations
3. **Clear memory between tasks**: Switch gen ↔ transcription
4. **Use appropriate batch sizes**: Match to your VRAM
5. **Close other apps**: Maximize available resources

### For Best Workflow

1. **Plan your session**: Generation vs transcription vs both
2. **Use examples**: Learn from provided examples
3. **Experiment with presets**: Find your preferred settings
4. **Save metadata**: Keep track of what works
5. **Organize outputs**: Files auto-saved to `./output` with timestamps

## 🔄 Updates & Maintenance

### Updating WebUI
```bash
git pull
pip install -r requirements-xpu.txt --upgrade
```

### Clearing Cache
```bash
rm -rf ~/.cache/huggingface/
rm -rf ~/.cache/torch/
```

### Backup Settings
Save your favorite settings by noting:
- Preset configurations
- Model paths
- Seed numbers for good generations

## 📞 Support

### Resources
- **Documentation**: This file
- **System Validation**: `python validate_xpu_setup.py`
- **GPU Monitoring**: `xpu-smi stats -d 0`
- **Issues**: GitHub Issues

### Community
- Discord: Join HeartMuLa community
- Email: heartmula.ai@gmail.com

## 🎉 Conclusion

The HeartMuLa WebUI provides a complete, user-friendly interface for:
- ✅ Music generation from lyrics
- ✅ Lyrics transcription from audio
- ✅ Full parameter control
- ✅ Performance monitoring
- ✅ XPU optimization

**Start creating music with a beautiful web interface!** 🎵

```bash
./launch_webui.sh
```

Then open http://localhost:7860 in your browser.
