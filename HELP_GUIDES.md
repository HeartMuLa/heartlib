# HELP_GUIDES — Help, Guides & Troubleshooting (consolidated)

Purpose: single help document that combines quick reference, user guides and troubleshooting.

## Quick commands (one-liners)
- Validate setup: `python validate_xpu_setup.py`  
- Generate (optimized): `./run_optimized.sh --lyrics ./inputs/lyrics/default.txt`  
- Transcribe (optimized): `./transcribe_optimized.sh --music_path ./audio/song.mp3`  
- Monitor GPU: `watch -n 0.5 'xpu-smi dump -m 0,1,5'` or `intel_gpu_top`

## Common flags explained
- `--cfg_scale` — 1.0 = speed, >1 = stronger guidance/quality  
- `--max_audio_length_ms` — longer => better GPU utilization  
- `--batch_size` (transcription) — larger = faster if VRAM permits  
- `--chunk_length_s` (transcription) — longer chunks = better accuracy, more memory

## Typical user workflows
1. Quick test: `python xpu_music_gen.py --max_audio_length_ms 30000`  
2. Full run with launcher: `./run_optimized.sh --lyrics my_lyrics.txt --save_path out.mp3`  
3. Verify with transcription: `./transcribe_optimized.sh --music_path out.mp3 --save_path verified.txt`

## Troubleshooting (common problems & solutions)
- XPU not detected: verify PyTorch 2.10+ with XPU support and Level Zero drivers.  
- Low GPU utilization: increase `--max_audio_length_ms`, check for CPU bottleneck (htop).  
- Out of memory (OOM): reduce `--batch_size` or `--max_audio_length_ms`; set `--cfg_scale 1.0`.  
- Slow checkpoint loading: ensure `transformers>=4.57.0` and SSD storage; use `low_cpu_mem_usage=True`.

## Transcription-specific tips
- Defaults: `--batch_size 16`, `--chunk_length_s 30`  
- For speed: increase batch_size and lower chunk_length  
- For accuracy: increase chunk_length and use cleaner audio

## When to open an issue
Include: output of `python validate_xpu_setup.py`, relevant command used, and `xpu-smi`/`intel_gpu_top` output.

## Where to find more detail
- Performance & tuning: `PERFORMANCE_OPTIMIZATION.md`  
- Full original doc (backup): `COMPLETE_OPTIMIZATION_SUMMARY.md`

---
If you want I can add these links to `README.md` or remove the old split docs now.