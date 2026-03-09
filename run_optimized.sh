#!/bin/bash

# XPU Performance Optimization Script for HeartMuLa
# PyTorch 2.10+ has native XPU support

echo "=========================================="
echo "HeartTranscriptor XPU Optimizer"
echo "  PyTorch Native XPU Support  "
echo "=========================================="

# Level Zero Optimizations
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
export SYCL_PI_LEVEL_ZERO_BATCH_SIZE=0
export SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=1
export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1

# Memory Management
export SYCL_PI_LEVEL_ZERO_USE_RELAXED_ALLOCATION_LIMITS=1

# Native PyTorch optimizations
export TORCH_USE_CUDA_DSA=1
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Thread Settings
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "✓ Level Zero optimizations enabled"
echo "✓ Immediate command lists: ON"
echo "✓ Persistent cache: ON"
echo "✓ Copy engine: ON"
echo "✓ Native PyTorch XPU: ON"
echo "=========================================="
echo ""

# Check GPU info
if command -v xpu-smi &> /dev/null; then
    echo "GPU Information:"
    xpu-smi discovery
    echo ""
fi

# Run the music generation
echo "Starting HeartMuLa with optimized settings..."
echo ""

# Ensure we are in the script directory
cd "$(dirname "$0")"

python xpu_music_gen.py \
    --model_path "./ckpt" \
    --lyrics "./inputs/lyrics/lyrics.txt" \
    --tags "./inputs/tags/tags.txt" \
    --save_path "./output/song.mp3" \
    --max_audio_length_ms 180000