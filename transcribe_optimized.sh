#!/bin/bash

# XPU Optimized Transcription Script for HeartMuLa
# PyTorch 2.10+ Native XPU Support

echo "=========================================="
echo "HeartTranscriptor XPU Optimizer"
echo "  PyTorch Native XPU Support  "
echo "=========================================="

# Level Zero Optimizations
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
export SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=1
export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1

# Memory Management
export SYCL_PI_LEVEL_ZERO_USE_RELAXED_ALLOCATION_LIMITS=1

# Thread Settings
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "✓ XPU optimizations enabled"
echo "✓ Native PyTorch XPU: ON"
echo "=========================================="
echo ""

# Check GPU info
if command -v xpu-smi &> /dev/null; then
    echo "GPU Information:"
    xpu-smi discovery
    echo ""
fi

# Run transcription
echo "Starting HeartTranscriptor with optimized settings..."
echo ""

# Change the filename below to match your generated file
SONG_PATH="./output/song.mp3"

python xpu_transcribe.py \
    --music_path "$SONG_PATH" \
    --model_path "./ckpt" \
    --compile_mode "on"

read -p "Press enter to close..."