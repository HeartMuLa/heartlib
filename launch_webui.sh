#!/bin/bash

# HeartMuLa WebUI Launcher
# Optimized for Intel XPU

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║              🎵 HeartMuLa WebUI - Intel XPU Edition          ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""

# Set XPU optimizations
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=1
export SYCL_CACHE_PERSISTENT=1
export SYCL_PI_LEVEL_ZERO_USE_COPY_ENGINE=1
export SYCL_PI_LEVEL_ZERO_DEVICE_SCOPE_EVENTS=1
export SYCL_PI_LEVEL_ZERO_USE_RELAXED_ALLOCATION_LIMITS=1
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "✓ XPU optimizations enabled"
echo "✓ Environment configured"
echo ""

# Check for gradio
if ! python -c "import gradio" 2>/dev/null; then
    echo "⚠️  Gradio not found. Installing..."
    pip install gradio
    echo ""
fi

# Check GPU
if command -v xpu-smi &> /dev/null; then
    echo "🎮 GPU Information:"
    xpu-smi discovery
    echo ""
fi

# Launch WebUI
echo "🚀 Launching WebUI..."
echo ""
echo "📍 Local URL: http://localhost:7860"
echo "🌐 Network URL will be displayed below"
echo ""
echo "Press Ctrl+C to stop the server"
echo "══════════════════════════════════════════════════════════════"
echo ""

python webui.py "$@"
