#!/bin/bash
set -e

CKPT_DIR="/app/ckpt"
mkdir -p "$CKPT_DIR"

check_models() {
    if [ -f "$CKPT_DIR/gen_config.json" ] && \
       [ -f "$CKPT_DIR/tokenizer.json" ] && \
       [ -d "$CKPT_DIR/HeartCodec-oss" ] && \
       [ -d "$CKPT_DIR/HeartMuLa-oss-3B" ]; then
        return 0
    else
        return 1
    fi
}

if check_models; then
    echo "=========================================="
    echo "✓ Checkpoints found in $CKPT_DIR"
    echo "✓ Skipping download"
    echo "=========================================="
else
    echo "=========================================="
    echo "⬇  Starting model download from ModelScope"
    echo "=========================================="
    echo ""

    python3 -u -c "
from modelscope import snapshot_download
import os
import sys

ckpt_dir = '$CKPT_DIR'

print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
print('📦 [1/3] Downloading HeartMuLaGen config and tokenizer...')
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
sys.stdout.flush()
snapshot_download('HeartMuLa/HeartMuLaGen', local_dir=ckpt_dir)
print('✓ HeartMuLaGen download completed')
print('')

print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
print('📦 [2/3] Downloading HeartMuLa-oss-3B model...')
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
sys.stdout.flush()
snapshot_download('HeartMuLa/HeartMuLa-oss-3B', local_dir=os.path.join(ckpt_dir, 'HeartMuLa-oss-3B'))
print('✓ HeartMuLa-oss-3B download completed')
print('')

print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
print('📦 [3/3] Downloading HeartCodec-oss model...')
print('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━')
sys.stdout.flush()
snapshot_download('HeartMuLa/HeartCodec-oss', local_dir=os.path.join(ckpt_dir, 'HeartCodec-oss'))
print('✓ HeartCodec-oss download completed')
print('')
"
    echo ""
    echo "=========================================="
    echo "✓ All models downloaded successfully!"
    echo "=========================================="
    echo ""
fi

echo "=========================================="
echo "🚀 Starting application..."
echo "=========================================="
exec "$@"
