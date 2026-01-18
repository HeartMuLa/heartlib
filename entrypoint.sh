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
    echo "Checkpoints found in $CKPT_DIR, skipping download."
else
    echo "Checkpoints not found or incomplete. Starting automatic download from ModelScope..."
    
    # 使用 python 脚本调用 modelscope 下载，避免命令行工具可能未加入 PATH 的问题
    python3 -c "
from modelscope import snapshot_download
import os

ckpt_dir = '$CKPT_DIR'
print('Downloading HeartMuLaGen config and tokenizer...')
snapshot_download('HeartMuLa/HeartMuLaGen', local_dir=ckpt_dir)

print('Downloading HeartMuLa-oss-3B...')
snapshot_download('HeartMuLa/HeartMuLa-oss-3B', local_dir=os.path.join(ckpt_dir, 'HeartMuLa-oss-3B'))

print('Downloading HeartCodec-oss...')
snapshot_download('HeartMuLa/HeartCodec-oss', local_dir=os.path.join(ckpt_dir, 'HeartCodec-oss'))
"
    echo "Download completed successfully."
fi

exec "$@"
