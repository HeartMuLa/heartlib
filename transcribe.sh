#!/bin/bash
# XPU Lyrics Transcriber Shortcut

# Ensure we are in the script directory
cd "$(dirname "$0")"

# Example: Transcribe the song we just generated
# Change the filename below to match your generated file
SONG_PATH="./output/song.mp3"

python xpu_transcribe.py \
    --music_path "$SONG_PATH" \
    --model_path "./ckpt" \
    --compile_mode "off"

read -p "Press enter to close..."