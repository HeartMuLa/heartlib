#!/usr/bin/env bash
# Launcher for the standalone LLM Gradio UI (matches user's requested filename)
# Usage: ./lauch_webui_llm.sh

set -euo pipefail
PYTHON=${PYTHON:-python3}
$PYTHON webui_llm.py "$@"