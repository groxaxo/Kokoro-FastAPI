#!/bin/bash

set -e

# Get project root directory
PROJECT_ROOT=$(pwd)

# Set environment variables
export USE_GPU=false

# ── ONNX backend settings ─────────────────────────────────────────────────────
export KOKORO_PROVIDER=CPUExecutionProvider
export KOKORO_ORT_EXECUTION_MODE=sequential
export KOKORO_MAX_CONCURRENT=4
export KOKORO_INFERENCE_THREADS=4
export KOKORO_ONNX_INTRA_THREADS=5
export KOKORO_ONNX_INTER_THREADS=2

export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/api
# export ESPEAK_DATA_PATH=/usr/lib/x86_64-linux-gnu/espeak-ng-data

# Install dependencies then start
uv pip install -e .
uv run --no-sync uvicorn api.src.main:app --host 0.0.0.0 --port 8880
