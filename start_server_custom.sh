#!/bin/bash
# Get project root directory
PROJECT_ROOT=$(pwd)

# Set environment variables
export USE_GPU=false
export USE_ONNX=false
export PYTHONPATH=$PROJECT_ROOT:$PROJECT_ROOT/api
export MODEL_DIR=$PROJECT_ROOT/api/src/models
export VOICES_DIR=$PROJECT_ROOT/api/src/voices/v1_0
export WEB_PLAYER_PATH=$PROJECT_ROOT/web
# Set the espeak-ng data path to your location
export ESPEAK_DATA_PATH=/usr/lib/x86_64-linux-gnu/espeak-ng-data
export ENABLE_FLASHSR=true

source .venv/bin/activate

# Create output directories if they don't exist
mkdir -p api/src/models/flashsr

# Start the server
uvicorn api.src.main:app --host 0.0.0.0 --port 8880
