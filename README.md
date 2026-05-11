# Kokoro FastAPI

<p align="center">
  <img src="https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/logo.png" alt="Kokoro FastAPI" width="300">
</p>

<p align="center">
  <a href="https://huggingface.co/hexgrad/Kokoro-82M"><img src="https://img.shields.io/badge/Model-Kokoro--82M-blue?logo=huggingface" alt="Kokoro-82M"></a>
  <img src="https://img.shields.io/badge/Runtime-ONNX-green" alt="ONNX Runtime">
  <img src="https://img.shields.io/badge/Voices-54-purple" alt="54 Voices">
  <a href="https://github.com/groxaxo/Kokoro-FastAPI/blob/master/LICENSE"><img src="https://img.shields.io/badge/License-Apache%202.0-yellow" alt="License"></a>
</p>

FastAPI server for Kokoro text-to-speech using `kokoro-onnx` and ONNX Runtime.

## Overview

- OpenAI-compatible `POST /v1/audio/speech`
- `GET /v1/voices` for the bundled 54-voice palette
- `GET /health` with provider and threading details
- Built-in web player at `/web/`
- CPU and CUDA execution through ONNX Runtime

## Quick Start

```bash
git clone https://github.com/groxaxo/Kokoro-FastAPI.git
cd Kokoro-FastAPI
pip install -e .
./start-onnx.sh
```

Place these files in the project root before starting:

- `kokoro-v1.0.onnx`
- `voices-v1.0.bin`

Default server URL:

- API: `http://localhost:8880`
- Web player: `http://localhost:8880/web/`
- Swagger: `http://localhost:8880/docs`

## Environment Variables

| Variable | Default | Info |
|:---|:---:|:---|
| `PORT` | `8880` | Server port |
| `KOKORO_PROVIDER` | `auto` | `auto`, `CPUExecutionProvider`, or `CUDAExecutionProvider` |
| `KOKORO_GPU_DEVICE_ID` | `0` | CUDA device index |
| `KOKORO_MAX_CONCURRENT` | `4` | Max simultaneous inference calls |
| `KOKORO_INFERENCE_THREADS` | `4` | Executor worker count |
| `KOKORO_ONNX_INTRA_THREADS` | `5` | ONNX intra-op threads |
| `KOKORO_ONNX_INTER_THREADS` | `2` | ONNX inter-op threads |
| `KOKORO_ORT_EXECUTION_MODE` | `sequential` | `sequential` or `parallel` |
| `KOKORO_MAX_REQUEST_BYTES` | `1048576` | Max request body size |
| `KOKORO_WARMUP_TEXT` | `Kokoro is ready.` | Startup warmup text |
| `KOKORO_WARMUP_VOICE` | `af_heart` | Startup warmup voice |
| `ONNX_MODEL_PATH` | `kokoro-v1.0.onnx` | Path to ONNX model file |
| `ONNX_VOICES_PATH` | `voices-v1.0.bin` | Path to voices file |

See `.env.example` for a ready-to-edit template.

## API Example

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8880/v1", api_key="not-needed")

response = client.audio.speech.create(
    model="kokoro",
    voice="af_heart",
    input="Hello from Kokoro FastAPI!",
    response_format="wav",
)

response.stream_to_file("output.wav")
```

## Endpoints

| Method | Path | Description |
|:---|:---|:---|
| `POST` | `/v1/audio/speech` | Generate audio from text |
| `GET` | `/v1/voices` | List available voices |
| `GET` | `/health` | Runtime health and provider details |

## Voice Palette

Included voices cover multiple languages and accents.

- American Female: `af_heart`, `af_bella`, `af_jessica`, `af_alloy`
- American Male: `am_adam`, `am_michael`
- British: `bf_emma`, `bf_isabella`, `bm_george`
- Spanish: `ef_dora`, `em_alex`
- Additional Japanese, Chinese, Hindi, Italian, Portuguese, and more

## Notes

- `mp3` and `opus` requests are accepted for compatibility and returned as WAV.
- The server normalizes voice names case-insensitively.
- The web player uses the same `POST /v1/audio/speech` endpoint as API clients.

## Credits

- Model: [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M)
- ONNX runtime backend: [kokoro-onnx](https://github.com/thewh1teagle/kokoro-onnx)
- License: Apache 2.0
