"""Kokoro ONNX backend — lightweight, GPU-auto-detecting TTS server.

When USE_ONNX=true, this module replaces the PyTorch kokoro pipeline with
kokoro-onnx + ONNX Runtime, providing an OpenAI-compatible /v1/audio/speech
endpoint with semaphore-gated concurrency and per-request tracing.

Features vs the PyTorch backend:
  - No torch dependency (ONNX Runtime only)
  - Automatic CUDA provider detection
  - Configurable ONNX threading (intra/inter-op)
  - Request-ID logging middleware
  - Request body size limit enforcement
  - Case-insensitive voice name normalization
  - Stream compatibility field (accepted, ignored)
  - Graceful executor shutdown (wait=True)
"""

import asyncio
import ctypes
import glob
import io
import logging
import os
import site
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

import numpy as np
import onnxruntime as rt
import soundfile as sf
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field

from .core.config import settings

logger = logging.getLogger("kokoro-fastapi-onnx")


def _read_int_env(name: str, default: int, min_value: int = 1) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        logger.warning("Invalid value for %s=%r. Using default %d", name, raw, default)
        return default
    if value < min_value:
        logger.warning("Value for %s=%d is too low. Clamping to %d", name, value, min_value)
        return min_value
    return value


# ── ONNX-specific env vars (read once at import) ──────────────────────────────
_ONNX_PROVIDER = os.getenv("KOKORO_PROVIDER", "auto").strip()
_ONNX_GPU_DEVICE_ID = _read_int_env("KOKORO_GPU_DEVICE_ID", settings.onnx_gpu_device_id, min_value=0)
_ONNX_WARMUP_TEXT = os.getenv("KOKORO_WARMUP_TEXT", settings.onnx_warmup_text)
_ONNX_WARMUP_VOICE = os.getenv("KOKORO_WARMUP_VOICE", settings.onnx_warmup_voice).strip()
_ONNX_EXECUTION_MODE = os.getenv("KOKORO_ORT_EXECUTION_MODE", settings.onnx_execution_mode).strip().lower()

_MAX_CONCURRENT = _read_int_env("KOKORO_MAX_CONCURRENT", settings.onnx_max_concurrent)
_INFERENCE_THREADS = _read_int_env("KOKORO_INFERENCE_THREADS", settings.onnx_inference_threads)
if _INFERENCE_THREADS < _MAX_CONCURRENT:
    _INFERENCE_THREADS = _MAX_CONCURRENT

_ONNX_INTRA_THREADS = _read_int_env("KOKORO_ONNX_INTRA_THREADS", settings.onnx_intra_threads)
_ONNX_INTER_THREADS = _read_int_env("KOKORO_ONNX_INTER_THREADS", settings.onnx_inter_threads)
_MAX_REQUEST_BYTES = _read_int_env("KOKORO_MAX_REQUEST_BYTES", settings.onnx_max_request_bytes, min_value=1024)

SUPPORTED_RESPONSE_FORMATS = frozenset({"wav", "flac", "ogg", "pcm"})

# ── Globals ───────────────────────────────────────────────────────────────────
_kokoro_onnx = None
_executor: ThreadPoolExecutor | None = None
_semaphore: asyncio.Semaphore | None = None
_voice_cache: dict[str, np.ndarray] = {}
_voice_index: dict[str, str] = {}
_voice_list: tuple[str, ...] = ()
_provider_name: str = "unknown"
_preloaded_cuda_libs: list[str] = []


def _normalize_response_format(raw_format: str) -> str:
    fmt = raw_format.strip().lower()
    if fmt in {"mp3", "opus"}:
        return "wav"
    if fmt not in SUPPORTED_RESPONSE_FORMATS:
        raise ValueError(f"Unsupported response_format: {raw_format}")
    return fmt


def _normalize_voice_name(raw_voice: str) -> str:
    return raw_voice.strip().lower()


def _nvidia_lib_dirs() -> list[str]:
    dirs: list[str] = []
    for root in site.getsitepackages():
        dirs.extend(glob.glob(os.path.join(root, "nvidia", "*", "lib")))
    return sorted(dict.fromkeys(dirs))


def _preload_cuda_libraries() -> list[str]:
    loaded: list[str] = []
    for pattern in (
        "libcudart.so*",
        "libcublas.so*",
        "libcublasLt.so*",
        "libcudnn.so*",
        "libnvrtc.so*",
    ):
        for lib_dir in _nvidia_lib_dirs():
            matches = sorted(glob.glob(os.path.join(lib_dir, pattern)), reverse=True)
            if not matches:
                continue
            try:
                ctypes.CDLL(matches[0], mode=os.RTLD_GLOBAL)
                loaded.append(matches[0])
                break
            except OSError as exc:
                logger.warning("Failed to preload %s: %s", matches[0], exc)
    return loaded


def _provider_config() -> tuple[list[str], list[dict[str, str]] | None, str]:
    available = rt.get_available_providers()
    normalized = {name.lower(): name for name in available}

    if _ONNX_PROVIDER and _ONNX_PROVIDER.lower() != "auto":
        requested = _ONNX_PROVIDER.strip().lower()
        if requested not in normalized:
            raise RuntimeError(
                f"Requested provider {_ONNX_PROVIDER!r} is unavailable. Available providers: {available}"
            )
        provider = normalized[requested]
        if provider.lower() == "cudaexecutionprovider":
            return (
                [provider, "CPUExecutionProvider"],
                [{"device_id": str(_ONNX_GPU_DEVICE_ID)}, {}],
                f"{provider}:{_ONNX_GPU_DEVICE_ID}",
            )
        return [provider], None, provider

    if "cudaexecutionprovider" in normalized:
        provider = normalized["cudaexecutionprovider"]
        return (
            [provider, "CPUExecutionProvider"],
            [{"device_id": str(_ONNX_GPU_DEVICE_ID)}, {}],
            f"{provider}:{_ONNX_GPU_DEVICE_ID}",
        )
    cpu_provider = normalized.get("cpuexecutionprovider", "CPUExecutionProvider")
    return [cpu_provider], None, cpu_provider


def _build_session(model_path: str) -> tuple[rt.InferenceSession, str]:
    opts = rt.SessionOptions()
    opts.intra_op_num_threads = _ONNX_INTRA_THREADS
    opts.inter_op_num_threads = _ONNX_INTER_THREADS
    opts.graph_optimization_level = rt.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.execution_mode = (
        rt.ExecutionMode.ORT_PARALLEL
        if _ONNX_EXECUTION_MODE == "parallel"
        else rt.ExecutionMode.ORT_SEQUENTIAL
    )
    opts.enable_mem_pattern = True
    opts.enable_cpu_mem_arena = True

    providers, provider_options, provider_name = _provider_config()
    if any(p.lower() == "cudaexecutionprovider" for p in providers):
        global _preloaded_cuda_libs
        _preloaded_cuda_libs = _preload_cuda_libraries()

    session = rt.InferenceSession(
        model_path,
        sess_options=opts,
        providers=providers,
        provider_options=provider_options,
    )
    active_provider = session.get_providers()[0] if session.get_providers() else provider_name
    return session, active_provider


def _synthesize(text: str, voice_style: np.ndarray, speed: float) -> tuple[np.ndarray, int]:
    if _kokoro_onnx is None:
        raise RuntimeError("Model not loaded")
    return _kokoro_onnx.create(text, voice=voice_style, speed=speed, lang="en-us")


def _encode(samples: np.ndarray, sr: int, fmt: str) -> tuple[bytes, str]:
    buf = io.BytesIO()
    if fmt == "pcm":
        pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
        buf.write(pcm.tobytes())
        return buf.getvalue(), "audio/pcm"
    if fmt == "flac":
        sf.write(buf, samples, sr, format="FLAC")
        return buf.getvalue(), "audio/flac"
    if fmt == "ogg":
        sf.write(buf, samples, sr, format="OGG", subtype="VORBIS")
        return buf.getvalue(), "audio/ogg"
    sf.write(buf, samples, sr, format="WAV")
    return buf.getvalue(), "audio/wav"


class _TTSRequest(BaseModel):
    model: str = "kokoro-v1.0"
    input: str = Field(..., min_length=1, description="Text to synthesize")
    voice: str = Field("af_sky", description="Voice name")
    speed: float = Field(1.0, ge=0.25, le=2.0, description="Speed factor (0.25-2.0)")
    response_format: str = Field("wav", description="wav, flac, ogg, pcm, mp3, opus")
    sample_rate: int | None = Field(None, ge=8000, le=192000, description="Override sample rate")
    stream: bool = Field(False, description="Compatibility field, always false")


def create_onnx_app() -> FastAPI:
    """Create a FastAPI app with the Kokoro-ONNX backend."""

    @asynccontextmanager
    async def _onnx_lifespan(app: FastAPI):
        global _kokoro_onnx, _executor, _semaphore, _voice_cache, _voice_index, _voice_list, _provider_name

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

        import kokoro_onnx

        logger.info("Loading Kokoro ONNX model (intra=%d inter=%d)", _ONNX_INTRA_THREADS, _ONNX_INTER_THREADS)
        sess, _provider_name = _build_session(settings.onnx_model_path)
        _kokoro_onnx = kokoro_onnx.Kokoro.from_session(sess, settings.onnx_voices_path)

        voices = _kokoro_onnx.get_voices()
        if not voices:
            raise RuntimeError("Model loaded but no voices are available")

        _voice_cache = {v: _kokoro_onnx.get_voice_style(v) for v in voices}
        _voice_list = tuple(sorted(_voice_cache))
        _voice_index = {v.lower(): v for v in _voice_cache}

        _executor = ThreadPoolExecutor(max_workers=_INFERENCE_THREADS, thread_name_prefix="kokoro-onnx")
        _semaphore = asyncio.Semaphore(_MAX_CONCURRENT)

        warmup_voice = _ONNX_WARMUP_VOICE
        if warmup_voice not in _voice_cache:
            warmup_voice = _voice_list[0]
        _synthesize(_ONNX_WARMUP_TEXT, _voice_cache[warmup_voice], 1.0)

        logger.info(
            "Kokoro ONNX ready. provider=%s voices=%d warmup_voice=%s max_concurrent=%d threads=%d execution_mode=%s",
            _provider_name, len(_voice_cache), warmup_voice, _MAX_CONCURRENT, _INFERENCE_THREADS, _ONNX_EXECUTION_MODE,
        )
        yield

        if _executor:
            _executor.shutdown(wait=True)
        _kokoro_onnx = None
        _voice_cache.clear()
        _voice_list = ()

    app = FastAPI(title="Kokoro TTS (ONNX)", version="1.0.0", lifespan=_onnx_lifespan)

    @app.middleware("http")
    async def _request_middleware(request: Request, call_next):
        request_id = uuid.uuid4().hex[:12]
        request.state.request_id = request_id
        start = time.monotonic()

        content_length = request.headers.get("content-length")
        if content_length and request.url.path == "/v1/audio/speech":
            try:
                size = int(content_length)
                if size > _MAX_REQUEST_BYTES:
                    logger.warning("[%s] request body too large %d > %d", request_id, size, _MAX_REQUEST_BYTES)
                    return JSONResponse(
                        {"error": f"Request body too large ({size} > {_MAX_REQUEST_BYTES} bytes)"},
                        status_code=413,
                    )
            except ValueError:
                pass

        logger.info("[%s] --> %s %s", request_id, request.method, request.url.path)
        response = await call_next(request)
        elapsed_ms = (time.monotonic() - start) * 1000
        logger.info("[%s] <-- %s %s %d %.1fms", request_id, request.method, request.url.path, response.status_code, elapsed_ms)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.post("/v1/audio/speech")
    async def speech(req: _TTSRequest):
        if not _kokoro_onnx or _executor is None or _semaphore is None:
            raise HTTPException(503, "Model not loaded yet")
        normalized_voice = _normalize_voice_name(req.voice)
        canonical_voice = _voice_index.get(normalized_voice)
        if canonical_voice is None:
            raise HTTPException(422, f"Unknown voice '{req.voice}'. GET /v1/voices for the list.")

        try:
            response_format = _normalize_response_format(req.response_format)
        except ValueError as exc:
            raise HTTPException(400, str(exc))

        voice_style = _voice_cache[canonical_voice]
        loop = asyncio.get_running_loop()

        try:
            async with _semaphore:
                samples, sr = await loop.run_in_executor(_executor, _synthesize, req.input, voice_style, req.speed)
        except Exception as exc:
            logger.error("Synthesis failed for voice=%s: %s", req.voice, exc, exc_info=True)
            raise HTTPException(500, str(exc))

        try:
            payload, content_type = await loop.run_in_executor(
                _executor, _encode, samples, req.sample_rate or sr, response_format,
            )
        except Exception as exc:
            logger.error("Encoding failed for format=%s: %s", req.response_format, exc, exc_info=True)
            raise HTTPException(500, str(exc))

        return Response(content=payload, media_type=content_type)

    @app.get("/v1/voices")
    async def list_voices():
        if not _kokoro_onnx:
            raise HTTPException(503, "Model not loaded yet")
        return {"object": "list", "data": [{"id": v, "name": v} for v in _voice_list]}

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "model_loaded": _kokoro_onnx is not None,
            "voices": len(_voice_cache),
            "max_concurrent": _MAX_CONCURRENT,
            "provider": _provider_name,
            "gpu_device_id": _ONNX_GPU_DEVICE_ID if "CUDAExecutionProvider" in _provider_name else None,
            "preloaded_cuda_libs": [os.path.basename(path) for path in _preloaded_cuda_libs],
            "executor_threads": _executor._max_workers if _executor else None,
            "inference_threads": _INFERENCE_THREADS,
            "execution_mode": _ONNX_EXECUTION_MODE,
        }

    # ── Web player static files ────────────────────────────────────────────────
    from fastapi.staticfiles import StaticFiles
    from pathlib import Path as _Path

    _web_dir = _Path(__file__).resolve().parent.parent.parent / "web"
    if _web_dir.is_dir():
        app.mount("/web", StaticFiles(directory=str(_web_dir), html=True), name="web")

    return app
