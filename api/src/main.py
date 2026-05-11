"""
FastAPI OpenAI Compatible API
"""

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings

if not settings.use_onnx:
    from loguru import logger
else:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logger = logging.getLogger("kokoro-fastapi")


def setup_logger():
    """Configure loguru logger with custom formatting"""
    valid_levels = ["TRACE", "DEBUG", "INFO", "SUCCESS", "WARNING", "ERROR", "CRITICAL"]
    level = os.getenv("API_LOG_LEVEL", "DEBUG").upper()
    if level not in valid_levels:
        level = "DEBUG"
    print(f"Global API loguru logger level: {level}")
    config = {
        "handlers": [
            {
                "sink": sys.stdout,
                "format": "<fg #2E8B57>{time:hh:mm:ss A}</fg #2E8B57> | "
                "{level: <8} | "
                "<fg #4169E1>{module}:{line}</fg #4169E1> | "
                "{message}",
                "colorize": True,
                "level": level,
            },
        ],
    }
    logger.remove()
    logger.configure(**config)
    logger.level("ERROR", color="<red>")


# Configure logger
if not settings.use_onnx:
    setup_logger()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for model initialization.

    When USE_ONNX=true the ONNX backend handles its own startup — this
    lifespan is for the PyTorch kokoro pipeline only.
    """
    if settings.use_onnx:
        yield  # ONNX app manages its own lifespan
        return

    from .inference.model_manager import get_manager
    from .inference.voice_manager import get_manager as get_voice_manager
    from .services.temp_manager import cleanup_temp_files
    from .services.flashsr_service import get_flashsr_service

    await cleanup_temp_files()

    logger.info("Loading TTS model and voice packs...")

    try:
        model_manager = await get_manager()
        voice_manager = await get_voice_manager()
        device, model, voicepack_count = await model_manager.initialize_with_warmup(voice_manager)

        flashsr_status = "disabled"
        if settings.enable_flashsr:
            try:
                logger.info("Initializing FlashSR audio super-resolution...")
                flashsr_service = await get_flashsr_service()
                if flashsr_service and flashsr_service.is_available():
                    flashsr_status = "enabled (24kHz -> 48kHz)"
                    logger.info("FlashSR initialized successfully")
                else:
                    flashsr_status = "initialization failed"
                    logger.warning("FlashSR initialization failed")
            except Exception as e:
                flashsr_status = f"error: {str(e)}"
                logger.error(f"Failed to initialize FlashSR: {e}")

    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise

    boundary = "░" * 2 * 12
    startup_msg = f"""

{boundary}

    ╔═╗┌─┐┌─┐┌┬┐
    ╠╣ ├─┤└─┐ │ 
    ╚  ┴ ┴└─┘ ┴
    ╦╔═┌─┐┬┌─┌─┐
    ╠╩╗│ │├┴┐│ │
    ╩ ╩└─┘┴ ┴└─┘

{boundary}
                """
    startup_msg += f"\nModel warmed up on {device}: {model}"
    if device == "mps":
        startup_msg += "\nUsing Apple Metal Performance Shaders (MPS)"
    elif device == "cuda":
        import torch
        startup_msg += f"\nCUDA: {torch.cuda.is_available()}"
    else:
        startup_msg += "\nRunning on CPU"
    startup_msg += f"\n{voicepack_count} voice packs loaded"
    startup_msg += f"\nFlashSR Audio Super-Resolution: {flashsr_status}"

    if settings.enable_web_player:
        startup_msg += f"\n\nBeta Web Player: http://{settings.host}:{settings.port}/web/"
        startup_msg += f"\nor http://localhost:{settings.port}/web/"
    else:
        startup_msg += "\n\nWeb Player: disabled"

    startup_msg += f"\n{boundary}\n"
    logger.info(startup_msg)

    yield


# Initialize FastAPI app — switch backends based on USE_ONNX
if settings.use_onnx:
    from .onnx_server import create_onnx_app
    app = create_onnx_app()
    logger.info("Backend: kokoro-onnx (ONNX Runtime)")
else:
    import torch

    from .routers.debug import router as debug_router
    from .routers.development import router as dev_router
    from .routers.openai_compatible import router as openai_router
    from .routers.web_player import router as web_router

    app = FastAPI(
        title=settings.api_title,
        description=settings.api_description,
        version=settings.api_version,
        lifespan=lifespan,
        openapi_url="/openapi.json",
    )

    # Add CORS middleware if enabled
    if settings.cors_enabled:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Include routers
    app.include_router(openai_router, prefix="/v1")
    app.include_router(dev_router)
    app.include_router(debug_router)
    if settings.enable_web_player:
        app.include_router(web_router, prefix="/web")

    logger.info("Backend: kokoro (PyTorch)")


# Health check endpoint (only for PyTorch backend; ONNX has its own)
if not settings.use_onnx:

    @app.get("/health")
    async def health_check():
        """Health check endpoint"""
        return {"status": "healthy"}


    @app.get("/v1/test")
    async def test_endpoint():
        """Test endpoint to verify routing"""
        return {"status": "ok"}


if __name__ == "__main__":
    uvicorn.run("api.src.main:app", host=settings.host, port=settings.port, reload=True)
