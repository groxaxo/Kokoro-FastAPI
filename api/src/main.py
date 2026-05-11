import uvicorn
"""Application entrypoint for the ONNX Kokoro FastAPI server."""

from .core.config import settings
from .onnx_server import create_onnx_app

app = create_onnx_app()


if __name__ == "__main__":
    uvicorn.run("api.src.main:app", host=settings.host, port=settings.port, reload=True)
