from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # API Settings
    api_title: str = "Kokoro TTS API"
    api_description: str = "API for text-to-speech generation using Kokoro"
    api_version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8880

    # Application Settings
    output_dir: str = "output"
    output_dir_size_limit_mb: float = 500.0  # Maximum size of output directory in MB
    default_voice: str = "af_heart"
    default_voice_code: str | None = (
        None  # If set, overrides the first letter of voice name, though api call param still takes precedence
    )
    use_gpu: bool = False  # Whether to request GPU acceleration if available

    # ── ONNX backend settings ─────────────────────────────────────────────────
    onnx_model_path: str = "kokoro-v1.0.onnx"       # Path to ONNX model file
    onnx_voices_path: str = "voices-v1.0.bin"       # Path to voices file
    onnx_gpu_device_id: int = 0                     # GPU device ID for CUDA provider
    onnx_max_concurrent: int = 4                    # Max simultaneous inference calls
    onnx_inference_threads: int = 4                 # Thread pool size
    onnx_intra_threads: int = 5                     # ONNX intra-op threads
    onnx_inter_threads: int = 2                     # ONNX inter-op threads
    onnx_warmup_text: str = "Kokoro is ready."      # Warmup text
    onnx_warmup_voice: str = "af_heart"             # Warmup voice name
    onnx_execution_mode: str = "sequential"         # ONNX execution mode (sequential|parallel)
    onnx_max_request_bytes: int = 1_048_576         # Max request body size (default 1 MiB)
    sample_rate: int = 24000

    # Web Player Settings
    enable_web_player: bool = True  # Whether to serve the web player UI
    web_player_path: str = "web"  # Path to web player static files

    class Config:
        env_file = ".env"


settings = Settings()
