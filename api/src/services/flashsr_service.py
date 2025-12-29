"""FlashSR audio super-resolution service for upsampling 24kHz to 48kHz."""

import asyncio
import os
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from huggingface_hub import hf_hub_download
from loguru import logger

from ..core.config import settings


class FlashSRService:
    """Service for audio super-resolution using FlashSR model."""

    _instance: Optional["FlashSRService"] = None
    _model = None
    _device = None
    _init_lock = asyncio.Lock()

    def __init__(self):
        """Initialize FlashSR service."""
        self.model_path = None
        self.model = None
        self.device = None

    @classmethod
    async def get_instance(cls) -> "FlashSRService":
        """Get singleton instance of FlashSR service (thread-safe)."""
        if cls._instance is None:
            async with cls._init_lock:
                # Double-check after acquiring lock
                if cls._instance is None:
                    cls._instance = cls()
                    await cls._instance.initialize()
        return cls._instance

    async def initialize(self):
        """Initialize the FlashSR model."""
        try:
            # Set device with validation
            device_str = settings.get_device()
            if device_str not in ["cpu", "cuda", "mps"]:
                logger.warning(f"Invalid device '{device_str}', falling back to CPU")
                device_str = "cpu"
            
            self.device = torch.device(device_str)
            logger.info(f"Initializing FlashSR service on device: {self.device}")

            # Download model from HuggingFace Hub
            model_dir = Path(settings.model_dir) / "flashsr"
            model_dir.mkdir(parents=True, exist_ok=True)

            self.model_path = hf_hub_download(
                repo_id="YatharthS/FlashSR",
                filename="upsampler.pth",
                local_dir=str(model_dir),
            )

            logger.info(f"FlashSR model downloaded to: {self.model_path}")

            # Import FlashSR model class
            from .flashsr import FASR

            # Load model
            self.model = FASR(self.model_path)
            self.model.model = self.model.model.to(self.device)

            # Use half precision on GPU for faster inference
            if self.device.type == "cuda":
                self.model.model = self.model.model.half()
                logger.info("FlashSR model loaded with half precision (FP16)")
            else:
                logger.info("FlashSR model loaded with full precision (FP32)")

            self.model.model.eval()
            logger.info("FlashSR service initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize FlashSR service: {e}")
            raise

    def upsample_audio(
        self, audio_data: np.ndarray, input_sample_rate: int = 24000
    ) -> np.ndarray:
        """
        Upsample audio from 24kHz to 48kHz using FlashSR.

        Args:
            audio_data: Input audio data as numpy array (int16 or float32)
            input_sample_rate: Sample rate of input audio (default: 24000)

        Returns:
            Upsampled audio data at 48kHz as numpy array (float32)
        """
        if self.model is None:
            logger.warning("FlashSR model not initialized, returning original audio")
            return audio_data

        try:
            # Convert to float32 if needed
            if audio_data.dtype == np.int16:
                audio_float = audio_data.astype(np.float32) / 32767.0
            else:
                audio_float = audio_data.astype(np.float32)

            # FlashSR expects 16kHz input, so we need to resample 24kHz -> 16kHz first
            if input_sample_rate != 16000:
                audio_16k = librosa.resample(
                    audio_float, orig_sr=input_sample_rate, target_sr=16000
                )
            else:
                audio_16k = audio_float

            # Convert to torch tensor
            audio_tensor = torch.from_numpy(audio_16k).unsqueeze(0)

            # Use half precision on GPU
            if self.device.type == "cuda":
                audio_tensor = audio_tensor.half()

            audio_tensor = audio_tensor.to(self.device)

            # Run super-resolution
            with torch.no_grad():
                upsampled_audio = self.model.run(audio_tensor)

            # Convert back to numpy
            if isinstance(upsampled_audio, torch.Tensor):
                upsampled_audio = upsampled_audio.cpu().float().numpy()

            # Ensure correct shape (remove batch dimension if present)
            if upsampled_audio.ndim > 1:
                upsampled_audio = upsampled_audio.squeeze()

            logger.debug(
                f"Audio upsampled from {input_sample_rate}Hz to 48kHz "
                f"(shape: {audio_data.shape} -> {upsampled_audio.shape})"
            )

            return upsampled_audio

        except Exception as e:
            logger.error(f"Error during audio upsampling: {e}")
            # Return original audio on error
            return audio_data

    def is_available(self) -> bool:
        """Check if FlashSR service is available."""
        return self.model is not None


# Global instance accessor
_flashsr_service: Optional[FlashSRService] = None
_service_lock = asyncio.Lock()


async def get_flashsr_service() -> Optional[FlashSRService]:
    """Get the global FlashSR service instance (thread-safe)."""
    global _flashsr_service
    if _flashsr_service is None:
        async with _service_lock:
            # Double-check after acquiring lock
            if _flashsr_service is None:
                try:
                    _flashsr_service = await FlashSRService.get_instance()
                except Exception as e:
                    logger.warning(f"FlashSR service not available: {e}")
                    return None
    return _flashsr_service
