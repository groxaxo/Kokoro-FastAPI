"""FlashSR audio super-resolution service for upsampling 24kHz to 48kHz using ONNX."""

import asyncio
import os
from pathlib import Path
from typing import Optional

import librosa
import numpy as np
import soundfile as sf
import onnxruntime as ort
from huggingface_hub import hf_hub_download
from loguru import logger

from ..core.config import settings


class FlashSRService:
    """Service for audio super-resolution using FlashSR ONNX model."""

    _instance: Optional["FlashSRService"] = None
    _session: Optional[ort.InferenceSession] = None
    _init_lock = asyncio.Lock()

    def __init__(self):
        """Initialize FlashSR service."""
        self.model_path = None
        self.session = None

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
        """Initialize the FlashSR ONNX model."""
        try:
            logger.info("Initializing FlashSR service (ONNX)...")

            # Download model from HuggingFace Hub
            model_dir = Path(settings.model_dir) / "flashsr"
            model_dir.mkdir(parents=True, exist_ok=True)

            self.model_path = hf_hub_download(
                repo_id="YatharthS/FlashSR",
                filename="model.onnx",
                subfolder="onnx",
                local_dir=str(model_dir),
            )

            logger.info(f"FlashSR ONNX model downloaded to: {self.model_path}")

            # Create ONNX session
            # We can configure providers here if needed, e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']
            # For now, let's let onnxruntime decide or prioritize CPU if that's the goal for "running fast and furiously" on edge 
            # (though the prompt mentioned "instead of running on gpu", often onnx on cpu is what is meant for "edge" ease, 
            # but user also said "runs on onnx instead of running on gpu". 
            # I will include CPUExecutionProvider as the primary since the user asked to run "instead of gpu".
            providers = ["CPUExecutionProvider"]
            
            # If user wants it explicitly NOT on GPU, we stick to CPUProvider. 
            # However, if 'fast and furiously' implies speed, maybe they just meant the ONNX speedup?
            # The prompt says "runs on onnx instead of running on gpu". 
            # I will stick to the default behavior or CPU to be safe given the prompt phrasing.
            # Actually, standard ONNX usage usually defaults to available providers. 
            # But to strictly follow "instead of running on gpu", I should probably prefer CPU. 
            # Although, usually "edge devices" might have NPUs. 
            # Let's go with default providers which usually falls back to CPU if no GPU or if configured.
            # But to be safe and explicit about INTENT of "instead of gpu":
            
            self.session = ort.InferenceSession(self.model_path, providers=providers)
            
            logger.info(f"FlashSR ONNX service initialized successfully with providers: {self.session.get_providers()}")

        except Exception as e:
            logger.error(f"Failed to initialize FlashSR service: {e}")
            raise

    def upsample_audio(self, audio_data: np.ndarray, input_sample_rate: int = 24000) -> np.ndarray:
        """Upsample audio using FlashSR ONNX model, processing in chunks if necessary."""
        if not self.is_available():
            logger.warning("FlashSR model not initialized, returning original audio")
            return audio_data

        try:
            # Process in chunks of ~5 seconds to avoid sequence length limits
            # 5s @ 24kHz = 120,000 samples
            chunk_size_samples = 5 * input_sample_rate
            
            if len(audio_data) <= chunk_size_samples:
                return self._upsample_segment(audio_data, input_sample_rate)
            
            # Split into chunks
            upsampled_chunks = []
            for i in range(0, len(audio_data), chunk_size_samples):
                segment = audio_data[i : i + chunk_size_samples]
                if len(segment) < 1000: # Skip tiny segments
                    continue
                upsampled_segment = self._upsample_segment(segment, input_sample_rate)
                upsampled_chunks.append(upsampled_segment)
            
            # Concatenate all upsampled chunks
            upsampled_full_audio = np.concatenate(upsampled_chunks)

            logger.debug(
                f"Audio upsampled from {input_sample_rate}Hz to 48kHz "
                f"(shape: {audio_data.shape} -> {upsampled_full_audio.shape})"
            )
            
            return upsampled_full_audio

        except Exception as e:
            logger.error(f"FlashSR upsampling failed: {e}")
            return audio_data

    def _upsample_segment(self, audio_data: np.ndarray, input_sample_rate: int) -> np.ndarray:
        """Upsample a single segment of audio."""
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

        # logger.debug(f"FlashSR input shape (16kHz): {audio_16k.shape}")

        # Add batch dimension: (1, samples)
        lowres_wav = audio_16k[np.newaxis, :]

        # Run inference
        onnx_output = self.session.run(
            ["reconstruction"], 
            {"audio_values": lowres_wav}
        )[0]

        # Output is (1, samples), squeeze to (samples,)
        upsampled_audio = onnx_output.squeeze(0)
        
        # logger.debug(f"FlashSR output shape (48kHz): {upsampled_audio.shape}")
        
        return upsampled_audio

    def is_available(self) -> bool:
        """Check if FlashSR service is available."""
        return self.session is not None


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
