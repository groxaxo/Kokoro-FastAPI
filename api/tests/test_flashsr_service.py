"""Tests for FlashSR service"""

import numpy as np
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from api.src.services.flashsr_service import FlashSRService


@pytest.fixture
def sample_audio_24k():
    """Generate a simple sine wave at 24kHz for testing"""
    sample_rate = 24000
    duration = 0.5  # 500ms
    t = np.linspace(0, duration, int(sample_rate * duration))
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio, sample_rate


@pytest.fixture
def mock_flashsr_model():
    """Mock FlashSR model"""
    mock_model = MagicMock()
    mock_model.run = MagicMock(return_value=np.random.randn(36000).astype(np.float32))
    return mock_model


@pytest.mark.asyncio
async def test_flashsr_service_initialization(mock_flashsr_model):
    """Test FlashSR service initialization"""
    with patch(
        "api.src.services.flashsr_service.hf_hub_download"
    ) as mock_download, patch(
        "api.src.services.flashsr_service.settings"
    ) as mock_settings, patch(
        "api.src.services.flashsr_service.torch"
    ) as mock_torch:
        # Setup mocks
        mock_settings.get_device.return_value = "cpu"
        mock_settings.model_dir = "/tmp/models"
        mock_download.return_value = "/tmp/models/flashsr/upsampler.pth"
        mock_torch.device.return_value = MagicMock()

        # Create service
        service = FlashSRService()

        # Mock FASR class
        with patch("api.src.services.flashsr_service.FASR") as mock_fasr:
            mock_fasr.return_value = MagicMock(model=mock_flashsr_model)
            await service.initialize()

            # Verify initialization
            assert service.model is not None
            assert service.device is not None
            assert service.is_available() is True
            mock_download.assert_called_once()


@pytest.mark.asyncio
async def test_flashsr_upsample_audio(sample_audio_24k, mock_flashsr_model):
    """Test audio upsampling functionality"""
    audio, sample_rate = sample_audio_24k

    with patch(
        "api.src.services.flashsr_service.hf_hub_download"
    ) as mock_download, patch(
        "api.src.services.flashsr_service.settings"
    ) as mock_settings, patch(
        "api.src.services.flashsr_service.torch"
    ) as mock_torch:
        # Setup mocks
        mock_settings.get_device.return_value = "cpu"
        mock_settings.model_dir = "/tmp/models"
        mock_download.return_value = "/tmp/models/flashsr/upsampler.pth"

        # Mock torch device and tensor operations
        mock_device = MagicMock()
        mock_device.type = "cpu"
        mock_torch.device.return_value = mock_device

        mock_tensor = MagicMock()
        mock_tensor.half.return_value = mock_tensor
        mock_tensor.to.return_value = mock_tensor
        mock_tensor.cpu.return_value = mock_tensor
        mock_tensor.float.return_value = mock_tensor
        mock_tensor.numpy.return_value = np.random.randn(36000).astype(np.float32)
        mock_torch.from_numpy.return_value = mock_tensor

        # Create service and initialize
        service = FlashSRService()

        with patch("api.src.services.flashsr_service.FASR") as mock_fasr, patch(
            "api.src.services.flashsr_service.librosa.resample"
        ) as mock_resample:
            # Setup FASR mock
            mock_fasr_instance = MagicMock()
            mock_fasr_instance.model = mock_flashsr_model
            mock_fasr_instance.run = MagicMock(return_value=mock_tensor)
            mock_fasr.return_value = mock_fasr_instance

            # Mock resampling
            mock_resample.return_value = np.random.randn(8000).astype(np.float32)

            await service.initialize()

            # Test upsampling
            # Convert int16 to float32
            audio_int16 = (audio * 32768.0).astype(np.int16)
            upsampled = service.upsample_audio(audio_int16, sample_rate)

            # Verify output
            assert isinstance(upsampled, np.ndarray)
            assert upsampled.dtype == np.float32
            # Upsampled audio should be roughly 3x longer (16kHz -> 48kHz)
            # But since we're mocking, just check it's not empty
            assert len(upsampled) > 0


@pytest.mark.asyncio
async def test_flashsr_error_handling(sample_audio_24k):
    """Test FlashSR error handling when model fails"""
    audio, sample_rate = sample_audio_24k

    # Create service without initialization
    service = FlashSRService()

    # Test upsampling without initialized model
    audio_int16 = (audio * 32768.0).astype(np.int16)
    result = service.upsample_audio(audio_int16, sample_rate)

    # Should return original audio on error
    assert np.array_equal(result, audio_int16)


@pytest.mark.asyncio
async def test_flashsr_service_availability():
    """Test FlashSR service availability check"""
    service = FlashSRService()

    # Should not be available before initialization
    assert service.is_available() is False

    # Mock initialization
    service.model = MagicMock()

    # Should be available after initialization
    assert service.is_available() is True
