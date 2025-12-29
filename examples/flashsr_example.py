"""
Example: FlashSR Audio Super-Resolution
Demonstrates how FlashSR automatically enhances audio quality by upsampling from 24kHz to 48kHz
"""

import requests

# Test with and without FlashSR to compare quality
base_url = "http://localhost:8880"

text_to_generate = "Hello! This is a test of the FlashSR audio super-resolution system."

# Generate audio with FlashSR enabled (default)
print("Generating audio with FlashSR enabled (48kHz)...")
response = requests.post(
    f"{base_url}/v1/audio/speech",
    json={
        "model": "kokoro",
        "input": text_to_generate,
        "voice": "af_bella",
        "response_format": "wav",
    },
)

with open("output_with_flashsr_48khz.wav", "wb") as f:
    f.write(response.content)
print(f"✓ Saved to output_with_flashsr_48khz.wav ({len(response.content)} bytes)")

print("\nFlashSR is enabled by default!")
print("To disable FlashSR and get 24kHz output, set the environment variable:")
print("  export ENABLE_FLASHSR=false")
print("\nFlashSR Features:")
print("  • Ultra-fast upsampling (200-400x realtime)")
print("  • Lightweight model (~2MB)")
print("  • Automatic 24kHz → 48kHz conversion")
print("  • Crystal-clear audio quality")
print("\nThe model automatically:")
print("  1. Takes Kokoro's 24kHz output")
print("  2. Resamples to 16kHz (FlashSR input)")
print("  3. Applies super-resolution")
print("  4. Outputs 48kHz audio")
print("\nNo client-side changes needed - it just works!")
