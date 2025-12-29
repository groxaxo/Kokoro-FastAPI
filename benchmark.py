
import time
import requests
import json
import soundfile as sf
import io
import numpy as np

API_URL = "http://localhost:8880/v1/audio/speech"
TEXT = "The quick brown fox jumps over the lazy dog. This is a speed test for Kokoro TTS with FlashSR enabled."

def run_benchmark(iterations=3):
    print(f"Running benchmark with {iterations} iterations...")
    latencies = []
    
    for i in range(iterations):
        start_time = time.time()
        response = requests.post(
            API_URL,
            json={
                "model": "kokoro",
                "input": TEXT,
                "voice": "af_heart",
                "response_format": "wav",
                "speed": 1.0
            }
        )
        end_time = time.time()
        
        if response.status_code == 200:
            latency = end_time - start_time
            latencies.append(latency)
            
            # Verify audio
            audio_data = io.BytesIO(response.content)
            data, samplerate = sf.read(audio_data)
            duration = len(data) / samplerate
            
            print(f"Iteration {i+1}: Latency={latency:.3f}s, Audio Duration={duration:.3f}s, Sample Rate={samplerate}Hz")
            
            if samplerate != 48000:
                print("WARNING: Sample rate is not 48kHz! FlashSR might not be working.")
        else:
            print(f"Iteration {i+1} failed: {response.status_code} - {response.text}")

    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"\nAverage Latency: {avg_latency:.3f}s")
        print(f"Real-time Factor (RTF): {avg_latency / duration:.3f} (Lower is better)")

if __name__ == "__main__":
    # Wait for server to fully initialize
    time.sleep(2)
    run_benchmark()
