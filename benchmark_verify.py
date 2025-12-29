
import asyncio
import time
import requests
import json
import soundfile as sf
import io
import os
import sys

# We'll use requests for the benchmark for simplicity
URL = "http://localhost:8880/v1/audio/speech"

def run_benchmark():
    print("Running Benchmark & Verification...")
    
    start_time = time.time()
    
    # 1. Verify default is Opus/48kHz
    # We will explicitely NOT send response_format to test the default we changed
    payload = {
        "model": "kokoro",
        "input": "This is a test of the automatic 48 kilohertz upsampling using FlashSR on CPU. It should return an Opus file.",
        "voice": "af_heart"
    }
    
    print(f"Sending request to {URL}...")
    try:
        response = requests.post(URL, json=payload)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Is it running?")
        sys.exit(1)
        
    duration = time.time() - start_time
    print(f"Request completed in {duration:.2f} seconds.")
    
    # Check headers or content type if possible, but easier to inspect bytes
    content = response.content
    print(f"Received {len(content)} bytes.")

    # Verify Opus
    # Opus usually starts with 'OggS' container or similar signatures if in Ogg container
    if content.startswith(b'OggS'):
        print("PASS: Output appears to be Ogg/Opus format.")
    else:
        # Check if it's wav or mp3
        head = content[:4]
        print(f"Output header: {head}")
        if b'RIFF' in head:
            print("FAIL: Output is WAV.")
        elif b'ID3' in head or b'\xff\xfb' in head:
             print("FAIL: Output is MP3.")
        else:
             print("WARNING: Unknown format, possibly rawopus or other.")

    # Verify 48kHz by saving and inspecting (requires librosa/sf)
    filename = "benchmark_output.opus"
    with open(filename, "wb") as f:
        f.write(content)
        
    try:
        data, sr = sf.read(io.BytesIO(content))
        print(f"Sample Rate: {sr} Hz")
        if sr == 48000:
            print("PASS: Sample rate is 48kHz (FlashSR Active)")
        else:
             print(f"FAIL: Sample rate is {sr} Hz (Expected 48000)")
             
        # Speed Benchmark
        audio_duration = len(data) / sr
        rtf = audio_duration / duration
        print(f"Audio Duration: {audio_duration:.2f}s")
        print(f"Realtime Factor (RTF): {rtf:.2f}x (Higher is better)")
        
    except Exception as e:
        print(f"Error analyzing audio: {e}")
        # sf might fail on opus if underlying libs aren't fully there, but let's try
        
if __name__ == "__main__":
    run_benchmark()
