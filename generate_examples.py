#!/usr/bin/env python3
"""Generate audio examples with and without FlashSR for GitHub documentation."""

import requests
import os
import time

# We'll generate examples by calling the API
# First with FlashSR enabled (current server state), then we'll restart without

BASE_URL = "http://localhost:8880/v1/audio/speech"
OUTPUT_DIR = "examples/audio_samples"

EXAMPLES = [
    {
        "text": "Kokoro FastAPI enhanced audio sounds significantly better because it utilizes the FlashSR ONNX model to upscale the output to 48kHz. This upscaling process restores high-frequency detail and harmonic richness that is often missing in standard text-to-speech outputs, resulting in a much more natural, professional, and pleasant listening experience for users across all applications.",
        "voice": "af_heart",
        "lang_code": "a",
        "name": "english_benefits"
    },
    {
        "text": "Kokoro FastAPI mejorado suena significativamente mejor porque utiliza el modelo FlashSR ONNX para escalar la salida a 48kHz. Este proceso de escalado restaura los detalles de alta frecuencia y la riqueza armónica que a menudo falta en las salidas estándar de texto a voz, lo que resulta en una experiencia de escucha mucho más natural, profesional y agradable para los usuarios en todas las aplicaciones.",
        "voice": "ef_dora",
        "lang_code": "e",
        "name": "spanish_benefits"
    },
    {
        "text": "Why do programmers always mix up Christmas and Halloween? Because Oct 31 equals Dec 25! And did you hear about the programmer who got stuck in the shower? The instructions said: Lather, Rinse, Repeat. He is still there to this day, following the infinite loop of cleanliness.",
        "voice": "af_heart",
        "lang_code": "a",
        "name": "english_jokes"
    },
    {
        "text": "¿Por qué los programadores prefieren el modo oscuro? ¡Porque la luz atrae a los bichos! ¿Y sabes qué hace un programador cuando tiene hambre? ¡Un bit-e! No son los mejores chistes del mundo, pero al menos no tienen errores de sintaxis ni fugas de memoria, lo cual es más de lo que puedo decir de algunos de mis códigos antiguos.",
        "voice": "ef_dora",
        "lang_code": "e",
        "name": "spanish_jokes"
    },
    {
        "text": "Deploying text-to-speech models on the CPU usually involves a trade-off between speed and quality. However, by leveraging ONNX Runtime and FlashSR, we can achieve high-fidelity 48 kilohertz audio with low latency, making it the perfect solution for edge deployments, real-time assistants, and accessible web interfaces that demand excellence.",
        "voice": "af_heart",
        "lang_code": "a",
        "name": "english_technical"
    }
]

def generate_samples(suffix=""):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for example in EXAMPLES:
        print(f"Generating: {example['name']}{suffix}...")
        
        try:
            response = requests.post(
                BASE_URL,
                json={
                    "model": "kokoro",
                    "input": example["text"],
                    "voice": example["voice"],
                    "lang_code": example["lang_code"],
                    "response_format": "opus",
                    "stream": False
                },
                timeout=120
            )
            
            if response.status_code == 200:
                filename = f"{example['name']}{suffix}.opus"
                filepath = os.path.join(OUTPUT_DIR, filename)
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"  ✓ Saved: {filepath} ({len(response.content)} bytes)")
            else:
                print(f"  ✗ Error: {response.status_code}")
                
        except Exception as e:
            print(f"  ✗ Exception: {e}")

if __name__ == "__main__":
    print("Generating audio samples WITHOUT FlashSR (24kHz)...")
    print("=" * 50)
    generate_samples("_24khz")
    print("\nDone!")
