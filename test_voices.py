#!/usr/bin/env python3
"""Test script for generating audio with specific voices and monk perspective texts."""

import requests
import time
import os

BASE_URL = "http://localhost:8880/v1/audio/speech"

# Test texts about happiness from a monk's perspective
TESTS = [
    {
        "voice": "ef_dora",
        "lang_code": "e",  # Spanish
        "text": "La verdadera felicidad no está en poseer cosas materiales, sino en la paz interior que cultivamos día a día. En esta sociedad moderna llena de ruido y prisa, el monje encuentra serenidad en la simplicidad. Respiramos, observamos, y dejamos ir.",
        "filename": "spanish_monk_happiness.opus"
    },
    {
        "voice": "af_heart", 
        "lang_code": "a",  # English
        "text": "In our modern world of constant notifications and endless distractions, true happiness remains what it always was: a quiet mind and an open heart. The monk watches the busy streets from the temple steps and smiles. All this rushing, all this striving - yet peace was always here, waiting to be noticed.",
        "filename": "english_monk_happiness.opus"
    },
    {
        "voice": "ef_dora",
        "lang_code": "e",  # Spanish
        "text": "Los jóvenes preguntan: ¿cómo ser feliz con tan poco? Y el monje responde: la pregunta correcta es, ¿cómo ser feliz con tanto? Menos es más. La tecnología nos conecta pero también nos separa. Vuelve a lo esencial.",
        "filename": "spanish_monk_youth.opus"
    },
    {
        "voice": "af_heart",
        "lang_code": "a",  # English  
        "text": "Happiness is not a destination but a manner of traveling. The ancient wisdom still applies: be present, be grateful, be kind. In coffee shops and coworking spaces, the modern seeker can still find the timeless truth - that contentment blooms from within.",
        "filename": "english_monk_wisdom.opus"
    }
]

def run_tests():
    print("Starting TTS tests with monk perspective texts...")
    print("=" * 60)
    
    output_dir = "test_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, test in enumerate(TESTS, 1):
        print(f"\n[Test {i}/{len(TESTS)}]")
        print(f"  Voice: {test['voice']}")
        print(f"  Language: {'Spanish' if test['lang_code'] == 'e' else 'English'}")
        print(f"  Text: {test['text'][:60]}...")
        
        start_time = time.time()
        
        try:
            response = requests.post(
                BASE_URL,
                json={
                    "model": "kokoro",
                    "input": test["text"],
                    "voice": test["voice"],
                    "lang_code": test["lang_code"],
                    "response_format": "opus",
                    "stream": False
                },
                timeout=120
            )
            
            duration = time.time() - start_time
            
            if response.status_code == 200:
                filepath = os.path.join(output_dir, test["filename"])
                with open(filepath, "wb") as f:
                    f.write(response.content)
                print(f"  ✓ Success! Generated in {duration:.2f}s")
                print(f"  ✓ Saved to: {filepath}")
                print(f"  ✓ File size: {len(response.content)} bytes")
            else:
                print(f"  ✗ Error: {response.status_code}")
                print(f"  ✗ Response: {response.text[:200]}")
                
        except Exception as e:
            print(f"  ✗ Exception: {e}")
    
    print("\n" + "=" * 60)
    print("Tests completed!")
    print(f"Output files saved to: {output_dir}/")

if __name__ == "__main__":
    run_tests()
