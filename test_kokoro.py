#!/usr/bin/env python3
"""Test script to diagnose Kokoro TTS loading issues"""

import sys
import traceback

print("Testing Kokoro TTS setup...\n")

# Test 1: Import
print("1. Testing kokoro_onnx import...")
try:
    from kokoro_onnx import Kokoro
    print("   ✓ kokoro_onnx imported successfully")
except Exception as e:
    print(f"   ✗ Failed to import: {e}")
    sys.exit(1)

# Test 2: Check files exist
print("\n2. Checking model files...")
import os
model_path = "voices/english/kokoro-v0_19.onnx"
voices_path = "voices/english/voices.json"

if os.path.exists(model_path):
    size_mb = os.path.getsize(model_path) / (1024 * 1024)
    print(f"   ✓ Model file exists: {model_path} ({size_mb:.1f} MB)")
else:
    print(f"   ✗ Model file not found: {model_path}")
    sys.exit(1)

if os.path.exists(voices_path):
    size_mb = os.path.getsize(voices_path) / (1024 * 1024)
    print(f"   ✓ Voices file exists: {voices_path} ({size_mb:.1f} MB)")
else:
    print(f"   ✗ Voices file not found: {voices_path}")
    sys.exit(1)

# Test 3: Load Kokoro
print("\n3. Loading Kokoro model...")
try:
    kokoro = Kokoro(model_path, voices_path)
    print("   ✓ Kokoro loaded successfully")
except Exception as e:
    print(f"   ✗ Failed to load Kokoro: {e}")
    print("\nFull traceback:")
    traceback.print_exc()
    sys.exit(1)

# Test 4: Generate test audio
print("\n4. Generating test audio...")
try:
    test_text = "Hello, this is a test."
    samples, sample_rate = kokoro.create(test_text, voice="af_bella", speed=1.0)
    print(f"   ✓ Generated {len(samples)} samples at {sample_rate}Hz")
    print(f"   ✓ Duration: {len(samples) / sample_rate:.2f} seconds")
    print(f"   ✓ Audio range: [{samples.min():.3f}, {samples.max():.3f}]")
except Exception as e:
    print(f"   ✗ Failed to generate audio: {e}")
    traceback.print_exc()
    sys.exit(1)

# Test 5: Try to play audio
print("\n5. Testing audio playback...")
try:
    import sounddevice as sd
    print("   Playing audio...")
    sd.play(samples, sample_rate)
    sd.wait()
    print("   ✓ Audio playback completed")
except Exception as e:
    print(f"   ✗ Failed to play audio: {e}")
    traceback.print_exc()

print("\n✅ All tests passed!")
