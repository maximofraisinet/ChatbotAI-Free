"""
Voice Detection Module
Detects Kokoro (voices/kokoro-v1.0/) and Sherpa-ONNX voices (all other
subfolders classified via voices/voices_config.json).
"""

import json
import os
from pathlib import Path
from typing import List


def _get_sherpa_voices_for_language(language: str) -> List[str]:
    """
    Return Sherpa voice folder names classified as `language`
    (read from voices/voices_config.json).
    """
    config_path = Path("voices/voices_config.json")
    if not config_path.exists():
        return []
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        return [folder for folder, lang in config.items() if lang == language]
    except Exception:
        return []


def get_english_voices() -> List[str]:
    """
    Return available English voices:
      • Kokoro voices (af_*, am_*) from voices-v1.0.bin
      • Any Sherpa folder classified as 'english' in voices_config.json
    """
    voices_bin = Path("voices/kokoro-v1.0/voices-v1.0.bin")
    kokoro_voices: List[str] = []

    if voices_bin.exists():
        try:
            import numpy as np
            data = np.load(str(voices_bin), allow_pickle=False)
            kokoro_voices = [v for v in data.files if v.startswith('a')]
            if kokoro_voices:
                print(f"✓ Detected {len(kokoro_voices)} English voices")
        except Exception as e:
            print(f"⚠️ Error reading voices-v1.0.bin: {e}")
    else:
        print("⚠️ voices-v1.0.bin not found for English voices")
        kokoro_voices = ["af_bella"]

    sherpa = _get_sherpa_voices_for_language("english")
    if sherpa:
        print(f"✓ Detected {len(sherpa)} Sherpa English voice(s): {sherpa}")

    return kokoro_voices + sherpa if kokoro_voices else (["af_bella"] + sherpa)


def get_spanish_voices() -> List[str]:
    """
    Return available Spanish voices:
      • Kokoro voices (ef_*, em_*) from voices-v1.0.bin
      • Any Sherpa folder classified as 'spanish' in voices_config.json
    """
    voices_bin = Path("voices/kokoro-v1.0/voices-v1.0.bin")
    kokoro_voices: List[str] = []

    if voices_bin.exists():
        try:
            import numpy as np
            data = np.load(str(voices_bin), allow_pickle=False)
            kokoro_voices = [v for v in data.files if v.startswith('e')]
            if kokoro_voices:
                print(f"\u2713 Detected {len(kokoro_voices)} Spanish voices")
        except Exception as e:
            print(f"\u26a0\ufe0f Error reading voices-v1.0.bin: {e}")
    else:
        print("\u26a0\ufe0f voices-v1.0.bin not found for Spanish voices")
        kokoro_voices = ["ef_dora"]

    sherpa = _get_sherpa_voices_for_language("spanish")
    if sherpa:
        print(f"\u2713 Detected {len(sherpa)} Sherpa Spanish voice(s): {sherpa}")

    return kokoro_voices + sherpa if kokoro_voices else (["ef_dora"] + sherpa)


def get_available_voices(language: str) -> List[str]:
    """
    Get list of available voices for the specified language
    
    Args:
        language: "english" or "spanish"
    
    Returns:
        List of voice names
    """
    if language == "english":
        return get_english_voices()
    elif language == "spanish":
        return get_spanish_voices()
    else:
        return []


if __name__ == "__main__":
    print("=== English Voices ===")
    english = get_english_voices()
    print(f"Found {len(english)} voices: {english[:5]}...")
    
    print("\n=== Spanish Voices ===")
    spanish = get_spanish_voices()
    print(f"Found {len(spanish)} voices: {spanish}")
