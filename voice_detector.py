"""
Voice Detection Module
Automatically detects available voices in voices/english/ and voices/spanish/
"""

import json
import os
from pathlib import Path
from typing import List, Dict


def get_english_voices() -> List[str]:
    """
    Detect available English voices from voices.json
    
    Returns:
        List of voice names (e.g., ['af_bella', 'af_sarah', 'am_adam'])
    """
    voices_json = Path("voices/english/voices.json")
    
    if not voices_json.exists():
        print("⚠️ voices.json not found for English voices")
        return ["af_bella"]  # Default fallback
    
    try:
        with open(voices_json, 'r') as f:
            voices_data = json.load(f)
            # Get all voice keys from the JSON
            voice_list = list(voices_data.keys())
            print(f"✓ Detected {len(voice_list)} English voices")
            return voice_list
    except Exception as e:
        print(f"⚠️ Error reading voices.json: {e}")
        return ["af_bella"]


def get_spanish_voices() -> Dict[str, str]:
    """
    Detect available Spanish voices by scanning subdirectories in voices/spanish/
    
    Returns:
        Dict with voice name as key and path as value
        Example: {'Daniela': 'voices/spanish/Daniela', 'Marta': 'voices/spanish/Marta'}
    """
    spanish_dir = Path("voices/spanish")
    
    if not spanish_dir.exists():
        print("⚠️ voices/spanish directory not found")
        return {}
    
    voices = {}
    
    try:
        # Scan for subdirectories
        for item in spanish_dir.iterdir():
            if item.is_dir():
                # Check if directory contains required files
                onnx_files = list(item.glob("*.onnx"))
                tokens_file = item / "tokens.txt"
                
                if onnx_files and tokens_file.exists():
                    voice_name = item.name
                    voices[voice_name] = str(item)
                    print(f"✓ Detected Spanish voice: {voice_name} ({onnx_files[0].name})")
        
        if not voices:
            print("⚠️ No Spanish voices found in voices/spanish/")
        
        return voices
    
    except Exception as e:
        print(f"⚠️ Error scanning Spanish voices: {e}")
        return {}


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
        spanish_voices = get_spanish_voices()
        return list(spanish_voices.keys())
    else:
        return []


if __name__ == "__main__":
    print("=== English Voices ===")
    english = get_english_voices()
    print(f"Found {len(english)} voices: {english[:5]}...")
    
    print("\n=== Spanish Voices ===")
    spanish = get_spanish_voices()
    for name, path in spanish.items():
        print(f"  {name}: {path}")
