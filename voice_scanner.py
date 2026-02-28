"""
Voice Scanner Module
====================
Scans the voices/ folder on startup to detect installed voice packs.

Structure rules:
  - voices/kokoro-v1.0/   →  Kokoro v1.0 engine (known, no classification needed)
  - voices/<anything>/    →  Treated as a Sherpa-ONNX voice pack (unknown engine)

A folder is considered a valid Sherpa voice pack when it contains:
  • at least one .onnx file
  • an espeak-ng-data/ sub-directory

Classification results are stored in voices/voices_config.json so the user is
only asked once per newly-discovered folder.
"""

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

# ── Constants ──────────────────────────────────────────────────────────────────
VOICES_DIR = Path("voices")
VOICES_CONFIG_FILE = VOICES_DIR / "voices_config.json"

KOKORO_FOLDER = "kokoro-v1.0"
KOKORO_MODEL_FILE = "kokoro-v1.0.onnx"
KOKORO_VOICES_FILE = "voices-v1.0.bin"

# Folders the scanner recognises without asking the user
KNOWN_ENGINE_FOLDERS = {KOKORO_FOLDER}


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_voices_config() -> dict:
    """Load the persistent voice classification config (or return empty dict)."""
    if VOICES_CONFIG_FILE.exists():
        try:
            with open(VOICES_CONFIG_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_voices_config(config: dict) -> None:
    """Persist the voice classification config to disk."""
    VOICES_DIR.mkdir(parents=True, exist_ok=True)
    with open(VOICES_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def is_sherpa_voice_folder(folder_path: Path) -> bool:
    """
    Return True if the folder looks like a Sherpa-ONNX voice pack.

    Criteria:
      • Contains at least one .onnx file
      • Contains an espeak-ng-data/ sub-directory
    """
    if not folder_path.is_dir():
        return False
    has_onnx = any(folder_path.glob("*.onnx"))
    has_espeak = (folder_path / "espeak-ng-data").is_dir()
    return has_onnx and has_espeak


# ── Main scanner ───────────────────────────────────────────────────────────────

def scan_voices_folder(voices_dir: Optional[str] = None) -> dict:
    """
    Scan the voices/ directory and return a structured result.

    Returns a dict with the following keys:

    {
        "kokoro": {
            "found": bool,
            "model": str | None,   # absolute or relative path
            "voices": str | None,
        },
        "sherpa_voices": [
            {
                "folder": str,          # folder name (e.g. vits-piper-es_AR-…)
                "path": str,            # full path to the folder
                "onnx": str | None,     # first .onnx file found
                "language": str | None, # cached classification ("english"/"spanish")
                "is_new": bool,         # True if not yet classified by user
            },
            …
        ],
        "has_any": bool,  # True if at least one usable voice is present
    }
    """
    base = Path(voices_dir) if voices_dir else VOICES_DIR
    config = _load_voices_config()

    result: dict = {
        "kokoro": {
            "found": False,
            "model": None,
            "voices": None,
        },
        "sherpa_voices": [],
        "has_any": False,
    }

    if not base.is_dir():
        return result

    for entry in sorted(base.iterdir()):
        if not entry.is_dir():
            continue

        name = entry.name

        # ── Kokoro engine ──────────────────────────────────────────────────
        if name == KOKORO_FOLDER:
            model_path = entry / KOKORO_MODEL_FILE
            voices_path = entry / KOKORO_VOICES_FILE
            if model_path.exists() and voices_path.exists():
                result["kokoro"]["found"] = True
                result["kokoro"]["model"] = str(model_path)
                result["kokoro"]["voices"] = str(voices_path)
            continue

        # ── Sherpa voice pack ──────────────────────────────────────────────
        if is_sherpa_voice_folder(entry):
            onnx_files = list(entry.glob("*.onnx"))
            onnx_path = str(onnx_files[0]) if onnx_files else None
            cached_lang = config.get(name)  # "english" / "spanish" / None
            result["sherpa_voices"].append(
                {
                    "folder": name,
                    "path": str(entry),
                    "onnx": onnx_path,
                    "language": cached_lang,
                    "is_new": cached_lang is None,
                }
            )

    result["has_any"] = result["kokoro"]["found"] or len(result["sherpa_voices"]) > 0

    # ── Auto-update config ────────────────────────────────────────────────────
    # Keep only entries that correspond to folders currently on disk.
    # This removes stale entries when a voice pack is deleted, without
    # losing classifications for packs that are still present.
    present_folders = {v["folder"] for v in result["sherpa_voices"]}
    reconciled_config = {
        folder: lang
        for folder, lang in config.items()
        if folder in present_folders
    }
    _save_voices_config(reconciled_config)

    return result


def save_voice_language(folder_name: str, language: str) -> None:
    """Persist the user's language classification for a Sherpa voice folder."""
    config = _load_voices_config()
    config[folder_name] = language
    _save_voices_config(config)
