"""
TTS Manager - Unified Text-to-Speech engine.
Routes synthesis to Kokoro (built-in voices) or Sherpa-ONNX (external voice
packs) based on the selected voice name.

  Kokoro voices  — short names like af_bella, ef_dora  (no hyphens)
  Sherpa voices  — folder names like vits-piper-es_AR-daniela-high  (contain hyphens)
"""

import numpy as np
from pathlib import Path
from typing import Tuple

# Language → Kokoro lang code mapping
LANG_CODES = {
    "english": "en-us",
    "spanish": "es",
}


def _is_sherpa_voice(voice_name: str) -> bool:
    """Sherpa voice folder names always contain hyphens; Kokoro names never do."""
    return "-" in voice_name


class TTSManager:
    """
    Manages TTS synthesis using Kokoro (default) or Sherpa-ONNX (external
    voice packs classified via the voice scanner).
    """

    def __init__(
        self,
        language: str = "english",
        kokoro_model_path: str = "voices/kokoro-v1.0/kokoro-v1.0.onnx",
        voices_path: str = "voices/kokoro-v1.0/voices-v1.0.bin",
        voice_name: str = "af_bella",
    ):
        self.language = language
        self.voice_name = voice_name

        # Kokoro engine
        self.kokoro = None
        self.kokoro_available = False

        # Sherpa engines (lazy, one per folder path)
        self._sherpa_cache: dict = {}

        self._init_kokoro(kokoro_model_path, voices_path)

        print(f"\n🔊 TTS Manager initialized (Kokoro + Sherpa)")
        print(f"   Language : {language}")
        print(f"   Lang code: {LANG_CODES.get(language, 'en-us')}")
        print(f"   Voice    : {voice_name}")
        print(f"   Available: {'✓' if self.kokoro_available else '✗'}")

    # ------------------------------------------------------------------ init
    def _init_kokoro(self, model_path: str, voices_path: str):
        """Initialize Kokoro TTS"""
        try:
            from kokoro_wrapper import KokoroWrapper
            self.kokoro = KokoroWrapper(model_path, voices_path)
            self.kokoro_available = True
            print("✓ Kokoro TTS loaded successfully!")
        except ImportError as e:
            print(f"⚠ Kokoro not available: {e}")
        except Exception as e:
            print(f"⚠ Could not load Kokoro: {e}")

    def _get_sherpa(self, folder_name: str):
        """Return a (cached) SherpaWrapper for the given voice folder name."""
        if folder_name not in self._sherpa_cache:
            from sherpa_wrapper import SherpaWrapper
            model_dir = str(Path("voices") / folder_name)
            print(f"Loading Sherpa voice: {model_dir}")
            self._sherpa_cache[folder_name] = SherpaWrapper(model_dir)
        return self._sherpa_cache[folder_name]

    # -------------------------------------------------------------- setters
    def set_language(self, language: str):
        """Change the active language."""
        if language not in LANG_CODES:
            print(f"⚠ Unknown language: {language}. Falling back to english.")
            language = "english"
        self.language = language
        print(f"🌐 TTS language set to: {language} (code: {LANG_CODES[language]})")

    def set_voice(self, voice_name: str):
        """Change the active voice (Kokoro or Sherpa)."""
        self.voice_name = voice_name
        print(f"🎤 TTS voice set to: {voice_name}")

    # ----------------------------------------------------------- synthesis
    def create(self, text: str, speed: float = 1.0) -> Tuple[np.ndarray, int]:
        """
        Synthesize speech. Routes to Sherpa-ONNX for external voice packs,
        or Kokoro for built-in voices.
        """
        if not text or not text.strip():
            return np.zeros(24000, dtype=np.float32), 24000

        # ── Sherpa voice ───────────────────────────────────────────────────
        if _is_sherpa_voice(self.voice_name):
            try:
                wrapper = self._get_sherpa(self.voice_name)
                samples, sr = wrapper.create(text, speed=speed)
                return samples, sr
            except Exception as e:
                print(f"Sherpa TTS error: {e}")
                return np.zeros(24000, dtype=np.float32), 24000

        # ── Kokoro voice ───────────────────────────────────────────────────
        if not self.kokoro_available:
            print("⚠ Kokoro not available, returning silence")
            return np.zeros(24000, dtype=np.float32), 24000

        lang_code = LANG_CODES.get(self.language, "en-us")
        try:
            samples, sample_rate = self.kokoro.create(
                text,
                voice=self.voice_name,
                speed=speed,
                lang=lang_code,
            )
            return samples, sample_rate
        except Exception as e:
            print(f"Kokoro TTS error: {e}")
            return np.zeros(24000, dtype=np.float32), 24000

    # ----------------------------------------------------------- status
    def is_available(self) -> bool:
        """Check if any TTS engine is available."""
        if _is_sherpa_voice(self.voice_name):
            return True  # will attempt to load on first use
        return self.kokoro_available

    def get_current_engine(self) -> str:
        """Get the name of the current TTS engine."""
        return "Sherpa-ONNX" if _is_sherpa_voice(self.voice_name) else "Kokoro"

    def get_status(self) -> dict:
        """Get status information about the TTS engine."""
        return {
            "current_language": self.language,
            "lang_code": LANG_CODES.get(self.language, "en-us"),
            "current_engine": self.get_current_engine(),
            "voice": self.voice_name,
            "kokoro_available": self.kokoro_available,
            "is_available": self.is_available(),
        }
