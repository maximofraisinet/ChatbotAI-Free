"""
TTS Manager - Hybrid Text-to-Speech engine
Supports multiple TTS backends based on language:
  - English: Kokoro ONNX
  - Spanish: Sherpa-ONNX (voicepowered.ai voices)
"""

import numpy as np
from typing import Tuple, Optional, Callable


class TTSManager:
    """
    Manages multiple TTS engines and routes synthesis 
    requests based on the current language setting.
    """
    
    def __init__(
        self,
        language: str = "english",
        # Kokoro config (English)
        kokoro_model_path: str = "voices/english/kokoro-v0_19.onnx",
        voices_path: str = "voices/english/voices.json",
        kokoro_voice: str = "af_bella",
        # Sherpa config (Spanish)
        sherpa_model_dir: str = "voices/spanish",
        sherpa_voice_name: str = "vits-piper-es_AR-daniela-high",
        sherpa_speaker_id: int = 0,
    ):
        """
        Initialize the TTS Manager with multiple backends.
        
        Args:
            language: Current language ("english" or "spanish")
            kokoro_model_path: Path to Kokoro ONNX model
            voices_path: Path to Kokoro voices.json
            kokoro_voice: Kokoro voice name (e.g., "af_bella")
            sherpa_model_dir: Base directory for Spanish voices (voices/spanish)
            sherpa_voice_name: Spanish voice subdirectory name (e.g., "Daniela", "Marta")
            sherpa_speaker_id: Speaker ID for Sherpa (multi-speaker models)
        """
        self.language = language
        self.kokoro_voice = kokoro_voice
        self.sherpa_voice_name = sherpa_voice_name
        self.sherpa_model_dir = sherpa_model_dir
        self.sherpa_speaker_id = sherpa_speaker_id
        
        # Engine instances
        self.kokoro = None
        self.sherpa = None
        
        # Availability flags
        self.kokoro_available = False
        self.sherpa_available = False
        
        # Load Kokoro (English TTS)
        self._init_kokoro(kokoro_model_path, voices_path)
        
        # Load Sherpa (Spanish TTS)
        self._init_sherpa(sherpa_model_dir)
        
        print(f"\n🔊 TTS Manager initialized")
        print(f"   Current language: {language}")
        print(f"   Kokoro (English): {'✓' if self.kokoro_available else '✗'}")
        print(f"   Sherpa (Spanish): {'✓' if self.sherpa_available else '✗'}")
    
    def _init_kokoro(self, model_path: str, voices_path: str):
        """Initialize Kokoro TTS for English"""
        try:
            from kokoro_wrapper import KokoroWrapper
            self.kokoro = KokoroWrapper(model_path, voices_path)
            self.kokoro_available = True
            print("✓ Kokoro TTS (English) loaded successfully!")
        except ImportError as e:
            print(f"⚠ Kokoro not available: {e}")
            self.kokoro_available = False
        except Exception as e:
            print(f"⚠ Could not load Kokoro: {e}")
            self.kokoro_available = False
    
    def _init_sherpa(self, model_dir: str):
        """Initialize Sherpa TTS for Spanish"""
        try:
            from sherpa_wrapper import SherpaWrapper, SHERPA_AVAILABLE
            
            if not SHERPA_AVAILABLE:
                print("⚠ sherpa-onnx not installed. Spanish TTS unavailable.")
                print("  Install with: pip install sherpa-onnx")
                self.sherpa_available = False
                return
            
            # Use subdirectory for specific voice
            voice_dir = f"{model_dir}/{self.sherpa_voice_name}"
            self.sherpa = SherpaWrapper(voice_dir)
            self.sherpa_available = True
            print(f"✓ Sherpa TTS (Spanish - {self.sherpa_voice_name}) loaded successfully!")
            
        except ImportError as e:
            print(f"⚠ Sherpa not available: {e}")
            self.sherpa_available = False
        except FileNotFoundError as e:
            print(f"⚠ Sherpa model not found: {e}")
            print("  Download Spanish voice model to voices/spanish/")
            self.sherpa_available = False
        except Exception as e:
            print(f"⚠ Could not load Sherpa: {e}")
            self.sherpa_available = False
    
    def set_language(self, language: str):
        """
        Change the current language for TTS.
        
        Args:
            language: "english" or "spanish"
        """
        if language not in ("english", "spanish"):
            print(f"⚠ Unknown language: {language}. Using english.")
            language = "english"
        
        self.language = language
        print(f"🌐 TTS language set to: {language}")
    
    def set_kokoro_voice(self, voice_name: str):
        """Set the Kokoro voice for English TTS"""
        self.kokoro_voice = voice_name
        print(f"🎤 Kokoro voice set to: {voice_name}")
    
    def set_sherpa_voice(self, voice_name: str):
        """Change the Spanish voice (reloads Sherpa with new voice directory)"""
        if voice_name == self.sherpa_voice_name:
            return  # Already using this voice
        
        self.sherpa_voice_name = voice_name
        print(f"🎤 Changing Spanish voice to: {voice_name}")
        
        # Reload Sherpa with new voice directory
        try:
            self._init_sherpa(self.sherpa_model_dir)
        except Exception as e:
            print(f"⚠ Failed to load voice {voice_name}: {e}")
            self.sherpa_available = False
    
    def set_sherpa_speaker(self, speaker_id: int):
        """Set the speaker ID for Sherpa (multi-speaker models)"""
        self.sherpa_speaker_id = speaker_id
        print(f"🎤 Sherpa speaker ID set to: {speaker_id}")
    
    def create(self, text: str, speed: float = 1.0) -> Tuple[np.ndarray, int]:
        """
        Generate speech from text using the appropriate TTS engine.
        
        Args:
            text: Text to synthesize
            speed: Speech speed multiplier (default 1.0)
            
        Returns:
            tuple: (audio_samples as numpy array, sample_rate)
        """
        if not text or not text.strip():
            return np.zeros(24000, dtype=np.float32), 24000
        
        # Route to appropriate engine based on language
        if self.language == "spanish":
            return self._synthesize_spanish(text, speed)
        else:
            return self._synthesize_english(text, speed)
    
    def _synthesize_english(self, text: str, speed: float) -> Tuple[np.ndarray, int]:
        """Synthesize English text using Kokoro"""
        if not self.kokoro_available:
            print("⚠ Kokoro not available, returning silence")
            return np.zeros(24000, dtype=np.float32), 24000
        
        try:
            samples, sample_rate = self.kokoro.create(
                text,
                voice=self.kokoro_voice,
                speed=speed,
                lang="en-us"
            )
            return samples, sample_rate
            
        except Exception as e:
            print(f"Kokoro TTS error: {e}")
            return np.zeros(24000, dtype=np.float32), 24000
    
    def _synthesize_spanish(self, text: str, speed: float) -> Tuple[np.ndarray, int]:
        """Synthesize Spanish text using Sherpa"""
        if not self.sherpa_available:
            print("⚠ Sherpa not available, returning silence")
            return np.zeros(22050, dtype=np.float32), 22050
        
        try:
            samples, sample_rate = self.sherpa.create(
                text,
                speed=speed,
                speaker_id=self.sherpa_speaker_id
            )
            return samples, sample_rate
            
        except Exception as e:
            print(f"Sherpa TTS error: {e}")
            return np.zeros(22050, dtype=np.float32), 22050
    
    def is_available(self) -> bool:
        """Check if TTS is available for the current language"""
        if self.language == "spanish":
            return self.sherpa_available
        else:
            return self.kokoro_available
    
    def get_current_engine(self) -> str:
        """Get the name of the current TTS engine"""
        if self.language == "spanish":
            return "Sherpa-ONNX"
        else:
            return "Kokoro"
    
    def get_status(self) -> dict:
        """Get status information about all TTS engines"""
        return {
            "current_language": self.language,
            "current_engine": self.get_current_engine(),
            "kokoro_available": self.kokoro_available,
            "sherpa_available": self.sherpa_available,
            "is_available": self.is_available(),
        }
