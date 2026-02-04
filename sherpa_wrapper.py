"""
Wrapper for Sherpa-ONNX TTS (Spanish voices from voicepowered.ai)
Provides high-quality offline Spanish text-to-speech synthesis
"""

import numpy as np
import os
from pathlib import Path

# Try to import sherpa_onnx
try:
    import sherpa_onnx
    SHERPA_AVAILABLE = True
except ImportError:
    SHERPA_AVAILABLE = False
    print("Warning: sherpa-onnx not installed. Spanish TTS will not be available.")


class SherpaWrapper:
    """Wrapper around Sherpa-ONNX for Spanish TTS"""
    
    def __init__(self, model_dir: str = "voices/spanish"):
        """
        Initialize Sherpa-ONNX TTS with Spanish voice model
        
        Args:
            model_dir: Directory containing the Sherpa model files:
                       - model.onnx (or model.int8.onnx)
                       - tokens.txt
                       - espeak-ng-data/ (optional, for phoneme-based models)
        
        Model files can be downloaded from:
        https://github.com/k2-fsa/sherpa-onnx/releases
        
        For Spanish voices from voicepowered.ai (Marta voice):
        https://voicepowered.ai/app/voice
        """
        self.model_dir = Path(model_dir)
        self.sample_rate = 22050  # Default sample rate for most Sherpa models
        self.tts = None
        
        if not SHERPA_AVAILABLE:
            raise ImportError("sherpa-onnx is not installed. Install with: pip install sherpa-onnx")
        
        self._load_model()
    
    def _load_model(self):
        """Load the Sherpa-ONNX TTS model"""
        print(f"Loading Sherpa-ONNX TTS model from: {self.model_dir}")
        
        # Check if model directory exists
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Model directory not found: {self.model_dir}\n"
                "Please download a Spanish voice model and place it in this directory.\n"
                "See README.md for download instructions."
            )
        
        # Find model files
        model_path = self._find_model_file()
        tokens_path = self.model_dir / "tokens.txt"
        
        if not model_path:
            raise FileNotFoundError(
                f"No model.onnx or model.int8.onnx found in {self.model_dir}"
            )
        
        if not tokens_path.exists():
            raise FileNotFoundError(
                f"tokens.txt not found in {self.model_dir}"
            )
        
        # Check for data directory (espeak-ng-data or similar)
        data_dir = self._find_data_dir()
        
        # Check for lexicon file (some models need it)
        lexicon_path = self.model_dir / "lexicon.txt"
        
        print(f"  Model: {model_path}")
        print(f"  Tokens: {tokens_path}")
        if data_dir:
            print(f"  Data dir: {data_dir}")
        
        # Create TTS configuration based on model type
        # Sherpa-ONNX supports several TTS model types
        
        # Try VITS model configuration first (most common for voicepowered.ai)
        try:
            tts_config = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=str(model_path),
                        tokens=str(tokens_path),
                        lexicon=str(lexicon_path) if lexicon_path.exists() else "",
                        data_dir=str(data_dir) if data_dir else "",
                    ),
                    provider="cpu",
                    num_threads=4,
                ),
                max_num_sentences=1,
            )
            
            self.tts = sherpa_onnx.OfflineTts(tts_config)
            self.sample_rate = self.tts.sample_rate
            print(f"✓ Sherpa-ONNX TTS loaded successfully!")
            print(f"  Sample rate: {self.sample_rate} Hz")
            
        except Exception as e:
            print(f"Error loading VITS model: {e}")
            # Try alternative model configurations if needed
            raise RuntimeError(f"Failed to load Sherpa TTS model: {e}")
    
    def _find_model_file(self) -> Path:
        """Find the ONNX model file in the model directory"""
        # Piper/Daniela voice models
        candidates = [
            "es_AR-daniela-high.onnx",
            "es_AR-daniela-medium.onnx",
            "es_AR-daniela-low.onnx",
            "model.int8.onnx",
            "model.onnx",
            "vits-model.onnx",
            "vits-model.int8.onnx",
        ]
        
        for candidate in candidates:
            path = self.model_dir / candidate
            if path.exists():
                return path
        
        # Search for any .onnx file
        onnx_files = list(self.model_dir.glob("*.onnx"))
        if onnx_files:
            return onnx_files[0]
        
        return None
    
    def _find_data_dir(self) -> Path:
        """Find the data directory (espeak-ng-data or similar)"""
        candidates = [
            "espeak-ng-data",
            "data",
        ]
        
        for candidate in candidates:
            path = self.model_dir / candidate
            if path.exists() and path.is_dir():
                return path
        
        return None
    
    def create(self, text: str, speed: float = 1.0, speaker_id: int = 0) -> tuple:
        """
        Generate speech from text
        
        Args:
            text: Text to synthesize (in Spanish)
            speed: Speech speed multiplier (default 1.0)
            speaker_id: Speaker ID for multi-speaker models (default 0)
            
        Returns:
            tuple: (audio_samples as numpy array, sample_rate)
        """
        if self.tts is None:
            raise RuntimeError("TTS model not loaded")
        
        if not text or not text.strip():
            return np.zeros(self.sample_rate, dtype=np.float32), self.sample_rate
        
        print(f"Generating Spanish speech for: '{text[:50]}...'")
        
        try:
            # Generate audio using Sherpa-ONNX
            audio = self.tts.generate(
                text,
                sid=speaker_id,
                speed=speed
            )
            
            # Convert to numpy array
            samples = np.array(audio.samples, dtype=np.float32)
            
            # Normalize if needed
            if samples.max() > 1.0 or samples.min() < -1.0:
                samples = samples / max(abs(samples.max()), abs(samples.min()))
            
            print(f"Generated {len(samples)} audio samples at {self.sample_rate}Hz")
            
            return samples, self.sample_rate
            
        except Exception as e:
            print(f"Sherpa TTS error: {e}")
            import traceback
            traceback.print_exc()
            # Return silence as fallback
            return np.zeros(self.sample_rate, dtype=np.float32), self.sample_rate
    
    def get_sample_rate(self) -> int:
        """Get the model's sample rate"""
        return self.sample_rate
    
    @staticmethod
    def is_available() -> bool:
        """Check if Sherpa-ONNX is available"""
        return SHERPA_AVAILABLE


def download_spanish_model():
    """
    Helper function to print instructions for downloading Spanish voice model
    """
    print("""
    ╔════════════════════════════════════════════════════════════════════╗
    ║           SPANISH VOICE MODEL DOWNLOAD INSTRUCTIONS                ║
    ╠════════════════════════════════════════════════════════════════════╣
    ║                                                                    ║
    ║  Option 1: VoicePowered.ai (Marta voice - recommended)             ║
    ║  ─────────────────────────────────────────────────────             ║
    ║  1. Visit: https://voicepowered.ai/app/voice                       ║
    ║  2. Search for "Marta" (Spanish female voice)                      ║
    ║  3. Download the model files                                       ║
    ║  4. Extract to: voices/spanish/                             ║
    ║                                                                    ║
    ║  Option 2: Sherpa-ONNX Pre-trained Models                          ║
    ║  ────────────────────────────────────────                          ║
    ║  1. Visit: https://github.com/k2-fsa/sherpa-onnx/releases          ║
    ║  2. Download a Spanish VITS model (e.g., vits-mms-spa)             ║
    ║  3. Extract to: voices/spanish/                             ║
    ║                                                                    ║
    ║  Quick download (MMS Spanish):                                     ║
    ║  cd voices && mkdir -p spanish && cd spanish         ║
    ║  wget https://huggingface.co/csukuangfj/vits-mms-spa/...           ║
    ║                                                                    ║
    ║  Required files in voices/spanish/:                         ║
    ║    - model.onnx (or model.int8.onnx)                               ║
    ║    - tokens.txt                                                    ║
    ║    - (optional) espeak-ng-data/                                    ║
    ║    - (optional) lexicon.txt                                        ║
    ║                                                                    ║
    ╚════════════════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    # Test the wrapper
    if SHERPA_AVAILABLE:
        try:
            wrapper = SherpaWrapper()
            audio, sr = wrapper.create("Hola, esto es una prueba de síntesis de voz en español.")
            print(f"Test successful! Generated {len(audio)} samples at {sr}Hz")
        except Exception as e:
            print(f"Test failed: {e}")
            download_spanish_model()
    else:
        print("sherpa-onnx not installed. Install with: pip install sherpa-onnx")
        download_spanish_model()
