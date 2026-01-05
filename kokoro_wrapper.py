"""
Wrapper for Kokoro TTS to handle the pickle issue with voices.json
"""

import numpy as np
from kokoro_onnx import Kokoro as KokoroOriginal
from kokoro_onnx.config import EspeakConfig


class KokoroWrapper:
    """Wrapper around Kokoro to handle allow_pickle issue"""
    
    def __init__(self, model_path: str, voices_path: str):
        """
        Initialize Kokoro with allow_pickle workaround
        
        Args:
            model_path: Path to the ONNX model
            voices_path: Path to the voices file (.json)
        """
        # Load voices from JSON
        print("Loading voices from JSON...")
        import json
        with open(voices_path, 'r') as f:
            voices_data = json.load(f)
        
        # Convert to numpy arrays
        self.voices = {}
        for voice_name, voice_array in voices_data.items():
            self.voices[voice_name] = np.array(voice_array, dtype=np.float32)
        
        print(f"Loaded {len(self.voices)} voices: {list(self.voices.keys())[:5]}...")
        
        # Create Kokoro instance without loading voices
        # We'll monkey-patch it
        print("Initializing Kokoro model...")
        
        # Import internals
        import onnxruntime as ort
        from kokoro_onnx.tokenizer import Tokenizer
        
        # Create espeak config
        espeak_config = EspeakConfig()
        
        # Create tokenizer
        self.tokenizer = Tokenizer(espeak_config=espeak_config)
        
        # Load ONNX model
        self.session = ort.InferenceSession(
            model_path,
            providers=['CPUExecutionProvider']  # Start with CPU, can add CUDA later
        )
        
        print("Kokoro wrapper initialized successfully!")
    
    def create(self, text: str, voice: str = "af_bella", speed: float = 1.0, lang: str = "en-us"):
        """
        Generate speech from text
        
        Args:
            text: Text to synthesize
            voice: Voice name (e.g., 'af_bella', 'af_sarah')
            speed: Speech speed multiplier
            lang: Language code
            
        Returns:
            tuple: (audio_samples, sample_rate)
        """
        # Get voice embedding
        if voice not in self.voices:
            available = list(self.voices.keys())
            raise ValueError(f"Voice '{voice}' not found. Available: {available}")
        
        voice_data = self.voices[voice]
        
        # Convert text to phonemes using espeak
        phonemes = self.tokenizer.phonemize(text, lang)
        
        # Tokenize phonemes
        tokens = np.array(self.tokenizer.tokenize(phonemes), dtype=np.int64)
        
        # Prepare voice style for the token length
        voice_style = voice_data[len(tokens)]
        
        # Add start and end tokens
        tokens_with_padding = np.array([[0, *tokens, 0]], dtype=np.int64)
        
        # Prepare input
        speed_array = np.ones(1, dtype=np.float32) * speed
        
        # Run inference
        outputs = self.session.run(
            None,
            {
                'tokens': tokens_with_padding,
                'style': np.array(voice_style, dtype=np.float32),
                'speed': speed_array
            }
        )
        
        audio = outputs[0].flatten()
        sample_rate = 24000  # Kokoro outputs at 24kHz
        
        return audio.astype(np.float32), sample_rate
