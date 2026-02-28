"""
Wrapper for Kokoro TTS v1.0
Supports chunking long text into multiple segments.
Voices are loaded from a NumPy binary file (voices-v1.0.bin, formato npz).
"""

import numpy as np
import re
from kokoro_onnx import Kokoro as KokoroOriginal
from kokoro_onnx.config import EspeakConfig

MAX_PHONEME_LENGTH = 500  # Safe limit below 510


class KokoroWrapper:
    """Wrapper around Kokoro to handle allow_pickle issue and long text"""
    
    def __init__(self, model_path: str, voices_path: str):
        """
        Initialize Kokoro v1.0

        Args:
            model_path: Path to the ONNX model (e.g., voices/kokoro-v1.0.onnx)
            voices_path: Path to the voices binary file (e.g., voices/voices-v1.0.bin)
        """
        # Load voices from NumPy binary (npz format)
        print(f"Loading voices from binary file: {voices_path}")
        data = np.load(voices_path, allow_pickle=False)
        self.voices = {name: data[name].astype(np.float32) for name in data.files}
        print(f"Loaded {len(self.voices)} voices: {list(self.voices.keys())[:5]}...")
        
        # Create Kokoro instance without loading voices
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
            providers=['CPUExecutionProvider']
        )
        
        print("Kokoro wrapper initialized successfully!")
    
    def _split_text(self, text: str, lang: str = "en-us") -> list:
        """
        Split text into chunks that fit within phoneme limit.
        Tries to split at sentence boundaries first, then at phrases.
        """
        # Split by sentences first
        sentences = re.split(r'(?<=[.!?])'+ r'\s+', text)
        
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            # Test if adding this sentence would exceed limit
            test_chunk = current_chunk + " " + sentence if current_chunk else sentence
            
            try:
                phonemes = self.tokenizer.phonemize(test_chunk, lang)
                if len(phonemes) < MAX_PHONEME_LENGTH:
                    current_chunk = test_chunk
                else:
                    # Current chunk is full, save it
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                    
                    # Check if single sentence is too long
                    sentence_phonemes = self.tokenizer.phonemize(sentence, lang)
                    if len(sentence_phonemes) < MAX_PHONEME_LENGTH:
                        current_chunk = sentence
                    else:
                        # Split long sentence by commas/phrases
                        phrases = re.split(r'[,;:]\s*', sentence)
                        for phrase in phrases:
                            if phrase.strip():
                                phrase_phonemes = self.tokenizer.phonemize(phrase.strip(), lang)
                                if len(phrase_phonemes) < MAX_PHONEME_LENGTH:
                                    chunks.append(phrase.strip())
                                else:
                                    # Last resort: truncate
                                    words = phrase.split()
                                    partial = ""
                                    for word in words:
                                        test = partial + " " + word if partial else word
                                        test_ph = self.tokenizer.phonemize(test, lang)
                                        if len(test_ph) < MAX_PHONEME_LENGTH:
                                            partial = test
                                        else:
                                            if partial:
                                                chunks.append(partial.strip())
                                            partial = word
                                    if partial:
                                        chunks.append(partial.strip())
                        current_chunk = ""
            except Exception as e:
                # If phonemization fails, just add sentence
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks if chunks else [text[:200]]  # Fallback
    
    def _create_audio_chunk(self, text: str, voice_data: np.ndarray, speed: float, lang: str):
        """Generate audio for a single chunk of text"""
        # Convert text to phonemes
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
        
        return outputs[0].flatten().astype(np.float32)
    
    def create(self, text: str, voice: str = "af_bella", speed: float = 1.0, lang: str = "en-us"):
        """
        Generate speech from text (handles long text by chunking)
        
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
        sample_rate = 24000
        
        # Split text into manageable chunks
        chunks = self._split_text(text, lang)
        print(f"Split text into {len(chunks)} chunks")
        
        # Generate audio for each chunk
        audio_parts = []
        for i, chunk in enumerate(chunks):
            try:
                audio = self._create_audio_chunk(chunk, voice_data, speed, lang)
                audio_parts.append(audio)
                
                # Add small pause between chunks (0.15 seconds of silence)
                if i < len(chunks) - 1:
                    pause = np.zeros(int(sample_rate * 0.15), dtype=np.float32)
                    audio_parts.append(pause)
                    
            except Exception as e:
                print(f"Error generating chunk {i}: {e}")
                continue
        
        if not audio_parts:
            return np.zeros(sample_rate, dtype=np.float32), sample_rate
        
        # Concatenate all audio
        full_audio = np.concatenate(audio_parts)
        
        return full_audio, sample_rate
