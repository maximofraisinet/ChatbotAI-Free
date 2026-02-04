"""
AI Manager - Handles Whisper STT, Ollama LLM, and TTS (Kokoro/Sherpa)
Loads models once at initialization to save VRAM
Supports English and Spanish languages
"""

import os
import json
import numpy as np
from faster_whisper import WhisperModel
import ollama

# Import TTS Manager (handles both Kokoro and Sherpa)
try:
    from tts_manager import TTSManager
    TTS_MANAGER_AVAILABLE = True
except ImportError:
    TTS_MANAGER_AVAILABLE = False
    print("Warning: tts_manager not found, TTS will be disabled")


class AIManager:
    """Manages all AI models for the chatbot"""
    
    def __init__(self, 
                 whisper_model="base",  # Use multilingual by default (not base.en)
                 ollama_model="llama3.1:8b",
                 kokoro_model_path="voices/english/kokoro-v0_19.onnx",
                 voices_path="voices/english/voices.json",
                 voice_name="af_bella",
                 language="english",
                 sherpa_model_dir="voices/spanish"):
        """
        Initialize all AI models
        
        Args:
            whisper_model: faster-whisper model size (use 'base' for multilingual)
            ollama_model: Ollama model name
            kokoro_model_path: Path to Kokoro ONNX model
            voices_path: Path to voices.json
            voice_name: Voice to use (af_bella or af_sarah)
            language: Current language ("english" or "spanish")
            sherpa_model_dir: Directory with Sherpa Spanish model
        """
        print("Initializing AI Manager...")
        
        self.language = language
        
        # Check CUDA availability
        try:
            import torch
            cuda_available = torch.cuda.is_available()
            print(f"CUDA available: {cuda_available}")
            self._device = "cuda" if cuda_available else "cpu"
            self._compute_type = "float16" if cuda_available else "int8"
        except ImportError:
            print("PyTorch not found, assuming CUDA available")
            self._device = "cuda"
            self._compute_type = "float16"
        
        # IMPORTANT: Always use multilingual model for multi-language support
        # Remove .en suffix if present - the .en models ONLY understand English
        actual_whisper_model = whisper_model.replace(".en", "")
        print(f"Using multilingual Whisper model: {actual_whisper_model}")
        
        # Load Whisper STT
        print(f"Loading Whisper model: {actual_whisper_model} on {self._device}")
        print("This may take a few minutes on first run (downloading model)...")
        self.whisper = WhisperModel(
            actual_whisper_model,
            device=self._device,
            compute_type=self._compute_type
        )
        self._whisper_model_name = actual_whisper_model
        print("✓ Whisper model loaded successfully!")
        
        # Ollama LLM
        print(f"✓ Ollama configured with model: {ollama_model}")
        self.ollama_model = ollama_model
        
        # Load TTS Manager (handles both Kokoro and Sherpa)
        print("Loading TTS engines...")
        
        if TTS_MANAGER_AVAILABLE:
            try:
                self.tts_manager = TTSManager(
                    language=language,
                    kokoro_model_path=kokoro_model_path,
                    voices_path=voices_path,
                    kokoro_voice=voice_name,
                    sherpa_model_dir=sherpa_model_dir,
                )
                self.tts_available = self.tts_manager.is_available()
            except Exception as e:
                print(f"Warning: Could not initialize TTS Manager: {e}")
                import traceback
                traceback.print_exc()
                self.tts_manager = None
                self.tts_available = False
        else:
            print("TTS Manager not available, TTS disabled")
            self.tts_manager = None
            self.tts_available = False
        
        # Conversation history for context
        self.conversation_history = []
        
        print("\n🎉 AI Manager initialized successfully!")
        print(f"   Language: {language}")
        print("Ready to chat!\n")
    
    def transcribe(self, audio_data, sample_rate=16000):
        """
        Transcribe audio to text using Whisper
        
        Args:
            audio_data: numpy array of audio
            sample_rate: audio sample rate
            
        Returns:
            Transcribed text string or empty string if filtered out
        """
        print("Transcribing audio...")
        
        # List of common Whisper hallucinations to filter out (language-specific)
        HALLUCINATION_PHRASES_EN = [
            'you', 'thank you', 'thanks', 'subtitle', 'subtitles',
            'mbc', 'bbc', 'thank you for watching', 'thanks for watching',
            'please subscribe', 'like and subscribe', 'bye', 'goodbye',
            'see you next time', 'see you', '.', '...'
        ]
        
        HALLUCINATION_PHRASES_ES = [
            'gracias', 'subtítulos', 'suscríbete', 'adiós', 'hasta luego',
            'gracias por ver', 'nos vemos', '.', '...'
        ]
        
        try:
            # Ensure audio is float32
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)
            
            # Determine language code for Whisper
            whisper_lang = "es" if self.language == "spanish" else "en"
            hallucinations = HALLUCINATION_PHRASES_ES if self.language == "spanish" else HALLUCINATION_PHRASES_EN
            
            print(f"Whisper transcribing with language='{whisper_lang}'...")
            
            # Use task="transcribe" explicitly and set language
            segments, info = self.whisper.transcribe(
                audio_data,
                language=whisper_lang,
                task="transcribe",  # Explicit transcription (not translation)
                beam_size=5,
                best_of=5,  # Better quality
                vad_filter=True,
                vad_parameters=dict(
                    min_silence_duration_ms=500,  # Shorter silence detection
                    speech_pad_ms=400,
                ),
                condition_on_previous_text=False,  # Prevent hallucinations
                no_speech_threshold=0.6,  # Filter out non-speech
                log_prob_threshold=-1.0,  # More lenient
                compression_ratio_threshold=2.4,
            )
            
            # Combine all segments
            text = " ".join([segment.text for segment in segments])
            text = text.strip()
            
            # Sanitization: Remove common hallucinations
            text_clean = text.lower().strip().strip('.,!?;:')
            
            if text_clean in hallucinations:
                print(f"Filtered hallucination: '{text}'")
                return ""
            
            # Check minimum length (at least 2 characters)
            if len(text_clean) < 2:
                print(f"Text too short, discarding: '{text}'")
                return ""
            
            print(f"✓ Transcription [{whisper_lang}]: {text}")
            return text
            
        except Exception as e:
            print(f"Transcription error: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def set_language(self, language: str):
        """
        Change the current language for STT and TTS
        
        Args:
            language: "english" or "spanish"
        """
        self.language = language
        print(f"🌐 AI Manager language set to: {language}")
        
        # Update TTS Manager language
        if self.tts_manager:
            self.tts_manager.set_language(language)
            self.tts_available = self.tts_manager.is_available()
    
    def get_llm_response(self, user_text):
        """
        Get response from Ollama LLM (non-streaming version for compatibility)
        
        Args:
            user_text: User's message
            
        Returns:
            Bot's response text
        """
        print("Getting LLM response...")
        
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_text
            })
            
            # Keep only last 10 messages to save memory
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            # Get response from Ollama
            response = ollama.chat(
                model=self.ollama_model,
                messages=self.conversation_history
            )
            
            bot_response = response['message']['content']
            
            # Add bot response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": bot_response
            })
            
            print(f"Bot response: {bot_response[:100]}...")
            return bot_response
            
        except Exception as e:
            print(f"LLM error: {e}")
            return "I'm sorry, I couldn't process that request."
    
    def get_llm_response_streaming(self, user_text, on_chunk=None, on_sentence=None):
        """
        Get response from Ollama LLM with streaming support
        
        Args:
            user_text: User's message
            on_chunk: Callback for each token chunk (full_text_so_far)
            on_sentence: Callback when a complete sentence is ready (sentence_text)
            
        Returns:
            Bot's full response text
        """
        print("Getting LLM response (streaming)...")
        
        try:
            # Add user message to history
            self.conversation_history.append({
                "role": "user",
                "content": user_text
            })
            
            # Keep only last 10 messages to save memory
            if len(self.conversation_history) > 10:
                self.conversation_history = self.conversation_history[-10:]
            
            # Stream response from Ollama
            full_response = ""
            current_sentence = ""
            sentence_delimiters = {'.', '!', '?', '\n'}
            
            stream = ollama.chat(
                model=self.ollama_model,
                messages=self.conversation_history,
                stream=True
            )
            
            for chunk in stream:
                token = chunk['message']['content']
                full_response += token
                current_sentence += token
                
                # Notify about new text chunk for UI update
                if on_chunk:
                    on_chunk(full_response)
                
                # Check if we have a complete sentence
                # Look for sentence-ending punctuation followed by space or end
                for delim in sentence_delimiters:
                    if delim in current_sentence:
                        # Split on delimiter, keeping the delimiter
                        parts = current_sentence.split(delim)
                        
                        # Process complete sentences (all but possibly the last part)
                        for i, part in enumerate(parts[:-1]):
                            complete_sentence = part.strip() + delim
                            if complete_sentence.strip() and len(complete_sentence.strip()) > 1:
                                if on_sentence:
                                    on_sentence(complete_sentence)
                        
                        # Keep the remainder for the next iteration
                        current_sentence = parts[-1]
                        break
            
            # Handle any remaining text as final sentence
            if current_sentence.strip():
                if on_sentence:
                    on_sentence(current_sentence.strip())
            
            # Add bot response to history
            self.conversation_history.append({
                "role": "assistant",
                "content": full_response
            })
            
            print(f"Bot response complete: {full_response[:100]}...")
            return full_response
            
        except Exception as e:
            print(f"LLM error: {e}")
            return "I'm sorry, I couldn't process that request."
    
    def text_to_speech(self, text):
        """
        Convert text to speech using the appropriate TTS engine
        (Kokoro for English, Sherpa for Spanish)
        
        Args:
            text: Text to synthesize
            
        Returns:
            tuple: (numpy array of audio samples, sample_rate)
        """
        print(f"Generating speech for: '{text[:50]}...'")
        
        if not self.tts_available or not self.tts_manager:
            print("TTS not available, returning silence")
            return np.zeros(24000, dtype=np.float32), 24000
        
        try:
            # Use TTS Manager to route to appropriate engine
            speed = 0.75 if self.language == "spanish" else 1.0
            samples, sample_rate = self.tts_manager.create(text, speed=speed)
            
            print(f"TTS returned: sample_rate={sample_rate}, samples={len(samples)}")
            
            # Convert to numpy array if needed
            if not isinstance(samples, np.ndarray):
                samples = np.array(samples, dtype=np.float32)
            
            # Ensure float32
            if samples.dtype != np.float32:
                samples = samples.astype(np.float32)
            
            print(f"Generated {len(samples)} audio samples at {sample_rate}Hz")
            print(f"Audio range: [{samples.min():.3f}, {samples.max():.3f}]")
            
            return samples, sample_rate
            
        except Exception as e:
            print(f"TTS error: {e}")
            import traceback
            traceback.print_exc()
            # Return silence as fallback
            return np.zeros(24000, dtype=np.float32), 24000
    
    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared")
