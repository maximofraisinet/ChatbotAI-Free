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

# Import TTS Manager (Kokoro unified)
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
                 kokoro_model_path="voices/kokoro-v1.0/kokoro-v1.0.onnx",
                 voices_path="voices/kokoro-v1.0/voices-v1.0.bin",
                 voice_name="af_bella",
                 language="english"):
        """
        Initialize all AI models
        
        Args:
            whisper_model: faster-whisper model size (use 'base' for multilingual)
            ollama_model: Ollama model name
            kokoro_model_path: Path to Kokoro v1.0 ONNX model
            voices_path: Path to unified voices.json (English + Spanish)
            voice_name: Voice to use (af_bella, ef_dora, etc.)
            language: Current language ("english" or "spanish")
        """
        print("Initializing AI Manager...")
        
        self.language = language
        self.current_voice = voice_name
        
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
        
        # Load TTS Manager (unified Kokoro for all languages)
        print("Loading TTS engine...")
        
        if TTS_MANAGER_AVAILABLE:
            try:
                self.tts_manager = TTSManager(
                    language=language,
                    kokoro_model_path=kokoro_model_path,
                    voices_path=voices_path,
                    voice_name=voice_name,
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
        # Token usage from last LLM response (updated after each stream)
        self.last_token_usage = {"prompt": 0, "completion": 0, "total": 0}
        
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
    
    def set_voice(self, voice_name: str):
        """
        Change the current voice.
        
        Args:
            voice_name: Voice name (e.g., "af_bella" for English, "ef_dora" for Spanish)
        """
        self.current_voice = voice_name
        
        if self.tts_manager:
            self.tts_manager.set_voice(voice_name)
        
        print(f"🎤 Voice changed to: {voice_name}")
    
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
    
    def get_llm_response_streaming(self, user_text, on_chunk=None, on_sentence=None, on_thinking=None):
        """
        Get response from Ollama LLM with streaming support.
        Handles <think>...</think> blocks: thinking content is routed to on_thinking
        while the actual response goes to on_chunk / on_sentence.

        Args:
            user_text: User's message
            on_chunk: Callback for each token chunk of the final response (text_so_far)
            on_sentence: Callback when a complete sentence of the final response is ready
            on_thinking: Callback with accumulated thinking text so far (thinking_so_far)

        Returns:
            Bot's full response text (excluding thinking block)
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

            # ── Streaming state ────────────────────────────────────────────────
            THINK_OPEN  = "<think>"
            THINK_CLOSE = "</think>"
            GUARD_OPEN  = len(THINK_OPEN)
            GUARD_CLOSE = len(THINK_CLOSE)

            in_think         = False
            buffer           = ""
            thinking_text    = ""
            response_text    = ""
            current_sentence = ""
            sentence_delimiters = {'.', '!', '?', '\n'}

            def _emit_response(text_piece):
                nonlocal response_text, current_sentence
                if not text_piece:
                    return
                response_text    += text_piece
                current_sentence += text_piece
                if on_chunk:
                    on_chunk(response_text)
                for delim in sentence_delimiters:
                    if delim in current_sentence:
                        parts = current_sentence.split(delim)
                        for part in parts[:-1]:
                            complete = part.strip() + delim
                            if complete.strip() and len(complete.strip()) > 1:
                                if on_sentence:
                                    on_sentence(complete)
                        current_sentence = parts[-1]
                        break

            def _emit_thinking(text_piece):
                nonlocal thinking_text
                if not text_piece:
                    return
                thinking_text += text_piece
                if on_thinking:
                    on_thinking(thinking_text)

            def _process_stream(use_think: bool):
                """Run the stream loop. Returns True on success, raises on real errors."""
                stream = ollama.chat(
                    model=self.ollama_model,
                    messages=self.conversation_history,
                    stream=True,
                    **({"think": True} if use_think else {}),
                )
                for chunk in stream:
                    # Capture token stats from the final 'done' chunk
                    if chunk.get('done'):
                        p = chunk.get('prompt_eval_count') or 0
                        c = chunk.get('eval_count') or 0
                        self.last_token_usage = {
                            "prompt": p,
                            "completion": c,
                            "total": p + c,
                        }
                    msg = chunk['message']

                    # Path 1: Ollama native thinking field
                    native_thinking = msg.get('thinking') or ''
                    if native_thinking:
                        _emit_thinking(native_thinking)

                    # Path 2: Content (may contain <think> tags as fallback)
                    content_piece = msg.get('content') or ''
                    if not content_piece:
                        continue

                    nonlocal buffer, in_think
                    buffer += content_piece

                    while True:
                        if not in_think:
                            tag_pos = buffer.find(THINK_OPEN)
                            if tag_pos == -1:
                                safe_end = max(0, len(buffer) - GUARD_OPEN)
                                if safe_end > 0:
                                    _emit_response(buffer[:safe_end])
                                    buffer = buffer[safe_end:]
                                break
                            else:
                                _emit_response(buffer[:tag_pos])
                                buffer = buffer[tag_pos + GUARD_OPEN:]
                                in_think = True
                                print("🧠 Thinking block started (<think> tag)")
                        else:
                            tag_pos = buffer.find(THINK_CLOSE)
                            if tag_pos == -1:
                                safe_end = max(0, len(buffer) - GUARD_CLOSE)
                                if safe_end > 0:
                                    _emit_thinking(buffer[:safe_end])
                                    buffer = buffer[safe_end:]
                                break
                            else:
                                _emit_thinking(buffer[:tag_pos])
                                buffer = buffer[tag_pos + GUARD_CLOSE:]
                                in_think = False
                                print("🧠 Thinking block ended")

            # Try with thinking; if the model rejects it, retry without
            try:
                _process_stream(use_think=True)
            except Exception as think_err:
                err_str = str(think_err).lower()
                if '400' in err_str or 'thinking' in err_str or 'does not support' in err_str:
                    print(f"Model does not support thinking, retrying without: {think_err}")
                    # Reset state for clean retry
                    buffer = ""
                    in_think = False
                    thinking_text = ""
                    response_text = ""
                    current_sentence = ""
                    _process_stream(use_think=False)
                else:
                    raise

            # ── Flush remaining buffer ─────────────────────────────────────────
            if buffer:
                if in_think:
                    _emit_thinking(buffer)
                else:
                    _emit_response(buffer)

            # Emit any remaining partial sentence
            if current_sentence.strip():
                if on_sentence:
                    on_sentence(current_sentence.strip())

            # Save only the response (not the thinking) to conversation history
            self.conversation_history.append({
                "role": "assistant",
                "content": response_text
            })

            print(f"Bot response complete: {response_text[:100]}...")
            return response_text

        except Exception as e:
            print(f"LLM error: {e}")
            return "I'm sorry, I couldn't process that request."
    
    def text_to_speech(self, text, speed=1.0):
        """
        Convert text to speech using Kokoro TTS
        
        Args:
            text: Text to synthesize
            speed: Voice speed multiplier (default 1.0)
            
        Returns:
            tuple: (numpy array of audio samples, sample_rate)
        """
        print(f"Generating speech for: '{text[:50]}...'")
        
        if not self.tts_available or not self.tts_manager:
            print("TTS not available, returning silence")
            return np.zeros(24000, dtype=np.float32), 24000
        
        try:
            # Use TTS Manager to route to appropriate engine with user-defined speed
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
    
    def get_model_context_size(self):
        """Return the context-window size (num_ctx) of the current model."""
        if hasattr(self, '_ctx_cache') and self._ctx_cache[0] == self.ollama_model:
            return self._ctx_cache[1]
        try:
            info = ollama.show(self.ollama_model)
            size = 4096  # sensible default
            model_info = getattr(info, 'model_info', None) or {}
            found = False
            for key, val in model_info.items():
                if 'context_length' in key.lower():
                    size = int(val)
                    found = True
                    break
            if not found:
                import re as _re
                modelfile = getattr(info, 'modelfile', '') or ''
                m = _re.search(r'PARAMETER\s+num_ctx\s+(\d+)', modelfile, _re.IGNORECASE)
                if m:
                    size = int(m.group(1))
            self._ctx_cache = (self.ollama_model, size)
            print(f"Model context size for {self.ollama_model}: {size} tokens")
            return size
        except Exception as e:
            print(f"Could not query model context size: {e}")
            return 4096

    def reset_conversation(self):
        """Clear conversation history"""
        self.conversation_history = []
        print("Conversation history cleared")
