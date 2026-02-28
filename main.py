"""
Voice Chatbot - Google Gemini Inspired UI
Walkie-Talkie Mode: Manual control of recording
"""

import sys
import numpy as np
import threading
import queue
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QHBoxLayout, QLabel, QScrollArea, QPushButton,
    QComboBox, QFrame, QSpacerItem, QSizePolicy, QTextEdit,
    QDialog, QCheckBox, QStackedWidget, QGraphicsDropShadowEffect,
    QTextBrowser, QSlider, QRadioButton, QButtonGroup
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot, QTimer, QPropertyAnimation, QEasingCurve, QRect
from PyQt6.QtGui import QFont, QColor, QPainter, QPen

from styles import GEMINI_STYLE, COLORS
from audio_utils import AudioRecorder, AudioPlayer
from ai_manager import AIManager
from preferences import load_preferences, save_preferences, get_font_size_config, FONT_SIZES, get_language_config, get_available_languages, LANGUAGES
import ollama


def get_available_ollama_models():
    """Get list of available Ollama models"""
    try:
        response = ollama.list()
        model_names = [model.model for model in response.models]
        print(f"Found Ollama models: {model_names}")
        return model_names if model_names else ['llama3.1:8b']
    except Exception as e:
        print(f"Error getting Ollama models: {e}")
        return ['llama3.1:8b']


class ManualRecorderThread(QThread):
    """Thread for manual recording (walkie-talkie style)"""
    
    recording_finished = pyqtSignal(object)  # Emits recorded audio data
    
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder
        self.is_recording = False
        self.audio_chunks = []
    
    def run(self):
        """Record until stop() is called"""
        self.is_recording = True
        self.audio_chunks = []
        self.recorder.start_stream()
        
        print("🎤 Recording started...")
        
        while self.is_recording:
            try:
                chunk = self.recorder.audio_queue.get(timeout=0.1)
                self.audio_chunks.append(chunk)
            except:
                continue
        
        self.recorder.stop_stream()
        
        if self.audio_chunks:
            audio_data = np.concatenate(self.audio_chunks, axis=0).flatten()
            record_rate = getattr(self.recorder, '_record_rate', self.recorder.sample_rate)
            duration = len(audio_data) / record_rate
            print(f"🎤 Recording stopped. Duration: {duration:.2f}s at {record_rate}Hz")

            # Resample to target rate (16000 Hz) if recorded at a different rate
            if record_rate != self.recorder.sample_rate:
                new_length = int(len(audio_data) * self.recorder.sample_rate / record_rate)
                old_idx = np.arange(len(audio_data))
                new_idx = np.linspace(0, len(audio_data) - 1, new_length)
                audio_data = np.interp(new_idx, old_idx, audio_data).astype(np.float32)
                print(f"Resampled mic audio {record_rate}Hz → {self.recorder.sample_rate}Hz")

            # Only emit if we have meaningful audio (> 0.5 seconds)
            if duration > 0.5:
                self.recording_finished.emit(audio_data)
            else:
                print("Recording too short, discarding")
                self.recording_finished.emit(None)
        else:
            self.recording_finished.emit(None)
    
    def stop_recording(self):
        """Stop the recording"""
        self.is_recording = False


class WorkerThread(QThread):
    """Thread for processing: Transcribe -> Think -> Speak with real-time streaming"""
    
    status_changed = pyqtSignal(str)
    user_message = pyqtSignal(str)
    bot_message = pyqtSignal(str)
    bot_message_update = pyqtSignal(str)       # For streaming text updates
    bot_thinking_update = pyqtSignal(str)      # Streaming thinking content
    processing_complete = pyqtSignal()
    speaking_started = pyqtSignal()  # Signal when TTS starts playing
    context_usage_updated = pyqtSignal(int, int)  # (used_tokens, model_ctx_size)
    
    def __init__(self, ai_manager, audio_player, voice_speed=1.0):
        super().__init__()
        self.ai_manager = ai_manager
        self.audio_player = audio_player
        self.audio_data = None
        self.text_input = None  # For text messages
        self.interrupted = False
        self.voice_speed = voice_speed
    
    def set_audio(self, audio_data):
        """Set audio data to process"""
        self.audio_data = audio_data
        self.text_input = None
    
    def set_text(self, text):
        """Set text to process (skip transcription)"""
        self.text_input = text
        self.audio_data = None
    
    def interrupt(self):
        """Interrupt the current processing (stop audio)"""
        self.interrupted = True
        self.audio_player.stop()
        print("⚠️ Interrupted by user")
    
    def run(self):
        """Process the audio or text through the pipeline with streaming"""
        self.interrupted = False
        
        # Determine user text source
        if self.text_input:
            user_text = self.text_input
            self.user_message.emit(user_text)
        elif self.audio_data is not None:
            # Step 1: Transcribe
            self.status_changed.emit("Transcribing...")
            user_text = self.ai_manager.transcribe(self.audio_data)
            
            if not user_text:
                self.status_changed.emit("Ready")
                self.processing_complete.emit()
                return
            
            self.user_message.emit(user_text)
        else:
            self.processing_complete.emit()
            return
        
        if self.interrupted:
            self.processing_complete.emit()
            return
        
        # Step 2 & 3: Stream LLM response and generate TTS in parallel
        self.status_changed.emit("Thinking...")
        
        # Queue for sentences ready for TTS
        sentence_queue = queue.Queue()
        # Queue for audio ready to play
        audio_queue = queue.Queue(maxsize=3)
        
        llm_complete = threading.Event()
        tts_complete = threading.Event()
        first_audio_ready = threading.Event()
        
        # Track the full response for history
        full_response = [""]
        bot_bubble_created = [False]
        
        def on_thinking_chunk(thinking_so_far):
            """Called for each thinking token - update thinking panel"""
            if self.interrupted:
                return
            # Create bot bubble (empty) the first time thinking arrives
            if not bot_bubble_created[0]:
                self.bot_message.emit("")
                bot_bubble_created[0] = True
            self.bot_thinking_update.emit(thinking_so_far)

        def on_text_chunk(text_so_far):
            """Called for each token - update UI in real-time"""
            if self.interrupted:
                return
            full_response[0] = text_so_far
            
            # Create or update the bot bubble
            if not bot_bubble_created[0]:
                self.bot_message.emit(text_so_far)
                bot_bubble_created[0] = True
            else:
                self.bot_message_update.emit(text_so_far)
        
        def on_sentence_ready(sentence):
            """Called when a complete sentence is ready for TTS"""
            if self.interrupted:
                return
            clean_sentence = self._clean_markdown(sentence)
            if clean_sentence.strip():
                sentence_queue.put(clean_sentence)
        
        # Thread 1: Stream LLM response
        def stream_llm():
            try:
                self.ai_manager.get_llm_response_streaming(
                    user_text,
                    on_chunk=on_text_chunk,
                    on_sentence=on_sentence_ready,
                    on_thinking=on_thinking_chunk
                )
            except Exception as e:
                print(f"LLM streaming error: {e}")
            finally:
                llm_complete.set()
        
        # Thread 2: Generate TTS for sentences as they arrive
        def generate_tts():
            while not (llm_complete.is_set() and sentence_queue.empty()):
                if self.interrupted:
                    break
                
                try:
                    sentence = sentence_queue.get(timeout=0.1)
                    if sentence and not self.interrupted:
                        audio_output, sample_rate = self.ai_manager.text_to_speech(sentence, speed=self.voice_speed)
                        if audio_output is not None and len(audio_output) > 0:
                            audio_queue.put((audio_output, sample_rate))
                            if not first_audio_ready.is_set():
                                first_audio_ready.set()
                except queue.Empty:
                    continue
                except Exception as e:
                    print(f"TTS error: {e}")
            
            tts_complete.set()
        
        # Start LLM and TTS threads
        llm_thread = threading.Thread(target=stream_llm, daemon=True)
        tts_thread = threading.Thread(target=generate_tts, daemon=True)
        
        llm_thread.start()
        tts_thread.start()
        
        # Wait for first audio to be ready, then start playing
        first_audio_ready.wait(timeout=30)  # Max 30 seconds for first response
        
        if not self.interrupted:
            self.status_changed.emit("Speaking...")
            self.speaking_started.emit()
        
        # Play audio chunks as they become available
        while not (tts_complete.is_set() and audio_queue.empty()):
            if self.interrupted:
                break
            
            try:
                audio_output, sample_rate = audio_queue.get(timeout=0.3)
                if not self.interrupted:
                    self.audio_player.play(audio_output, sample_rate)
            except queue.Empty:
                if tts_complete.is_set():
                    break
                continue
        
        # Wait for threads to finish
        llm_thread.join(timeout=1.0)
        tts_thread.join(timeout=1.0)
        
        self.status_changed.emit("Ready")
        # Emit context usage for the donut indicator
        usage = self.ai_manager.last_token_usage
        ctx_size = self.ai_manager.get_model_context_size()
        self.context_usage_updated.emit(usage.get('total', 0), ctx_size)
        self.processing_complete.emit()
    
    def _clean_markdown(self, text):
        """Remove markdown symbols like *, **, etc. and emojis"""
        import re
        
        # Remove emojis (Unicode emoji ranges)
        # This covers most common emojis including skin tone modifiers
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002600-\U000026FF"  # Miscellaneous Symbols
            "\U00002700-\U000027BF"  # Dingbats
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        
        # Remove bold/italic markers
        text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)  # ***text***
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)  # **text**
        text = re.sub(r'\*(.+?)\*', r'\1', text)  # *text*
        text = re.sub(r'__(.+?)__', r'\1', text)  # __text__
        text = re.sub(r'_(.+?)_', r'\1', text)  # _text_
        
        # Remove remaining asterisks
        text = text.replace('*', '')
        text = text.replace('_', '')
        
        # Remove markdown headers
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        
        # Remove code blocks
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`(.+?)`', r'\1', text)
        
        return text.strip()


class MarkdownRenderer:
    """Utility class to convert Markdown to HTML for display"""
    
    @staticmethod
    def to_html(text, font_size=15):
        """Convert markdown text to styled HTML"""
        import re
        
        # Escape HTML special characters first (except for what we'll process)
        text = text.replace('&', '&amp;')
        text = text.replace('<', '&lt;')
        text = text.replace('>', '&gt;')
        
        # Code blocks (```code```)
        def code_block_replace(match):
            lang = match.group(1) or ''
            code = match.group(2).strip()
            return f'''<div style="background-color: #1E1F20; border: 1px solid #3C4043; 
                       border-radius: 8px; padding: 12px; margin: 8px 0; font-family: 'Consolas', 'Monaco', monospace; 
                       font-size: {font_size - 1}px; overflow-x: auto; white-space: pre-wrap;">
                       <code>{code}</code></div>'''
        text = re.sub(r'```(\w*)\n?(.*?)```', code_block_replace, text, flags=re.DOTALL)

        # Markdown tables  (must run before newline conversion)
        def table_replace(match):
            lines = [l for l in match.group(0).split('\n') if l.strip()]
            if len(lines) < 3:
                return match.group(0)
            headers = [h.strip() for h in lines[0].strip('|').split('|')]
            rows = []
            for line in lines[2:]:
                cells = [c.strip() for c in line.strip('|').split('|')]
                rows.append(cells)
            th_html = ''.join(f'<th>{h}</th>' for h in headers)
            tr_html = ''.join(
                f'<tr class="{"tr-even" if i % 2 == 0 else "tr-odd"}">'
                + ''.join(f'<td>{c}</td>' for c in row)
                + '</tr>'
                for i, row in enumerate(rows)
            )
            return f'<table><thead><tr>{th_html}</tr></thead><tbody>{tr_html}</tbody></table>\n'
        text = re.sub(
            r'(?m)^\|[^\n]+\|\s*\n\|[-|: \t]+\|\s*\n(?:\|[^\n]+\|\s*\n?)+',
            table_replace, text
        )

        # Inline code (`code`)
        text = re.sub(r'`([^`]+)`', 
                     rf'<code style="background-color: #282A2C; padding: 2px 6px; border-radius: 4px; font-family: monospace; font-size: {font_size - 1}px;">\1</code>', 
                     text)
        
        # Horizontal rules (--- or ***)
        text = re.sub(r'^---+$', '<hr style="border: none; border-top: 1px solid #3C4043; margin: 8px 0;">', text, flags=re.MULTILINE)
        text = re.sub(r'^\*\*\*+$', '<hr style="border: none; border-top: 1px solid #3C4043; margin: 8px 0;">', text, flags=re.MULTILINE)
        
        # Headers (#### Header, ### Header, etc.)
        text = re.sub(r'^#### (.+)$', rf'<h4 style="font-size: {font_size + 2}px; font-weight: 600; margin: 8px 0 4px 0; color: #E3E3E3;">\1</h4>', text, flags=re.MULTILINE)
        text = re.sub(r'^### (.+)$', rf'<h3 style="font-size: {font_size + 3}px; font-weight: 600; margin: 10px 0 5px 0; color: #E3E3E3;">\1</h3>', text, flags=re.MULTILINE)
        text = re.sub(r'^## (.+)$', rf'<h2 style="font-size: {font_size + 5}px; font-weight: 600; margin: 12px 0 6px 0; color: #E3E3E3;">\1</h2>', text, flags=re.MULTILINE)
        text = re.sub(r'^# (.+)$', rf'<h1 style="font-size: {font_size + 8}px; font-weight: 700; margin: 14px 0 8px 0; color: #E3E3E3;">\1</h1>', text, flags=re.MULTILINE)
        
        # Bold (**text** or __text__)
        text = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', text)
        text = re.sub(r'__(.+?)__', r'<strong>\1</strong>', text)
        
        # Italic (*text* or _text_)
        text = re.sub(r'\*([^\*]+)\*', r'<em>\1</em>', text)
        text = re.sub(r'_([^_]+)_', r'<em>\1</em>', text)
        
        # Lists disabled - showing as plain text for now
        # Will keep the - or * or 1. 2. etc. as plain text
        
        # # Unordered lists (- item or * item) - DISABLED
        # # Ordered lists (1. item) - DISABLED
        
        # Clean up extra newlines around block elements (lists, headers, hr)
        text = re.sub(r'\n+(</?[uoh][lrl1-4]?>)', r'\1', text)  # Remove newlines before/after list/header tags
        text = re.sub(r'(</?[uoh][lrl1-4]?>)\n+', r'\1', text)
        text = re.sub(r'\n+(<hr[^>]*>)', r'\1', text)  # Remove newlines before hr
        text = re.sub(r'(<hr[^>]*>)\n+', r'\1', text)  # Remove newlines after hr
        text = re.sub(r'(</ul>|</ol>)\n+', r'\1', text)  # Remove newlines after closing list tags
        text = re.sub(r'\n+(</li>)', r'\1', text)  # Remove newlines before closing li tags
        
        # Paragraphs (double newline) - but not around block elements
        text = re.sub(r'\n\n+', '</p><p style="margin: 8px 0; line-height: 1.6;">', text)
        
        # Single newlines to <br> - but not after block elements
        # First, protect block elements with a marker
        text = re.sub(r'(</ul>|</ol>|</h[1-4]>|<hr[^>]*>)\n', r'\1<!-- BLOCK -->', text)
        text = re.sub(r'\n(<ul|<ol|<h[1-4]|<hr)', r'<!-- BLOCK -->\1', text)
        
        # Now convert remaining newlines to <br>
        text = text.replace('\n', '<br>')
        
        # Remove the markers
        text = text.replace('<!-- BLOCK -->', '')
        
        # Wrap in paragraph (with minimal margin)
        html = f'''<div style="margin: 0; line-height: 1.6;">{text}</div>'''

        return html

    @staticmethod
    def extract_tables(text):
        """Return list of raw markdown table strings found in text."""
        import re
        return re.findall(
            r'(?m)^\|[^\n]+\|\s*\n\|[-|: \t]+\|\s*\n(?:\|[^\n]+\|\s*\n?)+',
            text
        )


class ThinkingWidget(QWidget):
    """Collapsible panel showing the model's thinking/reasoning in streaming mode.
    Hidden until thinking content arrives.  Collapsed by default."""

    LABEL_COLLAPSED = "✨ Show Reasoning  ▾"
    LABEL_EXPANDED  = "✨ Hide Reasoning  ▴"

    def __init__(self, parent=None):
        super().__init__(parent)
        self._expanded = False
        self._has_content = False
        self.setStyleSheet("background-color: transparent;")
        self.setVisible(False)  # hidden until thinking content actually arrives

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 2, 0, 6)
        layout.setSpacing(4)

        # ── Toggle button ───────────────────────────────────────────────────
        self.toggle_btn = QPushButton(self.LABEL_COLLAPSED)
        self.toggle_btn.setFlat(True)
        self.toggle_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                color: #9AA0A6;
                border: none;
                text-align: left;
                font-size: 12px;
                padding: 0px;
            }
            QPushButton:hover { color: #8AB4F8; }
        """)
        self.toggle_btn.clicked.connect(self.toggle)
        layout.addWidget(self.toggle_btn)

        # ── Content area ────────────────────────────────────────────────────
        self.content = QTextBrowser()
        self.content.setOpenExternalLinks(False)
        self.content.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.content.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.content.setFrameShape(QFrame.Shape.NoFrame)
        self.content.setMaximumHeight(220)
        self.content.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        self.content.setStyleSheet("""
            QTextBrowser {
                background-color: #1A1B1E;
                border: 1px solid #3C4043;
                border-radius: 8px;
                color: #9AA0A6;
                font-size: 12px;
                padding: 8px 10px;
                font-style: italic;
                selection-background-color: #3C4043;
            }
            QScrollBar:vertical {
                background: #2A2B2E;
                width: 6px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: #5F6368;
                border-radius: 3px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical,
            QScrollBar::sub-line:vertical { height: 0px; }
        """)
        self.content.setVisible(False)
        layout.addWidget(self.content)

    # ── Public API ──────────────────────────────────────────────────────────
    def update_thinking(self, text: str):
        """Set accumulated thinking text (call on every streaming update)."""
        self._has_content = True
        self.content.setPlainText(text)
        # Auto-scroll to the bottom while streaming, only when expanded
        if self._expanded:
            sb = self.content.verticalScrollBar()
            sb.setValue(sb.maximum())
        # Show the widget (button) the first time real content arrives
        if not self.isVisible():
            self.setVisible(True)

    def toggle(self):
        """Expand / collapse the content area."""
        self._expanded = not self._expanded
        self.content.setVisible(self._expanded)
        self.toggle_btn.setText(
            self.LABEL_EXPANDED if self._expanded else self.LABEL_COLLAPSED
        )
        if self._expanded:
            sb = self.content.verticalScrollBar()
            sb.setValue(sb.maximum())


class UserMessageBubble(QFrame):
    """User message with bubble style - aligned right"""
    
    def __init__(self, text, font_size=15):
        super().__init__()
        self.setObjectName("userBubble")
        self.font_size = font_size
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(text)
        self.label.setObjectName("userBubbleText")
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.update_font_size(font_size)
        
        layout.addWidget(self.label)
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
        self.setMaximumWidth(900)
    
    def update_text(self, text):
        """Update the bubble text content"""
        self.label.setText(text)
    
    def update_font_size(self, font_size):
        """Update the font size"""
        self.font_size = font_size
        self.label.setStyleSheet(f"""
            color: #FFFFFF;
            font-size: {font_size}px;
            padding: 14px 20px;
            background: transparent;
            line-height: 1.5;
        """)
        self.setStyleSheet(f"""
            QFrame#userBubble {{
                background-color: #303136;
                border-radius: 22px;
                border: none;
            }}
        """)


class ContextPopup(QFrame):
    """Floating popup card shown when the user clicks the ContextDonut."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.WindowType.Popup)
        self.setObjectName("contextPopup")
        self.setStyleSheet("""
            QFrame#contextPopup {
                background-color: #202124;
                border: 1px solid #5C6166;
                border-radius: 10px;
            }
        """)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(6)

        self._title = QLabel("Context Window")
        self._title.setStyleSheet(
            "font-size:14px;font-weight:700;color:#E3E3E3;background:transparent;border:none;"
        )
        layout.addWidget(self._title)

        self._desc = QLabel(
            "The model can only see the last N tokens of your\n"
            "conversation. Once full, older messages are dropped."
        )
        self._desc.setStyleSheet(
            "font-size:11px;color:#9AA0A6;background:transparent;border:none;"
        )
        self._desc.setWordWrap(True)
        layout.addWidget(self._desc)

        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background:#3C4043;border:none;max-height:1px;")
        layout.addWidget(sep)

        self._usage_label = QLabel("No data yet")
        self._usage_label.setStyleSheet(
            "font-size:13px;font-weight:600;color:#9AA0A6;background:transparent;border:none;"
        )
        layout.addWidget(self._usage_label)

        self._hint = QLabel("Send a message to start tracking.")
        self._hint.setStyleSheet(
            "font-size:11px;color:#9AA0A6;background:transparent;border:none;"
        )
        self._hint.setWordWrap(True)
        layout.addWidget(self._hint)

        self.setFixedWidth(280)

    @staticmethod
    def _fmt_k(n: int) -> str:
        return f"{n / 1000:.1f}K" if n >= 1000 else str(n)

    def refresh(self, used: int, total: int):
        pct = int(min(used / max(total, 1), 1.0) * 100)
        u_str = self._fmt_k(used)
        t_str = self._fmt_k(total) if total > 0 else "—"

        if total == 0:
            hint = "Send a message to start tracking."
            color = "#9AA0A6"
            usage_text = "No data yet"
        elif pct < 50:
            hint = f"You're well within limits — {100 - pct}% still available."
            color = "#34A853"
            usage_text = f"{u_str} / {t_str} tokens  ({pct}%)"
        elif pct < 80:
            hint = f"{100 - pct}% remaining. Consider starting a new chat soon."
            color = "#FBBC04"
            usage_text = f"{u_str} / {t_str} tokens  ({pct}%)"
        else:
            hint = "Almost full! Start a new chat to avoid losing early context."
            color = "#EA4335"
            usage_text = f"{u_str} / {t_str} tokens  ({pct}%)"

        self._usage_label.setText(usage_text)
        self._usage_label.setStyleSheet(
            f"font-size:13px;font-weight:600;color:{color};background:transparent;border:none;"
        )
        self._hint.setText(hint)
        self.adjustSize()


class ContextDonut(QWidget):
    """Small circular indicator showing context-window usage as a filled arc."""
    _SIZE = 40
    _THICK = 6

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self._SIZE, self._SIZE)
        self._used = 0
        self._total = 0
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Click to see context window details")
        self._popup = ContextPopup()

    @staticmethod
    def _fmt_k(n: int) -> str:
        return f"{n / 1000:.1f}K" if n >= 1000 else str(n)

    def update_usage(self, used: int, total: int):
        self._used = used
        self._total = max(total, 1)
        self._popup.refresh(used, total)
        self.update()

    def mousePressEvent(self, event):
        if self._popup.isVisible():
            self._popup.hide()
            return
        self._popup.refresh(self._used, self._total)
        # Position popup above the donut, right-aligned
        global_pos = self.mapToGlobal(self.rect().topRight())
        popup_x = global_pos.x() - self._popup.sizeHint().width()
        popup_y = global_pos.y() - self._popup.sizeHint().height() - 8
        self._popup.move(popup_x, popup_y)
        self._popup.show()
        super().mousePressEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        m = self._THICK + 3
        from PyQt6.QtCore import QRect
        rect = QRect(m, m, self._SIZE - 2 * m, self._SIZE - 2 * m)

        ratio = min(self._used / max(self._total, 1), 1.0)

        # Background full circle
        bg_pen = QPen(QColor('#3C4043'))
        bg_pen.setWidth(self._THICK)
        bg_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
        painter.setPen(bg_pen)
        painter.drawArc(rect, 0, 360 * 16)

        # Colored arc proportional to usage
        if ratio > 0:
            if ratio < 0.5:
                arc_color = QColor('#34A853')   # green
            elif ratio < 0.80:
                arc_color = QColor('#FBBC04')   # yellow
            else:
                arc_color = QColor('#EA4335')   # red
            arc_pen = QPen(arc_color)
            arc_pen.setWidth(self._THICK)
            arc_pen.setCapStyle(Qt.PenCapStyle.FlatCap)
            painter.setPen(arc_pen)
            # Qt: 90*16 = top, negative span = clockwise
            painter.drawArc(rect, 90 * 16, -int(ratio * 360 * 16))

        # Center percentage text
        text = f"{int(ratio * 100)}%" if self._total > 0 else "?"
        from PyQt6.QtGui import QFont as _QFont
        f = _QFont()
        f.setPixelSize(9)
        f.setBold(True)
        painter.setFont(f)
        painter.setPen(QPen(QColor('#9AA0A6')))
        painter.drawText(rect, Qt.AlignmentFlag.AlignCenter, text)
        painter.end()


class CustomTextEdit(QTextEdit):
    """QTextEdit personalizado que envía con Enter y permite nueva línea con Shift+Enter"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.send_callback = None
    
    def keyPressEvent(self, event):
        """Manejar Enter para enviar, Shift+Enter para nueva línea"""
        if event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                # Shift+Enter: insertar nueva línea
                super().keyPressEvent(event)
            else:
                # Enter solo: enviar mensaje
                if self.send_callback:
                    self.send_callback()
                event.accept()
        else:
            super().keyPressEvent(event)


class BotMessageWidget(QFrame):
    """Bot message with avatar and markdown-rendered text - no bubble"""
    
    def __init__(self, text, font_size=15):
        super().__init__()
        self.setObjectName("botMessage")
        self.font_size = font_size
        self._raw_text = text  # Store raw text for TTS
        
        self.setStyleSheet("background-color: transparent; border: none;")
        
        # Main horizontal layout: Avatar | Text
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(0, 8, 0, 8)
        main_layout.setSpacing(12)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        
        # Avatar
        self.avatar = QLabel("✨")
        self.avatar.setObjectName("botAvatar")
        self.avatar.setFixedSize(32, 32)
        self.avatar.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.avatar.setStyleSheet("""
            QLabel#botAvatar {
                background-color: #8AB4F8;
                border-radius: 16px;
                color: #131314;
                font-size: 16px;
            }
        """)
        main_layout.addWidget(self.avatar, 0, Qt.AlignmentFlag.AlignTop)
        
        # Text content container
        text_container = QWidget()
        text_container.setStyleSheet("background-color: transparent;")
        text_layout = QVBoxLayout(text_container)
        text_layout.setContentsMargins(0, 0, 0, 0)
        text_layout.setSpacing(0)
        # Collapsible thinking panel (hidden until thinking content arrives)
        self.thinking_widget = ThinkingWidget(text_container)
        text_layout.addWidget(self.thinking_widget)

        # Markdown-rendered text using QTextBrowser
        self.text_browser = QTextBrowser()
        self.text_browser.setObjectName("markdownText")
        self.text_browser.setOpenExternalLinks(True)
        self.text_browser.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_browser.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_browser.setFrameShape(QFrame.Shape.NoFrame)
        self.text_browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        
        self.update_font_size(font_size)
        self.update_text(text)
        
        text_layout.addWidget(self.text_browser)
        main_layout.addWidget(text_container, 1)
        
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
    
    def update_text(self, text):
        """Update the message text with markdown rendering"""
        self._raw_text = text
        html_content = MarkdownRenderer.to_html(text, self.font_size)
        
        # Full HTML document with styling
        full_html = f'''
        <html>
        <head>
            <style>
                body {{
                    color: #E3E3E3;
                    font-family: 'Google Sans', 'Segoe UI', 'Roboto', Arial, sans-serif;
                    font-size: {self.font_size}px;
                    line-height: 1.5;
                    margin: 0;
                    padding: 0;
                    background-color: transparent;
                }}
                ul, ol {{
                    margin: 2px 0;
                    padding-left: 20px;
                    line-height: 1.2;
                }}
                li {{
                    margin: 0;
                    padding: 0;
                    line-height: 1.2;
                }}
                strong {{
                    font-weight: 600;
                    color: #FFFFFF;
                }}
                em {{
                    font-style: italic;
                }}
                code {{
                    background-color: #282A2C;
                    padding: 2px 6px;
                    border-radius: 4px;
                    font-family: 'Consolas', 'Monaco', monospace;
                    font-size: {self.font_size - 1}px;
                }}
                a {{
                    color: #8AB4F8;
                    text-decoration: none;
                }}
                a:hover {{
                    text-decoration: underline;
                }}
                h1, h2, h3, h4 {{
                    margin: 8px 0 4px 0;
                }}
                hr {{
                    border: none;
                    border-top: 1px solid #3C4043;
                    margin: 6px 0;
                }}
                p {{
                    margin: 4px 0;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 10px 0;
                }}
                th {{
                    background-color: #303136;
                    color: #FFFFFF;
                    padding: 8px 14px;
                    text-align: left;
                    font-weight: 600;
                    border: 1px solid #5C6166;
                    font-size: {self.font_size}px;
                }}
                td {{
                    padding: 6px 14px;
                    border: 1px solid #3C4043;
                    color: #E3E3E3;
                    font-size: {self.font_size}px;
                }}
                tr.tr-even td {{
                    background-color: #26272B;
                }}
                tr.tr-odd td {{
                    background-color: #1E1F20;
                }}
            </style>
        </head>
        <body>{html_content}</body>
        </html>
        '''
        
        self.text_browser.setHtml(full_html)

        # Adjust height to content
        self.text_browser.document().setTextWidth(self.text_browser.viewport().width())
        doc_height = self.text_browser.document().size().height()
        self.text_browser.setMinimumHeight(int(doc_height) + 10)

    def update_font_size(self, font_size):
        """Update the font size"""
        self.font_size = font_size
        self.text_browser.setStyleSheet(f"""
            QTextBrowser#markdownText {{
                background-color: transparent;
                border: none;
                color: #E3E3E3;
                font-size: {font_size}px;
                selection-background-color: #3C4043;
            }}
        """)
        # Re-render text with new font size
        if hasattr(self, '_raw_text'):
            self.update_text(self._raw_text)
    
    def get_raw_text(self):
        """Get the raw text (for TTS)"""
        return self._raw_text

    def update_thinking(self, thinking_text: str):
        """Forward streaming thinking content to the thinking panel."""
        self.thinking_widget.update_thinking(thinking_text)


class ChatBubble(QFrame):
    """Legacy chat bubble widget - kept for backwards compatibility"""
    
    def __init__(self, text, is_user=False, font_size=15):
        super().__init__()
        
        if is_user:
            self.setObjectName("userBubble")
            self.setStyleSheet("background-color: #303136; border-radius: 22px;")
        else:
            self.setObjectName("botBubble")
            self.setStyleSheet("background-color: #1E1F20; border-radius: 20px;")
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        self.label = QLabel(text)
        self.label.setObjectName("bubbleText")
        self.label.setWordWrap(True)
        self.label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.update_font_size(font_size)
        
        layout.addWidget(self.label)
        
        # Responsive: use size policy instead of fixed max width
        self.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Minimum)
    
    def update_text(self, text):
        """Update the bubble text content (for streaming)"""
        self.label.setText(text)
    
    def update_font_size(self, font_size):
        """Update the font size of the bubble text"""
        self.label.setStyleSheet(f"""
            color: #E3E3E3;
            font-size: {font_size}px;
            padding: 14px 18px;
            background: transparent;
        """)


class LiveWorkerThread(QThread):
    """Thread for Live Mode: Continuous conversation with VAD and barge-in detection"""
    
    status_changed = pyqtSignal(str)  # listening, thinking, speaking, muted
    transcription_ready = pyqtSignal(str)
    response_chunk = pyqtSignal(str)  # Streaming response
    speaking_finished = pyqtSignal()
    error_occurred = pyqtSignal(str)
    
    def __init__(self, ai_manager, audio_recorder, audio_player, voice_speed=1.0):
        super().__init__()
        self.ai_manager = ai_manager
        self.recorder = audio_recorder
        self.player = audio_player
        self.is_running = False
        self.is_muted = False
        self.interrupted = False
        self.user_speaking = threading.Event()  # Signal for barge-in
        self.monitor_thread = None
        self.voice_speed = voice_speed
    
    def run(self):
        """Main loop for continuous conversation with barge-in detection"""
        self.is_running = True
        self.recorder.start_stream()
        
        # Start continuous audio monitoring thread for barge-in detection
        self.monitor_thread = threading.Thread(target=self._monitor_for_barge_in, daemon=True)
        self.monitor_thread.start()
        
        print("🎙️ Live Mode started with barge-in detection")
        
        while self.is_running:
            # Cuando está muted, no procesamos nueva entrada del usuario
            # pero permitimos que la IA termine de hablar si ya empezó
            if self.is_muted:
                self.status_changed.emit("muted")
                QThread.msleep(200)
                continue
            
            # Step 1: Listen for speech
            self.status_changed.emit("listening")
            
            # Record until silence
            audio_data = self._record_with_vad()
            
            if audio_data is None or not self.is_running:
                continue
            
            if self.is_muted:
                continue
            
            # Step 2: Transcribe
            self.status_changed.emit("thinking")
            user_text = self.ai_manager.transcribe(audio_data)
            
            if not user_text or not self.is_running:
                continue
            
            # No mostrar transcripción en UI
            # self.transcription_ready.emit(user_text)
            
            # Step 3: Get response with streaming
            self.interrupted = False
            self.user_speaking.clear()  # Reset barge-in flag
            full_response = ""
            
            # Queue for sentences ready for TTS
            sentence_queue = queue.Queue()
            audio_queue = queue.Queue(maxsize=3)
            llm_complete = threading.Event()
            tts_complete = threading.Event()
            
            def on_text_chunk(text_so_far):
                if self.interrupted or not self.is_running:
                    return
                self.response_chunk.emit(text_so_far)
            
            def on_sentence_ready(sentence):
                if self.interrupted or not self.is_running:
                    return
                clean_sentence = self._clean_markdown(sentence)
                if clean_sentence.strip():
                    sentence_queue.put(clean_sentence)
            
            # Thread for LLM streaming
            def stream_llm():
                nonlocal full_response
                try:
                    full_response = self.ai_manager.get_llm_response_streaming(
                        user_text,
                        on_chunk=on_text_chunk,
                        on_sentence=on_sentence_ready
                    )
                except Exception as e:
                    print(f"LLM error: {e}")
                finally:
                    llm_complete.set()
            
            # Thread for TTS generation
            def generate_tts():
                while not (llm_complete.is_set() and sentence_queue.empty()):
                    if self.interrupted or not self.is_running:
                        break
                    try:
                        sentence = sentence_queue.get(timeout=0.1)
                        if sentence:
                            audio_output, sample_rate = self.ai_manager.text_to_speech(sentence, speed=self.voice_speed)
                            if audio_output is not None and len(audio_output) > 0:
                                audio_queue.put((audio_output, sample_rate))
                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"TTS error: {e}")
                tts_complete.set()
            
            # Start threads
            llm_thread = threading.Thread(target=stream_llm, daemon=True)
            tts_thread = threading.Thread(target=generate_tts, daemon=True)
            llm_thread.start()
            tts_thread.start()
            
            # Play audio (keep monitoring for user interruption)
            self.status_changed.emit("speaking")
            # NO pausar el micrófono para poder detectar interrupciones
            
            interruption_detected = False
            
            while not (tts_complete.is_set() and audio_queue.empty()):
                # Check for user interruption (barge-in) FIRST
                if self.user_speaking.is_set():
                    if not interruption_detected:
                        print("⚠️ User interrupted! Stopping playback immediately...")
                        interruption_detected = True
                        self.interrupted = True
                        self.player.stop()
                        # Clear queues
                        while not audio_queue.empty():
                            try:
                                audio_queue.get_nowait()
                            except queue.Empty:
                                break
                        break
                
                if self.interrupted or not self.is_running:
                    break
                    
                try:
                    audio_output, sample_rate = audio_queue.get(timeout=0.2)
                    # Verificar de nuevo antes de reproducir
                    if self.user_speaking.is_set():
                        print("⚠️ Interruption detected before playback")
                        break
                    if not self.interrupted and self.is_running:
                        self.player.play(audio_output, sample_rate)
                except queue.Empty:
                    if tts_complete.is_set():
                        break
                    continue
            
            llm_thread.join(timeout=1.0)
            tts_thread.join(timeout=1.0)
            
            self.speaking_finished.emit()
            
            # If user interrupted, restart immediately to listen
            if self.user_speaking.is_set():
                print("♻️ User interrupted, restarting listening immediately")
                self.user_speaking.clear()
                continue
            
            # Small delay before listening again
            if self.is_running and not self.is_muted:
                QThread.msleep(300)
        
        self.recorder.stop_stream()
        print("🎙️ Live Mode stopped")
    
    def _monitor_for_barge_in(self):
        """Continuously monitor audio for user speech (barge-in detection)"""
        print("🎧 Starting barge-in monitoring thread")
        speech_threshold = self.recorder.silence_threshold * 2.0  # Higher threshold for clear interruption
        consecutive_speech_frames = 0
        frames_needed = 4  # Need 4 consecutive frames to trigger barge-in (more reliable)
        
        while self.is_running:
            # No verificar is_muted aquí - siempre monitorear para detectar interrupciones
            try:
                # Revisar directamente la cola de audio
                if not self.recorder.audio_queue.empty():
                    try:
                        # Peek sin remover
                        items = list(self.recorder.audio_queue.queue)
                        if items:
                            audio_chunk = items[-1]  # Tomar el más reciente
                            
                            rms = np.sqrt(np.mean(audio_chunk ** 2))
                            
                            if rms > speech_threshold:
                                consecutive_speech_frames += 1
                                if consecutive_speech_frames >= frames_needed:
                                    # User is speaking! Signal barge-in
                                    if not self.user_speaking.is_set():
                                        print(f"🗣️ User speech detected! RMS: {rms:.4f} (threshold: {speech_threshold:.4f})")
                                        self.user_speaking.set()
                            else:
                                # Reset if below threshold
                                if consecutive_speech_frames > 0:
                                    consecutive_speech_frames = max(0, consecutive_speech_frames - 1)
                    except (IndexError, AttributeError):
                        pass
                            
                QThread.msleep(30)  # Check every 30ms for faster response
            except Exception as e:
                QThread.msleep(100)
                continue
        
        print("🎧 Barge-in monitoring stopped")
    
    def _record_with_vad(self):
        """Record audio with Voice Activity Detection"""
        audio_buffer = []
        silence_frames = 0
        silence_threshold = self.recorder.silence_threshold
        record_rate = getattr(self.recorder, '_record_rate', self.recorder.sample_rate)
        silence_frames_needed = int(1.5 * record_rate / 1024)  # 1.5 sec silence
        max_frames = int(30 * record_rate / 1024)  # 30 sec max
        frame_count = 0
        speech_detected = False
        
        # Clear queue
        while not self.recorder.audio_queue.empty():
            try:
                self.recorder.audio_queue.get_nowait()
            except queue.Empty:
                break
        
        while frame_count < max_frames and self.is_running and not self.is_muted:
            try:
                audio_chunk = self.recorder.audio_queue.get(timeout=0.1)
                audio_buffer.append(audio_chunk)
                frame_count += 1
                
                rms = np.sqrt(np.mean(audio_chunk ** 2))
                
                if rms > silence_threshold:
                    speech_detected = True
                    silence_frames = 0
                else:
                    if speech_detected:
                        silence_frames += 1
                        if silence_frames >= silence_frames_needed:
                            break
            except queue.Empty:
                continue
        
        if not audio_buffer or not speech_detected:
            return None
        
        audio_data = np.concatenate(audio_buffer, axis=0).flatten()

        # Resample to target rate (16000 Hz for Whisper) if needed
        record_rate = getattr(self.recorder, '_record_rate', self.recorder.sample_rate)
        if record_rate != self.recorder.sample_rate:
            new_length = int(len(audio_data) * self.recorder.sample_rate / record_rate)
            old_idx = np.arange(len(audio_data))
            new_idx = np.linspace(0, len(audio_data) - 1, new_length)
            audio_data = np.interp(new_idx, old_idx, audio_data).astype(np.float32)
            print(f"Resampled VAD audio {record_rate}Hz → {self.recorder.sample_rate}Hz")

        duration = len(audio_data) / self.recorder.sample_rate
        
        if duration < 0.5:
            return None
        
        return audio_data
    
    def _clean_markdown(self, text):
        """Remove markdown symbols and emojis"""
        import re
        
        # Remove emojis
        emoji_pattern = re.compile(
            "["
            "\U0001F600-\U0001F64F"  # emoticons
            "\U0001F300-\U0001F5FF"  # symbols & pictographs
            "\U0001F680-\U0001F6FF"  # transport & map symbols
            "\U0001F1E0-\U0001F1FF"  # flags (iOS)
            "\U00002702-\U000027B0"
            "\U000024C2-\U0001F251"
            "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
            "\U0001FA00-\U0001FA6F"  # Chess Symbols
            "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
            "\U00002600-\U000026FF"  # Miscellaneous Symbols
            "\U00002700-\U000027BF"  # Dingbats
            "]+", flags=re.UNICODE
        )
        text = emoji_pattern.sub('', text)
        
        text = re.sub(r'\*\*\*(.+?)\*\*\*', r'\1', text)
        text = re.sub(r'\*\*(.+?)\*\*', r'\1', text)
        text = re.sub(r'\*(.+?)\*', r'\1', text)
        text = re.sub(r'__(.+?)__', r'\1', text)
        text = re.sub(r'_(.+?)_', r'\1', text)
        text = text.replace('*', '').replace('_', '')
        text = re.sub(r'^#+\s+', '', text, flags=re.MULTILINE)
        text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
        text = re.sub(r'`(.+?)`', r'\1', text)
        return text.strip()
    
    def set_muted(self, muted):
        """Toggle mute state - only affects user input capture, not AI playback"""
        self.is_muted = muted
        if muted:
            print("🔇 Muted - User input disabled (AI continues)")
        else:
            print("🎙️ Unmuted - User input enabled")
    
    def interrupt(self):
        """Interrupt current response"""
        self.interrupted = True
        self.player.stop()
    
    def stop(self):
        """Stop the live mode"""
        self.is_running = False
        self.interrupted = True
        self.player.stop()


class LiveModeWidget(QWidget):
    """Gemini Live-style continuous conversation interface"""
    
    exit_requested = pyqtSignal()
    
    def __init__(self, ai_manager, audio_recorder, audio_player, parent=None):
        super().__init__(parent)
        self.ai_manager = ai_manager
        self.recorder = audio_recorder
        self.player = audio_player
        self.parent_window = parent  # Referencia a MainWindow para acceder a voice_speed
        self.worker_thread = None
        self.is_active = False
        
        self.setObjectName("liveModeWidget")
        self.init_ui()
        self.setup_animations()
    
    def init_ui(self):
        """Create the Live Mode UI"""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 40, 20, 30)
        layout.setSpacing(20)
        
        # Top spacer
        layout.addStretch(1)
        
        # Central indicator container
        indicator_container = QWidget()
        indicator_container.setObjectName("liveIndicatorContainer")
        indicator_layout = QVBoxLayout(indicator_container)
        indicator_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        indicator_layout.setSpacing(20)
        
        # Pulsing circle indicator
        self.indicator = QLabel()
        self.indicator.setFixedSize(150, 150)
        self.indicator.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.indicator.setStyleSheet("""
            background-color: #1A73E8;
            border-radius: 75px;
        """)
        
        # Add glow effect
        self.glow_effect = QGraphicsDropShadowEffect()
        self.glow_effect.setBlurRadius(40)
        self.glow_effect.setColor(QColor(138, 180, 248, 150))
        self.glow_effect.setOffset(0, 0)
        self.indicator.setGraphicsEffect(self.glow_effect)
        
        indicator_layout.addWidget(self.indicator, alignment=Qt.AlignmentFlag.AlignCenter)
        
        # State label
        self.state_label = QLabel("Ready to start")
        self.state_label.setObjectName("liveStateLabel")
        self.state_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        indicator_layout.addWidget(self.state_label)
        
        layout.addWidget(indicator_container, alignment=Qt.AlignmentFlag.AlignCenter)
        
        layout.addStretch(2)
        
        # Bottom controls
        controls = QWidget()
        controls_layout = QHBoxLayout(controls)
        controls_layout.setSpacing(40)
        controls_layout.setContentsMargins(0, 0, 0, 20)
        
        # Mute button
        self.mute_btn = QPushButton("🎤")
        self.mute_btn.setObjectName("liveMuteButton")
        self.mute_btn.setCheckable(True)
        self.mute_btn.setChecked(False)
        self.mute_btn.clicked.connect(self.toggle_mute)
        self.mute_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        controls_layout.addWidget(self.mute_btn)
        
        # Exit button
        self.exit_btn = QPushButton("✕")
        self.exit_btn.setObjectName("liveExitButton")
        self.exit_btn.clicked.connect(self.exit_live_mode)
        self.exit_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        controls_layout.addWidget(self.exit_btn)
        
        layout.addWidget(controls, alignment=Qt.AlignmentFlag.AlignCenter)
    
    def setup_animations(self):
        """Setup pulse animation for indicator"""
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self.animate_pulse)
        self.pulse_phase = 0
    
    def animate_pulse(self):
        """Animate the indicator pulse"""
        self.pulse_phase = (self.pulse_phase + 1) % 20
        
        # Vary blur radius for pulsing effect
        base_blur = 30
        pulse_amount = 20 * (0.5 + 0.5 * np.sin(self.pulse_phase * np.pi / 10))
        self.glow_effect.setBlurRadius(base_blur + pulse_amount)
    
    def start_live_mode(self):
        """Start the live conversation mode"""
        if self.worker_thread and self.worker_thread.isRunning():
            return
        
        self.is_active = True
        self.update_state("listening")
        
        # Start worker thread
        self.worker_thread = LiveWorkerThread(
            self.ai_manager, self.recorder, self.player, self.parent_window.voice_speed
        )
        self.worker_thread.status_changed.connect(self.update_state)
        # No conectar transcription ni response_chunk para no mostrar texto
        # self.worker_thread.transcription_ready.connect(self.on_transcription)
        # self.worker_thread.response_chunk.connect(self.on_response_chunk)
        self.worker_thread.speaking_finished.connect(self.on_speaking_finished)
        self.worker_thread.start()
        
        self.pulse_timer.start(50)
    
    def stop_live_mode(self):
        """Stop the live conversation mode"""
        self.is_active = False
        self.pulse_timer.stop()
        
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait(2000)
        
        self.update_state("stopped")
        self.state_label.setText("Ready to start")
    
    @pyqtSlot(str)
    def update_state(self, state):
        """Update visual state of the indicator"""
        if state == "listening":
            self.indicator.setStyleSheet("""
                background-color: #1A73E8;
                border-radius: 75px;
            """)
            self.glow_effect.setColor(QColor(26, 115, 232, 180))
            self.state_label.setText("Listening...")
            
        elif state == "thinking":
            self.indicator.setStyleSheet("""
                background-color: #9334E6;
                border-radius: 75px;
            """)
            self.glow_effect.setColor(QColor(147, 52, 230, 180))
            self.state_label.setText("Thinking...")
            
        elif state == "speaking":
            self.indicator.setStyleSheet("""
                background-color: #E8EAED;
                border-radius: 75px;
            """)
            self.glow_effect.setColor(QColor(232, 234, 237, 150))
            self.state_label.setText("Speaking...")
            
        elif state == "muted":
            self.indicator.setStyleSheet("""
                background-color: #5F6368;
                border-radius: 75px;
            """)
            self.glow_effect.setColor(QColor(95, 99, 104, 100))
            self.state_label.setText("Muted")
            
        elif state == "stopped":
            self.indicator.setStyleSheet("""
                background-color: #3C4043;
                border-radius: 75px;
            """)
            self.glow_effect.setColor(QColor(60, 64, 67, 80))
    
    @pyqtSlot()
    def on_speaking_finished(self):
        """Called when TTS finishes"""
        pass  # No need to update text
    
    def toggle_mute(self):
        """Toggle mute state"""
        is_muted = self.mute_btn.isChecked()
        self.mute_btn.setText("🔇" if is_muted else "🎤")
        
        if self.worker_thread:
            self.worker_thread.set_muted(is_muted)
    
    def exit_live_mode(self):
        """Exit live mode and return to chat"""
        self.stop_live_mode()
        self.mute_btn.setChecked(False)
        self.mute_btn.setText("🎤")
        self.exit_requested.emit()


class SettingsDialog(QDialog):
    """Settings dialog with font size and language options"""
    
    def __init__(self, parent=None, auto_send=True, font_size="medium", language="english", voice_speed=1.0, output_device=-1, input_device=-1):
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(440)
        self.setMinimumHeight(500)
        self.auto_send = auto_send
        self.font_size = font_size
        self.language = language
        self.voice_speed = voice_speed
        self.output_device = output_device
        self.input_device = input_device

        # Outer layout: scroll area (contenido) + botón guardar (fijo abajo)
        from PyQt6.QtWidgets import QScrollArea
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(20, 20, 20, 20)
        outer_layout.setSpacing(10)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setStyleSheet("""
            QScrollArea { background: transparent; border: none; }
            QScrollBar:vertical {
                background: #2A2B2E;
                width: 8px;
                border-radius: 4px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #5F6368;
                border-radius: 4px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover { background: #9AA0A6; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)

        scroll_content = QWidget()
        scroll_content.setStyleSheet("background: transparent;")
        layout = QVBoxLayout(scroll_content)
        layout.setContentsMargins(0, 0, 12, 0)
        layout.setSpacing(15)

        scroll_area.setWidget(scroll_content)
        outer_layout.addWidget(scroll_area)

        # Title
        title = QLabel("⚙️ Settings")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #E3E3E3;")
        layout.addWidget(title)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setStyleSheet("background-color: #3C4043;")
        layout.addWidget(separator)
        
        # ===== LANGUAGE SETTING =====
        lang_label = QLabel("🌐 Language / Idioma:")
        lang_label.setStyleSheet("font-size: 14px; color: #E3E3E3; margin-top: 10px;")
        layout.addWidget(lang_label)
        
        self.language_combo = QComboBox()
        self.language_combo.addItems(["English", "Español"])
        lang_map = {"english": 0, "spanish": 1}
        self.language_combo.setCurrentIndex(lang_map.get(language, 0))
        self.language_combo.setStyleSheet("""
            QComboBox {
                background-color: #282A2C;
                color: #E3E3E3;
                border: 1px solid #3C4043;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
                min-width: 120px;
            }
            QComboBox:hover {
                background-color: #3C4043;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #9AA0A6;
            }
            QComboBox QAbstractItemView {
                background-color: #282A2C;
                color: #E3E3E3;
                selection-background-color: #3C4043;
                border: 1px solid #3C4043;
                border-radius: 8px;
                padding: 4px;
            }
        """)
        layout.addWidget(self.language_combo)
        
        lang_help = QLabel("Both languages powered by Kokoro TTS v1.0")
        lang_help.setWordWrap(True)
        lang_help.setStyleSheet("font-size: 11px; color: #9AA0A6; margin-left: 4px;")
        layout.addWidget(lang_help)
        
        # Separator 2
        separator2 = QFrame()
        separator2.setFrameShape(QFrame.Shape.HLine)
        separator2.setStyleSheet("background-color: #3C4043;")
        layout.addWidget(separator2)
        
        # Font size setting
        font_label = QLabel("Font Size:")
        font_label.setStyleSheet("font-size: 14px; color: #E3E3E3; margin-top: 10px;")
        layout.addWidget(font_label)
        
        self.font_size_combo = QComboBox()
        self.font_size_combo.addItems(["Small", "Medium", "Large"])
        size_map = {"small": 0, "medium": 1, "large": 2}
        self.font_size_combo.setCurrentIndex(size_map.get(font_size, 1))
        self.font_size_combo.setStyleSheet("""
            QComboBox {
                background-color: #282A2C;
                color: #E3E3E3;
                border: 1px solid #3C4043;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
                min-width: 120px;
            }
            QComboBox:hover {
                background-color: #3C4043;
            }
            QComboBox::drop-down {
                border: none;
                padding-right: 10px;
            }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #9AA0A6;
            }
            QComboBox QAbstractItemView {
                background-color: #282A2C;
                color: #E3E3E3;
                selection-background-color: #3C4043;
                border: 1px solid #3C4043;
                border-radius: 8px;
                padding: 4px;
            }
        """)
        layout.addWidget(self.font_size_combo)
        
        # Separator 3
        separator3 = QFrame()
        separator3.setFrameShape(QFrame.Shape.HLine)
        separator3.setStyleSheet("background-color: #3C4043;")
        layout.addWidget(separator3)
        
        # Voice speed setting
        speed_label = QLabel("🔊 Voice Speed:")
        speed_label.setStyleSheet("font-size: 14px; color: #E3E3E3; margin-top: 10px;")
        layout.addWidget(speed_label)
        
        # Speed value display
        self.speed_value_label = QLabel(f"{self.voice_speed:.2f}x")
        self.speed_value_label.setStyleSheet("font-size: 13px; color: #8AB4F8; font-weight: bold;")
        layout.addWidget(self.speed_value_label)
        
        # Speed slider
        self.speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.speed_slider.setMinimum(50)  # 0.5x
        self.speed_slider.setMaximum(200)  # 2.0x
        self.speed_slider.setValue(int(self.voice_speed * 100))
        self.speed_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.speed_slider.setTickInterval(25)
        self.speed_slider.valueChanged.connect(self.update_speed_label)
        self.speed_slider.setStyleSheet("""
            QSlider::groove:horizontal {
                border: 1px solid #3C4043;
                height: 6px;
                background: #282A2C;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #1A73E8;
                border: 2px solid #1A73E8;
                width: 16px;
                height: 16px;
                margin: -6px 0;
                border-radius: 8px;
            }
            QSlider::handle:horizontal:hover {
                background: #1557b0;
                border-color: #1557b0;
            }
            QSlider::sub-page:horizontal {
                background: #1A73E8;
                border-radius: 3px;
            }
        """)
        layout.addWidget(self.speed_slider)
        
        speed_help = QLabel("Adjust voice playback speed (0.5x slow - 2.0x fast)")
        speed_help.setWordWrap(True)
        speed_help.setStyleSheet("font-size: 11px; color: #9AA0A6; margin-left: 4px;")
        layout.addWidget(speed_help)

        # Separator audio device
        sep_dev = QFrame()
        sep_dev.setFrameShape(QFrame.Shape.HLine)
        sep_dev.setStyleSheet("background-color: #3C4043;")
        layout.addWidget(sep_dev)

        # ===== AUDIO OUTPUT DEVICE =====
        import sounddevice as sd_dev
        dev_label = QLabel("🔈 Audio Output Device:")
        dev_label.setStyleSheet("font-size: 14px; color: #E3E3E3; margin-top: 10px;")
        layout.addWidget(dev_label)

        self.device_combo = QComboBox()
        combo_style = """
            QComboBox {
                background-color: #282A2C;
                color: #E3E3E3;
                border: 1px solid #3C4043;
                border-radius: 8px;
                padding: 8px 16px;
                font-size: 13px;
                min-width: 120px;
            }
            QComboBox:hover { background-color: #3C4043; }
            QComboBox::drop-down { border: none; padding-right: 10px; }
            QComboBox::down-arrow {
                image: none;
                border-left: 5px solid transparent;
                border-right: 5px solid transparent;
                border-top: 6px solid #9AA0A6;
            }
            QComboBox QAbstractItemView {
                background-color: #282A2C;
                color: #E3E3E3;
                selection-background-color: #3C4043;
                border: 1px solid #3C4043;
                border-radius: 8px;
                padding: 4px;
            }
        """
        self.device_combo.setStyleSheet(combo_style)
        # Populate with output devices
        self._device_indices = [-1]
        self.device_combo.addItem("System Default")
        selected_combo_idx = 0
        try:
            for i, d in enumerate(sd_dev.query_devices()):
                if d['max_output_channels'] > 0:
                    self._device_indices.append(i)
                    label = f"{d['name']}  ({int(d['default_samplerate'])}Hz)"
                    self.device_combo.addItem(label)
                    if i == output_device:
                        selected_combo_idx = len(self._device_indices) - 1
        except Exception:
            pass
        self.device_combo.setCurrentIndex(selected_combo_idx)
        layout.addWidget(self.device_combo)

        dev_help = QLabel("Select where TTS audio is played back.")
        dev_help.setWordWrap(True)
        dev_help.setStyleSheet("font-size: 11px; color: #9AA0A6; margin-left: 4px;")
        layout.addWidget(dev_help)

        # Separator mic
        sep_mic = QFrame()
        sep_mic.setFrameShape(QFrame.Shape.HLine)
        sep_mic.setStyleSheet("background-color: #3C4043;")
        layout.addWidget(sep_mic)

        # ===== AUDIO INPUT DEVICE (MICROPHONE) =====
        mic_label = QLabel("🎤 Microphone Input Device:")
        mic_label.setStyleSheet("font-size: 14px; color: #E3E3E3; margin-top: 10px;")
        layout.addWidget(mic_label)

        self.mic_combo = QComboBox()
        self.mic_combo.setStyleSheet(combo_style)
        self._mic_indices = [-1]
        self.mic_combo.addItem("System Default")
        selected_mic_idx = 0
        try:
            for i, d in enumerate(sd_dev.query_devices()):
                if d['max_input_channels'] > 0:
                    self._mic_indices.append(i)
                    mic_item = f"{d['name']}  ({int(d['default_samplerate'])}Hz)"
                    self.mic_combo.addItem(mic_item)
                    if i == input_device:
                        selected_mic_idx = len(self._mic_indices) - 1
        except Exception:
            pass
        self.mic_combo.setCurrentIndex(selected_mic_idx)
        layout.addWidget(self.mic_combo)

        mic_help = QLabel("Select the microphone used to record your voice.")
        mic_help.setWordWrap(True)
        mic_help.setStyleSheet("font-size: 11px; color: #9AA0A6; margin-left: 4px;")
        layout.addWidget(mic_help)

        # Separator 4
        separator4 = QFrame()
        separator4.setFrameShape(QFrame.Shape.HLine)
        separator4.setStyleSheet("background-color: #3C4043;")
        layout.addWidget(separator4)
        
        # Voice mode setting
        mode_label = QLabel("Voice Recording Mode:")
        mode_label.setStyleSheet("font-size: 14px; color: #E3E3E3; margin-top: 10px;")
        layout.addWidget(mode_label)
        
        self.auto_send_checkbox = QCheckBox("Send automatically after recording")
        self.auto_send_checkbox.setChecked(auto_send)
        self.auto_send_checkbox.setStyleSheet("""
            QCheckBox {
                color: #E3E3E3;
                font-size: 13px;
                spacing: 8px;
            }
            QCheckBox::indicator {
                width: 18px;
                height: 18px;
                border-radius: 4px;
                border: 2px solid #5F6368;
                background-color: #202124;
            }
            QCheckBox::indicator:checked {
                background-color: #1A73E8;
                border-color: #1A73E8;
            }
        """)
        layout.addWidget(self.auto_send_checkbox)
        
        help_text = QLabel("When unchecked, transcribed text will appear in the input field for you to review before sending.")
        help_text.setWordWrap(True)
        help_text.setStyleSheet("font-size: 11px; color: #9AA0A6; margin-left: 26px;")
        layout.addWidget(help_text)
        
        layout.addStretch()

        # Close button fijo fuera del scroll
        close_btn = QPushButton("Save & Close")
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: #1A73E8;
                color: white;
                border: none;
                border-radius: 8px;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #1557b0;
            }
        """)
        close_btn.clicked.connect(self.accept)
        outer_layout.addWidget(close_btn)

        # Set dialog background
        self.setStyleSheet("""
            QDialog {
                background-color: #202124;
            }
        """)
    
    def update_speed_label(self, value):
        """Actualizar etiqueta de velocidad al mover el slider"""
        speed = value / 100.0
        self.speed_value_label.setText(f"{speed:.2f}x")
    
    def get_auto_send(self):
        return self.auto_send_checkbox.isChecked()
    
    def get_font_size(self):
        size_names = ["small", "medium", "large"]
        return size_names[self.font_size_combo.currentIndex()]
    
    def get_language(self):
        lang_names = ["english", "spanish"]
        return lang_names[self.language_combo.currentIndex()]
    
    def get_voice_speed(self):
        return self.speed_slider.value() / 100.0

    def get_output_device(self):
        return self._device_indices[self.device_combo.currentIndex()]

    def get_input_device(self):
        return self._mic_indices[self.mic_combo.currentIndex()]


class VoiceSetupDialog(QDialog):
    """
    Shown at startup when NO voice packs are detected.
    Guides the user to download Kokoro v1.0 or a Sherpa-ONNX voice.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Voice Setup Required")
        self.setModal(True)
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 28, 30, 24)
        layout.setSpacing(16)

        # ── Icon + title ───────────────────────────────────────────────────
        title_row = QHBoxLayout()
        icon_lbl = QLabel("🎙️")
        icon_lbl.setStyleSheet("font-size: 36px;")
        title_row.addWidget(icon_lbl)
        title_row.addSpacing(12)

        title_col = QVBoxLayout()
        title = QLabel("No Voice Packs Found")
        title.setStyleSheet("font-size: 20px; font-weight: bold; color: #E3E3E3;")
        subtitle = QLabel("Your AI assistant needs a voice to speak.")
        subtitle.setStyleSheet("font-size: 13px; color: #9AA0A6;")
        title_col.addWidget(title)
        title_col.addWidget(subtitle)
        title_row.addLayout(title_col, 1)
        layout.addLayout(title_row)

        # ── Separator ──────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #3C4043; margin: 4px 0;")
        layout.addWidget(sep)

        # ── Instructions ───────────────────────────────────────────────────
        intro = QLabel(
            "Drop your voice model files into the <b>voices/</b> folder, "
            "then restart the app. Choose one of the options below to get started:"
        )
        intro.setWordWrap(True)
        intro.setStyleSheet("font-size: 13px; color: #C8C8C8; line-height: 1.5;")
        layout.addWidget(intro)

        # ── Option A — Kokoro v1.0 ─────────────────────────────────────────
        card_a = QFrame()
        card_a.setStyleSheet("""
            QFrame { background-color: #282A2C; border-radius: 10px;
                     border: 1px solid #3C4043; }
        """)
        card_a_layout = QVBoxLayout(card_a)
        card_a_layout.setContentsMargins(16, 14, 16, 14)
        card_a_layout.setSpacing(6)

        lbl_a_title = QLabel("⭐  Recommended — Kokoro TTS v1.0")
        lbl_a_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #8AB4F8;")
        card_a_layout.addWidget(lbl_a_title)

        lbl_a_desc = QLabel(
            "High-quality neural TTS. Supports English and Spanish out of the box.\n"
            "Download <b>kokoro-v1.0.onnx</b> and <b>voices-v1.0.bin</b>, then "
            "place them inside <code>voices/kokoro-v1.0/</code>."
        )
        lbl_a_desc.setWordWrap(True)
        lbl_a_desc.setStyleSheet("font-size: 12px; color: #C8C8C8;")
        card_a_layout.addWidget(lbl_a_desc)

        btn_a = QPushButton("Open Kokoro Releases on GitHub →")
        btn_a.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_a.setStyleSheet("""
            QPushButton {
                background-color: #1A73E8; color: white; border: none;
                border-radius: 6px; padding: 7px 16px; font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #1557b0; }
        """)
        btn_a.clicked.connect(
            lambda: __import__("webbrowser").open(
                "https://github.com/thewh1teagle/kokoro-onnx/releases"
            )
        )
        card_a_layout.addWidget(btn_a)
        layout.addWidget(card_a)

        # ── Option B — Sherpa-ONNX ─────────────────────────────────────────
        card_b = QFrame()
        card_b.setStyleSheet("""
            QFrame { background-color: #282A2C; border-radius: 10px;
                     border: 1px solid #3C4043; }
        """)
        card_b_layout = QVBoxLayout(card_b)
        card_b_layout.setContentsMargins(16, 14, 16, 14)
        card_b_layout.setSpacing(6)

        lbl_b_title = QLabel("🌍  More Languages — Sherpa-ONNX Voices")
        lbl_b_title.setStyleSheet("font-size: 14px; font-weight: bold; color: #34A853;")
        card_b_layout.addWidget(lbl_b_title)

        lbl_b_desc = QLabel(
            "Add extra language voices using Sherpa-ONNX VITS packs.\n"
            "Download a voice folder and place it directly inside "
            "<code>voices/</code> (e.g. <code>voices/vits-piper-es_AR-daniela-high/</code>)."
        )
        lbl_b_desc.setWordWrap(True)
        lbl_b_desc.setStyleSheet("font-size: 12px; color: #C8C8C8;")
        card_b_layout.addWidget(lbl_b_desc)

        btn_b = QPushButton("Browse Sherpa-ONNX Voices on HuggingFace →")
        btn_b.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_b.setStyleSheet("""
            QPushButton {
                background-color: #2D5A27; color: #34A853; border: none;
                border-radius: 6px; padding: 7px 16px; font-size: 12px;
                font-weight: bold;
            }
            QPushButton:hover { background-color: #3A7033; color: white; }
        """)
        btn_b.clicked.connect(
            lambda: __import__("webbrowser").open(
                "https://huggingface.co/csukuangfj/vits-piper-es_AR-daniela-high/tree/main"
            )
        )
        card_b_layout.addWidget(btn_b)
        layout.addWidget(card_b)

        # ── Open folder button ─────────────────────────────────────────────
        open_folder_btn = QPushButton("📁  Open voices/ Folder")
        open_folder_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        open_folder_btn.setStyleSheet("""
            QPushButton {
                background-color: #3C4043; color: #E3E3E3; border: none;
                border-radius: 6px; padding: 8px 16px; font-size: 13px;
            }
            QPushButton:hover { background-color: #5F6368; }
        """)
        open_folder_btn.clicked.connect(self._open_voices_folder)
        layout.addWidget(open_folder_btn)

        # ── Dismiss ────────────────────────────────────────────────────────
        close_btn = QPushButton("Continue Anyway (text-only mode)")
        close_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        close_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent; color: #9AA0A6; border: none;
                padding: 6px; font-size: 12px;
            }
            QPushButton:hover { color: #E3E3E3; }
        """)
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn, alignment=Qt.AlignmentFlag.AlignCenter)

        self.setStyleSheet("QDialog { background-color: #202124; }")

    def _open_voices_folder(self):
        import subprocess, os
        voices_dir = os.path.abspath("voices")
        os.makedirs(voices_dir, exist_ok=True)
        try:
            subprocess.Popen(["xdg-open", voices_dir])
        except Exception:
            pass


class NewVoiceClassifierDialog(QDialog):
    """
    Shown once per newly-discovered Sherpa voice folder.
    Asks the user which language the voice pack belongs to.
    The choice is persisted so the dialog never re-appears for the same folder.
    """

    def __init__(self, folder_name: str, parent=None):
        super().__init__(parent)
        self.folder_name = folder_name
        self.selected_language: str = "english"  # default

        self.setWindowTitle("New Voice Pack Detected")
        self.setModal(True)
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 26, 28, 22)
        layout.setSpacing(14)

        # ── Header ─────────────────────────────────────────────────────────
        header_row = QHBoxLayout()
        icon = QLabel("🎉")
        icon.setStyleSheet("font-size: 32px;")
        header_row.addWidget(icon)
        header_row.addSpacing(12)

        hdr_col = QVBoxLayout()
        hdr_title = QLabel("New Voice Pack Detected!")
        hdr_title.setStyleSheet("font-size: 18px; font-weight: bold; color: #8AB4F8;")
        hdr_sub = QLabel(f"<code>{folder_name}</code>")
        hdr_sub.setStyleSheet("font-size: 12px; color: #9AA0A6;")
        hdr_col.addWidget(hdr_title)
        hdr_col.addWidget(hdr_sub)
        header_row.addLayout(hdr_col, 1)
        layout.addLayout(header_row)

        # ── Separator ──────────────────────────────────────────────────────
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet("background-color: #3C4043; margin: 2px 0;")
        layout.addWidget(sep)

        # ── Language prompt ────────────────────────────────────────────────
        prompt = QLabel(
            "Which language should <b>ChatbotAI</b> use this voice pack for?"
        )
        prompt.setWordWrap(True)
        prompt.setStyleSheet("font-size: 13px; color: #C8C8C8;")
        layout.addWidget(prompt)

        # ── Radio buttons ──────────────────────────────────────────────────
        radio_style = """
            QRadioButton {
                color: #E3E3E3; font-size: 14px; spacing: 10px;
                padding: 10px 14px; border-radius: 8px;
            }
            QRadioButton:hover { background-color: #2A2B2E; }
            QRadioButton:checked { color: #8AB4F8; }
            QRadioButton::indicator { width: 18px; height: 18px; }
            QRadioButton::indicator:unchecked {
                border: 2px solid #5F6368; border-radius: 9px;
                background-color: #202124;
            }
            QRadioButton::indicator:checked {
                border: 2px solid #1A73E8; border-radius: 9px;
                background-color: #1A73E8;
            }
        """

        radio_box = QFrame()
        radio_box.setStyleSheet(
            "QFrame { background-color: #282A2C; border-radius: 10px;"
            "border: 1px solid #3C4043; }"
        )
        radio_layout = QVBoxLayout(radio_box)
        radio_layout.setContentsMargins(8, 8, 8, 8)
        radio_layout.setSpacing(4)

        self._btn_group = QButtonGroup(self)

        self._radio_en = QRadioButton("🇺🇸  English")
        self._radio_en.setStyleSheet(radio_style)
        self._radio_en.setChecked(True)
        self._btn_group.addButton(self._radio_en)
        radio_layout.addWidget(self._radio_en)

        self._radio_es = QRadioButton("🇪🇸  Spanish / Español")
        self._radio_es.setStyleSheet(radio_style)
        self._btn_group.addButton(self._radio_es)
        radio_layout.addWidget(self._radio_es)

        layout.addWidget(radio_box)

        # ── Confirm button ─────────────────────────────────────────────────
        confirm_btn = QPushButton("Save & Continue")
        confirm_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        confirm_btn.setStyleSheet("""
            QPushButton {
                background-color: #1A73E8; color: white; border: none;
                border-radius: 8px; padding: 10px 24px;
                font-size: 14px; font-weight: bold;
            }
            QPushButton:hover { background-color: #1557b0; }
        """)
        confirm_btn.clicked.connect(self._on_confirm)
        layout.addWidget(confirm_btn)

        self.setStyleSheet("QDialog { background-color: #202124; }")

    def _on_confirm(self):
        self.selected_language = "english" if self._radio_en.isChecked() else "spanish"
        self.accept()

    def get_language(self) -> str:
        return self.selected_language


class MainWindow(QMainWindow):
    """Main application window - Gemini Style"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Voice Chat AI")
        self.setGeometry(100, 100, 1400, 900)
        self.setMinimumSize(1400, 900)
        
        # Load user preferences
        self.preferences = load_preferences()
        
        # State
        self.is_recording = False
        self.is_processing = False
        self.is_speaking = False
        self.auto_send = self.preferences.get("auto_send", True)
        self.font_size_name = self.preferences.get("font_size", "medium")
        self.language = self.preferences.get("language", "english")
        self.voice_speed = self.preferences.get("voice_speed", 1.0)
        self.chat_bubbles = []  # Track bubbles for font size updates
        
        # Voice preferences
        self.english_voice = self.preferences.get("english_voice", "af_bella")
        self.spanish_voice = self.preferences.get("spanish_voice", "ef_dora")
        
        # Detect available voices
        from voice_detector import get_english_voices, get_spanish_voices
        self.available_english_voices = get_english_voices()
        self.available_spanish_voices = get_spanish_voices()
        
        # Initialize components
        self.output_device = self.preferences.get("output_device", -1)
        self.input_device = self.preferences.get("input_device", -1)
        self.recorder = AudioRecorder(device=self.input_device if self.input_device >= 0 else None)
        self.player = AudioPlayer(device=self.output_device if self.output_device >= 0 else None)
        self.recorder_thread = None
        self.worker_thread = None
        
        # Load AI Manager with language and voice settings
        print("Loading AI models...")
        try:
            current_voice = self.english_voice if self.language == "english" else self.spanish_voice
            self.ai_manager = AIManager(
                language=self.language,
                voice_name=current_voice,
            )
        except Exception as e:
            print(f"Error initializing AI: {e}")
            self.ai_manager = None
        
        # Setup UI
        self.init_ui()
        self.setStyleSheet(GEMINI_STYLE)
        self.apply_font_size()  # Apply saved font size
        
        # Force style update for buttons
        self.send_btn.setStyleSheet("")  # Clear any inline styles
        self.mic_btn.setStyleSheet("")
        self.live_btn.setStyleSheet("")
        self.style().unpolish(self.send_btn)
        self.style().polish(self.send_btn)
        self.style().unpolish(self.mic_btn)
        self.style().polish(self.mic_btn)
        self.style().unpolish(self.live_btn)
        self.style().polish(self.live_btn)
        
        # Recording animation timer
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self.update_recording_animation)
        self.pulse_state = 0

        # Voice scanner — runs after the window is fully rendered
        QTimer.singleShot(300, self._check_voices_on_startup)
    
    def init_ui(self):
        """Initialize the Gemini-inspired UI"""
        
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # ===== HEADER =====
        header = QWidget()
        header.setObjectName("headerWidget")
        header.setFixedHeight(70)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(20, 15, 20, 15)
        
        # Model Selector (centered)
        header_layout.addStretch()
        
        self.model_selector = QComboBox()
        self.model_selector.setObjectName("modelSelector")
        available_models = get_available_ollama_models()
        self.model_selector.addItems(available_models)
        
        # Set model from preferences or use first available
        saved_model = self.preferences.get("model")
        if saved_model and saved_model in available_models:
            idx = self.model_selector.findText(saved_model)
            if idx >= 0:
                self.model_selector.setCurrentIndex(idx)
                if self.ai_manager:
                    self.ai_manager.ollama_model = saved_model
        elif available_models:
            # Use first available model as default
            self.model_selector.setCurrentIndex(0)
            if self.ai_manager:
                self.ai_manager.ollama_model = available_models[0]
        
        self.model_selector.currentTextChanged.connect(self.on_model_changed)
        header_layout.addWidget(self.model_selector)
        
        # Voice Selector (next to model selector)
        header_layout.addSpacing(20)
        
        voice_label = QLabel("🎤")
        voice_label.setStyleSheet("font-size: 18px; color: #9AA0A6;")
        header_layout.addWidget(voice_label)
        
        self.voice_selector = QComboBox()
        self.voice_selector.setObjectName("voiceSelector")
        self.voice_selector.setMinimumWidth(150)
        self.update_voice_list()  # Populate based on current language
        self.voice_selector.currentTextChanged.connect(self.on_voice_changed)
        header_layout.addWidget(self.voice_selector)
        
        header_layout.addStretch()
        
        # Settings button (top right)
        settings_btn = QPushButton("⚙️")
        settings_btn.setObjectName("settingsButton")
        settings_btn.setFixedSize(40, 40)
        settings_btn.setStyleSheet("""
            QPushButton {
                background-color: transparent;
                border: none;
                border-radius: 20px;
                font-size: 20px;
            }
            QPushButton:hover {
                background-color: #3C4043;
            }
        """)
        settings_btn.clicked.connect(self.open_settings)
        header_layout.addWidget(settings_btn)
        
        main_layout.addWidget(header)
        
        # ===== STACKED WIDGET (Chat + Live Mode) =====
        self.stacked_widget = QStackedWidget()
        
        # --- Page 0: Chat Area ---
        chat_page = QWidget()
        chat_page_layout = QVBoxLayout(chat_page)
        chat_page_layout.setContentsMargins(0, 0, 0, 0)
        chat_page_layout.setSpacing(0)
        
        # Chat scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        
        self.chat_widget = QWidget()
        self.chat_widget.setObjectName("chatContainer")
        self.chat_layout = QVBoxLayout(self.chat_widget)
        self.chat_layout.setContentsMargins(16, 16, 16, 16)
        self.chat_layout.setSpacing(12)
        self.chat_layout.addStretch()
        
        scroll.setWidget(self.chat_widget)
        self.scroll_area = scroll
        chat_page_layout.addWidget(scroll, 1)
        
        self.stacked_widget.addWidget(chat_page)  # Page 0: Chat
        
        # --- Page 1: Live Mode ---
        self.live_mode_widget = LiveModeWidget(
            self.ai_manager, self.recorder, self.player, self
        )
        self.live_mode_widget.exit_requested.connect(self.exit_live_mode)
        self.stacked_widget.addWidget(self.live_mode_widget)  # Page 1: Live
        
        main_layout.addWidget(self.stacked_widget, 1)
        
        # ===== INPUT BAR =====
        # Wrapper to center input bar
        input_bar_wrapper = QWidget()
        input_bar_wrapper.setStyleSheet("background-color: #131314;")
        input_bar_wrapper_layout = QHBoxLayout(input_bar_wrapper)
        input_bar_wrapper_layout.setContentsMargins(0, 0, 0, 0)
        input_bar_wrapper_layout.addStretch()
        
        input_bar = QWidget()
        input_bar.setObjectName("inputBar")
        input_bar.setMaximumWidth(1200)
        input_bar.setMinimumWidth(1200)
        input_bar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Minimum)
        input_layout = QVBoxLayout(input_bar)
        input_layout.setContentsMargins(16, 8, 16, 12)
        input_layout.setSpacing(8)
        
        # Text input row - PRIMERO (crece hacia arriba)
        text_row = QHBoxLayout()
        text_row.setSpacing(10)
        
        # Text input field
        self.text_input = CustomTextEdit()
        self.text_input.setObjectName("textInput")
        self.text_input.setPlaceholderText("Type a message...")
        self.text_input.send_callback = self.send_text_message
        self.text_input.setStyleSheet("""
            QTextEdit {
                background-color: #282A2C;
                color: #E3E3E3;
                border: 1px solid #3C4043;
                border-radius: 20px;
                padding: 12px 18px;
                font-size: 14px;
            }
            QTextEdit:focus {
                border-color: #8AB4F8;
            }
        """)
        self.text_input.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.text_input.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.text_input.setLineWrapMode(QTextEdit.LineWrapMode.WidgetWidth)
        self.text_input.setFixedHeight(48)
        self.text_input.textChanged.connect(self.adjust_input_height)
        text_row.addWidget(self.text_input)
        
        # Send button
        self.send_btn = QPushButton("➤")
        self.send_btn.setObjectName("sendButton")
        self.send_btn.clicked.connect(self.send_text_message)
        self.send_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        text_row.addWidget(self.send_btn)
        
        input_layout.addLayout(text_row)
        
        # Button row (mic + status) - SEGUNDO (abajo, fijo)
        button_row = QHBoxLayout()
        button_row.setSpacing(15)
        
        # Clear button (left)
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("clearButton")
        self.clear_btn.clicked.connect(self.clear_chat)
        self.clear_btn.setFixedWidth(70)
        button_row.addWidget(self.clear_btn)
        
        # Status label (center-left)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        button_row.addWidget(self.status_label, 1)

        # Context window usage indicator
        self.context_donut = ContextDonut()
        button_row.addWidget(self.context_donut)

        # MIC BUTTON (center-right) - Main voice button
        self.mic_btn = QPushButton("🎤")
        self.mic_btn.setObjectName("micButton")
        self.mic_btn.clicked.connect(self.toggle_recording)
        self.mic_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        button_row.addWidget(self.mic_btn)
        
        # LIVE MODE BUTTON - Start continuous conversation
        self.live_btn = QPushButton("✨")
        self.live_btn.setObjectName("liveStartButton")
        self.live_btn.setToolTip("Start Live Mode")
        self.live_btn.clicked.connect(self.enter_live_mode)
        self.live_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        button_row.addWidget(self.live_btn)
        
        input_layout.addLayout(button_row)
        
        input_bar_wrapper_layout.addWidget(input_bar)
        input_bar_wrapper_layout.addStretch()
        
        main_layout.addWidget(input_bar_wrapper)
        
        # Store reference to input_bar for showing/hiding
        self.input_bar = input_bar
        
        # Welcome message (language-dependent)
        self.add_bot_message(self._get_welcome_message())
    
    def _get_welcome_message(self):
        """Get welcome message based on current language"""
        if self.language == "spanish":
            return "¡Hola! Escribe un mensaje o toca el micrófono para hablar."
        else:
            return "Hello! Type a message or tap the microphone to speak."
    
    def toggle_recording(self):
        """Toggle between recording and not recording, or interrupt if speaking"""
        
        # If speaking, interrupt
        if self.is_speaking:
            self.interrupt_speaking()
            return
        
        if self.is_processing and not self.is_speaking:
            return
        
        if self.is_recording:
            self.stop_recording()
        else:
            self.start_recording()
    
    def interrupt_speaking(self):
        """Interrupt the bot while speaking"""
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.interrupt()
            # Wait with timeout to avoid blocking
            if not self.worker_thread.wait(1000):  # Wait max 1 second
                print("⚠️ Thread didn't finish in time, forcing cleanup")
            self.status_label.setText("Ready")
            self.is_speaking = False
            self.is_processing = False
            self.mic_btn.setText("🎤")
            self.mic_btn.setEnabled(True)
            self.send_btn.setEnabled(True)
            print("🛑 User interrupted playback")
    
    def adjust_input_height(self):
        """Ajusta la altura del input según el contenido (hasta 10 líneas)"""
        doc = self.text_input.document()
        doc_height = doc.size().height()
        
        # Calcular altura necesaria con padding
        content_height = int(doc_height + 24)  # 12px padding top + 12px bottom
        
        # Límites: mínimo 48px (1 línea), máximo ~240px (10 líneas)
        min_height = 48
        max_height = 240
        
        new_height = max(min_height, min(content_height, max_height))
        self.text_input.setFixedHeight(new_height)
    
    def send_text_message(self):
        """Send a text message typed by user"""
        text = self.text_input.toPlainText().strip()
        
        if not text:
            return
        
        if self.ai_manager is None:
            self.status_label.setText("AI not loaded!")
            return
        
        # If speaking, interrupt first
        if self.is_speaking:
            self.interrupt_speaking()
        
        if self.is_processing:
            return
        
        # Clear input
        self.text_input.clear()
        self.text_input.setFixedHeight(48)
        
        # Start processing
        self.is_processing = True
        self.mic_btn.setEnabled(False)
        self.send_btn.setEnabled(False)
        self.status_label.setText("Processing...")
        
        # Start worker thread with text
        self.worker_thread = WorkerThread(self.ai_manager, self.player, self.voice_speed)
        self.worker_thread.set_text(text)
        self.worker_thread.status_changed.connect(self.update_status)
        self.worker_thread.user_message.connect(self.add_user_message)
        self.worker_thread.bot_message.connect(self.add_bot_message)
        self.worker_thread.bot_message_update.connect(self.update_bot_message)
        self.worker_thread.bot_thinking_update.connect(self.update_bot_thinking)
        self.worker_thread.context_usage_updated.connect(self.update_context_donut)
        self.worker_thread.processing_complete.connect(self.on_processing_complete)
        self.worker_thread.speaking_started.connect(self.on_speaking_started)
        self.worker_thread.start()
    
    def start_recording(self):
        """Start manual recording"""
        
        if self.ai_manager is None:
            self.status_label.setText("AI not loaded!")
            return
        
        self.is_recording = True
        
        # Update UI
        self.mic_btn.setText("⏹")
        self.mic_btn.setStyleSheet("""
            QPushButton#micButton {
                background-color: #EA4335;
                color: white;
                border: none;
                border-radius: 32px;
                min-width: 64px;
                max-width: 64px;
                min-height: 64px;
                max-height: 64px;
                font-size: 26px;
            }
            QPushButton#micButton:hover {
                background-color: #F44336;
            }
        """)
        self.status_label.setText("🔴 Recording... Tap to send")
        
        # Start recording thread
        self.recorder_thread = ManualRecorderThread(self.recorder)
        self.recorder_thread.recording_finished.connect(self.on_recording_finished)
        self.recorder_thread.start()
        
        # Start pulse animation
        self.pulse_timer.start(500)
    
    def stop_recording(self):
        """Stop recording and process"""
        
        self.is_recording = False
        self.pulse_timer.stop()
        
        # Update UI
        self.mic_btn.setText("🎤")
        self.mic_btn.setStyleSheet("")  # Remove inline styles to use global CSS
        
        # Stop the recorder thread
        if self.recorder_thread and self.recorder_thread.isRunning():
            self.recorder_thread.stop_recording()
    
    def update_recording_animation(self):
        """Pulse animation for recording state"""
        self.pulse_state = (self.pulse_state + 1) % 2
        if self.pulse_state == 0:
            self.status_label.setText("🔴 Recording... Tap to send")
        else:
            self.status_label.setText("⚫ Recording... Tap to send")
    
    @pyqtSlot(object)
    def on_recording_finished(self, audio_data):
        """Handle finished recording"""
        
        if audio_data is None:
            self.status_label.setText("Ready")
            return
        
        # If auto_send is disabled, transcribe and put in text field
        if not self.auto_send:
            self.status_label.setText("Transcribing...")
            self.mic_btn.setEnabled(False)
            self.send_btn.setEnabled(False)
            
            # Transcribe in a background thread
            def transcribe_only():
                text = self.ai_manager.transcribe(audio_data)
                return text
            
            # Use QTimer to run transcription without blocking
            import concurrent.futures
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            future = executor.submit(transcribe_only)
            
            def on_transcribe_done():
                try:
                    transcribed_text = future.result(timeout=0.1)
                    if transcribed_text:
                        self.text_input.setText(transcribed_text)
                        self.text_input.setFocus()
                        self.status_label.setText("Review and send")
                    else:
                        self.status_label.setText("Ready")
                except concurrent.futures.TimeoutError:
                    # Not ready yet, check again
                    QTimer.singleShot(100, on_transcribe_done)
                    return
                except Exception as e:
                    print(f"Transcription error: {e}")
                    self.status_label.setText("Ready")
                finally:
                    if future.done():
                        self.mic_btn.setEnabled(True)
                        self.send_btn.setEnabled(True)
                        executor.shutdown(wait=False)
            
            QTimer.singleShot(100, on_transcribe_done)
            return
        
        # Auto-send mode (original behavior)
        self.is_processing = True
        self.mic_btn.setEnabled(False)
        self.status_label.setText("Processing...")
        
        # Start worker thread
        self.worker_thread = WorkerThread(self.ai_manager, self.player, self.voice_speed)
        self.worker_thread.set_audio(audio_data)
        self.worker_thread.status_changed.connect(self.update_status)
        self.worker_thread.user_message.connect(self.add_user_message)
        self.worker_thread.bot_message.connect(self.add_bot_message)
        self.worker_thread.bot_message_update.connect(self.update_bot_message)
        self.worker_thread.bot_thinking_update.connect(self.update_bot_thinking)
        self.worker_thread.context_usage_updated.connect(self.update_context_donut)
        self.worker_thread.processing_complete.connect(self.on_processing_complete)
        self.worker_thread.speaking_started.connect(self.on_speaking_started)
        self.worker_thread.start()
    
    @pyqtSlot()
    def on_processing_complete(self):
        """Called when processing is done"""
        self.is_processing = False
        self.is_speaking = False
        self.mic_btn.setEnabled(True)
        self.send_btn.setEnabled(True)
        self.mic_btn.setText("🎤")
        self.status_label.setText("Ready")
    
    @pyqtSlot()
    def on_speaking_started(self):
        """Called when TTS starts playing audio"""
        self.is_speaking = True
        self.mic_btn.setEnabled(True)  # Enable to allow interruption
        self.mic_btn.setText("⏹️")  # Change to stop icon
    
    def enter_live_mode(self):
        """Switch to Live Mode (continuous conversation)"""
        if self.ai_manager is None:
            self.status_label.setText("AI not loaded!")
            return
        
        # Stop any ongoing recording/processing
        if self.is_recording:
            self.stop_recording()
        if self.is_speaking:
            self.interrupt_speaking()
        
        # Hide input bar when entering live mode
        self.input_bar.setVisible(False)
        
        # Switch to Live Mode page
        self.stacked_widget.setCurrentIndex(1)
        self.live_mode_widget.start_live_mode()
        print("🎙️ Entering Live Mode")
    
    def exit_live_mode(self):
        """Return to Chat Mode from Live Mode"""
        # Show input bar again
        self.input_bar.setVisible(True)
        
        self.stacked_widget.setCurrentIndex(0)
        self.status_label.setText("Ready")
        print("💬 Returning to Chat Mode")
    
    def open_settings(self):
        """Open settings dialog"""
        dialog = SettingsDialog(self, self.auto_send, self.font_size_name, self.language, self.voice_speed, self.output_device, self.input_device)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.auto_send = dialog.get_auto_send()
            new_font_size = dialog.get_font_size()
            new_language = dialog.get_language()
            new_voice_speed = dialog.get_voice_speed()
            new_output_device = dialog.get_output_device()
            new_input_device = dialog.get_input_device()
            
            # Update font size if changed
            if new_font_size != self.font_size_name:
                self.font_size_name = new_font_size
                self.apply_font_size()
            
            # Update voice speed
            self.voice_speed = new_voice_speed
            
            # Update language if changed
            if new_language != self.language:
                old_language = self.language
                self.language = new_language
                
                # Resetear velocidad a 1.0 al cambiar idioma
                self.voice_speed = 1.0
                
                if self.ai_manager:
                    self.ai_manager.set_language(new_language)
                print(f"🌐 Language changed to: {new_language}")
                print(f"🔊 Voice speed reset to: 1.0x")
                
                # Update voice selector for new language
                self.update_voice_list()
                
                # Show language change notification in chat
                lang_display = "Español" if new_language == "spanish" else "English"
                self.add_bot_message(f"🌐 {self._get_language_change_message(new_language)}")
            
            # Update output device if changed
            if new_output_device != self.output_device:
                self.output_device = new_output_device
                self.player.device = new_output_device if new_output_device >= 0 else None
                print(f"🔈 Audio output device changed to index: {new_output_device}")

            # Update input device (microphone) if changed
            if new_input_device != self.input_device:
                self.input_device = new_input_device
                self.recorder.device = new_input_device if new_input_device >= 0 else None
                # Reiniciar stream si está activo
                if self.recorder.stream is not None:
                    self.recorder.stop_stream()
                print(f"🎤 Microphone device changed to index: {new_input_device}")

            # Save preferences
            self.preferences["auto_send"] = self.auto_send
            self.preferences["font_size"] = self.font_size_name
            self.preferences["language"] = self.language
            self.preferences["voice_speed"] = self.voice_speed
            self.preferences["english_voice"] = self.english_voice
            self.preferences["spanish_voice"] = self.spanish_voice
            self.preferences["output_device"] = self.output_device
            self.preferences["input_device"] = self.input_device
            save_preferences(self.preferences)
            
            mode_name = "Auto-send" if self.auto_send else "Manual review"
            lang_display = "English" if self.language == "english" else "Español"
            print(f"⚙️ Settings updated - Mode: {mode_name}, Font: {self.font_size_name}, Lang: {lang_display}, Speed: {self.voice_speed:.2f}x")
    
    def update_voice_list(self):
        """Update voice selector based on current language"""
        self.voice_selector.clear()
        
        if self.language == "english":
            self.voice_selector.addItems(self.available_english_voices)
            # Set saved English voice
            if self.english_voice in self.available_english_voices:
                idx = self.voice_selector.findText(self.english_voice)
                if idx >= 0:
                    self.voice_selector.setCurrentIndex(idx)
        else:  # spanish
            self.voice_selector.addItems(self.available_spanish_voices)
            # Set saved Spanish voice
            if self.spanish_voice in self.available_spanish_voices:
                idx = self.voice_selector.findText(self.spanish_voice)
                if idx >= 0:
                    self.voice_selector.setCurrentIndex(idx)
    
    def on_voice_changed(self, voice_name):
        """Handle voice selection change"""
        if not voice_name:
            return
        
        # Save voice preference based on language
        if self.language == "english":
            self.english_voice = voice_name
        else:
            self.spanish_voice = voice_name
        
        # Update AI manager
        if self.ai_manager:
            self.ai_manager.set_voice(voice_name)
        
        # Save preferences
        self.preferences["english_voice"] = self.english_voice
        self.preferences["spanish_voice"] = self.spanish_voice
        save_preferences(self.preferences)
        
        print(f"🎤 Voice changed to: {voice_name}")
    
    def _get_language_change_message(self, language):
        """Get language change notification message"""
        if language == "spanish":
            return "Idioma cambiado a Español. ¡Ahora puedes hablar en español!"
        else:
            return "Language changed to English. You can now speak in English!"
    
    def apply_font_size(self):
        """Apply font size to all UI elements"""
        font_config = get_font_size_config(self.font_size_name)
        
        # Update existing bubbles
        for bubble in self.chat_bubbles:
            if bubble:
                try:
                    bubble.update_font_size(font_config["bubble_text"])
                except RuntimeError:
                    pass  # Widget was deleted
        
        # Update input field
        self.text_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: #282A2C;
                color: #E3E3E3;
                border: 1px solid #3C4043;
                border-radius: 20px;
                padding: 12px 18px;
                font-size: {font_config["input_text"]}px;
            }}
            QLineEdit:focus {{
                border-color: #8AB4F8;
            }}
        """)
        
        # Update status label
        self.status_label.setStyleSheet(f"""
            color: #9AA0A6;
            font-size: {font_config["status_text"]}px;
            padding: 8px 16px;
            background: transparent;
        """)
    
    @pyqtSlot(str)
    def update_status(self, status):
        """Update status label"""
        self.status_label.setText(status)
    
    @pyqtSlot(str)
    def on_model_changed(self, model_name):
        """Handle model selection change"""
        if self.ai_manager:
            self.ai_manager.ollama_model = model_name
            print(f"Switched to: {model_name}")
            self.add_bot_message(f"[Model: {model_name}]")
            # Save preference
            self.preferences["model"] = model_name
            save_preferences(self.preferences)
    
    @pyqtSlot(str)
    def add_user_message(self, message):
        """Add user message with bubble style - aligned right"""
        wrapper = QWidget()
        wrapper.setStyleSheet("background-color: transparent;")
        wrapper_layout = QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(60, 4, 16, 4)
        wrapper_layout.addStretch()
        
        font_config = get_font_size_config(self.font_size_name)
        bubble = UserMessageBubble(message, font_size=font_config["bubble_text"])
        self.chat_bubbles.append(bubble)
        wrapper_layout.addWidget(bubble)
        
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, wrapper)
        self.scroll_to_bottom()
    
    @pyqtSlot(str)
    def add_bot_message(self, message):
        """Add bot message with avatar and markdown - centered with fixed width"""
        wrapper = QWidget()
        wrapper.setStyleSheet("background-color: transparent;")
        wrapper_layout = QHBoxLayout(wrapper)
        wrapper_layout.setContentsMargins(0, 4, 0, 4)
        
        # Add stretch on both sides to center the bot message
        wrapper_layout.addStretch()
        
        # Container with fixed width
        bot_container = QWidget()
        bot_container.setMaximumWidth(1000)
        bot_container.setMinimumWidth(1000)
        bot_container.setStyleSheet("background-color: transparent;")
        bot_container_layout = QHBoxLayout(bot_container)
        bot_container_layout.setContentsMargins(16, 0, 16, 0)
        
        font_config = get_font_size_config(self.font_size_name)
        bot_widget = BotMessageWidget(message, font_size=font_config["bubble_text"])
        self.chat_bubbles.append(bot_widget)
        self.current_bot_bubble = bot_widget  # Track for streaming updates
        bot_container_layout.addWidget(bot_widget)
        
        wrapper_layout.addWidget(bot_container)
        wrapper_layout.addStretch()
        
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, wrapper)
        self.scroll_to_bottom()
    
    @pyqtSlot(str)
    def update_bot_message(self, message):
        """Update the current bot bubble text (for streaming)"""
        if hasattr(self, 'current_bot_bubble') and self.current_bot_bubble:
            try:
                self.current_bot_bubble.update_text(message)
                self.scroll_to_bottom()
            except RuntimeError:
                pass  # Widget was deleted

    @pyqtSlot(str)
    @pyqtSlot(int, int)
    def update_context_donut(self, used: int, total: int):
        """Update the context-window donut indicator."""
        self.context_donut.update_usage(used, total)

    def update_bot_thinking(self, thinking_text):
        """Update the thinking panel on the current bot bubble (streaming)."""
        if hasattr(self, 'current_bot_bubble') and self.current_bot_bubble:
            try:
                self.current_bot_bubble.update_thinking(thinking_text)
                self.scroll_to_bottom()
            except RuntimeError:
                pass
    
    def scroll_to_bottom(self):
        """Scroll to bottom of chat only if user is already at the bottom"""
        scrollbar = self.scroll_area.verticalScrollBar()
        # Check if user is near the bottom (within 50 pixels)
        is_at_bottom = scrollbar.value() >= scrollbar.maximum() - 50
        
        if is_at_bottom:
            QTimer.singleShot(100, lambda: 
                scrollbar.setValue(scrollbar.maximum())
            )
    
    def clear_chat(self):
        """Clear all messages"""
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Clear bubbles list
        self.chat_bubbles.clear()
        
        if self.ai_manager:
            self.ai_manager.reset_conversation()
        
        self.add_bot_message("Chat cleared. Tap the mic to start a new conversation!")

    # ── Startup voice scanner ──────────────────────────────────────────────────

    def _check_voices_on_startup(self):
        """
        Scan the voices/ folder and show the appropriate dialog:
          • No voices at all  → VoiceSetupDialog (download instructions)
          • New unknown folder → NewVoiceClassifierDialog (per unknown folder)
        """
        from voice_scanner import scan_voices_folder, save_voice_language

        scan = scan_voices_folder()

        # ── Case 1: no voices found ────────────────────────────────────────
        if not scan["has_any"]:
            dlg = VoiceSetupDialog(self)
            dlg.exec()
            return

        # ── Case 2: classify every newly-detected Sherpa folder ──────────
        for voice in scan["sherpa_voices"]:
            if not voice["is_new"]:
                continue  # already classified, skip

            dlg = NewVoiceClassifierDialog(voice["folder"], self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                lang = dlg.get_language()
                save_voice_language(voice["folder"], lang)
                print(f"✓ Voice pack '{voice['folder']}' classified as: {lang}")

    def closeEvent(self, event):
        """Cleanup on close"""
        # Stop Live Mode if active
        if hasattr(self, 'live_mode_widget') and self.live_mode_widget.is_active:
            self.live_mode_widget.stop_live_mode()        
        if self.recorder_thread and self.recorder_thread.isRunning():
            self.recorder_thread.stop_recording()
            self.recorder_thread.wait()
        
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.wait()
        
        event.accept()


def main():
    app = QApplication(sys.argv)
    
    font = QFont("Google Sans", 10)
    font.setStyleHint(QFont.StyleHint.SansSerif)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
