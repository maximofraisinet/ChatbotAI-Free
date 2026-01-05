"""
Main Application - Voice Chatbot with PyQt6
Multi-threaded architecture to prevent UI freezing
"""

import sys
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, 
    QHBoxLayout, QLabel, QScrollArea, QPushButton
)
from PyQt6.QtCore import Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QFont

from styles import DARK_STYLE
from audio_utils import AudioRecorder, AudioPlayer
from ai_manager import AIManager


class ListenerThread(QThread):
    """Thread for continuous microphone listening"""
    
    audio_detected = pyqtSignal(object)  # Emits recorded audio data
    
    def __init__(self, recorder):
        super().__init__()
        self.recorder = recorder
        self.is_running = False
    
    def run(self):
        """Continuously listen for speech"""
        self.is_running = True
        self.recorder.start_stream()
        
        while self.is_running:
            audio_data = self.recorder.record_until_silence()
            
            if audio_data is not None and len(audio_data) > 0:
                self.audio_detected.emit(audio_data)
            
            if not self.is_running:
                break
        
        self.recorder.stop_stream()
    
    def stop(self):
        """Stop the listening thread"""
        self.is_running = False


class WorkerThread(QThread):
    """Thread for processing: Transcribe -> Think -> Speak"""
    
    status_changed = pyqtSignal(str)  # Status updates
    user_message = pyqtSignal(str)    # User's transcribed message
    bot_message = pyqtSignal(str)     # Bot's text response
    playback_started = pyqtSignal()   # When audio playback begins
    playback_finished = pyqtSignal()  # When audio playback ends
    
    def __init__(self, ai_manager, audio_player, audio_recorder):
        super().__init__()
        self.ai_manager = ai_manager
        self.audio_player = audio_player
        self.audio_recorder = audio_recorder
        self.audio_queue = []
        self.is_running = False
    
    def add_audio(self, audio_data):
        """Add audio data to processing queue"""
        self.audio_queue.append(audio_data)
    
    def run(self):
        """Process audio queue"""
        self.is_running = True
        
        while self.is_running:
            if self.audio_queue:
                audio_data = self.audio_queue.pop(0)
                self._process_audio(audio_data)
            else:
                self.msleep(100)  # Sleep briefly if no audio to process
    
    def _process_audio(self, audio_data):
        """Process a single audio chunk through the pipeline"""
        
        # Step 1: Transcribe
        self.status_changed.emit("Thinking...")
        user_text = self.ai_manager.transcribe(audio_data)
        
        if not user_text:
            self.status_changed.emit("Listening...")
            return
        
        self.user_message.emit(user_text)
        
        # Step 2: Get LLM response
        bot_text = self.ai_manager.get_llm_response(user_text)
        self.bot_message.emit(bot_text)
        
        # Step 3: Generate speech
        self.status_changed.emit("Speaking...")
        audio_output, sample_rate = self.ai_manager.text_to_speech(bot_text)
        
        if audio_output is None or len(audio_output) == 0:
            print("No audio generated, skipping playback")
            self.status_changed.emit("Listening...")
            return
        
        # Step 4: Play audio (pause recording to prevent feedback)
        self.audio_recorder.pause_recording()
        self.playback_started.emit()
        
        self.audio_player.play(audio_output, sample_rate)
        
        self.playback_finished.emit()
        self.audio_recorder.resume_recording()
        
        self.status_changed.emit("Listening...")
    
    def stop(self):
        """Stop the worker thread"""
        self.is_running = False


class ChatBubble(QLabel):
    """Custom chat bubble widget"""
    
    def __init__(self, text, is_user=False):
        super().__init__(text)
        
        self.setWordWrap(True)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        
        if is_user:
            self.setProperty("class", "user_message")
            self.setAlignment(Qt.AlignmentFlag.AlignRight)
        else:
            self.setProperty("class", "bot_message")
            self.setAlignment(Qt.AlignmentFlag.AlignLeft)
        
        self.setMaximumWidth(600)


class MainWindow(QMainWindow):
    """Main application window"""
    
    def __init__(self):
        super().__init__()
        
        self.setWindowTitle("Voice Chatbot - AI Assistant")
        self.setGeometry(100, 100, 900, 700)
        
        # Initialize audio components
        self.recorder = AudioRecorder()
        self.player = AudioPlayer()
        
        # Initialize AI Manager (loads all models)
        self.status_label = None  # Will be created in init_ui
        self.update_status("Initializing AI models...")
        
        try:
            self.ai_manager = AIManager()
        except Exception as e:
            print(f"Error initializing AI Manager: {e}")
            self.show_error_and_exit(str(e))
            return
        
        # Initialize threads
        self.listener_thread = None
        self.worker_thread = None
        
        # Setup UI
        self.init_ui()
        
        # Apply styles
        self.setStyleSheet(DARK_STYLE)
        
        # Auto-start listening
        self.start_listening()
    
    def init_ui(self):
        """Initialize the user interface"""
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout
        main_layout = QVBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Status indicator
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setProperty("class", "status_idle")
        main_layout.addWidget(self.status_label)
        
        # Chat scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        
        # Chat container
        self.chat_widget = QWidget()
        self.chat_widget.setObjectName("chatContainer")
        self.chat_layout = QVBoxLayout()
        self.chat_layout.addStretch()
        self.chat_widget.setLayout(self.chat_layout)
        
        scroll_area.setWidget(self.chat_widget)
        main_layout.addWidget(scroll_area)
        
        # Store scroll area reference for auto-scrolling
        self.scroll_area = scroll_area
        
        # Control buttons
        button_layout = QHBoxLayout()
        
        self.start_button = QPushButton("Start Listening")
        self.start_button.setObjectName("startButton")
        self.start_button.clicked.connect(self.start_listening)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self.stop_listening)
        self.stop_button.setEnabled(False)
        
        self.clear_button = QPushButton("Clear Chat")
        self.clear_button.clicked.connect(self.clear_chat)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        button_layout.addWidget(self.clear_button)
        
        main_layout.addLayout(button_layout)
        
        # Add welcome message
        self.add_bot_message("Hello! I'm your AI voice assistant. Start speaking anytime!")
    
    def start_listening(self):
        """Start the listening and processing threads"""
        
        if self.listener_thread is not None and self.listener_thread.isRunning():
            return
        
        # Create listener thread
        self.listener_thread = ListenerThread(self.recorder)
        self.listener_thread.audio_detected.connect(self.on_audio_detected)
        
        # Create worker thread
        self.worker_thread = WorkerThread(self.ai_manager, self.player, self.recorder)
        self.worker_thread.status_changed.connect(self.update_status)
        self.worker_thread.user_message.connect(self.add_user_message)
        self.worker_thread.bot_message.connect(self.add_bot_message)
        
        # Start threads
        self.listener_thread.start()
        self.worker_thread.start()
        
        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.update_status("Listening...")
    
    def stop_listening(self):
        """Stop all threads"""
        
        self.update_status("Stopping...")
        
        # Stop listener thread
        if self.listener_thread is not None:
            self.listener_thread.stop()
            self.listener_thread.wait()
            self.listener_thread = None
        
        # Stop worker thread
        if self.worker_thread is not None:
            self.worker_thread.stop()
            self.worker_thread.wait()
            self.worker_thread = None
        
        # Stop any audio playback
        self.player.stop()
        
        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.update_status("Stopped")
    
    @pyqtSlot(object)
    def on_audio_detected(self, audio_data):
        """Handle audio detected by listener thread"""
        if self.worker_thread is not None:
            self.worker_thread.add_audio(audio_data)
    
    @pyqtSlot(str)
    def update_status(self, status):
        """Update status label"""
        if self.status_label is None:
            print(f"Status: {status}")
            return
            
        self.status_label.setText(status)
        
        # Update status color
        if "Listening" in status:
            self.status_label.setProperty("class", "status_listening")
        elif "Thinking" in status:
            self.status_label.setProperty("class", "status_thinking")
        elif "Speaking" in status:
            self.status_label.setProperty("class", "status_speaking")
        else:
            self.status_label.setProperty("class", "status_idle")
        
        # Force style update
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)
    
    @pyqtSlot(str)
    def add_user_message(self, message):
        """Add user message bubble to chat"""
        bubble = ChatBubble(message, is_user=True)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        self.scroll_to_bottom()
    
    @pyqtSlot(str)
    def add_bot_message(self, message):
        """Add bot message bubble to chat"""
        bubble = ChatBubble(message, is_user=False)
        self.chat_layout.insertWidget(self.chat_layout.count() - 1, bubble)
        self.scroll_to_bottom()
    
    def scroll_to_bottom(self):
        """Scroll chat to bottom"""
        scrollbar = self.scroll_area.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear_chat(self):
        """Clear all chat messages"""
        # Remove all message bubbles except the stretch
        while self.chat_layout.count() > 1:
            item = self.chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Reset conversation history
        self.ai_manager.reset_conversation()
        
        # Add welcome message
        self.add_bot_message("Chat cleared. Ready for a new conversation!")
    
    def show_error_and_exit(self, error_message):
        """Show error message and exit"""
        from PyQt6.QtWidgets import QMessageBox
        
        msg = QMessageBox()
        msg.setIcon(QMessageBox.Icon.Critical)
        msg.setText("Initialization Error")
        msg.setInformativeText(error_message)
        msg.setWindowTitle("Error")
        msg.exec()
        
        sys.exit(1)
    
    def closeEvent(self, event):
        """Handle window close event"""
        self.stop_listening()
        event.accept()


def main():
    """Main application entry point"""
    
    app = QApplication(sys.argv)
    
    # Set application font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    # Create and show main window
    window = MainWindow()
    window.show()
    
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
