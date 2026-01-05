"""
Dark Mode QSS Styles for Voice Chatbot
Modern WhatsApp/iMessage inspired design
"""

DARK_STYLE = """
/* Main Window */
QMainWindow {
    background-color: #121212;
}

/* Chat Container */
QScrollArea {
    background-color: #121212;
    border: none;
}

QWidget#chatContainer {
    background-color: #121212;
}

/* Message Bubbles - User (Right) */
QLabel.user_message {
    background-color: #005C4B;
    color: #FFFFFF;
    padding: 12px 16px;
    border-radius: 18px;
    font-size: 14px;
    font-family: 'Segoe UI', Arial, sans-serif;
    margin: 6px 60px 6px 20px;
}

/* Message Bubbles - Bot (Left) */
QLabel.bot_message {
    background-color: #1F1F1F;
    color: #E8E8E8;
    padding: 12px 16px;
    border-radius: 18px;
    font-size: 14px;
    font-family: 'Segoe UI', Arial, sans-serif;
    margin: 6px 20px 6px 60px;
}

/* Status Indicator */
QLabel#statusLabel {
    background-color: #2A2A2A;
    color: #8AB4F8;
    padding: 8px 16px;
    border-radius: 12px;
    font-size: 13px;
    font-weight: bold;
    margin: 10px;
}

/* Status Colors */
QLabel.status_listening {
    color: #81C784;
}

QLabel.status_thinking {
    color: #FFD54F;
}

QLabel.status_speaking {
    color: #64B5F6;
}

QLabel.status_idle {
    color: #B0B0B0;
}

/* Control Buttons */
QPushButton {
    background-color: #2A2A2A;
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #3A3A3A;
}

QPushButton:pressed {
    background-color: #1A1A1A;
}

QPushButton#startButton {
    background-color: #005C4B;
}

QPushButton#startButton:hover {
    background-color: #007A63;
}

QPushButton#stopButton {
    background-color: #D32F2F;
}

QPushButton#stopButton:hover {
    background-color: #E53935;
}

/* Scrollbar */
QScrollBar:vertical {
    border: none;
    background-color: #121212;
    width: 10px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #3A3A3A;
    border-radius: 5px;
    min-height: 20px;
}

QScrollBar::handle:vertical:hover {
    background-color: #4A4A4A;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    background: none;
}
"""
