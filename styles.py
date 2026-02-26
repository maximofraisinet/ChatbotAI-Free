"""
Google Gemini/ChatGPT-Inspired Dark Mode Styles
Modern, minimal, clean design with distinct message styles
"""

# Color Palette (Gemini Dark)
COLORS = {
    'background': '#131314',
    'surface': '#1E1F20',
    'surface_variant': '#282A2C',
    'user_bubble': '#303136',  # Gris oscuro suave para usuario
    'bot_bubble': 'transparent',  # Sin fondo para bot
    'text_primary': '#E3E3E3',
    'text_secondary': '#9AA0A6',
    'accent': '#8AB4F8',
    'mic_button': '#1A73E8',
    'mic_recording': '#EA4335',
    'border': '#3C4043',
    'code_bg': '#1E1F20',
    'code_border': '#3C4043',
}

GEMINI_STYLE = """
/* ===== MAIN WINDOW ===== */
QMainWindow {
    background-color: #131314;
}

QWidget {
    background-color: #131314;
    color: #E3E3E3;
    font-family: 'Google Sans', 'Segoe UI', 'Roboto', Arial, sans-serif;
}

/* ===== SCROLL AREA ===== */
QScrollArea {
    background-color: #131314;
    border: none;
}

QScrollArea > QWidget > QWidget {
    background-color: #131314;
}

QWidget#chatContainer {
    background-color: #131314;
}

/* ===== HEADER ===== */
QWidget#headerWidget {
    background-color: #131314;
    border-bottom: 1px solid #282A2C;
}

/* ===== MODEL SELECTOR - Pill Style ===== */
QComboBox#modelSelector {
    background-color: #282A2C;
    color: #E3E3E3;
    border: 1px solid #3C4043;
    border-radius: 18px;
    padding: 10px 20px;
    font-size: 13px;
    min-width: 200px;
}

QComboBox#modelSelector:hover {
    background-color: #3C4043;
    border-color: #5F6368;
}

QComboBox#modelSelector::drop-down {
    border: none;
    padding-right: 15px;
}

QComboBox#modelSelector::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #9AA0A6;
    margin-right: 10px;
}

QComboBox#modelSelector QAbstractItemView {
    background-color: #282A2C;
    color: #E3E3E3;
    selection-background-color: #3C4043;
    border: 1px solid #3C4043;
    border-radius: 12px;
    padding: 8px;
    outline: none;
}

QComboBox#modelSelector QAbstractItemView::item {
    padding: 10px 16px;
    border-radius: 8px;
    margin: 2px 4px;
}

QComboBox#modelSelector QAbstractItemView::item:hover {
    background-color: #3C4043;
}

/* ===== VOICE SELECTOR ===== */
QComboBox#voiceSelector {
    background-color: #282A2C;
    color: #E3E3E3;
    border: 1px solid #3C4043;
    border-radius: 18px;
    padding: 10px 20px;
    font-size: 13px;
    min-width: 150px;
}

QComboBox#voiceSelector:hover {
    background-color: #3C4043;
    border-color: #5F6368;
}

QComboBox#voiceSelector::drop-down {
    border: none;
    padding-right: 15px;
}

QComboBox#voiceSelector::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #9AA0A6;
    margin-right: 10px;
}

QComboBox#voiceSelector QAbstractItemView {
    background-color: #282A2C;
    color: #E3E3E3;
    selection-background-color: #3C4043;
    border: 1px solid #3C4043;
    border-radius: 12px;
    padding: 8px;
    outline: none;
}

QComboBox#voiceSelector QAbstractItemView::item {
    padding: 10px 16px;
    border-radius: 8px;
    margin: 2px 4px;
}

QComboBox#voiceSelector QAbstractItemView::item:hover {
    background-color: #3C4043;
}

/* ===== STATUS LABEL ===== */
QLabel#statusLabel {
    color: #9AA0A6;
    font-size: 13px;
    padding: 8px 16px;
    background: transparent;
}

/* ===== USER MESSAGE BUBBLE (mantiene burbuja) ===== */
QFrame#userBubble {
    background-color: #303136;
    border-radius: 22px;
    border: none;
}

/* ===== BOT MESSAGE CONTAINER (sin burbuja, estilo ChatGPT) ===== */
QFrame#botMessage {
    background-color: transparent;
    border: none;
    border-radius: 0px;
}

/* ===== BOT AVATAR ===== */
QLabel#botAvatar {
    background-color: #8AB4F8;
    border-radius: 16px;
    color: #131314;
    font-weight: bold;
    font-size: 14px;
}

/* ===== MESSAGE TEXT LABELS ===== */
QLabel#userBubbleText {
    color: #FFFFFF;
    font-size: 15px;
    padding: 14px 20px;
    background: transparent;
    line-height: 1.5;
}

QLabel#botMessageText {
    color: #E3E3E3;
    font-size: 15px;
    padding: 4px 0px;
    background: transparent;
    line-height: 1.6;
}

/* ===== MARKDOWN STYLED TEXT (para QTextBrowser) ===== */
QTextBrowser#markdownText {
    background-color: transparent;
    border: none;
    color: #E3E3E3;
    font-size: 15px;
    line-height: 1.6;
    selection-background-color: #3C4043;
}

/* ===== LEGACY BUBBLE TEXT (compatibilidad) ===== */
QLabel#bubbleText {
    color: #E3E3E3;
    font-size: 15px;
    padding: 14px 18px;
    background: transparent;
}

/* ===== INPUT BAR ===== */
QWidget#inputBar {
    background-color: #1E1F20;
    border-top: 1px solid #282A2C;
    border-radius: 16px;
}

/* ===== SEND BUTTON ===== */
QPushButton#sendButton {
    background-color: transparent;
    color: #E3E3E3;
    border: none;
    border-radius: 20px;
    min-width: 40px;
    max-width: 40px;
    min-height: 40px;
    max-height: 40px;
    font-size: 18px;
}

QPushButton#sendButton:hover {
    background-color: #282A2C;
}

QPushButton#sendButton:pressed {
    background-color: #3C4043;
}

QPushButton#sendButton:disabled {
    background-color: transparent;
    color: #5F6368;
}

/* ===== MIC BUTTON - Large Circular ===== */
QPushButton#micButton {
    background-color: transparent;
    color: #E3E3E3;
    border: none;
    border-radius: 32px;
    min-width: 64px;
    max-width: 64px;
    min-height: 64px;
    max-height: 64px;
    font-size: 26px;
}

QPushButton#micButton:hover {
    background-color: #282A2C;
}

QPushButton#micButton:pressed {
    background-color: #3C4043;
}

/* ===== CLEAR BUTTON - Ghost ===== */
QPushButton#clearButton {
    background-color: transparent;
    color: #9AA0A6;
    border: 1px solid #3C4043;
    border-radius: 20px;
    padding: 10px 20px;
    font-size: 13px;
}

QPushButton#clearButton:hover {
    background-color: #282A2C;
    color: #E3E3E3;
    border-color: #5F6368;
}

/* ===== SCROLLBAR - Minimal ===== */
QScrollBar:vertical {
    border: none;
    background-color: #131314;
    width: 8px;
    margin: 0px;
}

QScrollBar::handle:vertical {
    background-color: #3C4043;
    border-radius: 4px;
    min-height: 40px;
}

QScrollBar::handle:vertical:hover {
    background-color: #5F6368;
}

QScrollBar::add-line:vertical,
QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical,
QScrollBar::sub-page:vertical {
    background: none;
    height: 0px;
}

/* ===== RECORDING PULSE ===== */
QLabel#recordingDot {
    background-color: #EA4335;
    border-radius: 6px;
    min-width: 12px;
    max-width: 12px;
    min-height: 12px;
    max-height: 12px;
}

/* ===== LIVE MODE ===== */
QWidget#liveModeWidget {
    background-color: #131314;
}

QWidget#liveIndicatorContainer {
    background-color: transparent;
}

QLabel#liveStateLabel {
    color: #9AA0A6;
    font-size: 16px;
    font-weight: bold;
    background: transparent;
}

QLabel#liveTranscriptLabel {
    color: #E3E3E3;
    font-size: 14px;
    background: transparent;
    padding: 20px;
}

QPushButton#liveMuteButton {
    background-color: #282A2C;
    color: #E3E3E3;
    border: 2px solid #3C4043;
    border-radius: 35px;
    min-width: 70px;
    max-width: 70px;
    min-height: 70px;
    max-height: 70px;
    font-size: 28px;
}

QPushButton#liveMuteButton:hover {
    background-color: #3C4043;
    border-color: #5F6368;
}

QPushButton#liveMuteButton:checked {
    background-color: #EA4335;
    border-color: #EA4335;
}

QPushButton#liveExitButton {
    background-color: #EA4335;
    color: white;
    border: none;
    border-radius: 35px;
    min-width: 70px;
    max-width: 70px;
    min-height: 70px;
    max-height: 70px;
    font-size: 28px;
}

QPushButton#liveExitButton:hover {
    background-color: #F44336;
}

QPushButton#liveStartButton {
    background-color: transparent;
    color: #E3E3E3;
    border: none;
    border-radius: 30px;
    min-width: 60px;
    max-width: 60px;
    min-height: 60px;
    max-height: 60px;
    font-size: 24px;
}

QPushButton#liveStartButton:hover {
    background-color: #282A2C;
}

QPushButton#liveStartButton:pressed {
    background-color: #3C4043;
}

/* ===== TOOLTIP ===== */
QToolTip {
    background-color: #202124;
    color: #E3E3E3;
    border: 1px solid #5C6166;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    font-family: 'Google Sans', 'Segoe UI', Arial, sans-serif;
}
"""

# For backwards compatibility
DARK_STYLE = GEMINI_STYLE
