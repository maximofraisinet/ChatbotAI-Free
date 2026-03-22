"""
Google Gemini Inspired Dark Theme
Modern, clean design with refined Gemini aesthetics
"""

COLORS = {
    'background': '#101417',
    'surface': '#1C1E21',
    'surface_variant': '#2D2F31',
    'user_bubble': '#2D2F31',
    'bot_bubble': 'transparent',
    'text_primary': '#E8EAED',
    'text_secondary': '#9AA0A6',
    'accent': '#8AB4F8',
    'accent_blue': '#4285F4',
    'mic_button': '#4285F4',
    'mic_recording': '#EA4335',
    'border': '#3C4043',
    'code_bg': '#1C1E21',
    'code_border': '#3C4043',
    'success': '#34A853',
    'warning': '#FBBC04',
    'error': '#EA4335',
}

GEMINI_STYLE = """
/* ===== MAIN WINDOW ===== */
QMainWindow {
    background-color: #101417;
}

/* ===== BASE WIDGET ===== */
QWidget {
    background-color: #101417;
    color: #E8EAED;
    font-family: 'Google Sans', 'Segoe UI', system-ui, sans-serif;
}

/* ===== SCROLL AREA ===== */
QScrollArea {
    background-color: #101417;
    border: none;
}

QScrollArea > QWidget > QWidget {
    background-color: #101417;
}

QWidget#chatContainer {
    background-color: #101417;
}

/* ===== HEADER ===== */
QWidget#headerWidget {
    background-color: #101417;
    border-bottom: 1px solid #3C4043;
}

/* ===== MODEL SELECTOR ===== */
QComboBox#modelSelector {
    background-color: #2D2F31;
    color: #E8EAED;
    border: 1px solid #3C4043;
    border-radius: 20px;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: 500;
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
    background-color: #1C1E21;
    color: #E8EAED;
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
    background-color: #2D2F31;
    color: #E8EAED;
    border: 1px solid #3C4043;
    border-radius: 20px;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: 500;
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
    background-color: #1C1E21;
    color: #E8EAED;
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

/* ===== USER MESSAGE BUBBLE ===== */
QFrame#userBubble {
    background-color: #2D2F31;
    border: 1px solid #3C4043;
    border-radius: 18px;
}

/* ===== BOT MESSAGE CONTAINER ===== */
QFrame#botMessage {
    background-color: transparent;
    border: none;
    border-radius: 0px;
}

/* ===== BOT AVATAR ===== */
QLabel#botAvatar {
    background: qlineargradient(x1:0%, y1:0%, x2:100%, y2:100%, stop:0% #8AB4F8, stop:50% #7C9AFE, stop:100% #4285F4);
    border-radius: 16px;
    color: #101417;
    font-weight: bold;
    font-size: 14px;
}

/* ===== MESSAGE TEXT LABELS ===== */
QLabel#userBubbleText {
    color: #E8EAED;
    font-size: 15px;
    padding: 14px 20px;
    background: transparent;
    line-height: 1.5;
}

QLabel#botMessageText {
    color: #E8EAED;
    font-size: 15px;
    padding: 4px 0px;
    background: transparent;
    line-height: 1.6;
}

/* ===== MARKDOWN TEXT ===== */
QTextBrowser#markdownText {
    background-color: transparent;
    border: none;
    color: #E8EAED;
    font-size: 15px;
    line-height: 1.6;
    selection-background-color: #3C4043;
}

/* ===== LEGACY BUBBLE TEXT ===== */
QLabel#bubbleText {
    color: #E8EAED;
    font-size: 15px;
    padding: 14px 18px;
    background: transparent;
}

/* ===== INPUT BAR ===== */
QWidget#inputBar {
    background-color: #1C1E21;
    border-top: 1px solid #3C4043;
    border-radius: 16px;
}

/* ===== SEND BUTTON ===== */
QPushButton#sendButton {
    background-color: transparent;
    color: #9AA0A6;
    border: none;
    border-radius: 20px;
    min-width: 40px;
    max-width: 40px;
    min-height: 40px;
    max-height: 40px;
    font-size: 18px;
}

QPushButton#sendButton:hover {
    background-color: #2D2F31;
    color: #8AB4F8;
}

QPushButton#sendButton:pressed {
    background-color: #3C4043;
}

QPushButton#sendButton:disabled {
    background-color: transparent;
    color: #5F6368;
}

/* ===== MIC BUTTON ===== */
QPushButton#micButton {
    background-color: #2D2F31;
    color: #4285F4;
    border: 2px solid #3C4043;
    border-radius: 32px;
    min-width: 64px;
    max-width: 64px;
    min-height: 64px;
    max-height: 64px;
    font-size: 26px;
}

QPushButton#micButton:hover {
    background-color: #3C4043;
    border-color: #4285F4;
}

QPushButton#micButton:pressed {
    background-color: #4285F4;
    color: #FFFFFF;
}

/* ===== CLEAR BUTTON ===== */
QPushButton#clearButton {
    background-color: transparent;
    color: #9AA0A6;
    border: 1px solid #3C4043;
    border-radius: 20px;
    padding: 10px 20px;
    font-size: 13px;
}

QPushButton#clearButton:hover {
    background-color: #2D2F31;
    color: #E8EAED;
    border-color: #5F6368;
}

/* ===== SCROLLBAR ===== */
QScrollBar:vertical {
    border: none;
    background-color: #101417;
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
    background-color: #101417;
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
    color: #E8EAED;
    font-size: 14px;
    background: transparent;
    padding: 20px;
}

QPushButton#liveMuteButton {
    background-color: #2D2F31;
    color: #E8EAED;
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
    color: white;
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
    background-color: #2D2F31;
    color: #4285F4;
    border: 2px solid #3C4043;
    border-radius: 30px;
    min-width: 60px;
    max-width: 60px;
    min-height: 60px;
    max-height: 60px;
    font-size: 24px;
}

QPushButton#liveStartButton:hover {
    background-color: #3C4043;
    border-color: #4285F4;
}

QPushButton#liveStartButton:pressed {
    background-color: #4285F4;
    color: white;
}

QPushButton#practiceStartButton {
    background-color: #2D2F31;
    color: #8AB4F8;
    border: 2px solid #3C4043;
    border-radius: 30px;
    min-width: 60px;
    max-width: 60px;
    min-height: 60px;
    max-height: 60px;
    font-size: 24px;
}

QPushButton#practiceStartButton:hover {
    background-color: #3C4043;
    border-color: #8AB4F8;
}

QPushButton#practiceStartButton:pressed {
    background-color: #8AB4F8;
    color: #101417;
}

/* ===== TOOLTIP ===== */
QToolTip {
    background-color: #1C1E21;
    color: #E8EAED;
    border: 1px solid #5F6368;
    border-radius: 8px;
    padding: 8px 12px;
    font-size: 12px;
    font-family: 'Google Sans', 'Segoe UI', system-ui, sans-serif;
}

/* ===== SIDEBAR ===== */
QWidget#sidebarWidget {
    background-color: #1C1E21;
    border-right: 1px solid #3C4043;
}

QPushButton#newChatButton {
    background-color: #2D2F31;
    color: #E8EAED;
    border: 1px solid #3C4043;
    border-radius: 20px;
    padding: 10px 16px;
    font-size: 14px;
    font-weight: 600;
    text-align: left;
}

QPushButton#newChatButton:hover {
    background-color: #3C4043;
    border-color: #8AB4F8;
}

QPushButton#hamburgerButton {
    background-color: transparent;
    color: #9AA0A6;
    border: none;
    border-radius: 20px;
    font-size: 22px;
    min-width: 40px;
    max-width: 40px;
    min-height: 40px;
    max-height: 40px;
}

QPushButton#hamburgerButton:hover {
    background-color: #2D2F31;
    color: #E8EAED;
}

QListWidget#chatListWidget {
    background-color: transparent;
    border: none;
    outline: none;
    font-size: 13px;
    color: #E8EAED;
}

QListWidget#chatListWidget::item {
    background-color: transparent;
    border-radius: 12px;
    padding: 10px 14px;
    margin: 2px 6px;
    color: #C4C7C5;
}

QListWidget#chatListWidget::item:hover {
    background-color: #2D2F31;
}

QListWidget#chatListWidget::item:selected {
    background-color: #3C4043;
    color: #E8EAED;
}

QLabel#sidebarTitle {
    color: #9AA0A6;
    font-size: 11px;
    font-weight: bold;
    padding-left: 14px;
    background-color: transparent;
}

/* ===== CHARACTER SELECTOR ===== */
QComboBox#characterSelector {
    background-color: #2D2F31;
    color: #E8EAED;
    border: 1px solid #3C4043;
    border-radius: 20px;
    padding: 10px 20px;
    font-size: 13px;
    font-weight: 500;
    min-width: 150px;
}

QComboBox#characterSelector:hover {
    background-color: #3C4043;
    border-color: #5F6368;
}

QComboBox#characterSelector::drop-down {
    border: none;
    padding-right: 15px;
}

QComboBox#characterSelector::down-arrow {
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid #9AA0A6;
    margin-right: 10px;
}

QComboBox#characterSelector QAbstractItemView {
    background-color: #1C1E21;
    color: #E8EAED;
    selection-background-color: #3C4043;
    border: 1px solid #3C4043;
    border-radius: 12px;
    padding: 8px;
    outline: none;
}

QComboBox#characterSelector QAbstractItemView::item {
    padding: 10px 16px;
    border-radius: 8px;
    margin: 2px 4px;
}

QComboBox#characterSelector QAbstractItemView::item:hover {
    background-color: #3C4043;
}
"""

DARK_STYLE = GEMINI_STYLE
