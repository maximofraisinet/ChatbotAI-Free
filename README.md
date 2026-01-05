# Voice Chatbot - Local AI Assistant 🎙️🤖

A fully local voice chatbot powered by AI, featuring real-time speech-to-text, LLM conversation, and text-to-speech synthesis. Optimized for AMD Ryzen 7 + NVIDIA RTX 4060 (8GB VRAM).

## Features ✨

- **🎤 Real-time Voice Input**: Continuous listening with Voice Activity Detection (VAD)
- **🧠 Smart Conversations**: Powered by Llama3 via Ollama
- **🔊 Natural Speech Output**: High-quality TTS with Kokoro ONNX
- **🎨 Modern Dark UI**: WhatsApp/iMessage-style chat bubbles
- **⚡ Multi-threaded**: Non-blocking UI with efficient resource usage
- **🔁 Feedback Prevention**: Automatic microphone muting during bot speech

## Architecture 🏗️

```
┌─────────────────────────────────────────────┐
│            PyQt6 GUI (Main Thread)          │
│  - Chat bubbles (user/bot)                  │
│  - Status indicator                         │
│  - Control buttons                          │
└──────────────┬──────────────────────────────┘
               │
       ┌───────┴───────┐
       │               │
┌──────▼───────┐ ┌────▼──────────┐
│   Listener   │ │    Worker     │
│    Thread    │ │    Thread     │
│              │ │               │
│ - Mic input  │ │ - Transcribe  │
│ - VAD        │ │ - LLM         │
│              │ │ - TTS         │
│              │ │ - Playback    │
└──────────────┘ └───────────────┘
```

## Installation 📦

### 1. Prerequisites

- **Python**: 3.10 or 3.11 (recommended)
- **CUDA**: 11.8+ for GPU acceleration
- **Ollama**: Installed and running ([Download here](https://ollama.ai))

### 2. Install Ollama Model

```bash
ollama pull llama3
```

### 3. Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Model Files Setup

Place the following files in the project root directory:

```
ChatbotAI-English/
├── kokoro-v0_19.onnx    # Kokoro TTS model (already present)
├── voices.json           # Voice configurations (already present)
├── main.py
├── ai_manager.py
├── audio_utils.py
├── styles.py
└── requirements.txt
```

**Note**: The `kokoro-v0_19.onnx` and `voices.json` files should already be in your workspace.

## Usage 🚀

### Run the Application

```bash
python main.py
```

### How It Works

1. **Click "Start Listening"** (or wait for auto-start)
2. **Speak into your microphone** in English
3. **The bot will**:
   - Transcribe your speech (Whisper)
   - Process with Llama3 (Ollama)
   - Generate voice response (Kokoro)
   - Play the audio back
4. **Repeat**: The bot continues listening after speaking

### Controls

- **Start Listening**: Begin voice interaction
- **Stop**: Pause all processing
- **Clear Chat**: Reset conversation history

## Configuration ⚙️

### Adjust Models (in `ai_manager.py`)

```python
ai_manager = AIManager(
    whisper_model="base.en",      # Options: tiny.en, base.en, small.en
    ollama_model="llama3",         # Any Ollama model
    voice_name="af_bella"          # Options: af_bella, af_sarah
)
```

### Adjust VAD Sensitivity (in `audio_utils.py`)

```python
recorder = AudioRecorder(
    silence_threshold=0.015,   # Lower = more sensitive
    silence_duration=1.5       # Seconds of silence before stop
)
```

### Change UI Colors (in `styles.py`)

Modify the `DARK_STYLE` CSS variables:
- Background: `#121212`
- User bubble: `#005C4B`
- Bot bubble: `#1F1F1F`

## Project Structure 📁

```
├── main.py              # Main application & GUI logic
├── ai_manager.py        # AI model management (Whisper, Ollama, Kokoro)
├── audio_utils.py       # Audio recording & playback with VAD
├── styles.py            # PyQt6 QSS dark theme styles
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── kokoro-v0_19.onnx   # Kokoro TTS model
└── voices.json          # Voice configurations
```

## Performance Tips 🎯

### For 8GB VRAM

- Use `base.en` or `small.en` for Whisper (not `medium` or `large`)
- Keep conversation history to 10 messages max (already configured)
- Close other GPU-intensive applications

### Troubleshooting

**GPU Not Detected**:
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

**Ollama Connection Error**:
```bash
# Ensure Ollama is running
ollama list
```

**Microphone Not Working**:
```bash
# Test audio devices
python -c "import sounddevice as sd; print(sd.query_devices())"
```

## Technical Details 🔧

### Audio Pipeline

- **Input**: 16 kHz, mono, float32
- **VAD**: RMS-based energy detection
- **Feedback Prevention**: Microphone paused during TTS playback

### AI Models

- **STT**: faster-whisper (base.en) on CUDA with FP16
- **LLM**: Ollama llama3 with streaming disabled
- **TTS**: Kokoro ONNX (24 kHz output) on CUDA

### Threading Model

- **Main Thread**: UI updates only (PyQt6)
- **Listener Thread**: Continuous audio capture
- **Worker Thread**: STT → LLM → TTS pipeline

## License 📄

This project uses various open-source models and libraries. Please review individual licenses:

- **faster-whisper**: MIT License
- **Ollama**: MIT License
- **Kokoro TTS**: Check model provider's license
- **PyQt6**: GPL v3

## Credits 👏

- **Whisper**: OpenAI
- **Llama3**: Meta AI
- **Kokoro TTS**: [Model provider]
- **Ollama**: Ollama team

---

**Enjoy your local AI voice assistant! 🎉**
