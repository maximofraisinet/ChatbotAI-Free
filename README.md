# 🎙️🤖 ChatbotAI-Free — Local Voice AI Assistant

A fully **offline** voice chatbot powered by local LLMs, high-quality neural TTS, and real-time speech recognition. Practice conversations, explore ideas, or just talk — entirely on your own hardware, with no cloud required.

---

### ✨ ChatbotAI-Free in Action

**Classic Chat Mode**
<img width="1912" height="996" alt="Normal mode" src="https://github.com/user-attachments/assets/71630fb8-6b97-42fc-b4e7-3f47f736936e" />

https://github.com/user-attachments/assets/56a0d0cb-73ae-42df-8c5a-3f0938419d29

**Live Mode**
![Screenshot of Live Mode](https://github.com/user-attachments/assets/33e8d5dc-4310-4248-ba22-4b16085958b9)

https://github.com/user-attachments/assets/c33bd6a7-a4ae-48cf-89c5-c72d019a0d53

---

## 🚀 Features

- **🌐 Multilingual TTS — one engine, all languages**
  [Kokoro TTS v1.0](https://github.com/thewh1teagle/kokoro-onnx) handles both **English** and **Spanish** out of the box (54 voices included). Add any additional language via a Sherpa-ONNX voice pack — the app auto-detects it and asks you which language it belongs to.

- **🎯 Two Conversation Modes**
  - **Classic Chat** — turn-by-turn, with full markdown rendering and streaming responses.
  - **Live Mode** — hands-free, continuous conversation with barge-in detection (interrupt the AI mid-sentence naturally).

- **🗣️ Advanced Voice Pipeline**
  - Real-time Speech-to-Text via [`faster-whisper`](https://github.com/guillaumekln/faster-whisper).
  - Voice Activity Detection (VAD) for precise end-of-speech detection.
  - PipeWire-native audio playback — TTS never blocks other apps.

- **🧠 Fully Local & Private**
  - LLM inference via [Ollama](https://ollama.ai/) — Llama, Mistral, Gemma, and any model you pull.
  - Streaming responses with simultaneous TTS generation.
  - Persistent conversation history with context-window indicator.

- **📄 PDF Document Chat**
  Attach a PDF directly into the conversation — the app extracts text, counts tokens, and shows a detailed confirmation dialog with context-window stats before injecting it. Ask questions about the document without any external vector DB or RAG pipeline.

- **🎨 Modern, Customizable UI**
  - Dark theme inspired by Google Gemini.
  - Adjustable voice speed (0.5× – 2.0×), font size, and audio devices.
  - Collapsible reasoning panel for thinking-capable models.

- **🔍 Smart Voice Scanner**
  On startup the app scans the `voices/` folder. New voice packs are detected automatically — you'll be prompted once to classify each one by language. No manual config needed.

---

## 🛠️ Technology Stack

| Component | Technology |
|---|---|
| **Application & UI** | Python 3.10+, PyQt6 |
| **LLM Inference** | [Ollama](https://ollama.ai/) |
| **Speech-to-Text** | [faster-whisper](https://github.com/guillaumekln/faster-whisper) |
| **Text-to-Speech (primary)** | [Kokoro ONNX v1.0](https://github.com/thewh1teagle/kokoro-onnx) |
| **Text-to-Speech (extra voices)** | [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) (optional) |
| **PDF Text Extraction** | [PyMuPDF](https://pymupdf.readthedocs.io/) |
| **Token Counting** | [tiktoken](https://github.com/openai/tiktoken) |
| **Audio I/O** | sounddevice, NumPy, paplay (PipeWire) |

---

## 📦 Getting Started

### 1. Prerequisites

- **Python** 3.10 or 3.11
- **Ollama** installed and running — [ollama.ai](https://ollama.ai/)
- **Git**

### 2. Clone & Install

```bash
git clone https://github.com/maximofraisinet/ChatbotAI-Free
cd ChatbotAI-Free

python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

For NVIDIA GPU acceleration (recommended):
```bash
pip install onnxruntime-gpu
```

### 3. Download the Kokoro v1.0 Voice Model

Kokoro powers **all built-in voices** (English + Spanish). The model files are too large for GitHub, so download them manually:

1. Go to the [kokoro-onnx releases page](https://github.com/thewh1teagle/kokoro-onnx/releases).
2. Download **`kokoro-v1.0.onnx`** and **`voices-v1.0.bin`**.
3. Place both files inside `voices/kokoro-v1.0/`:

```
voices/
└── kokoro-v1.0/
    ├── kokoro-v1.0.onnx    ← ~300 MB neural TTS model
    └── voices-v1.0.bin     ← ~27 MB  (54 English + Spanish voices)
```

### 4. Pull an Ollama Model

```bash
ollama pull llama3.1:8b
```

### 5. Run

```bash
python main.py
```

On first launch the voice scanner checks `voices/`. If the Kokoro files are in place you're ready to go immediately.

---

## 🌍 Adding More Voices (Sherpa-ONNX)

Want voices in **other languages** — French, Italian, German, Portuguese, and more? Use any [Piper-compatible Sherpa-ONNX VITS pack](https://huggingface.co/csukuangfj):

### Step 1 — Install Sherpa-ONNX

```bash
pip install sherpa-onnx
```

### Step 2 — Download a voice pack

Browse available voices at [huggingface.co/csukuangfj](https://huggingface.co/csukuangfj). For example, the Argentine Spanish "Daniela" voice:

```
https://huggingface.co/csukuangfj/vits-piper-es_AR-daniela-high/tree/main
```

Download these three items from the repo:
- The `.onnx` model file
- `tokens.txt`
- The `espeak-ng-data/` directory

### Step 3 — Drop the folder into `voices/`

Place the downloaded folder **directly** inside `voices/` (not nested deeper):

```
voices/
├── kokoro-v1.0/                         ← built-in (Kokoro)
│   ├── kokoro-v1.0.onnx
│   └── voices-v1.0.bin
└── vits-piper-es_AR-daniela-high/       ← your new Sherpa voice
    ├── es_AR-daniela-high.onnx
    ├── tokens.txt
    └── espeak-ng-data/
```

### Step 4 — Restart the app

On the next launch, the voice scanner detects the new folder and shows a one-time dialog asking which language to assign it to. After you confirm, the voice appears in the voice selector dropdown — no further setup needed.

> **Any valid Sherpa-ONNX VITS model works.** The app identifies a Sherpa pack by the presence of a `.onnx` file and an `espeak-ng-data/` sub-directory inside the folder.

---

## ⌨️ Usage

| Control | Action |
|---|---|
| Top dropdowns | Select LLM model and active voice |
| ⚙️ Settings | Language, voice speed, font size, audio devices, recording mode |
| 🎤 Mic button | Tap to record; tap again to send (or enable auto-send in Settings) |
| ✨ Live button | Enter hands-free Live Mode |
| 📎 Attach button | Upload a PDF document into the conversation context |
| ⏹ Stop (during playback) | Interrupt the AI mid-response |
| Context donut (bottom bar) | Click to see context window usage |

---

## 🤝 Contributing

Contributions are welcome! Open an issue or submit a pull request.

1. Fork the project
2. Create your feature branch: `git checkout -b feature/AmazingFeature`
3. Commit your changes: `git commit -m 'Add AmazingFeature'`
4. Push: `git push origin feature/AmazingFeature`
5. Open a Pull Request

---

## 📄 License

This project is released under **The Unlicense**. See `LICENSE` for details.
