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

- **💬 Chat History & Sidebar**
  Conversations are auto-saved as Markdown files. A collapsible sidebar (☰) shows your recent chats — click to resume any conversation. Chat titles are generated automatically by the lightest available Ollama model. Right-click any chat to **rename** or **delete** it.

- **🧠 Fully Local & Private**
  - LLM inference via [Ollama](https://ollama.ai/) — Llama, Mistral, Gemma, and any model you pull.
  - Streaming responses with simultaneous TTS generation.
  - Persistent conversation history with context-window indicator.

- **📄 PDF Document Chat**
  Attach a PDF directly into the conversation — the app extracts text, counts tokens, and shows a detailed confirmation dialog with context-window stats before injecting it. Ask questions about the document without any external vector DB or RAG pipeline.

- **📖 Reading Practice Mode (Shadowing Coach)**
  Paste or type any text, hit Start, and read it aloud. The app listens in real time via VAD + Whisper and colors each word as you go — **green** for correct, **red** for mispronounced, **grey** for not yet read. Click any word to hear its pronunciation via TTS. When you finish, a feedback dialog shows your grade (A+ → F), accuracy bar, per-word stats, and a list of missed words.

- **⚙️ Configurable Whisper Model**
  Choose your STT quality/speed trade-off from Settings — **base**, **small**, **medium**, or **large-v3**. The model change takes effect after a restart (the app offers to restart immediately).

- **🎨 Modern, Customizable UI**
  - Dark theme inspired by Google Gemini.
  - Adjustable voice speed (0.5× – 2.0×), font size, and audio devices.
  - Collapsible reasoning panel for thinking-capable models.

- **🔍 Smart Voice Scanner**
  On startup the app scans the `voices/` folder. New voice packs are detected automatically — you'll be prompted once to classify each one by language. No manual config needed.

- **🎭 Characters & Personas**
  Switch the AI's personality from the **🎭 selector** in the header. Six sample personas ship out of the box (Job Interviewer, English Teacher, Casual Friend — in English and Spanish). Each persona is a plain JSON file in the `characters/` folder — edit the existing ones or add your own with any system prompt. Changing the persona automatically starts a fresh conversation with the new system prompt injected invisibly (not shown in saved history).

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


- **⚡ Lightweight Mode (Low-resource)**
  - Disable voice generation in `Settings` to skip Kokoro TTS and keep the app text-only (great for netbooks and low-RAM systems).
  - Optionally disable "Auto-generate chat titles" to avoid extra Ollama calls and save CPU/network on limited machines.
  - When TTS is disabled a small red indicator appears next to the voice selector (`Voice off`) so you always know the state.
  - The app is responsive: it detects screen size on startup and uses 10% side margins for chat/input so it works on small screens.

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
| 🎭 Character selector | Switch the AI persona (header); opens a fresh chat with the new system prompt |
| ☰ Hamburger button | Toggle the chat-history sidebar |
| ➕ New Chat | Start a fresh conversation (sidebar) |
| Right-click a chat | Rename or delete a saved conversation |
| ⚙️ Settings | Language, voice speed, font size, audio devices, recording mode |
| 🎤 Mic button | Tap to record; tap again to send (or enable auto-send in Settings) |
| ✨ Live button | Enter hands-free Live Mode |
| 📎 Attach button | Upload a PDF document into the conversation context |
| 📖 Practice button | Enter Reading Practice (Shadowing Coach) mode |
| ⏹ Stop (during playback) | Interrupt the AI mid-response |
| Context donut (bottom bar) | Click to see context window usage |

---

## 💡 Low-resource & Responsive Tips

- **Disable TTS**: Open `Settings` → `Voice Generation (TTS)` and uncheck the box to skip audio generation. The AI will still produce text responses and you can interrupt it at any time. This saves significant CPU/RAM on low-end machines.

- **Disable AI-generated chat titles**: In `Settings` uncheck "Auto-generate chat titles" to avoid extra Ollama calls. Chats will keep a timestamp-based name instead.

- **Choose a smaller Whisper model**: In `Settings` select `base` or `small` for faster, lower-memory STT.

- **Run CPU-only**: If you don't have an NVIDIA GPU, install CPU runtimes (e.g., `onnxruntime`/`onnxruntime-cpu`) and don't install the GPU packages; the app will run Whisper and Kokoro on CPU (may be slower but avoids CUDA errors).

- **Netbook-friendly UI**: The app now detects your screen and uses 10% horizontal margins for chat and input; minimum window size is reduced to 720×520 so it fits small screens.

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
