# 🎙️🤖 Voice Chatbot: AI Language Practice Assistant

A fully local voice chatbot designed for practicing **English** and **Spanish** conversation skills. It features high-quality, natural-sounding AI voices and operates in two distinct modes: **Classic Chat** and continuous **Live Mode**.

---

### ✨ Live Mode in Action

**Normal Mode**
![Screenshot of Normal Mode](https://github.com/user-attachments/assets/0030e384-f9c8-47ee-aa90-2c4a955bc27b)

https://github.com/user-attachments/assets/56a0d0cb-73ae-42df-8c5a-3f0938419d29

**Live Mode**
![Screenshot of Live Mode](https://github.com/user-attachments/assets/33e8d5dc-4310-4248-ba22-4b16085958b9)

https://github.com/user-attachments/assets/c33bd6a7-a4ae-48cf-89c5-c72d019a0d53

---

## 🚀 Features

- **🌐 Multi-Language Support**: Seamlessly switch between English (Kokoro TTS) and Spanish (Sherpa-ONNX TTS).
- **🎯 Dual Conversation Modes**:
    - **Classic Chat**: Traditional turn-by-turn conversation.
    - **Live Mode**: Hands-free, continuous conversation with barge-in capability (interrupt the AI naturally).
- **🗣️ Advanced Voice Capabilities**:
    - Real-time, multilingual Speech-to-Text (via `faster-whisper`).
    - High-quality, natural Text-to-Speech output.
    - Voice Activity Detection (VAD) for precise end-of-speech detection.
- **🧠 Local & Private AI**:
    - Powered by local LLMs through **Ollama** (e.g., Llama 3.1, Mistral, Gemma 2).
    - Streaming responses for instant audio and text feedback.
    - Context-aware conversations with persistent history.
- **🎨 Modern & Customizable UI**:
    - Sleek dark theme inspired by Google Gemini.
    - Adjustable font sizes and voice playback speeds.
    - Visual feedback for voice activity.
- **⚙️ Smart & Persistent**: User preferences (model, language, voice, etc.) are saved locally.

## 🛠️ Technology Stack

| Component            | Technology                                                                                             |
| -------------------- | ------------------------------------------------------------------------------------------------------ |
| **Application & UI** | Python, PyQt6                                                                                          |
| **LLM Inference**    | [Ollama](https://ollama.ai/) (Llama 3.1, Mistral, etc.)                                                  |
| **Speech-to-Text**   | [faster-whisper](https://github.com/guillaumekln/faster-whisper)                                         |
| **Text-to-Speech**   | [Kokoro TTS](https://github.com/thewh1teagle/kokoro-onnx) (English), [Sherpa-ONNX](https://github.com/k2-fsa/sherpa-onnx) (Spanish) |
| **Audio I/O**        | PyAudio / sounddevice, NumPy                                                                           |

## 📦 Getting Started

### 1. Prerequisites

- **Python** 3.10 or 3.11
- **Ollama**: Make sure it is installed and running. Download from [ollama.ai](https://ollama.ai/).
- **Git**

### 2. Installation & Setup

#### Step 1: Clone the Repository
Clone this repository to your local machine:
```bash
git clone https://github.com/your-username/ChatbotAI-English.git
cd ChatbotAI-English
```
*(Remember to replace `your-username` with the actual repository URL)*

#### Step 2: Create a Virtual Environment (Recommended)
```bash
python3 -m venv venv
source venv/bin/activate
# On Windows, use: venv\Scripts\activate
```

#### Step 3: Install Dependencies
Install all required Python packages from `requirements.txt`:
```bash
pip install -r requirements.txt
```
For **NVIDIA GPU** support (recommended for `faster-whisper`), ensure you have CUDA installed and install the GPU-enabled `onnxruntime`:
```bash
pip install onnxruntime-gpu
```

#### Step 4: Download TTS Models
The TTS models are too large for GitHub and must be downloaded manually.

**A. English TTS Model (Kokoro)**
1.  Download the following files from the [Kokoro-82M releases](https://github.com/thewh1teagle/kokoro-onnx/releases) page:
    - `kokoro-v0_19.onnx`
    - `voices.json`
2.  Place both files inside the `voices/english/` directory.

**B. Spanish TTS Models (Sherpa-ONNX)** (Optional)
The application automatically detects any Spanish voice installed in the `voices/spanish/` directory.

1.  First, install the `sherpa-onnx` package:
    ```bash
    pip install sherpa-onnx
    ```
2.  For each voice you want to add, download the model files from the [Piper-Voices Hugging Face repos](https://huggingface.co/csukuangfj).
    - [Daniela (Argentine Spanish)](https://huggingface.co/csukuangfj/vits-piper-es_AR-daniela-high/tree/main)
    - [Miro (European Spanish)](https://huggingface.co/csukuangfj/vits-piper-es_ES-miro-high/tree/main)

3.  Create a sub-folder for each voice inside `voices/spanish/` (e.g., `voices/spanish/Daniela/`).
4.  Download and place the following three files into the new sub-folder:
    - The `.onnx` model file.
    - `tokens.txt`
    - The `espeak-ng-data/` directory (extract it from the `.tar.bz2` archive).

Your final `voices` directory should look like this:
```
voices/
├── english/
│   ├── kokoro-v0_19.onnx    # English TTS model
│   └── voices.json          # English voice configs
└── spanish/
    ├── Daniela/             # Spanish voice 1
    │   ├── es_AR-daniela-high.onnx
    │   ├── tokens.txt
    │   └── espeak-ng-data/
    └── Miro/                # Spanish voice 2
        ├── es_ES-miro-high.onnx
        ├── tokens.txt
        └── espeak-ng-data/
```

#### Step 5: Pull an Ollama Model
Download a model for Ollama to use. `llama3.1:8b` is a great starting point.
```bash
ollama pull llama3.1:8b
```

### 3. Run the Application
Launch the chatbot with:
```bash
python main.py
```

## ⌨️ Usage

- **🧠 Model & Voice Selection**: Use the dropdown menus at the top to select your desired Ollama model and TTS voice.
- **⚙️ Settings**: Click the gear icon to configure:
    - **Language**: English or Spanish.
    - **Voice Speed**: Adjust playback rate from 0.5x to 2.0x.
    - **Font Size**: Small, Medium, or Large.
    - **Voice Mode**: Toggle between auto-sending your recording or reviewing it first.

## 🤝 Contributing

Contributions are welcome! If you have ideas for new features, bug fixes, or improvements, please open an issue or submit a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request

## 📄 License

This project is licensed under **The Unlicense**. See the `LICENSE` file for details.
