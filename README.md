<div align="center">
  <img src="./images/omni.png" alt="Omnilingual ASR Header" width="100%" />
</div>

# 🎙️ Omnilingual ASR Dashboard

Omnilingual ASR is a state-of-the-art speech recognition system supporting over **1,600 languages**. This dashboard provides a professional, user-friendly interface to leverage the power of Meta's Omnilingual ASR models for transcription, dataset collection, and model management.

---

## ✨ Key Features

- **🚀 Multi-Model Support**: Switch between different model architectures (LLM, CTC, etc.) and sizes (1B, 3B, 7B) on the fly.
- **🎙️ Live Microphone Transcription**: Record and transcribe audio directly from your browser with real-time visualization.
- **📄 Long Audio Processing**: Automatically handles large audio files by intelligent chunking, ensuring stable transcription for files of any length.
- **🌐 Global Language Coverage**: Support for 1,600+ languages with easy search and selection.
- **💾 Transcription History**: Securely save, review, and download your previous transcriptions.
- **🤝 Contribution Workflow**: Dedicated "Contribute" tab for data collection, allowing users to record specific prompts to help improve model performance for low-resource languages.
- **🎨 Modern UI**: Responsive design with smooth animations and drag-and-drop support.
- **🔌 REST API**: Programmatic access to transcription and history endpoints.

---

## 🚀 Quick Start

### Installation
For a fresh setup, simply run:
```bash
# Optimized for Ubuntu 24.04
bash setup_env.sh
```

### Running the Dashboard
After installation, you can launch the dashboard using the dedicated virtual environment:

```bash
# From the project root
./asr_venv/bin/python app.py
```

Or with custom options:
```bash
./asr_venv/bin/python app.py --host 0.0.0.0 --port 5000 --model-card omniASR_LLM_1B_local
```

### Accessing the Dashboard
Open your web browser and navigate to: [http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 🛠️ Installation (Ubuntu 24.04 Optimized)

This project uses a standalone `venv` strategy to ensure ABI stability and manage complex CUDA dependencies.

### 1. System Requirements
Ensure you have Python 3.10 and the necessary system libraries installed:
```bash
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt update
sudo apt install python3.10-venv ffmpeg libsndfile1
```

### 2. Environment Setup
The recommended way to set up the environment is to use our automated script:
```bash
bash setup_env.sh
```

Alternatively, you can perform the steps manually:
```bash
python3.10 -m venv asr_venv
./asr_venv/bin/pip install --upgrade pip
./asr_venv/bin/pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
./asr_venv/bin/pip install fairseq2==0.6 fairseq2n==0.6 --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.6.0/cu124
./asr_venv/bin/pip install -r requirements-dashboard.txt
./asr_venv/bin/pip install omnilingual-asr --no-deps
```

---

## 📁 Model Configuration

The system is designed to use local high-performance checkpoints.

### Local Asset Registry
Models are managed via `fairseq2` asset cards. Ensure your configuration is created at:
`~/.config/fairseq2/assets/cards/models/omniasr_local.yaml`

| Identifier | Parameters | Family |
|------------|------------|--------|
| `omniASR_LLM_1B_local` | 1B | LLM (Default) |
| `omniASR_CTC_1B_local` | 1B | CTC (High Speed) |
| `omniASR_LLM_3B_local` | 3B | LLM (High Accuracy) |

---

## 🔌 API Endpoints

The dashboard exposes several REST API endpoints for integration:

- `POST /api/transcribe` - Upload and transcribe an audio file.
- `POST /api/transcribe_long` - Process audio files > 40 seconds with automatic chunking.
- `GET /api/history` - Retrieve transcription history.
- `GET /api/download/<id>` - Download a specific transcription.
- `POST /api/model` - Switch the active model card in memory.

---

## ⚠️ Troubleshooting & FAQ

### 🎤 Microphone "Access Denied"
Browsers only allow microphone access in **Secure Contexts**. If accessing remotely:
1. Use an **SSH Tunnel**: `ssh -L 5000:localhost:5000 user@remote-ip`
2. Or enable flags locally: `chrome://flags/#unsafely-treat-insecure-origin-as-secure` for your URL.

### ⚡ Audio Conversion Issues
If `.m4a` or `.mp3` fails, verify that `/usr/bin/ffmpeg` is healthy. The dashboard converts these to 16kHz Mono WAV automatically for the model.

### 🕰️ Supported Audio Length
- **Direct Transcription**: Optimized for clips < 40s.
- **Long Audio Tab**: Handles files of several minutes by splitting them into chunks.

---

## 📜 License & Citation

Omnilingual ASR is licensed under the Apache 2.0 License.

```bibtex
@misc{omnilingualasr2025,
    title={{Omnilingual ASR}: Open-Source Multilingual Speech Recognition for 1600+ Languages},
    author={Omnilingual ASR Team},
    year={2025},
    url={https://ai.meta.com/research/publications/omnilingual-asr-open-source-multilingual-speech-recognition-for-1600-languages/},
}
```
