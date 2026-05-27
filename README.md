# DIALOGUE-X: AI-Powered Multilingual Communication Suite

**Dialogue-X** is an advanced real-time communication platform designed to bridge spoken language and sign language barriers instantly. By unifying low-latency WebRTC video conferencing, streaming translation, and a dual-model Sign Language Recognition (SLR) engine, Dialogue-X allows hearing participants speaking different languages and deaf or hard-of-hearing individuals to communicate naturally and fluidly.

Unlike conventional communication tools that rely on third-party captioning plug-ins, Dialogue-X features a deep-pipeline integration that synchronizes video latency with audio translation delays and dynamically switches between word-level signing and fingerspelling.

---

## 🌟 Key Capabilities & Features

### 🎙️ Real-Time Video & Spoken Translation
*   **WebRTC Video Calling**: High-definition, low-latency rooms powered by Agora RTC.
*   **Dynamic Audio Normalization**: Real-time gain boosting and volume stabilization via `pydub` to optimize speech recognition.
*   **Multilingual STT/TTS**: Converts speech to text (using Google Speech Recognition) and synthesizes translations back to spoken voice (via `gTTS`) with support for over 16+ languages (including English, Spanish, Malayalam, Hindi, French, German, etc.).
*   **Automatic Subtitles**: Interactive transcription overlays delivered instantly to remote peers via Socket.IO.

### 🖖 Sign Language Recognition (SLR)
*   **Pose Motion Energy (PME) Switching**: A custom state machine that analyzes physical motion velocity to dynamically shift between whole-word translation and finger-spelling modes.
*   **Word-Level Classifier (NSLT)**: Spatial-temporal models (BiLSTM or Transformer) trained to recognize complex words from 69+ classes (such as `AGAIN`, `AND`, `WATER`, etc.) using sequence windows.
*   **Alphabet-Level Speller (A-Z)**: High-speed, 98%-accurate Multi-Layer Perceptron (MLP) model for real-time fingerspelling, incorporating wrist-centered, scale-invariant coordinates normalization.
*   **Client-Side Landmark Extraction**: MediaPipe Holistic WASM models run locally in the browser to extract hand and body coordinate markers, minimizing upstream bandwidth usage.

### ⏱️ Video Synchronization Buffer
*   **Lip-Sync Frame Delay**: Captures incoming video tracks at 30 FPS using an HTML5 Canvas buffer.
*   **Synchronized Playback**: Queues and delays the caller's video frames (typically 3.0 to 4.5 seconds) to align perfectly with the computation latency of the voice translation pipeline, avoiding disjointed "lips-moving-but-no-sound" phenomena.

---

## 🏗️ System Architecture & Data Pipelines

### 1. Spoken Translation & Video Sync Pipeline

```
[Speaker A]
    │  (WebRTC Audio Stream)
    ▼
[Agora Channel] ────► [Local Audio Capture (Web Audio API)]
                             │
                             ▼ (Base64 WebM/Opus Chunks)
                      [Socket.IO Server]
                             │
                             ▼
                      [Audio Handler]
                         ├─ Decode Base64 Audio
                         ├─ Auto-Format Conversion (WebM/Opus -> WAV)
                         └─ Downsample (16kHz, mono, 16-bit) & Normalize (min -30 dBFS)
                             │
                             ▼
                      [Translation Pipeline]
                         ├─ Speech-to-Text (STT) -> Transcription
                         ├─ Google Translate (src_lang -> dest_lang)
                         └─ Text-to-Speech (gTTS) -> Translated Audio
                             │
                             ▼ (Base64 MP3 Chunks + Subtitles)
                      [Socket.IO Broadcast]
                             │
                             ▼
[Listener B] ◄─────── [Client UI]
  (Hears Translated      ├─ Receives Translated Audio (Play immediately)
   Audio & Subtitles)    └─ Receives Delayed Video via VideoBufferManager (Lip-synced!)
```

### 2. Dual-Model Sign Language Recognition Pipeline

```
[Signer Camera]
       │
       ▼
[MediaPipe Holistic WASM] (Local Browser)
       │
       ▼ (225-Float Coordinate Vector: Left Hand, Right Hand, Pose)
[Socket.IO landmark Streams]
       │
       ▼
[SignTranslationHandler] (Backend Server)
       │
       ├─► Calculates Pose Motion Energy (PME) from 33 Pose landmarks
       │
       ├─► [PME > 0.015 (WORD_SIGNING)] ──► Accumulate 64-Frame Window ──► [Transformer / LSTM Predictor]
       ├─► [PME < 0.003 (IDLE)] ──────────► Stillness / Transition Timeout
       └─► [HANDS ACTIVE (SPELLING)] ─────► Wrist & Scale Normalization ──► [ASL Alphabet Speller MLP]
               │
               ├─► Predict Label + Confidence Overlays
               └─► final Sentence Dispatch (Pause detected) ──► [Translation Pipeline] ──► Spoken Audio (Listener B)
```

---

## 📁 Repository Directory Layout

The workspace is organized into self-contained backend server, frontend client, and specialized machine learning model directories:

```
DialogueX-DEV/
├── backend/                       # Flask + Socket.IO Translation Backend
│   ├── audio_handler.py           # Decodes, converts, and normalizes client audio chunks
│   ├── translation_pipeline.py    # Orchestrates speech recognition, translation, and TTS
│   ├── sign_translation.py        # Implements PME state machine and routes gestures/spelling
│   ├── config.py                  # Server, Socket.IO, and language configuration settings
│   └── server.py                  # Entry server script; manages rooms and active sockets
│
├── frontend/                      # WebRTC Frontend Client App
│   ├── js/
│   │   └── sign-detection.js      # Handles local camera, MediaPipe Holistic execution, and socket streaming
│   ├── video-buffer.js            # VideoBufferManager and VideoFrameBuffer canvas rendering delay
│   └── index.html                 # Core client dashboard and user interface
│
├── sign_alphabet_model/           # ASL Alphabet (A-Z) Fingerspelling Classifier
│   ├── preprocessing/             # Scripts to extract, normalize, and split dataset (.npy) files
│   ├── training/                  # MLP models, baseline training, and Optuna hyperparameter tuner
│   ├── evaluation/                # Confusion matrix heatmaps and per-class precision/recall reporters
│   ├── inference/                 # ASLPredictor wrapper and live OpenCV webcam script
│   ├── tests/                     # Preprocessing and model output pytests
│   ├── best_model.pt              # Saved weights for ASLPredictor
│   └── label_encoder.pkl          # Pickled mapping for integer labels to letters
│
├── sign_model/                    # Word-Level Sign Language Classifier
│   ├── configs/                   # YAML files for datasets, splits, architectures, and hyperparameters
│   ├── scripts/                   # Pipelines for landmark extraction, preprocessing, training, and evaluation
│   ├── models/                    # Model scalers, training logs, best checkpoints, and final models
│   ├── utils/                     # Supporting utilities for dataset operations and backups
│   ├── predictor.py               # WordSignPredictor wrapper loading LSTM/Transformer models
│   └── label_map.json             # Key-value map linking neural outputs to gesture words
│
├── index.html                     # Alternate WebRTC client UI
├── requirements.txt               # Main Python dependencies list
├── run.bat                        # Windows batch script for automated startup
└── universal_run.py               # Cross-platform startup script (with local IP autodetection)
```

---

## 🚀 Getting Started & Setup

### 1. Prerequisites
Ensure you have the following packages installed:
*   **Python 3.9+** (For core platform and ML modules)
*   **FFmpeg** (Required by `pydub` for browser-compatible WebM/Opus decoding):
    *   *Windows*: Install via Winget: `winget install ffmpeg` or download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) (add the `bin` directory to your System PATH variables).
    *   *Mac*: `brew install ffmpeg`
    *   *Linux*: `sudo apt update && sudo apt install ffmpeg`
*   **Agora App ID**: Sign up for free at [Agora.io](https://console.agora.io/) and create a project to obtain your App ID.

### 2. Installation
Clone the repository and install dependencies for the main platform, alphabet speller, and word model:
```bash
# Install root/backend requirements
pip install -r requirements.txt

# Install ASL Alphabet requirements
pip install -r sign_alphabet_model/requirements.txt

# Install Word Sign requirements
pip install -r sign_model/requirements.txt
```
*(If training on GPU, ensure you install `torch` built with CUDA support, e.g. `pip install torch --index-url https://download.pytorch.org/whl/cu118`)*

### 3. Environment Variables Configuration
1.  Copy the environment template from `.env.example`:
    ```bash
    cp .env.example .env
    ```
2.  Open `.env` in a text editor and fill in your values:
    *   `SECRET_KEY`: A secure random hash to encrypt server-side sessions.
    *   `AGORA_APP_ID`: Your actual App ID copied from the Agora Console.
    *   `BACKEND_URL`: `http://localhost:5000` (or your private IP address when testing over a Local Area Network).

### 4. Running the Platform

#### Option A: One-Click Startup (Windows)
Double-click the `run.bat` script. It checks Python and FFmpeg setups, launches both backend and frontend servers in separate console windows, and opens `http://localhost:8080` in your default browser.

#### Option B: Cross-Platform Launcher (Windows, Mac, Linux)
Run the launcher script:
```bash
python universal_run.py
```
This script dynamically detects your machine's private LAN IP address, rewrites `frontend/index.html`'s server address configuration to support multi-device testing (such as calling from a phone connected to the same Wi-Fi network), starts the services, and launches the browser.

#### Option C: Manual Startup
Open two separate terminal windows:
```bash
# Terminal 1: Backend Server (runs on port 5000)
cd backend
python server.py

# Terminal 2: Frontend Client (runs on port 8080)
cd frontend
python -m http.server 8080
```
Open `http://localhost:8080` to access the main interface.

---

## 🔬 Machine Learning Specifications & Models

### ASL Alphabet Model (A–Z Fingerspelling)
Designed for finger-spelling short or out-of-vocabulary words.
*   **Features**: 63 coordinate values per frame representing 21 hand joints ($x, y, z$).
*   **Normalization**: Wrist centering (zeroing index 0) followed by scaling using the max pairwise landmark distance, rendering predictions translation- and scale-invariant.
*   **Architecture**:
    *   Input Layer: 63 features
    *   Hidden Layer 1: 256 units with Batch Normalization, ReLU activation, and 40% Dropout
    *   Hidden Layer 2: 128 units with Batch Normalization, ReLU activation, and 30% Dropout
    *   Hidden Layer 3: 64 units with ReLU activation
    *   Output Layer: 26 units (Logits for letters A-Z)
*   **Performance**: $\ge 98\%$ classification accuracy, with per-frame inference executing in $<5$ ms on CPU.

### Word-Level Model (NSLT Word Recognition)
Recognizes 69 full-sentence gestures using continuous sequences.
*   **Features**: 225-float vector per frame representing left-hand (63), right-hand (63), and pose (99) joints.
*   **Normalization**: Sequence normalization to a fixed length of 64 frames (via center-cropping or zero-padding), followed by Standard Scaling.
*   **Architecture**: Comparison and winner evaluation between:
    *   *Model A (BiLSTM)*: 512,000 parameters capturing temporal dependencies.
    *   *Model B (Transformer)*: 420,000 parameters utilizing spatial-temporal attention.
*   **Validation Winner**: Transformer (V2 model) with average top-1 accuracy of $\ge 89\%$ and top-3 accuracy of $\ge 97\%$.
*   **V2 Advanced Model Upgrades**:
    *   *Augmentation expansion*: Increased from 2 basic copies to 6 copies/sample utilizing body scaling, temporal speed variations, wrist rotations, mixup, and cutmix (yielding over 26,000 training inputs).
    *   *Optimization*: Implemented label smoothing (0.1) and validation-loss early stopping to resolve overfitting.
    *   *Model weights directory*: Outputs are kept separate at `sign_model/models/v2/` to protect V1 models.

---

## 🛠️ Troubleshooting & Technical Reference

| Issue / Error | Likely Root Cause | Solution |
| :--- | :--- | :--- |
| **Audio translation does not work** | FFmpeg is missing from the system's path, blocking the `pydub` format converter. | Ensure FFmpeg is installed. On Windows, run `winget install ffmpeg` and ensure your environment variables include the path to FFmpeg's `bin/` folder. |
| **"MediaPipe hand not detected" warning** | User is not centered in the camera, or hand is obscured. | The system handles this gracefully by feeding zeros for missing hand segments. Improve lighting and ensure hands are fully visible in the frame. |
| **Agora Video/Audio calls fail to connect** | Missing or incorrect `AGORA_APP_ID` in `.env` or client HTML. | Log in to console.agora.io, copy your project's App ID, and paste it into `.env` (or directly update the frontend scripts). |
| **Webcam fails to open** | Permissions blocked, or the camera is occupied by another application (Zoom, Teams, etc.). | Close conflicting software, refresh the browser page, and explicitly grant camera/mic access permissions. |
| **Socket Connection Error** | Backend server is down, or `BACKEND_URL` points to the wrong address. | Check that `python backend/server.py` is running successfully. If testing over a network, use `universal_run.py` to match the frontend config with your host's local IP. |
| **CUDA out of memory error** | Batch sizes are too large for the system's GPU VRAM. | Reduce the `batch_size` parameter inside [sign_model/configs/config.yaml](file:///c:/Users/amals/OneDrive/Desktop/PROJECTS/DialogueX/DialogueX-DEV/sign_model/configs/config.yaml) or `config_v2.yaml` to 16 or 8. |
| **Low Gesture Recognition Accuracy** | Preprocessed coordinates do not match training scales. | Run `python scripts/02_preprocess.py` to regenerate the `.npz` file and fit a new standard scaler object. |

---

**Made with ❤️ to empower inclusive communication.**
