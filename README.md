# DIALOGUE-X: AI-Powered Multilingual Communication Suite

**Dialogue-X** is a cutting-edge communication platform that breaks language barriers through real-time video translation and advanced Sign Language Recognition (SLR). 

Transform your interactions with real-time translation featuring **synchronized video delay** and state-of-the-art **AI Sign Language Interpretation**.

---

## 🌟 Key Features

### 🎙️ Real-Time Video Translation
- ✅ **Synchronized Video Delay**: Perfectly lip-synced video by buffering frames to match translation latency (3-4s).
- ✅ **Live Audio Translation**: Support for 16+ languages using high-quality STT (Whisper/Google) and TTS (gTTS).
- ✅ **WebRTC Calling**: Robust room-based video meetings powered by Agora.
- ✅ **Enhanced Audio Quality**: 48kHz sampling, gain boost, and dynamic range compression for superior recognition.

### 🖖 Sign Language Recognition (SLR)
- ✅ **Word-Level Recognition (NSLT-300)**: Recognizes 300 common sign language words with BiLSTM + Attention architecture.
- ✅ **Alphabet-Level Classification (A-Z)**: High-speed, 98% accurate ASL alphabet recognition using MLP models.
- ✅ **Motion-Based Segmentation**: Intelligent real-time word boundary detection based on landmark velocity.
- ✅ **Landmark-Based (MediaPipe)**: Extremely efficient processing using hand and pose landmarks instead of raw video.

---

## 🏗️ System Architecture

### Video Translation Pipeline
```
Speaker (English) → Audio Capture (48kHz) → STT (Whisper) → Translation → TTS (Spanish)
                                ↓                                         ↓
                     Video Buffer (3-4s Delay) ────────────→ Listener hears Spanish
                                                               (Perfectly Synced!)
```

### Sign Language Recognition Pipeline
```
Webcam → MediaPipe Holistic → Landmark Extraction (162 features) → BiLSTM/MLP Model → Real-Time Prediction Overlay
```

---

## 📁 Project Structure

```
d:/haha/
├── backend/                # Translation server (Flask, Socket.IO)
├── frontend/               # WebRTC client (Agora, Video Buffer)
├── sign_alphabet_model/    # ASL Alphabet (A-Z) classifier
├── word_model_nslt300/     # Word-level SLR (300 words)
├── word_model_nslt100/     # Word-level SLR (100 words subset)
├── requirements.txt        # Core dependencies
└── run.bat                 # One-click launcher
```

---

## 🚀 Quick Start

### 1. Unified Setup
Clone the repository and install all dependencies:
```bash
pip install -r requirements.txt
pip install -r sign_alphabet_model/requirements.txt
```
*Note: FFmpeg is required for the translation system. See [FFMPEG_INSTALL.md](file:///d:/haha/FFMPEG_INSTALL.md).*

### 2. Video Calling & Translation
**Easiest way (Windows):**
Double-click `run.bat` or:
```bash
# Terminal 1: Backend
python backend/server.py

# Terminal 2: Frontend
cd frontend && python -m http.server 8080
```
Open `http://localhost:8080`, join a room, and click the **Translate** button.

### 3. Sign Language Recognition
**Word-Level (NSLT-300):**
```bash
python word_model_nslt300/live_inference.py
```
**Alphabet-Level (A-Z):**
```bash
python sign_alphabet_model/inference/webcam_test.py
```

---

## 🛠️ Technical Specifications

### Video Translation
| Component | Technology | Latency |
|-----------|------------|---------|
| Communication | Agora WebRTC | ~50ms |
| Audio Processing | Web Audio API / FFmpeg | - |
| STT | OpenAI Whisper / Google SR | 500-1000ms |
| Translation | Google Translate | 200-500ms |
| TTS | gTTS (Google) | 300-700ms |
| **Total Sync Delay** | **Video Buffered** | **3.0 - 4.5s** |

### SLR Models
- **Alphabet Model**: MLP (63 features) | ≥98% Accuracy | <5ms Inference
- **Word Model (NSLT-300)**: BiLSTM + Temporal Attention (162 features) | 30-frame window | Motion-triggered segmentation

---

## 📝 Documentation
For detailed guides on specific components, see:
- 📖 [Project Progress](file:///d:/haha/PROJECT_PROGRESS.md)
- 🎤 [Audio Improvements](file:///d:/haha/AUDIO_IMPROVEMENTS_SUMMARY.md)
- 🧪 [Testing Guide](file:///d:/haha/TESTING_GUIDE.md)
- 🗺️ [System Roadmap](file:///d:/haha/SYSTEM_ANALYSIS_AND_ROADMAP.md)

---

## 📜 License
MIT License - Free for personal and commercial use.

---

**Made with ❤️ for a world without barriers.**
