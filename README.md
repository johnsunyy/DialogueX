# DIALOGUE-X  
### AI-Powered Multilingual Speech and Sign Language Translation Platform

---

## Overview

**DIALOGUE-X** is an AI-driven communication platform designed to eliminate language and accessibility barriers through:

- Real-time multilingual speech translation
- Synchronized lip-synced video buffering
- AI-powered Sign Language Recognition (SLR)
- WebRTC-based live communication

The platform combines speech processing, machine learning, computer vision, and real-time communication technologies into a unified system for seamless cross-language interaction.

---

## Core Features

### Real-Time Video Translation
- Real-time multilingual voice translation
- Synchronized video buffering for accurate lip-sync
- Live speech-to-text and text-to-speech pipeline
- Low-latency WebRTC communication using Agora
- Enhanced audio preprocessing for better recognition quality

### Sign Language Recognition (SLR)
- Real-time ASL alphabet recognition
- Word-level sign language interpretation
- Motion-triggered sign segmentation
- Landmark-based inference using MediaPipe
- Lightweight and efficient AI models for live prediction

---

## System Architecture

### Translation Pipeline

```text
Speaker Audio
      ↓
Speech-to-Text (Whisper / Google SR)
      ↓
Language Translation
      ↓
Text-to-Speech Generation
      ↓
Synchronized Video Playback
      ↓
Translated Output
```

### Sign Language Recognition Pipeline

```text
Webcam Feed
      ↓
MediaPipe Landmark Extraction
      ↓
Feature Processing
      ↓
BiLSTM / MLP Models
      ↓
Real-Time Prediction Overlay
```

---

## Technologies Used

### Frontend
- HTML
- CSS
- JavaScript
- WebRTC
- Agora SDK

### Backend
- Python
- Flask
- Socket.IO

### AI / Machine Learning
- OpenAI Whisper
- Google Speech Recognition
- gTTS
- MediaPipe
- TensorFlow / Keras
- BiLSTM + Attention
- MLP Classifiers

---

## Project Structure

```text
DialogueX/
│
├── backend/                     # Translation backend server
├── frontend/                    # WebRTC frontend client
├── sign_alphabet_model/         # ASL alphabet recognition model
├── word_model_nslt300/          # Word-level sign recognition model
├── word_model_nslt100/          # Smaller NSLT subset model
├── requirements.txt
└── run.bat
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/johnsunyy/DialogueX.git
cd DialogueX
```

### Install Dependencies

```bash
pip install -r requirements.txt
pip install -r sign_alphabet_model/requirements.txt
```

---

## Running the Project

### Start Backend

```bash
python backend/server.py
```

### Start Frontend

```bash
cd frontend
python -m http.server 8080
```

Open:

```text
http://localhost:8080
```

---

## Running Sign Language Recognition

### Word-Level Recognition

```bash
python word_model_nslt300/live_inference.py
```

### Alphabet-Level Recognition

```bash
python sign_alphabet_model/inference/webcam_test.py
```

---

## Technical Highlights

| Module | Technology |
|---|---|
| Video Communication | Agora WebRTC |
| Speech Recognition | Whisper / Google SR |
| Translation | Google Translate |
| Text-to-Speech | gTTS |
| Landmark Detection | MediaPipe |
| Word Recognition | BiLSTM + Attention |
| Alphabet Recognition | MLP |

---

## Performance

| Feature | Performance |
|---|---|
| ASL Alphabet Accuracy | ~98% |
| Translation Sync Delay | 3–4.5 Seconds |
| Audio Sampling | 48kHz |
| Inference Speed | <5ms (Alphabet Model) |

---

## Future Improvements

- Full sentence-level sign language translation
- Mobile application support
- Custom multilingual TTS voices
- Transformer-based sign recognition
- Cloud deployment & scaling
- End-to-end AI speech synthesis

---

## Use Cases

- Cross-language communication
- Accessibility support for deaf and mute individuals
- International collaboration
- AI-assisted education platforms
- Real-time multilingual meetings

---

## License

This project is licensed under the MIT License.

---

## Contributors

- John Sunny
- Dialogue-X Development Team

---

## Repository

https://github.com/johnsunyy/DialogueX
