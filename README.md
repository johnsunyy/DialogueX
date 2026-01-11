# DIALOGUE-X - Real-Time Multilingual Video Calling with Synchronized Translation

Transform your video calls with real-time translation featuring **synchronized video delay** - speak in any language and let others hear you in their preferred language with perfectly lip-synced video!

##  Key Features

✅ **Real-time video calling** (Agora WebRTC)  
✅ **Live audio translation** (16+ languages)  
✅ **Synchronized video delay** - Video delayed to match audio translation  
✅ **Real-time subtitles**  
✅ **Low-latency translation pipeline** (3-4 seconds)  
✅ **Screen sharing support**  
✅ **Professional UI** with mute/unmute, video controls  
✅ **Room-based meetings**  
✅ **100% Free & Open-Source**

## How It Works

### The Innovation: Synchronized Video Delay

**Problem**: Translation takes 3-4 seconds, causing video-audio mismatch.  
**Solution**: Buffer and delay video by 3-4 seconds to match translated audio!

```
Speaker A speaks in English → Translation happens (3-4s) → Listener B hears Spanish

WITHOUT SYNC: Video shows lips moving NOW, but audio arrives 3-4s later 
WITH SYNC: Video is delayed 3-4s, perfectly matches translated audio 
```

### Architecture

```
User A (English) → Agora (Video + Audio)
                      ↓
                   User B receives
                      ↓
     ┌────────────────┴────────────────┐
     │                                  │
  Original                         Translation
  Video                            Pipeline
     │                                  │
  Buffer                          STT → Trans → TTS
  3-4 sec                              3-4 sec
     │                                  │
     └──────────┬─────────────┬────────┘
                │             │
            Delayed Video + Translated Audio (Spanish)
                      ↓
                Perfect Sync! 
```

##  Prerequisites

### 1. Python 3.8 or higher
- Download from: https://www.python.org/downloads/

### 2. FFmpeg (Required for audio conversion)
- **Windows**: 
  - Download from https://ffmpeg.org/download.html
  - Extract and add to PATH
  - Or use: `choco install ffmpeg` (if you have Chocolatey)
- **Linux**: `sudo apt-get install ffmpeg`
- **Mac**: `brew install ffmpeg`

### 3. PyAudio dependencies
- **Windows**: Usually works out of the box
- **Linux**: `sudo apt-get install portaudio19-dev python3-pyaudio`
- **Mac**: `brew install portaudio`

##   Quick Start

### Windows (Easiest)

1. **Clone/download this repository**
2. **Double-click** `run.bat`
3. **Wait** for servers to start (~10 seconds)
4. **Browser opens** automatically at `http://localhost:8080`

Done! 

### Manual Installation

#### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

#### Step 2: Start Backend (Terminal 1)
```bash
cd backend
python server.py
```

#### Step 3: Start Frontend (Terminal 2)
```bash
cd frontend
python -m http.server 8080
```

#### Step 4: Open Browser
Navigate to `http://localhost:8080`

##  Usage Guide

### Starting a Meeting

1. Open `http://localhost:8080`
2. Choose **"Host Meeting"** or **"Join Meeting"**
3. Enter a **Meeting ID** (same for all participants)
4. Enter **your name**
5. Click **"Start Meeting"**

### Enabling Translation with Video Sync

1. Once in the call, click the **"Translate"** button 
2. Select your **preferred language**
3. Click **"Enable Translation"**

**What happens:**
- ✅ You hear other participants in your selected language
- ✅ Subtitles appear in your selected language
- ✅ Video automatically delays to match translation timing
- ✅ Perfect lip-sync experience!

### Example Scenario

**User A** (speaks English):
- Normal video stream
- Hears everyone in original language

**User B** (enables Spanish translation):
- Video delayed by 3.5 seconds
- Hears User A in Spanish
- Sees Spanish subtitles
- Perfectly synchronized!

##  Supported Languages

- English (en)
- Spanish (es)
- French (fr)
- German (de)
- Hindi (hi - हिंदी)
- Malayalam (ml - മലയാളം)
- Tamil (ta - தமிழ்)
- Telugu (te - తెలుగు)
- Bengali (bn - বাংলা)
- Japanese (ja - 日本語)
- Korean (ko - 한국어)
- Chinese (zh-cn - 中文)
- Arabic (ar - العربية)
- Portuguese (pt)
- Russian (ru - Русский)
- Italian (it)

##  System Architecture

### Backend (`./backend/`)

| File | Purpose |
|------|---------|
| `server.py` | Flask + SocketIO server, handles translation requests |
| `audio_handler.py` | WebM → WAV conversion for speech recognition |
| `translation_pipeline.py` | STT → Translation → TTS with latency tracking |
| `config.py` | Configuration (languages, delay settings) |

### Frontend (`./frontend/`)

| File | Purpose |
|------|---------|
| `index.html` | Main UI with Agora WebRTC integration |
| `video-buffer.js` | Video delay synchronization system |

### Translation Pipeline

```
1. Audio Capture (MediaRecorder, 2s chunks)
2. Send to Backend (Socket.IO WebSocket)
3. Speech-to-Text (Google SR, ~500-1000ms)
4. Translation (Google Translate, ~200-500ms)
5. Text-to-Speech (gTTS, ~300-700ms)
6. Send Back (Socket.IO)
7. Play Translated Audio + Show Subtitle
8. Video plays with matching delay
```

**Total Latency**: 3-4.5 seconds  
**Video Delay**: Automatically set to match

##  Troubleshooting

### "FFmpeg not found"
→ Install FFmpeg and add to PATH. Restart terminal after installation.

### Backend won't start / Port 5000 in use
```bash
# Check what's using port 5000
netstat -ano | findstr :5000

# Kill the process or change port in backend/config.py
```

### No audio translation
→ Check browser console (F12) for errors  
→ Verify Socket.IO connection status  
→ Ensure microphone permissions granted  
→ Check backend logs for processing errors

### Video and audio out of sync
→ This is expected without translation enabled  
→ Enable translation on the listener side  
→ System automatically synchronizes video delay

### Translation not working
→ Check internet connection (Google APIs require network)  
→ Verify backend is running on port 5000  
→ Check browser console for Socket.IO connection errors

##  Performance

### Latency Breakdown

| Stage | Time |
|-------|------|
| Audio capture (2s chunks) | 2000ms |
| Network (localhost) | 50ms |
| Google Speech Recognition | 500-1000ms |
| Google Translate | 200-500ms |
| gTTS | 300-700ms |
| Network return | 50ms |
| **Total** | **3.1-4.25 seconds** |

### System Requirements

**Minimum**:
- 4GB RAM
- Dual-core CPU
- Internet connection (for Google APIs)

**Recommended**:
- 8GB+ RAM
- Quad-core CPU
- Good internet connection (for best translation quality)

##  Advanced: Lower Latency with Local Models

Want <1.5s latency? Replace Google APIs with local models:

| Component | Google API | Local Alternative | Latency |
|-----------|------------|-------------------|---------|
| STT | Google SR (1s) | Faster-Whisper tiny (200ms) | -80% |
| Translation | Google Translate (500ms) | NLLB-200 distilled (150ms) | -70% |
| TTS | gTTS (700ms) | Piper TTS (250ms) | -65% |
| **TOTAL** | **~3.5s** | **~0.7s** | **-80%** |

**Requirements**: 
- GPU (NVIDIA with CUDA)
- 16GB+ RAM
- See `docs/LOCAL_MODELS_SETUP.md` (coming soon)

##  Development

### Project Structure

```
d:/haha/
├── backend/
│   ├── server.py                # Main Flask server
│   ├── audio_handler.py         # Audio processing
│   ├── translation_pipeline.py  # STT→Trans→TTS
│   └── config.py                # Configuration
│
├── frontend/
│   ├── index.html               # Main UI
│   └── video-buffer.js          # Video sync system
│
├── requirements.txt             # Python dependencies
├── run.bat                      # Windows launcher
└── README.md                    # This file
```

### Adding New Languages

1. Check if supported by Google Translate
2. Add to `backend/config.py`:
```python
SUPPORTED_LANGUAGES = {
    'your-lang-code': 'Language Name',
    # ...
}
```
3. Add to `frontend/index.html` language selector

### Modifying Translation Delay

Edit `backend/config.py`:
```python
EXPECTED_TRANSLATION_DELAY_MS = 3500  # Change this value
```

##  Technical Details

### Video Buffering System

The `VideoBufferManager` class implements frame-by-frame video buffering:

1. **Capture** frames from remote video at 30 FPS
2. **Store** in a circular buffer (queue)
3. **Delay** playback by translation latency
4. **Synchronize** with translated audio arrival

### Socket.IO Events

| Event | Direction | Purpose |
|-------|-----------|---------|
| `join_room` | Client→Server | Join meeting room |
| `enable_translation` | Client→Server | Enable translation |
| `audio_stream` | Client→Server | Send audio for translation  |
| `translated_audio` | Server→Client | Return translated audio |
| `translated_subtitle` | Server→Client | Send subtitle text |
| `translation_enabled` | Server→Client | Confirm + send delay time |

##  License

MIT License - Free for personal and commercial use

##  Credits

- **Agora SDK**: Real-time video/audio communication
- **Google Speech Recognition**: Speech-to-Text
- **Google Translate**: Translation services
- **gTTS**: Text-to-Speech synthesis
- **Flask & Socket.IO**: Backend framework
- **WebRTC**: Browser media APIs

##  Support

Having issues?
1. Check **Troubleshooting** section above
2. Review backend logs in terminal
3. Check browser console (F12 → Console tab)
4. Verify FFmpeg is installed: `ffmpeg -version`

---

##  Enjoy Breaking Language Barriers in Real-Time!

**Made with love for seamless global communication**

*Now with perfectly synchronized video - because lip-sync matters!*
