# How to Run DIALOGUE-X

This guide provides step-by-step instructions on how to set up and run the DIALOGUE-X platform locally.

## Prerequisites

Before you begin, ensure you have the following installed on your system:
1. **Python 3.8+**: [Download Python](https://www.python.org/downloads/)
2. **FFmpeg**: Required for audio processing. 
   - **Windows**: Download from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/) or install via winget: `winget install ffmpeg`
   - **Mac**: `brew install ffmpeg`
   - **Linux**: `sudo apt install ffmpeg`
3. **Agora App ID**: You need a free account from [Agora.io](https://console.agora.io/) to handle the WebRTC video calling.

---

## Step 1: Environment Setup

1. **Configure Environment Variables**
   - Copy the provided template file: 
     ```bash
     cp .env.example .env
     ```
   - Open the `.env` file in a text editor.
   - Set `SECRET_KEY` to a random string.
   - Set `AGORA_APP_ID` to your App ID from the Agora Console.
   - Set `BACKEND_URL` to `http://localhost:5000` (or your local IP if running on a LAN).

2. **Install Python Dependencies**
   Open a terminal in the project root folder and run:
   ```bash
   pip install -r requirements.txt
   pip install -r sign_model/requirements.txt
   ```

3. **Configure the Frontend**
   - Open `frontend/index.html` and `index.html`.
   - Ensure the `APP_ID` constant is updated with your actual Agora App ID, or that the frontend is reading it correctly.
   *(Note: The codebase may have been updated to read from a config or `.env` depending on your exact setup, but if you see `YOUR_AGORA_APP_ID` in the HTML, replace it with your real key).*

---

## Step 2: Running the Platform

### Method 1: The One-Click Script (Windows Only)
If you are on Windows, you can simply double-click the `run.bat` file in the root directory. 
This script will automatically check your Python/FFmpeg installation, start the backend server, start the frontend server, and open your web browser.

### Method 2: Manual Startup (All Platforms)

You will need to open **two separate terminal windows**.

**Terminal 1: Start the Backend Translation Server**
```bash
cd backend
python server.py
```
*You should see output indicating the Flask/Socket.IO server is running on port 5000.*

**Terminal 2: Start the Frontend Web Server**
```bash
# If using the frontend folder
cd frontend
python -m http.server 8080

# OR if using the root folder index.html
python -m http.server 8080
```

---

## Step 3: Using the Application

1. Open your web browser and go to: `http://localhost:8080`
2. Click **Get Started** and log in (any email/password will work for the demo, or enter your configured auth credentials).
3. Choose your communication mode:
   - **Normal Video Call**: Standard WebRTC calling without translation delays.
   - **Live Translation Call**: Audio is routed through the STT -> Translation -> TTS pipeline. Video is intentionally buffered to maintain lip-sync with the translated audio.
   - **Sign -> Speech**: Utilizes your webcam to recognize Sign Language landmarks and converts them to speech.
4. **Grant Permissions**: Your browser will ask for Microphone and Camera permissions. You must allow these for the platform to work.
5. In the call interface, use the **Translate** button to select your preferred spoken language.

---

## Troubleshooting

- **Audio not working/translating?** Ensure FFmpeg is correctly installed and added to your system's PATH.
- **Video not showing?** Ensure another application (like Zoom or Teams) isn't holding exclusive control over your webcam.
- **Connection Error/Socket Error?** Ensure the `BACKEND_URL` in your frontend HTML files perfectly matches the address where `server.py` is running (usually `http://localhost:5000`).
- **No Sign Language Detection?** Ensure you have run the data extraction and model training scripts inside `sign_model/scripts/` to generate the `.pt` and `label_map.json` files if they are missing.

---

## ⚠️ Important: Large Data Files

The processed datasets (`dataset.npz`, `dataset_v2.npz`) and raw landmark files are **ignored by Git** because they exceed GitHub's 100MB file size limit.

- **If you are cloning this repo for the first time:** You will need to run the preprocessing scripts (`sign_model/scripts/02_preprocess.py`) to generate these files locally before you can train or evaluate the model.
- **If you are collaborating:** Do not manually remove these files from `.gitignore`. Instead, share the processed `.npz` files via a cloud drive or re-generate them from the raw landmark data.
