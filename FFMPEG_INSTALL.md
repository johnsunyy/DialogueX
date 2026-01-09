# FFmpeg Installation Guide for DIALOGUE-X

## Problem
Google Speech Recognition cannot process WebM audio directly. FFmpeg is required to convert WebM → WAV.

## Quick Installation (Windows)

### Method 1: Download Pre-built Binary (EASIEST)

1. **Download FFmpeg**:
   - Go to: https://www.gyan.dev/ffmpeg/builds/
   - Download: **ffmpeg-release-essentials.zip** (~ 80MB)

2. **Extract**:
   - Extract ZIP to `C:\ffmpeg`
   - You should have: `C:\ffmpeg\bin\ffmpeg.exe`

3. **Add to PATH**:
   ```
   - Press Win + X → System
   - Click "Advanced system settings"
   - Click "Environment Variables"
   - Under "System variables", find "Path"
   - Click "Edit" → "New"
   - Add: C:\ffmpeg\bin
   - Click OK on all dialogs
   ```

4. **Verify Installation**:
   ```bash
   # Close and reopen PowerShell
   ffmpeg -version
   ```

### Method 2: Winget (Windows Package Manager)

```bash
winget install ffmpeg
```

### Method 3: Scoop

```bash
scoop install ffmpeg
```

## After Installation

1. **Close all terminals**
2. **Reopen PowerShell**
3. **Verify**: `ffmpeg -version`
4. **Restart backend**:
   ```bash
   cd d:\haha\backend
   python server.py
   ```

## Alternative: Use the Original audio_handler.py

Once FFmpeg is installed, revert to the original audio_handler.py that uses pydub:

```bash
# I can restore the original version for you
```

---

**Once FFmpeg is installed, translation will work perfectly!** ✅
