# Quick Testing Guide - Audio Quality Improvements

## Setup (2 Browser Tabs Required)

The backend is already running! You should see two command windows:
- **Backend Server** (port 5000)
- **Frontend Server** (port 8080)

## Step-by-Step Test

### Tab 1 (Listener - Will Hear Translation)

1. **Open**: http://localhost:8080
2. **Fill form**:
   - Room code: `audio-test`
   - Your name: `Listener`
   - Role: Host Meeting
3. **Click**: "Start Meeting"
4. **Allow**: Camera and microphone permissions
5. **Open Console** (Press F12, click "Console" tab)
6. **Click**: "Translate" button (bottom controls)
7. **Select**: Spanish (or any language)
8. **Click**: "Enable Translation"

### Tab 2 (Speaker - Will Speak)

1. **Open NEW TAB**: http://localhost:8080
2. **Fill form**:
   - Room code: `audio-test` (SAME as Tab 1)
   - Your name: `Speaker`
   - Role: Join Meeting
3. **Click**: "Join Meeting"
4. **Allow**: Camera and microphone permissions
5. **Wait** 5 seconds for connection
6. **Speak clearly**: "Hello, how are you today?"

---

## ✅ What to Look For (Proof Our Changes Worked)

### In Tab 1 Console (F12)

Look for these NEW log messages (our improvements):

```
✅ Audio processing chain created: {
    sampleRate: 48000Hz,
    gainBoost: 1.5x (50%),
    compression: enabled
}
🔴 Recording started: 3-second chunks at 128kbps
📦 Audio chunk captured: 150KB+ (larger chunks = better quality!)
📤 Sending audio chunk to backend for translation
```

**If you see this** → Web Audio API processing is working! ✅

### In Backend Server Window

Look for these messages:

```
[AudioHandler] Received 150000+ bytes  ← Bigger than before (was ~40KB)
[AudioHandler] Audio level before normalization: -25.34 dBFS
[AudioHandler] Audio level after normalization: -20.12 dBFS  ← Volume normalization working!
[STT] Transcribed: 'Hello, how are you today' (detected: en)  ← Speech recognized!
[Translation] en→es: 'Hello, how are you today' → 'Hola, ¿cómo estás hoy?'
[TTS] Generated audio for 'Hola, ¿cómo estás hoy?' in es
[Pipeline] Total latency: 2604ms
✓ Translation sent
```

### In Tab 1 (Listener Browser)

You should:
1. **Hear**: Spanish audio saying "Hola, ¿cómo estás hoy?"
2. **See**: Subtitle at bottom showing translated text

---

## ⚠️ Troubleshooting

### "Could not understand audio"

**OLD SYSTEM** → Would fail often (50% failure rate)
**NEW SYSTEM** → Should rarely fail (<20%)

If you see this error now, you should ALSO see:
```
[STT] Could not understand audio, retrying with adjusted threshold...
[STT] ✓ Retry successful: '...' (detected: en)
```
This is the NEW retry mechanism working!

### Audio chunks too small

If console shows `📦 Audio chunk captured: 40KB`:
- Refresh the page and try again
- Browser may not support `audioBitsPerSecond` setting

**Expected**: Should be 150-200KB per chunk

### No audio processing logs

If you DON'T see "Audio processing chain created":
- Browser may not support Web Audio API
- Try Chrome or Edge instead

---

## Quick Comparison

| Metric | Before | After (Now) |
|--------|--------|-------------|
| Chunk Size | ~40-80KB | ~150-200KB |
| Sample Rate | 16kHz (default) | 48kHz |
| Bitrate | Default (~64kbps) | 128kbps |
| Processing | None | Gain + Compression |
| Volume Norm | No | Yes (backend) |
| STT Retry | No | Yes |

---

## Expected Timeline

1. **Tab 2 speaks** (0s)
2. **Audio captured** every 3 seconds
3. **Backend processes** (~2-3 seconds)
4. **Tab 1 hears translation** (~3-4 seconds total)

Total delay: 3-4 seconds is NORMAL and expected!
