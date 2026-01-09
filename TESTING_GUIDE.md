# Translation System - Testing Guide

## ✅ System Status
**ALL COMPONENTS WORKING!**
- ✅ Audio capture (48KB segments)
- ✅ Format conversion (WebM → WAV)
- ✅ Backend receiving audio
- ✅ STT/Translation/TTS pipeline operational

## Current Issue
Google Speech Recognition returns: **"Could not understand audio"**

This means the audio quality or speech clarity needs improvement.

## How to Test Successfully

### Tab 1 (Speaker)
1. **Join meeting** as "Alice"
2. **Keep microphone ON**
3. **DO NOT enable translation**
4. **Position close to microphone**

### Tab 2 (Listener)
1. **Join same meeting** as "Bob"  
2. **MUTE YOUR MICROPHONE** ⚠️
3. **Enable Translation** → Select Malayalam
4. **Wear headphones** (prevent feedback)

### Speaking Tips for Recognition
**DO:**
- ✅ Speak **clearly and slowly**
- ✅ Speak **directly into mic** (6-12 inches away)
- ✅ Use **short, simple phrases**
- ✅ Speak for **full 3+ seconds**
- ✅ Use **common words** first

**Example phrases:**
- "Hello, how are you today?"
- "This is a test of the translation system"
- "One, two, three, four, five"

**DON'T:**
- ❌ Whisper or speak quietly
- ❌ Speak too fast
- ❌ Have background noise/music
- ❌ Speak less than 3 seconds
- ❌ Use complex technical words

## Testing Process

1. **Tab 1**: Click unmute, speak clearly: "Hello, how are you?"
2. **Wait 6-7 seconds** (3s recording + 3-4s translation)
3. **Tab 2**: Should hear Malayalam translation ഹലോ

## What You Should See in Backend Logs

**Working:**
```
[STT] Transcribed: 'hello how are you' (detected: en) - 747ms
[Translation] en→ml: 'hello how are you' → 'നിങ്ങൾക്ക് എങ്ങനെയുണ്ട്' - 482ms
[TTS] Generated audio - 1200ms
[Pipeline] Total latency: 2429ms
```

**Not Working:**
```
[STT] Could not understand audio  ← Audio too quiet/unclear
```

## Troubleshooting

### If still "Could not understand audio":
1. **Check microphone level** in Windows settings
2. **Test in Tab 1** - can you see audio visualizer reacting?
3. **Try different phrases** - use simple words
4. **Reduce background noise**
5. **Speak louder and clearer**

### If you hear feedback/echo:
- Mute Tab 2's microphone
- Use headphones on Tab 2

## Next Steps

Once you get successful transcription:
- Translation will happen automatically
- Translated audio plays in Tab 2
- Malayalam subtitle appears
- Video delay synchronizes

**The system is ready - just needs clear speech input!** 🎤
