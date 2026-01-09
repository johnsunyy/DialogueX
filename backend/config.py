# Configuration file for DIALOGUE-X backend server

# Server configuration
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# Socket.IO configuration
SOCKETIO_CORS_ALLOWED = '*'  # Allow all origins for development
SOCKETIO_PING_TIMEOUT = 60
SOCKETIO_PING_INTERVAL = 25

# Language configuration
# These are EXAMPLES for the frontend dropdown - backend accepts ANY Google Translate language code
SUPPORTED_LANGUAGES = {
    'en': 'English',
    'hi': 'Hindi (हिंदी)',
    'ml': 'Malayalam (മലയാളം)',
    'es': 'Spanish (Español)',
    'fr': 'French (Français)',
    'de': 'German (Deutsch)',
    'ja': 'Japanese (日本語)',
    'ko': 'Korean (한국어)',
    'zh-cn': 'Chinese (中文)',
    'ar': 'Arabic (العربية)',
    'pt': 'Portuguese (Português)',
    'ru': 'Russian (Русский)',
    'it': 'Italian (Italiano)',
    'ta': 'Tamil (தமிழ்)',
    'te': 'Telugu (తెలుగు)',
    'bn': 'Bengali (বাংলা)'
}

# Audio processing
AUDIO_CHUNK_SIZE = 3000  # milliseconds (increased from 2000 for better STT)
TEMP_AUDIO_DIR = 'temp_audio'

# Audio quality settings (frontend Web Audio API processing)
AUDIO_SAMPLE_RATE = 48000  # High quality capture (48kHz)
AUDIO_BITRATE = 128000     # 128kbps for high fidelity
AUDIO_CHUNK_DURATION_MS = 3000  # 3-second chunks (optimal for speech)
MIN_AUDIO_DURATION_MS = 1500    # Minimum chunk size to process

# Speech recognition tuning
STT_ENERGY_THRESHOLD = 300   # Standard energy threshold
STT_PAUSE_THRESHOLD = 0.8    # Pause detection threshold
STT_DYNAMIC_THRESHOLD = True # Auto-adjust threshold

# Video synchronization
# Expected translation latency (will be measured and sent to frontend)
EXPECTED_TRANSLATION_DELAY_MS = 3500  # 3.5 seconds default

# Logging
LOG_LEVEL = 'INFO'
