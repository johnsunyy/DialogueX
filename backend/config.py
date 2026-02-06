# Configuration file for DIALOGUE-X backend server

# Server configuration
FLASK_HOST = '0.0.0.0'
FLASK_PORT = 5000
FLASK_DEBUG = True

# Socket.IO configuration
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

# Translation timing
EXPECTED_TRANSLATION_DELAY_MS = 3500  # 3.5 seconds default for video sync

# Logging
LOG_LEVEL = 'INFO'
