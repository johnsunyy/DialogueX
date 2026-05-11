from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import logging
import os
import secrets
from datetime import datetime

from audio_handler import AudioHandler
from translation_pipeline import TranslationPipeline
from sign_translation import SignTranslationHandler
from config import *

# Initialize Flask app
app = Flask(__name__)
# Load SECRET_KEY from environment variable; never hardcode secrets in source.
# Set the SECRET_KEY environment variable before running in production.
_secret_key = os.environ.get('SECRET_KEY')
if not _secret_key:
    _secret_key = secrets.token_hex(32)
    print("[WARNING] SECRET_KEY environment variable not set. "
          "Using a temporary random key — sessions will not persist across restarts.")
app.config['SECRET_KEY'] = _secret_key
CORS(app)

# Initialize SocketIO
socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    ping_timeout=SOCKETIO_PING_TIMEOUT,
    ping_interval=SOCKETIO_PING_INTERVAL
)

# Initialize handlers
audio_handler = AudioHandler()
translation_pipeline = TranslationPipeline()
sign_translation_handler = SignTranslationHandler(translation_pipeline)

# Store user states: {sid: {room, name, user_id, language, translation_enabled}}
user_states = {}

# Configure logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Socket.IO Event Handlers ====================

@socketio.on('connect')
def handle_connect():
    """Handle client connection"""
    logger.info(f"Client connected: {request.sid}")
    emit('connection_response', {'status': 'connected', 'sid': request.sid})

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection"""
    sid = request.sid
    if sid in user_states:
        user_info = user_states[sid]
        logger.info(f"Client disconnected: {user_info.get('name', sid)}")
        del user_states[sid]
    else:
        logger.info(f"Client disconnected: {sid}")

@socketio.on('join_room')
def handle_join_room(data):
    """
    Handle user joining a room
    Data: {room: str, name: str, user_id: int/str}
    """
    try:
        room = data.get('room')
        name = data.get('name', 'Unknown')
        user_id = data.get('user_id')
        
        if not room:
            emit('error', {'message': 'Room ID is required'})
            return
        
        # Join the room
        join_room(room)
        
        # Store user state
        user_states[request.sid] = {
            'room': room,
            'name': name,
            'user_id': user_id,
            'language': None,
            'source_language': data.get('source_language', 'en-US'),
            'translation_enabled': False
        }
        
        logger.info(f"User '{name}' (ID: {user_id}) joined room '{room}'")
        
        emit('joined_room', {
            'room': room,
            'name': name,
            'user_id': user_id,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in join_room: {e}")
        emit('error', {'message': f'Failed to join room: {str(e)}'})

@socketio.on('enable_translation')
def handle_enable_translation(data):
    """
    Enable translation for a user
    Data: {room: str, language: str}
    """
    try:
        room = data.get('room')
        language = data.get('language')
        
        if request.sid not in user_states:
            emit('error', {'message': 'User not in a room'})
            return
        
        # Accept ANY language code - Google Translate/TTS will handle validation
        if not language:
            emit('error', {'message': 'Language is required'})
            return
        
        # Update user state
        user_states[request.sid]['language'] = language
        user_states[request.sid]['translation_enabled'] = True
        
        user_name = user_states[request.sid]['name']
        
        # Try to get friendly name from config, otherwise use the code
        language_name = SUPPORTED_LANGUAGES.get(language, language.upper())
        
        logger.info(f"Translation enabled for '{user_name}' → {language_name}")
        
        # Send expected delay to frontend for video buffering
        emit('translation_enabled', {
            'language': language,
            'language_name': language_name,
            'expected_delay_ms': EXPECTED_TRANSLATION_DELAY_MS,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Error in enable_translation: {e}")
        emit('error', {'message': f'Failed to enable translation: {str(e)}'})

@socketio.on('disable_translation')
def handle_disable_translation(data):
    """
    Disable translation for a user
    Data: {room: str}
    """
    try:
        if request.sid not in user_states:
            return
            
        user_states[request.sid]['translation_enabled'] = False
        user_states[request.sid]['language'] = None
        
        user_name = user_states[request.sid]['name']
        logger.info(f"Translation disabled for '{user_name}'")
        
        emit('translation_disabled', {'timestamp': datetime.now().isoformat()})
        
    except Exception as e:
        logger.error(f"Error in disable_translation: {e}")
        emit('error', {'message': f'Failed to disable translation: {str(e)}'})

@socketio.on('client_log')
def handle_client_log(data):
    """
    Receive debug logs from frontend to help diagnose issues
    Data: {level: str, message: str}
    """
    level = data.get('level', 'INFO')
    message = data.get('message', '')
    sid = request.sid
    user = user_states.get(sid, {}).get('name', 'Unknown')
    print(f"[CLIENT LOG - {user}] {level}: {message}")

@socketio.on('translate_audio_chunk')
def handle_translate_audio_chunk(data):
    """
    Handle incoming audio chunk from a RECEIVER for specific translation.
    Receiver captures a remote user's audio and asks for translation to their own language.
    Data: {
        room: str,
        source_uid: int/str,  # Who is speaking
        audio: str (base64),
        target_lang: str      # Language to translate TO
    }
    """
    try:
        requester_sid = request.sid
        if requester_sid not in user_states:
            return

        source_uid = data.get('source_uid')
        target_lang = data.get('target_lang')
        base64_audio = data.get('audio')
        room = user_states[requester_sid]['room']

        if not base64_audio or not target_lang:
            return

        # Identify Source Name and Source Language
        # We find the name of the user with source_uid in the same room
        source_name = f"User {source_uid}"
        source_language = "en-US"
        
        # Optimize: In a real app, use a dict for ID lookups. Here we iterate (N is small).
        for sid, info in user_states.items():
            # Check if in same room and ID matches
            # Note: user_id might be int or str, safest to compare as str
            if info['room'] == room and str(info.get('user_id')) == str(source_uid):
                source_name = info['name']
                source_language = info.get('source_language', 'en-US')
                break

        # Process Audio (Decode Base64)
        try:
            audio_data = audio_handler.process_audio_chunk(base64_audio)
        except Exception as e:
            logger.error(f"Audio processing error: {e}")
            return
        
        # Translate
        # Note: pipeline.process_audio detects source language automatically from audio
        # It needs 'source_name' just for logging/subtitle attribution
        try:
            result = translation_pipeline.process_audio(audio_data, target_lang, source_name, source_language)
            
            if result:
                # UNICAST response to Requester (only they hear this translation)
                emit('translated_audio', {
                    'audio': result['audio'],
                    'latency_ms': result['total_latency_ms'],
                    'breakdown': result['breakdown'],
                    'timestamp': datetime.now().isoformat(),
                    'sender': source_name,
                    'target_lang': target_lang
                }, room=requester_sid)
                
                emit('translated_subtitle', result['subtitle'], room=requester_sid)
                
                logger.info(f"✓ Translated {source_name} -> {user_states[requester_sid]['name']} ({target_lang})")
            else:
                # Silence or no speech detected
                pass
                
        except Exception as e:
            logger.error(f"Translation pipeline error: {e}")
            
    except Exception as e:
        logger.error(f"Error in translate_audio_chunk: {e}")

@socketio.on('sign_landmarks')
def handle_sign_landmarks(data):
    """
    Handle incoming hand landmarks for sign language translation.
    Data: { room: str, uid: int/str, landmarks: list[float] }
    """
    try:
        room = data.get('room')
        uid = data.get('uid')
        landmarks = data.get('landmarks')
        
        if room and uid and landmarks:
            sign_translation_handler.process_landmarks(room, uid, landmarks, socketio)
    except Exception as e:
        logger.error(f"Error processing sign landmarks: {str(e)}")

@socketio.on('set_sign_mode')
def handle_set_sign_mode(data):
    """
    Let the signer choose which recognition model to use.
    Data: { room: str, uid: int/str, mode: 'alphabet' | 'word' | 'auto' }

    Modes:
      'alphabet' — always use the MLP (finger-spelling, letter by letter)
      'word'     — always use the Transformer (whole-word gestures)
      'auto'     — PME-based automatic switching (default)
    """
    try:
        room = data.get('room')
        uid  = data.get('uid')
        mode = data.get('mode', 'auto')

        if mode not in ('alphabet', 'word', 'auto'):
            emit('error', {'message': f"Unknown sign mode '{mode}'. Use alphabet/word/auto."})
            return

        sign_translation_handler.set_user_mode(room, uid, mode)
        logger.info(f"Sign mode for user {uid} in room {room} set to '{mode}'")

        emit('sign_mode_updated', {
            'mode': mode,
            'uid':  uid,
            'room': room
        })
    except Exception as e:
        logger.error(f"Error in set_sign_mode: {str(e)}")

def sign_language_background_task():
    """Background task to finalize sign language words and sentences."""
    while True:
        socketio.sleep(0.5)
        sign_translation_handler.check_pauses_and_broadcast(socketio, user_states)

# Start background task
socketio.start_background_task(sign_language_background_task)

# ==================== Flask Routes ====================

@app.route('/')
def index():
    """Health check route"""
    return {
        'status': 'running',
        'service': 'DIALOGUE-X Translation Backend',
        'example_languages': SUPPORTED_LANGUAGES,
        'note': 'Supports ALL Google Translate language codes, not just examples above',
        'expected_delay_ms': EXPECTED_TRANSLATION_DELAY_MS
    }

@app.route('/health')
def health():
    """Health check endpoint"""
    return {'status': 'healthy', 'timestamp': datetime.now().isoformat()}

# ==================== Main ====================

if __name__ == '__main__':
    logger.info("=" * 70)
    logger.info("🚀 Starting DIALOGUE-X Translation Backend")
    logger.info("=" * 70)
    logger.info(f"Server: http://{FLASK_HOST}:{FLASK_PORT}")
    logger.info(f"Supported languages: {len(SUPPORTED_LANGUAGES)}")
    logger.info(f"Expected translation delay: {EXPECTED_TRANSLATION_DELAY_MS}ms")
    logger.info("=" * 70)
    
    # Run server
    socketio.run(
        app,
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=FLASK_DEBUG,
        use_reloader=False  # Disable reloader in production
    )
