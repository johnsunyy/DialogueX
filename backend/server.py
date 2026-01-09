from flask import Flask, request
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_cors import CORS
import logging
from datetime import datetime

from audio_handler import AudioHandler
from translation_pipeline import TranslationPipeline
from config import *

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'dialogue-x-secret-key'
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
            emit('error', {'message': 'User not in a room'})
            return
        
        user_states[request.sid]['translation_enabled'] = False
        user_states[request.sid]['language'] = None
        
        user_name = user_states[request.sid]['name']
        logger.info(f"Translation disabled for '{user_name}'")
        
        emit('translation_disabled', {'timestamp': datetime.now().isoformat()})
        
    except Exception as e:
        logger.error(f"Error in disable_translation: {e}")
        emit('error', {'message': f'Failed to disable translation: {str(e)}'})

@socketio.on('audio_stream')
def handle_audio_stream(data):
    """
    Handle incoming audio stream for CLIENT-SIDE translation
    Client sends audio they RECEIVED, backend translates to THEIR language
    Data: {room: str, audio: str (base64)}
    """
    try:
        sender_sid = request.sid
        if sender_sid not in user_states:
            logger.warning("Audio stream from unknown user")
            return
        
        user_info = user_states[sender_sid]
        
        if not user_info['translation_enabled']:
            logger.warning(f"Audio stream from user without translation enabled: {user_info['name']}")
            return
        
        base64_audio = data.get('audio')
        if not base64_audio:
            logger.warning("Empty audio data received")
            return
        
        target_lang = user_info['language']
        user_name = user_info['name']
        
        logger.info(f"Processing audio for '{user_name}' → {SUPPORTED_LANGUAGES.get(target_lang, target_lang)}")
        
        # Process audio in pipeline
        try:
            # Step 1: Convert base64 to AudioData
            audio_data = audio_handler.process_audio_chunk(base64_audio)
            
            # Step 2: Run translation pipeline (translate to THIS user's language)
            result = translation_pipeline.process_audio(audio_data, target_lang, "Speaker")
            
            if result:
                # Send translated audio BACK to the SAME client who sent it
                emit('translated_audio', {
                    'audio': result['audio'],
                    'latency_ms': result['total_latency_ms'],
                    'breakdown': result['breakdown'],
                    'timestamp': datetime.now().isoformat()
                })
                
                # Send subtitle back to client
                emit('translated_subtitle', result['subtitle'])
                
                logger.info(f"✓ Translation sent to '{user_name}' (latency: {result['total_latency_ms']}ms)")
            else:
                logger.warning(f"Translation pipeline returned no result for '{user_name}'")
                
        except Exception as e:
            logger.error(f"Error in translation pipeline: {e}")
            emit('error', {'message': f'Translation failed: {str(e)}'})
        
    except Exception as e:
        logger.error(f"Error in audio_stream handler: {e}")
        emit('error', {'message': f'Audio processing failed: {str(e)}'})

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
