# backend/sign_translation.py

import os
import sys
import time
from collections import deque
import numpy as np

# Adjust imports to access existing ASL model in the parent directory
SIGN_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sign_alphabet_model'))
if SIGN_MODEL_DIR not in sys.path:
    sys.path.insert(0, SIGN_MODEL_DIR)

from inference.predictor import ASLPredictor
from preprocessing.normalize import normalize_sample

CONFIDENCE_THRESHOLD = 0.8
STABLE_FRAMES = 8
LETTER_COOLDOWN = 0.5
WORD_PAUSE_DURATION = 1.5

class SignTranslationHandler:
    def __init__(self, translation_pipeline):
        print("Initializing SignTranslationHandler with ASLPredictor...")
        self.predictor = ASLPredictor()
        self.translation_pipeline = translation_pipeline
        
        # State per user per room: { (room, uid): state_dict }
        self.user_states = {}

    def get_user_state(self, room, uid):
        key = (room, uid)
        if key not in self.user_states:
            self.user_states[key] = {
                'recent_predictions': deque(maxlen=STABLE_FRAMES),
                'last_accepted_letter': None,
                'last_accept_time': 0.0,
                'last_hand_time': time.time(),
                'current_word': "",
                'current_sentence': ""
            }
        return self.user_states[key]

    def process_landmarks(self, room, uid, landmarks_list):
        """Process an incoming hand landmark frame and update word buffers."""
        state = self.get_user_state(room, uid)
        current_time = time.time()
        
        # Update last seen timestamp
        state['last_hand_time'] = current_time
        
        # Convert list of 63 numbers to numpy and normalize
        raw_landmarks = np.array(landmarks_list, dtype=np.float32)
        norm_landmarks = normalize_sample(raw_landmarks)
        
        # Predict character
        label, confidence = self.predictor.predict(norm_landmarks)
        
        # Letter Stabilization: Ensure the same letter appears over consecutive frames
        stable_label = None
        if confidence >= CONFIDENCE_THRESHOLD:
            state['recent_predictions'].append(label)
            if len(state['recent_predictions']) == STABLE_FRAMES and len(set(state['recent_predictions'])) == 1:
                stable_label = state['recent_predictions'][0]
        else:
            state['recent_predictions'].clear()
            
        # Add to word buffer if cool-down has passed to avoid duplicates
        if stable_label and stable_label != state['last_accepted_letter']:
            if (current_time - state['last_accept_time']) > LETTER_COOLDOWN:
                state['current_word'] += stable_label
                state['last_accepted_letter'] = stable_label
                state['last_accept_time'] = current_time
                state['recent_predictions'].clear()
                print(f"[SIGN] Accepted letter for User {uid}: {stable_label}")

    def check_pauses_and_broadcast(self, socketio, app_user_states):
        """
        Periodically invoked to detect pauses in signing 
        to finalize words and sentences.
        """
        current_time = time.time()
        keys_to_clear = []
        
        for key, state in self.user_states.items():
            room, uid = key
            
            # 1. Finalize Word
            # If the user has started a word, but their hand hasn't been seen for WORD_PAUSE_DURATION
            if state['current_word'] and (current_time - state['last_hand_time']) > WORD_PAUSE_DURATION:
                print(f"[SIGN] Finalized Word for User {uid}: {state['current_word']}")
                state['current_sentence'] += state['current_word'] + " "
                state['current_word'] = ""
                state['last_accepted_letter'] = None
                state['recent_predictions'].clear()
                
            # 2. Finalize Sentence and Translate
            # If the sentence has words and we've waited a bit longer (e.g. 2.0s total pause)
            if state['current_sentence'] and (current_time - state['last_hand_time']) > 2.0:
                final_text = state['current_sentence'].strip()
                print(f"[SIGN] Finalized Sentence for User {uid}: '{final_text}'")
                
                # Send sentence to existing translation pipeline
                self.translate_and_emit(room, uid, final_text, app_user_states, socketio)
                
                # Reset for next sentence
                state['current_sentence'] = ""
                
            # 3. Cleanup stale users
            if (current_time - state['last_hand_time']) > 30.0:
                keys_to_clear.append(key)
                
        # Remove old records
        for key in keys_to_clear:
            del self.user_states[key]

    def translate_and_emit(self, room, uid, text, app_user_states, socketio):
        """Processes text through translation_pipeline and sends to active recipients."""
        from datetime import datetime
        
        # Determine sender's display name
        source_name = f"User {uid}"
        for sid, info in app_user_states.items():
            # user_id might be stored as int or str, normalize to str
            if info.get('room') == room and str(info.get('user_id')) == str(uid):
                source_name = f"{info.get('name')} (Sign)"
                break
                
        # The Sign Language model translates directly into English
        source_lang = 'en'
        
        # Send translated subtitles and audio to all participants with translation enabled
        for target_sid, target_info in app_user_states.items():
            if target_info.get('room') == room and target_info.get('translation_enabled'):
                
                # Optionally prevent sending it back to the signer (if desired), 
                # but we will send it so they see their own translation too
                target_lang = target_info.get('language', 'en')
                
                pipeline_start = time.time()
                
                # Translate text
                trans_result = self.translation_pipeline.translate_text(text, source_lang, target_lang)
                if not trans_result:
                    continue
                    
                translated_text = trans_result['text']
                
                # TTS
                tts_result = self.translation_pipeline.text_to_speech(translated_text, target_lang)
                if not tts_result:
                    continue
                    
                audio_base64 = tts_result['audio']
                
                total_latency_ms = int((time.time() - pipeline_start) * 1000)
                
                # Emit events matching the existing `translated_audio` and `translated_subtitle`
                subtitle_data = {
                    'text': translated_text,
                    'original_text': text,
                    'source_lang': source_lang,
                    'target_lang': target_lang,
                    'sender_name': source_name,
                    'timestamp': datetime.now().isoformat()
                }
                
                # Emit audio packet
                socketio.emit('translated_audio', {
                    'audio': audio_base64,
                    'latency_ms': total_latency_ms,
                    'breakdown': {
                        'translation_ms': trans_result['latency_ms'],
                        'tts_ms': tts_result['latency_ms'],
                        'total_ms': total_latency_ms
                    },
                    'timestamp': datetime.now().isoformat(),
                    'sender': source_name,
                    'target_lang': target_lang
                }, room=target_sid)
                
                # Emit subtitle packet
                socketio.emit('translated_subtitle', subtitle_data, room=target_sid)
