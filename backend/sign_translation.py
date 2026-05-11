# backend/sign_translation.py
import collections as _collections
#
# Dual-model sign language handler for Dialogue-X.
#
# State machine
# ─────────────
#   IDLE         → no hand motion detected (PME below threshold)
#   SPELLING     → small / static motion → route to alphabet MLP (ASLPredictor)
#   WORD_SIGNING → large / dynamic motion → accumulate into Transformer buffer
#
# Feature layout (225 floats per frame, sent by frontend MediaPipe Holistic):
#   [0  :63 ]  left-hand  landmarks (21 × 3)
#   [63 :126]  right-hand landmarks (21 × 3)
#   [126:225]  pose       landmarks (33 × 3)   ← PME uses these 99 values

import os
import sys
import time
import asyncio
import threading
from collections import deque
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
#  Import existing Alphabet MLP predictor
# ──────────────────────────────────────────────────────────────────────────────

SIGN_ALPHABET_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'sign_alphabet_model')
)
if SIGN_ALPHABET_DIR not in sys.path:
    sys.path.insert(0, SIGN_ALPHABET_DIR)

from inference.predictor import ASLPredictor
from preprocessing.normalize import normalize_sample

# ──────────────────────────────────────────────────────────────────────────────
#  Import new word-level Transformer predictor
# ──────────────────────────────────────────────────────────────────────────────

SIGN_MODEL_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '..', 'sign_model')
)
if SIGN_MODEL_DIR not in sys.path:
    sys.path.insert(0, os.path.dirname(SIGN_MODEL_DIR))

from sign_model.predictor import WordSignPredictor

# ──────────────────────────────────────────────────────────────────────────────
#  Tuning constants
# ──────────────────────────────────────────────────────────────────────────────

# --- Alphabet / spelling mode ---
ALPHA_CONFIDENCE_THRESHOLD = 0.80   # min softmax confidence for a letter
ALPHA_STABLE_FRAMES        = 8      # consecutive same-letter frames to accept
LETTER_COOLDOWN            = 0.5    # seconds between accepted letters

# --- PME switching thresholds ---
PME_WORD_THRESHOLD         = 0.015  # above → WORD_SIGNING mode
PME_IDLE_THRESHOLD         = 0.003  # below → IDLE (no hands / still)
PME_DEBOUNCE_FRAMES        = 6      # frames to confirm a state switch

# --- Transformer / word mode (sliding-window, matches 05_test_live.py) ---
WORD_TARGET_FRAMES         = 64     # fixed sequence length the model was trained on
WORD_INFER_EVERY           = 8      # run inference every N new frames (webcam parity)
WORD_CONFIDENCE_THRESHOLD  = 0.75   # raised to reduce false positives (was 0.65)
WORD_COOLDOWN_SECS         = 1.5    # seconds to block after a successful word prediction
WORD_PAUSE_DURATION        = 2.5    # seconds of stillness to finalise a sentence

# --- Sentence finalisation ---
SENTENCE_PAUSE_DURATION    = 4.0    # total pause to send sentence to pipeline
STALE_USER_TIMEOUT         = 3600.0 # 1 hour timeout to prevent wiping settings if user tabs out

# ──────────────────────────────────────────────────────────────────────────────
#  States
# ──────────────────────────────────────────────────────────────────────────────

STATE_IDLE         = "IDLE"
STATE_SPELLING     = "SPELLING"
STATE_WORD_SIGNING = "WORD_SIGNING"

# ── User-selectable forced modes ──────────────────────────────────────────────
FORCED_MODE_AUTO     = "auto"       # default — PME decides
FORCED_MODE_ALPHABET = "alphabet"   # always use alphabet MLP
FORCED_MODE_WORD     = "word"       # always use word Transformer


# ──────────────────────────────────────────────────────────────────────────────
#  Helper — Pose Motion Energy
# ──────────────────────────────────────────────────────────────────────────────

def _pose_motion_energy(prev_pose: np.ndarray, curr_pose: np.ndarray) -> float:
    """
    Mean absolute displacement of the 33 pose landmarks between two frames.
    pose vectors are 99-floats (33 landmarks × 3 coords).
    """
    if prev_pose is None:
        return 0.0
    return float(np.mean(np.abs(curr_pose - prev_pose)))


# ──────────────────────────────────────────────────────────────────────────────
#  Per-user state factory
# ──────────────────────────────────────────────────────────────────────────────

def _make_user_state() -> dict:
    return {
        # --- Timing ---
        'last_hand_time':      time.time(),
        'last_seen_time':      time.time(),

        # --- User-selected model override ('auto' / 'alphabet' / 'word') ---
        'forced_mode':         FORCED_MODE_AUTO,

        # --- PME / auto-mode ---
        'mode':                STATE_IDLE,
        'prev_pose':           None,
        'pme_debounce_word':   0,    # frames consistently above word-threshold
        'pme_debounce_idle':   0,    # frames consistently below idle-threshold

        # --- SPELLING mode (alphabet MLP) ---
        'recent_predictions':   deque(maxlen=ALPHA_STABLE_FRAMES),
        'last_accepted_letter': None,
        'last_accept_time':     0.0,
        'current_word':        "",

        # --- WORD_SIGNING mode (Transformer) ---
        # Sliding window: deque of the last WORD_TARGET_FRAMES landmark vectors
        'word_window':         _collections.deque(maxlen=WORD_TARGET_FRAMES),
        'word_frames_since':   0,    # frames accumulated since last inference
        'last_word_time':      0.0,  # timestamp of last frame added in word mode
        'word_cooldown_until': 0.0,  # time.time() after which the next word is accepted

        # --- Sentence ---
        'current_sentence':    "",
    }


# ──────────────────────────────────────────────────────────────────────────────
#  Handler
# ──────────────────────────────────────────────────────────────────────────────

class SignTranslationHandler:
    """
    Drop-in replacement for the original SignTranslationHandler.
    Preserves the public API expected by server.py:
        - process_landmarks(room, uid, landmarks_list)
        - check_pauses_and_broadcast(socketio, app_user_states)
    """

    def __init__(self, translation_pipeline):
        print("[SignTranslation] Initialising dual-model sign handler …")

        # ── Alphabet model (MLP, 26 letters) ─────────────────────────────────
        try:
            self.alpha_predictor = ASLPredictor()
            print("[SignTranslation] ✓ ASLPredictor (alphabet MLP) loaded")
        except Exception as e:
            print(f"[SignTranslation] ✗ ASLPredictor failed to load: {e}")
            self.alpha_predictor = None

        # ── Word model (Transformer, 69 words) ───────────────────────────────
        try:
            self.word_predictor = WordSignPredictor()
            print("[SignTranslation] ✓ WordSignPredictor (Transformer) loaded")
        except Exception as e:
            print(f"[SignTranslation] ✗ WordSignPredictor failed to load: {e}")
            self.word_predictor = None

        self.translation_pipeline = translation_pipeline
        self.user_states: dict = {}          # { (room, uid): state_dict }
        self._inference_lock = threading.Lock()

    # ── State management ──────────────────────────────────────────────────────

    def _get_state(self, room, uid) -> dict:
        key = (room, uid)
        if key not in self.user_states:
            self.user_states[key] = _make_user_state()
        return self.user_states[key]

    def set_user_mode(self, room, uid, mode: str):
        """
        Set the model selection mode for a specific user.

        Parameters
        ----------
        mode : 'auto' | 'alphabet' | 'word'
          'auto'     — PME state machine decides automatically (default)
          'alphabet' — always route to the MLP (finger-spelling)
          'word'     — always route to the Transformer (whole-word gestures)
        """
        state = self._get_state(room, uid)
        state['forced_mode'] = mode
        # Reset buffers so the new mode starts fresh
        state['word_window'].clear()
        state['word_frames_since']   = 0
        state['word_cooldown_until'] = 0.0
        state['current_word']        = ""
        state['recent_predictions'].clear()
        state['pme_debounce_word']   = 0
        state['pme_debounce_idle']   = 0
        state['mode']                = STATE_IDLE
        print(f"[SignTranslation] User {uid} in room {room}: sign mode set to '{mode}'")

    # ── Main entry point (called by server.py on each socket frame) ───────────

    def process_landmarks(self, room, uid, landmarks_list, socketio=None):
        """
        Process a single incoming 225-float landmark frame.

        landmarks_list : list of 225 floats
            [0:63]   left-hand  (21 × xyz)
            [63:126] right-hand (21 × xyz)
            [126:225] pose      (33 × xyz)
        """
        state = self._get_state(room, uid)
        current_time = time.time()
        state['last_seen_time'] = current_time

        landmarks = np.array(landmarks_list, dtype=np.float32)
        if landmarks.shape[0] != 225:
            # Pad or truncate gracefully for robustness
            tmp = np.zeros(225, dtype=np.float32)
            n   = min(225, landmarks.shape[0])
            tmp[:n] = landmarks[:n]
            landmarks = tmp

        # ── Extract sub-regions ───────────────────────────────────────────────
        left_hand  = landmarks[0:63]      # 63 floats
        right_hand = landmarks[63:126]    # 63 floats
        pose       = landmarks[126:225]   # 99 floats  (33 pose × xyz)

        # Dominant hand = whichever is more active (non-zero)
        left_active  = np.any(left_hand  != 0)
        right_active = np.any(right_hand != 0)
        dominant_hand = right_hand if right_active else left_hand

        # ── Motion tracking (reset pause timer if moving) ─────────────────────
        pme = _pose_motion_energy(state['prev_pose'], pose)
        state['prev_pose'] = pose.copy()

        # Only consider it "active motion" if hands are actually visible.
        # For ALPHABET mode, we strictly use letter-acceptance to reset the timer.
        # This prevents webcam jitter from gluing all spelled words together into one long string.
        if pme > PME_IDLE_THRESHOLD and (left_active or right_active):
            if state['forced_mode'] != FORCED_MODE_ALPHABET:
                state['last_hand_time'] = current_time

        # ── Forced-mode override (user-selected; bypasses PME entirely) ───────
        forced = state['forced_mode']
        if forced == FORCED_MODE_ALPHABET:
            self._process_spelling(state, uid, room, dominant_hand, current_time, socketio)
            return
        if forced == FORCED_MODE_WORD:
            self._process_word_signing(state, landmarks, current_time,
                                       uid=uid, room=room, socketio=socketio)
            return
        # FORCED_MODE_AUTO falls through to the PME state machine below

        # ── Mode switching via debounce ───────────────────────────────────────
        if pme >= PME_WORD_THRESHOLD:
            state['pme_debounce_word'] += 1
            state['pme_debounce_idle']  = 0
        elif pme <= PME_IDLE_THRESHOLD:
            state['pme_debounce_idle'] += 1
            state['pme_debounce_word']  = 0
        else:
            # Mid-range: slowly decay debounce counters
            state['pme_debounce_word'] = max(0, state['pme_debounce_word'] - 1)
            state['pme_debounce_idle'] = max(0, state['pme_debounce_idle'] - 1)

        prev_mode = state['mode']

        if state['pme_debounce_word'] >= PME_DEBOUNCE_FRAMES:
            state['mode'] = STATE_WORD_SIGNING
        elif state['pme_debounce_idle'] >= PME_DEBOUNCE_FRAMES:
            state['mode'] = STATE_IDLE
        else:
            # Stay in current mode; if IDLE and hands appear, go to SPELLING
            if state['mode'] == STATE_IDLE and (left_active or right_active):
                state['mode'] = STATE_SPELLING

        # Log mode transitions
        if state['mode'] != prev_mode:
            print(f"[SignTranslation] User {uid} → {state['mode']}  (PME={pme:.4f})")

        # ── Dispatch to the active mode ───────────────────────────────────────
        if state['mode'] == STATE_SPELLING:
            self._process_spelling(state, uid, room, dominant_hand, current_time, socketio)

        elif state['mode'] == STATE_WORD_SIGNING:
            self._process_word_signing(state, landmarks, current_time,
                                       uid=uid, room=room, socketio=socketio)

        # In IDLE: do nothing; pauses handled in check_pauses_and_broadcast

    # ── SPELLING mode (alphabet MLP) ─────────────────────────────────────────

    def _process_spelling(self, state, uid, room, dominant_hand_63: np.ndarray, current_time: float, socketio=None):
        if self.alpha_predictor is None:
            return

        norm = normalize_sample(dominant_hand_63)
        label, confidence = self.alpha_predictor.predict(norm)

        # Emit realtime reflection
        if socketio:
            socketio.emit('realtime_prediction', {
                'uid': uid,
                'label': label,
                'confidence': round(confidence * 100),
                'type': 'alphabet'
            }, room=room)

        if confidence >= ALPHA_CONFIDENCE_THRESHOLD:
            state['recent_predictions'].append(label)
            buf = state['recent_predictions']
            if len(buf) == ALPHA_STABLE_FRAMES and len(set(buf)) == 1:
                stable_label = buf[0]
                # Cooldown + de-duplicate
                if (stable_label != state['last_accepted_letter'] and
                        (current_time - state['last_accept_time']) > LETTER_COOLDOWN):
                    state['current_word']         += stable_label
                    state['last_accepted_letter']  = stable_label
                    state['last_accept_time']      = current_time
                    state['last_hand_time']        = current_time
                    state['recent_predictions'].clear()
                    print(f"[SignTranslation] User {uid} letter: {stable_label} "
                          f"→ word so far: '{state['current_word']}'")
        else:
            state['recent_predictions'].clear()

    # ── WORD_SIGNING mode (Transformer — sliding window) ─────────────────────

    def _process_word_signing(self, state, landmarks_225: np.ndarray,
                               current_time: float, uid=None, room=None, socketio=None):
        """
        Sliding-window inference — mirrors 05_test_live.py exactly.

        Each incoming frame is appended to a fixed-size deque (maxlen=WORD_TARGET_FRAMES=64).
        Once the deque is full, inference fires every WORD_INFER_EVERY (8) new frames.
        Results are emitted immediately via 'realtime_prediction'.
        High-confidence words are appended to current_sentence for later sentence dispatch.
        """
        # Belt-and-suspenders: never run in alphabet forced mode
        if state['forced_mode'] == FORCED_MODE_ALPHABET:
            return

        # ── Fix 3: Skip frames where no hands are visible ────────────────────
        # The left-hand slice is [0:63] and the right-hand slice is [63:126].
        # If both are all-zero MediaPipe produced no hand detections — there is
        # nothing meaningful to classify, so skip silently.
        left_hand_present  = np.any(landmarks_225[:63]  != 0.0)
        right_hand_present = np.any(landmarks_225[63:126] != 0.0)
        if not left_hand_present and not right_hand_present:
            return

        window = state['word_window']          # collections.deque(maxlen=64)
        window.append(landmarks_225.copy())
        state['last_word_time'] = current_time
        state['word_frames_since'] += 1

        # Only infer when the window is full AND the stride has elapsed
        if len(window) < WORD_TARGET_FRAMES:
            return
        if state['word_frames_since'] < WORD_INFER_EVERY:
            return

        state['word_frames_since'] = 0

        # ── Fix 2: Honour the post-acceptance cooldown ───────────────────────
        if current_time < state['word_cooldown_until']:
            return

        if self.word_predictor is None:
            return

        frames = np.stack(list(window), axis=0)  # (64, 225)

        try:
            label, confidence = self.word_predictor.predict(frames)
        except Exception as e:
            print(f"[SignTranslation] Word inference error for user {uid}: {e}")
            return

        # Emit real-time overlay for any plausible prediction (threshold 0.30 for display)
        if socketio and confidence > 0.30:
            socketio.emit('realtime_prediction', {
                'uid':        uid,
                'label':      label,
                'confidence': round(confidence * 100),
                'type':       'word'
            }, room=room)

        print(f"[SignTranslation] User {uid} \u25b6 '{label}' (conf={confidence:.2f})")

        # ── Fix 1 + 2: On a confident acceptance, clear window + start cooldown
        if confidence >= WORD_CONFIDENCE_THRESHOLD:
            state['current_sentence'] += label.capitalize() + " "
            print(f"[SignTranslation] Sentence so far: '{state['current_sentence']}'")
            # Clear the window so the next prediction needs a fresh set of frames
            window.clear()
            state['word_frames_since'] = 0
            # Block further acceptances for WORD_COOLDOWN_SECS
            state['word_cooldown_until'] = current_time + WORD_COOLDOWN_SECS

    # ── Pause detection & broadcast (called periodically by server.py) ────────

    def check_pauses_and_broadcast(self, socketio, app_user_states):
        """
        Periodically invoked background task (from server.py).
        Detects signing pauses to finalize words/sentences and push to pipeline.

        ALPHABET mode  — mirrors the original pre-integration behavior exactly:
                         letters → current_word → current_sentence → translate
        WORD mode      — frames → transformer buffer → word prediction → sentence
        AUTO mode      — both paths active, switched by PME
        """
        current_time   = time.time()
        keys_to_clear  = []

        for key, state in list(self.user_states.items()):
            room, uid = key
            time_since_hand = current_time - state['last_hand_time']
            forced = state['forced_mode']

            # ── ALPHABET MODE: letter-spelling path only ──────────────────────
            # Exactly replicates original SignTranslationHandler behaviour.
            if forced == FORCED_MODE_ALPHABET:
                # 1. Finalize a spelled word after a pause
                if state['current_word'] and time_since_hand > WORD_PAUSE_DURATION:
                    spelled = state['current_word'].strip()
                    print(f"[SignTranslation] [SPELL] User {uid} word: '{spelled}'")
                    state['current_sentence']    += spelled + " "
                    state['current_word']         = ""
                    state['last_accepted_letter'] = None
                    state['recent_predictions'].clear()

                # 2. Finalize & translate the full sentence
                if state['current_sentence'] and time_since_hand > SENTENCE_PAUSE_DURATION:
                    final_text = state['current_sentence'].strip()
                    print(f"[SignTranslation] [SPELL] User {uid} sentence: '{final_text}'")
                    state['current_sentence'] = ""
                    self.translate_and_emit(room, uid, final_text, app_user_states, socketio)

            # ── WORD MODE: transformer-only path ─────────────────────────────
            # Sliding-window inference fires continuously inside _process_word_signing.
            # This branch only handles sentence-level finalisation after a long pause.
            elif forced == FORCED_MODE_WORD:
                if state['current_sentence'] and time_since_hand > SENTENCE_PAUSE_DURATION:
                    final_text = state['current_sentence'].strip()
                    print(f"[SignTranslation] [WORD] User {uid} sentence: '{final_text}'")
                    state['current_sentence'] = ""
                    self.translate_and_emit(room, uid, final_text, app_user_states, socketio)

            # ── AUTO MODE: both paths, PME decides ───────────────────────────
            else:
                # 1. Finalize a spelled word on pause
                if state['current_word'] and time_since_hand > WORD_PAUSE_DURATION:
                    spelled = state['current_word'].strip()
                    print(f"[SignTranslation] [AUTO/SPELL] User {uid} word: '{spelled}'")
                    state['current_sentence']    += spelled + " "
                    state['current_word']         = ""
                    state['last_accepted_letter'] = None
                    state['recent_predictions'].clear()

                # 2. (Sliding-window word inference fires per-frame; no pause-flush needed)

                # 3. Finalize & translate sentence
                if state['current_sentence'] and time_since_hand > SENTENCE_PAUSE_DURATION:
                    final_text = state['current_sentence'].strip()
                    print(f"[SignTranslation] [AUTO] User {uid} sentence: '{final_text}'")
                    state['current_sentence'] = ""
                    self.translate_and_emit(room, uid, final_text, app_user_states, socketio)

            # ── Cleanup stale users (all modes) ──────────────────────────────
            time_since_seen = current_time - state.get('last_seen_time', state['last_hand_time'])
            if time_since_seen > STALE_USER_TIMEOUT:
                keys_to_clear.append(key)

        for key in keys_to_clear:
            del self.user_states[key]

    # ── Translation + socket emit (unchanged from original) ──────────────────

    def translate_and_emit(self, room, uid, text, app_user_states, socketio):
        """Sends text through the translation pipeline and emits socket events."""
        from datetime import datetime

        # Resolve display name
        source_name = f"User {uid}"
        for sid, info in app_user_states.items():
            if info.get('room') == room and str(info.get('user_id')) == str(uid):
                source_name = f"{info.get('name')} (Sign)"
                break

        source_lang = 'en'

        for target_sid, target_info in app_user_states.items():
            if not (target_info.get('room') == room and
                    target_info.get('translation_enabled')):
                continue

            target_lang = target_info.get('language', 'en')
            pipeline_start = time.time()

            trans_result = self.translation_pipeline.translate_text(
                text, source_lang, target_lang
            )
            if not trans_result:
                continue
            translated_text = trans_result['text']

            tts_result = self.translation_pipeline.text_to_speech(
                translated_text, target_lang
            )
            if not tts_result:
                continue
            audio_base64 = tts_result['audio']

            total_latency_ms = int((time.time() - pipeline_start) * 1000)

            subtitle_data = {
                'text':          translated_text,
                'original_text': text,
                'source_lang':   source_lang,
                'target_lang':   target_lang,
                'sender_name':   source_name,
                'timestamp':     datetime.now().isoformat(),
            }

            socketio.emit('translated_audio', {
                'audio':      audio_base64,
                'latency_ms': total_latency_ms,
                'breakdown': {
                    'translation_ms': trans_result['latency_ms'],
                    'tts_ms':         tts_result['latency_ms'],
                    'total_ms':       total_latency_ms,
                },
                'timestamp': datetime.now().isoformat(),
                'sender':    source_name,
                'target_lang': target_lang,
            }, room=target_sid)

            socketio.emit('translated_subtitle', subtitle_data, room=target_sid)
