"""
Prototype Sign Language Speller
Converts alphabet predictions into spelled words and sentences in real time.
"""

import os
import sys
import time
import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Allow importing from existing modules in the repository
# Assuming this script is at `sign_alphabet_model/sign_language_speller.py`
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from preprocessing.normalize import normalize_sample
from inference.predictor import ASLPredictor

# --- CONFIGURABLE PARAMETERS ---
CONFIDENCE_THRESHOLD = 0.8
STABLE_FRAMES = 8
LETTER_COOLDOWN = 0.5
WORD_PAUSE_DURATION = 1.5

def initialize_camera(camera_id=0):
    """Initialize OpenCV webcam capture."""
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {camera_id}.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return cap

def detect_hand(frame, hands_model):
    """Detect hand using MediaPipe Hands."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands_model.process(rgb)
    if results.multi_hand_landmarks and len(results.multi_hand_landmarks) > 0:
        return results.multi_hand_landmarks[0]
    return None

def extract_landmarks(hand_landmarks):
    """Extract flattened x,y,z coordinates from hand landmarks."""
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return np.array(coords, dtype=np.float32)

def predict_letter(predictor, raw_landmarks):
    """Normalize landmarks and predict letter using ASLPredictor."""
    norm_landmarks = normalize_sample(raw_landmarks)
    label, confidence = predictor.predict(norm_landmarks)
    return label, confidence

def stabilize_prediction(recent_predictions, label, confidence):
    """Stabilize noisy predictions by enforcing multiple consecutive identical frames."""
    if confidence >= CONFIDENCE_THRESHOLD:
        recent_predictions.append(label)
        if len(recent_predictions) == STABLE_FRAMES and len(set(recent_predictions)) == 1:
            return recent_predictions[0]
    else:
        recent_predictions.clear()
    return None

def update_word_buffer(stable_label, current_time, last_accepted_letter, last_accept_time, current_word, recent_predictions):
    """Update word buffer with a new accepted letter if cooldown passes."""
    if stable_label and stable_label != last_accepted_letter:
        if (current_time - last_accept_time) > LETTER_COOLDOWN:
            current_word += stable_label
            print(f"Detected letter: {stable_label}")
            last_accepted_letter = stable_label
            last_accept_time = current_time
            recent_predictions.clear()
    return current_word, last_accepted_letter, last_accept_time

def detect_word_pause(current_time, last_hand_time):
    """Check if hand has been absent long enough to finalize a word."""
    return (current_time - last_hand_time) > WORD_PAUSE_DURATION

def update_sentence(current_word, current_sentence):
    """Append completed word to the sentence buffer."""
    if current_word:
        print(f"Word detected: {current_word}")
        current_sentence += (current_word + " ")
        print(f"Sentence: {current_sentence.strip()}")
        current_word = ""
    return current_word, current_sentence

def draw_ui(frame, current_letter, current_word, current_sentence):
    """Overlay textual information on the webcam frame."""
    h, w = frame.shape[:2]
    
    # Background rectangle for text readability
    cv2.rectangle(frame, (0, 0), (w, 150), (20, 20, 20), -1)
    
    # Formatted overlay text
    letter_display = current_letter if current_letter else "-"
    
    cv2.putText(frame, f"Letter: {letter_display}", (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    cv2.putText(frame, f"Word: {current_word}", (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
    cv2.putText(frame, f"Sentence: {current_sentence}", (10, 120), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    # Footer instructions
    cv2.putText(frame, "Press 'q' to quit", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)

def main():
    # 1. Initialize Webcam & MediaPipe
    cap = initialize_camera()
    
    mp_hands = mp.solutions.hands
    hands_model = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5,
    )
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    
    # 2. Load ASL Prediction Model
    print("Loading ASL predictor...")
    # Using existing inference tool
    predictor = ASLPredictor()
    print("Predictor loaded. Starting webcam feed...")
    
    # 3. Initialize State variables
    recent_predictions = deque(maxlen=STABLE_FRAMES)
    last_accepted_letter = None
    last_accept_time = 0.0
    last_hand_time = time.time()
    
    current_word = ""
    current_sentence = ""
    active_letter = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Optional mirror effect (better for signing)
        frame = cv2.flip(frame, 1) 
        
        # 4. Hand Detection
        hand_landmarks = detect_hand(frame, hands_model)
        
        current_time = time.time()
        active_letter = None
        
        if hand_landmarks:
            # Update last hand seen time
            last_hand_time = current_time
            
            # Draw MediaPipe skeleton
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )
            
            # 5. Model Prediction
            raw_landmarks = extract_landmarks(hand_landmarks)
            label, confidence = predict_letter(predictor, raw_landmarks)
            active_letter = label
            
            # 6. Smooth predictions & prevent duplicates
            stable_label = stabilize_prediction(recent_predictions, label, confidence)
            
            current_word, last_accepted_letter, last_accept_time = update_word_buffer(
                stable_label, current_time, last_accepted_letter, last_accept_time, current_word, recent_predictions
            )
        else:
            # Hand disappeared: clear stability buffer
            recent_predictions.clear()
            
            # Reset last accepted letter to allow signing identical letter again
            last_accepted_letter = None
            
            # 7. Finalize Word & Sentence when hand is absent
            if detect_word_pause(current_time, last_hand_time):
                current_word, current_sentence = update_sentence(current_word, current_sentence)
                last_hand_time = current_time # Reset timer to avoid continuous triggers
                
        # 8. Render UI Overlay
        draw_ui(frame, active_letter, current_word, current_sentence.strip())
        
        cv2.imshow("Sign Language Speller", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    # Cleanup camera / windows
    cap.release()
    cv2.destroyAllWindows()
    hands_model.close()

if __name__ == "__main__":
    main()
