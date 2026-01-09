import speech_recognition as sr
from googletrans import Translator
from gtts import gTTS
import tempfile
import base64
import os
import time
from datetime import datetime

class TranslationPipeline:
    """Handles STT → Translation → TTS pipeline with latency tracking"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        self.translator = Translator()
        
        # Optimize recognizer
        self.recognizer.pause_threshold = 0.8
        self.recognizer.energy_threshold = 300
        
        # Track latency for video sync
        self.last_latency_ms = 0
    
    def speech_to_text(self, audio_data):
        """
        Convert speech to text using Google Speech Recognition
        
        Args:
            audio_data (sr.AudioData): Audio data
            
        Returns:
            dict: {'text': str, 'language': str, 'latency_ms': int} or None
        """
        start_time = time.time()
        try:
            # Use Google Speech Recognition with auto language detection
            text = self.recognizer.recognize_google(audio_data)
            
            # Detect language
            detected = self.translator.detect(text)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            print(f"[STT] Transcribed: '{text}' (detected: {detected.lang}) - {latency_ms}ms")
            
            return {
                'text': text,
                'language': detected.lang,
                'confidence': detected.confidence,
                'latency_ms': latency_ms
            }
            
        except sr.UnknownValueError:
            # Retry with adjusted energy threshold
            print("[STT] Could not understand audio, retrying with adjusted threshold...")
            original_threshold = self.recognizer.energy_threshold
            self.recognizer.energy_threshold = int(original_threshold * 0.6)  # Lower by 40%
            
            try:
                text = self.recognizer.recognize_google(audio_data)
                detected = self.translator.detect(text)
                latency_ms = int((time.time() - start_time) * 1000)
                
                print(f"[STT] ✓ Retry successful: '{text}' (detected: {detected.lang}) - {latency_ms}ms")
                
                return {
                    'text': text,
                    'language': detected.lang,
                    'confidence': detected.confidence,
                    'latency_ms': latency_ms
                }
            except (sr.UnknownValueError, sr.RequestError):
                print("[STT] ❌ Could not understand audio even after retry")
                return None
            finally:
                # Always restore original threshold
                self.recognizer.energy_threshold = original_threshold
        except sr.RequestError as e:
            print(f"[STT] API error: {e}")
            return None
        except Exception as e:
            print(f"[STT] Error: {e}")
            return None
    
    def translate_text(self, text, source_lang, target_lang):
        """
        Translate text from source to target language
        
        Args:
            text (str): Text to translate
            source_lang (str): Source language code
            target_lang (str): Target language code
            
        Returns:
            dict: {'text': str, 'latency_ms': int} or None
        """
        start_time = time.time()
        try:
            # Skip translation if already in target language
            if source_lang == target_lang:
                print(f"[Translation] Already in target language ({target_lang})")
                return {'text': text, 'latency_ms': 0}
            
            # Translate
            translation = self.translator.translate(text, src=source_lang, dest=target_lang)
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            print(f"[Translation] {source_lang}→{target_lang}: '{text}' → '{translation.text}' - {latency_ms}ms")
            
            return {'text': translation.text, 'latency_ms': latency_ms}
            
        except Exception as e:
            print(f"[Translation] Error: {e}")
            return None
    
    def text_to_speech(self, text, language):
        """
        Convert text to speech and return as base64 MP3
        
        Args:
            text (str): Text to convert
            language (str): Language code
            
        Returns:
            dict: {'audio': base64_str, 'latency_ms': int} or None
        """
        start_time = time.time()
        temp_file_path = None
        try:
            # Generate speech
            tts = gTTS(text, lang=language, slow=False)
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
            temp_file_path = temp_file.name
            temp_file.close()  # Close the file handle before gTTS writes to it
            
            tts.save(temp_file_path)
            
            # Small delay to ensure file is fully written
            time.sleep(0.1)
            
            # Read and encode as base64
            with open(temp_file_path, 'rb') as f:
                audio_bytes = f.read()
                base64_audio = base64.b64encode(audio_bytes).decode('utf-8')
            
            latency_ms = int((time.time() - start_time) * 1000)
            
            print(f"[TTS] Generated audio for '{text[:50]}...' in {language} - {latency_ms}ms")
            
            return {'audio': base64_audio, 'latency_ms': latency_ms}
            
        except Exception as e:
            print(f"[TTS] Error: {e}")
            return None
        finally:
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                except:
                    pass  # Ignore cleanup errors
    
    def process_audio(self, audio_data, target_lang, sender_name="Unknown"):
        """
        Complete pipeline: Audio → Transcribe → Translate → Synthesize
        Tracks total latency for video synchronization
        
        Args:
            audio_data (sr.AudioData): Audio data
            target_lang (str): Target language code
            sender_name (str): Name of the speaker
            
        Returns:
            dict: {
                'audio': base64_str,
                'subtitle': dict,
                'total_latency_ms': int,
                'breakdown': dict
            } or None
        """
        pipeline_start = time.time()
        
        try:
            # Step 1: Speech to Text
            stt_result = self.speech_to_text(audio_data)
            if not stt_result:
                return None
            
            original_text = stt_result['text']
            source_lang = stt_result['language']
            stt_latency = stt_result['latency_ms']
            
            # Step 2: Translate
            trans_result = self.translate_text(original_text, source_lang, target_lang)
            if not trans_result:
                return None
            
            translated_text = trans_result['text']
            trans_latency = trans_result['latency_ms']
            
            # Step 3: Text to Speech
            tts_result = self.text_to_speech(translated_text, target_lang)
            if not tts_result:
                return None
            
            audio_base64 = tts_result['audio']
            tts_latency = tts_result['latency_ms']
            
            # Calculate total latency
            total_latency_ms = int((time.time() - pipeline_start) * 1000)
            self.last_latency_ms = total_latency_ms
            
            # Step 4: Create subtitle data
            subtitle_data = {
                'text': translated_text,
                'original_text': original_text,
                'source_lang': source_lang,
                'target_lang': target_lang,
                'sender_name': sender_name,
                'timestamp': datetime.now().isoformat()
            }
            
            # Latency breakdown for debugging
            latency_breakdown = {
                'stt_ms': stt_latency,
                'translation_ms': trans_latency,
                'tts_ms': tts_latency,
                'total_ms': total_latency_ms
            }
            
            print(f"[Pipeline] Total latency: {total_latency_ms}ms (STT: {stt_latency}ms, Trans: {trans_latency}ms, TTS: {tts_latency}ms)")
            
            return {
                'audio': audio_base64,
                'subtitle': subtitle_data,
                'total_latency_ms': total_latency_ms,
                'breakdown': latency_breakdown
            }
            
        except Exception as e:
            print(f"[Pipeline] Error: {e}")
            return None
