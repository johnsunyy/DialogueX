import base64
import io
import os
import tempfile
from pydub import AudioSegment
import speech_recognition as sr

class AudioHandler:
    """Handles audio conversion and processing"""
    
    def __init__(self):
        self.recognizer = sr.Recognizer()
        # Optimize recognizer settings (standard values with audio preprocessing)
        self.recognizer.pause_threshold = 0.8
        self.recognizer.energy_threshold = 300  # Standard threshold (audio is now preprocessed)
        self.recognizer.dynamic_energy_threshold = True  # Auto-adjust for variations
    
    def base64_to_audio_file(self, base64_audio):
        """
        Convert base64 encoded audio to WAV file for speech recognition
        Auto-detects format (WebM, WAV, MP3, etc.)
        
        Args:
            base64_audio (str): Base64 encoded audio data
            
        Returns:
            str: Path to temporary WAV file
        """
        temp_input = None
        try:
            # Decode base64
            audio_bytes = base64.b64decode(base64_audio)
            
            # Save to temporary file for format detection
            temp_input = tempfile.NamedTemporaryFile(delete=False, suffix='.tmp')
            temp_input.write(audio_bytes)
            temp_input.close()
            
            print(f"[AudioHandler] Received {len(audio_bytes)} bytes")
            
            # Try to load audio with explicit codec specifications
            try:
                # Try WebM with Opus codec first (most common from browsers)
                audio = AudioSegment.from_file(
                    temp_input.name, 
                    format="webm",
                    codec="opus"
                )
                print(f"[AudioHandler] Loaded as WebM/Opus successfully")
            except Exception as e1:
                print(f"[AudioHandler] WebM/Opus failed: {e1}")
                try:
                    # Try without codec specification (auto-detect)
                    audio = AudioSegment.from_file(temp_input.name)
                    print(f"[AudioHandler] Auto-detected format successfully")
                except Exception as e2:
                    print(f"[AudioHandler] Auto-detection failed: {e2}")
                    try:
                        # Try WebM without codec
                        audio = AudioSegment.from_file(temp_input.name, format="webm")
                        print(f"[AudioHandler] Loaded as WebM")
                    except Exception as e3:
                        print(f"[AudioHandler] WebM failed: {e3}")
                        try:
                            # Last resort: try WAV
                            audio = AudioSegment.from_file(temp_input.name, format="wav")
                            print(f"[AudioHandler] Loaded as WAV")
                        except Exception as e4:
                            error_msg = f"Failed all format attempts. WebM/Opus: {e1}, Auto: {e2}, WebM: {e3}, WAV: {e4}"
                            print(f"[AudioHandler] ERROR: {error_msg}")
                            raise Exception(error_msg)
            
            # Convert to WAV format (16kHz, mono, 16-bit) for optimal speech recognition
            audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
            
            # Normalize volume to improve speech recognition
            print(f"[AudioHandler] Audio level before normalization: {audio.dBFS:.2f} dBFS")
            audio = audio.normalize()
            
            # Ensure minimum volume level (boost if too quiet)
            if audio.dBFS < -30:
                gain_needed = -30 - audio.dBFS
                audio = audio.apply_gain(gain_needed)
                print(f"[AudioHandler] Boosted quiet audio by {gain_needed:.2f} dB")
            
            print(f"[AudioHandler] Audio level after normalization: {audio.dBFS:.2f} dBFS")
            
            # Create temporary WAV file
            wav_temp = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            audio.export(wav_temp.name, format="wav")
            
            print(f"[AudioHandler] Converted to WAV: {wav_temp.name}")
            
            return wav_temp.name
            
        except Exception as e:
            print(f"[AudioHandler] Error converting audio: {e}")
            raise
        finally:
            # Clean up temp input file
            if temp_input and os.path.exists(temp_input.name):
                try:
                    os.remove(temp_input.name)
                except:
                    pass
    
    def audio_file_to_speech_recognition_data(self, wav_file_path):
        """
        Convert WAV file to AudioData for speech recognition
        
        Args:
            wav_file_path (str): Path to WAV file
            
        Returns:
            sr.AudioData: Audio data for recognition
        """
        try:
            with sr.AudioFile(wav_file_path) as source:
                audio_data = self.recognizer.record(source)
            return audio_data
        except Exception as e:
            print(f"[AudioHandler] Error reading audio file: {e}")
            raise
        finally:
            # Clean up temp file
            if os.path.exists(wav_file_path):
                try:
                    os.remove(wav_file_path)
                except:
                    pass
    
    def process_audio_chunk(self, base64_audio):
        """
        Complete pipeline: base64 → AudioData
        
        Args:
            base64_audio (str): Base64 encoded audio
            
        Returns:
            sr.AudioData: Audio data ready for recognition
        """
        wav_path = self.base64_to_audio_file(base64_audio)
        audio_data = self.audio_file_to_speech_recognition_data(wav_path)
        return audio_data
