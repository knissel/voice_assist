import asyncio
import collections
import json
import pyaudio
import wave
import subprocess
import os
import shutil
import threading
import queue
import time
import uuid
import requests
import numpy as np
import sounddevice as sd
import torch
import tempfile
from piper.voice import PiperVoice
from dotenv import load_dotenv

# Load environment variables FIRST before any internal imports that read them
load_dotenv()

# Internal imports (relative to services/edge/src/)
from core.conversation import parse_clear_phrases
from core.event_bus import (
    EventBus, emit_state_changed, emit_transcript, 
    emit_assistant_text, emit_tool_call, emit_tool_result, emit_error
)
from tools.audio import pause_media, resume_media
from dashboard import start_dashboard_thread, update_state as update_dashboard_state

try:
    from openwakeword.model import Model as OpenWakeWordModel
except Exception:
    OpenWakeWordModel = None

# === CONFIGURATION ===
COMPUTE_SERVER_URL = os.getenv("COMPUTE_SERVER_URL", "http://localhost:8000")
WAKEWORD_MODELS = os.getenv("WAKEWORD_MODELS", "hey_jarvis").split(",")
WAKEWORD_THRESHOLD = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))
WAKEWORD_SAMPLE_RATE = 16000
WAKEWORD_FRAME_LENGTH = 1280

# === UTILS ===
def _resolve_wakeword_models(models: list[str], repo_root: str) -> list[str]:
    resolved = []
    for item in models:
        expanded = os.path.expanduser(item)
        if not os.path.isabs(expanded):
            candidate = os.path.join(repo_root, expanded)
            if os.path.exists(candidate):
                expanded = candidate
        resolved.append(expanded)
    return resolved

# === CLASSES ===
class AudioRecorderState:
    LISTENING = 0
    RECORDING = 1
    PROCESSING = 2

class EdgeAssistant:
    def __init__(self):
        self.state = AudioRecorderState.LISTENING
        self.bus = EventBus()
        self.bus.start()
        
        self.repo_root = os.path.dirname(os.path.abspath(__file__))
        
        # Audio Output (Speaker)
        self._setup_speaker()
        
        # Audio Input (Mic)
        self._setup_mic()
        
        # Wake Word
        self._setup_wakeword()
        
        # VAD
        self._setup_vad()
        
        self.is_processing = False

    def _setup_speaker(self):
        # We'll use sounddevice for persistent output
        self.sample_rate = 22050 # Standard for Piper, will resample if needed
        self.out_stream = sd.OutputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype='int16'
        )
        self.out_stream.start()

    def _setup_mic(self):
        self.pa = pyaudio.PyAudio()
        self.in_stream = self.pa.open(
            rate=WAKEWORD_SAMPLE_RATE,
            channels=1,
            format=pyaudio.paInt16,
            input=True,
            frames_per_buffer=WAKEWORD_FRAME_LENGTH
        )

    def _setup_wakeword(self):
        if OpenWakeWordModel is None:
            raise RuntimeError("openwakeword not installed")
        
        resolved_models = _resolve_wakeword_models(WAKEWORD_MODELS, self.repo_root)
        print(f"[INFO] Loading wakeword models: {resolved_models}")
        
        self.wakeword_detector = OpenWakeWordModel(
            wakeword_models=resolved_models,
            inference_framework="onnx"
        )

    def _setup_vad(self):
        # We still need VAD on edge to know when to stop recording
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', 
            model='silero_vad', 
            force_reload=False, 
            onnx=False
        )
        self.vad_model = model
        self.vad_available = True

    def play_audio(self, audio_bytes: bytes):
        """Play WAV bytes received from compute server."""
        # For simplicity, save to temp and play or use sounddevice
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        # Read WAV
        with wave.open(tmp_path, 'rb') as wf:
            data = wf.readframes(wf.getnframes())
            # Assume 16-bit PCM for now
            audio_np = np.frombuffer(data, dtype=np.int16)
            self.out_stream.write(audio_np)
        
        os.remove(tmp_path)

    async def run(self):
        print("[LISTEN] Edge Assistant Listening...")
        update_dashboard_state("status", "listening")
        pre_roll_buffer = collections.deque(maxlen=20) # ~1.5s
        current_command_frames = []
        speech_detected = False
        silence_counter = 0
        
        VAD_NORMALIZE = 1.0 / 32768.0

        try:
            while True:
                pcm_bytes = self.in_stream.read(WAKEWORD_FRAME_LENGTH, exception_on_overflow=False)
                
                if self.state == AudioRecorderState.LISTENING:
                    pre_roll_buffer.append(pcm_bytes)
                    pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
                    scores = self.wakeword_detector.predict(pcm)
                    
                    found = False
                    for name, score in scores.items():
                        if score >= WAKEWORD_THRESHOLD:
                            print(f"[WAKE] Wake Word: {name}")
                            found = True
                            break
                    
                    if found and not self.is_processing:
                        pause_media()
                        emit_state_changed(self.bus, "idle", "listening")
                        update_dashboard_state("status", "recording")
                        
                        # Clear pre-roll buffer - don't include wake word audio
                        pre_roll_buffer.clear()
                        
                        # Small delay to let wake word audio finish (prevents "Oogway" in transcript)
                        await asyncio.sleep(0.3)
                        
                        self.state = AudioRecorderState.RECORDING
                        current_command_frames = []  # Start fresh, no pre-roll
                        speech_detected = False
                        silence_counter = 0

                elif self.state == AudioRecorderState.RECORDING:
                    current_command_frames.append(pcm_bytes)
                    
                    # VAD Check - Silero VAD needs exactly 512 samples at 16kHz
                    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
                    # Take only the first 512 samples for VAD (Silero requirement)
                    vad_chunk = audio_int16[:512] if len(audio_int16) >= 512 else audio_int16
                    vad_tensor = torch.from_numpy(vad_chunk.astype(np.float32) * VAD_NORMALIZE)
                    prob = self.vad_model(vad_tensor, WAKEWORD_SAMPLE_RATE).item()
                    
                    if prob >= 0.5:
                        speech_detected = True
                        silence_counter = 0
                    else:
                        if speech_detected:
                            silence_counter += 1
                    
                    # End recording conditions
                    if (speech_detected and silence_counter > 10) or len(current_command_frames) > 150:
                        print("[STOP] Recording finished")
                        update_dashboard_state("status", "processing")
                        self.state = AudioRecorderState.PROCESSING
                        self.is_processing = True
                        
                        # Process on Compute Node
                        threading.Thread(target=self._process_on_compute, args=(list(current_command_frames),)).start()
                        
                        self.state = AudioRecorderState.LISTENING
                        current_command_frames = []
                        pre_roll_buffer.clear()
                        self.wakeword_detector.reset()

                await asyncio.sleep(0.01)

        except Exception as e:
            print(f"[ERROR] Error in loop: {e}")
        finally:
            self.in_stream.stop_stream()
            self.in_stream.close()
            self.pa.terminate()

    def _process_on_compute(self, frames):
        try:
            # 1. Prepare WAV
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(WAKEWORD_SAMPLE_RATE)
                wf.writeframes(b''.join(frames))
            buffer.seek(0)
            
            # 2. Send to Compute
            print("[SEND] Sending to Compute...")
            files = {'audio': ('command.wav', buffer, 'audio/wav')}
            response = requests.post(f"{COMPUTE_SERVER_URL}/process", files=files, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                transcript = data.get('transcript', '')
                response_text = data.get('response_text', '')
                print(f"[RESPONSE] Response: {response_text}")
                
                # Update dashboard
                update_dashboard_state("last_transcript", transcript)
                update_dashboard_state("last_response", response_text)
                
                # 3. Handle Audio Playback
                audio_b64 = data.get('audio_base64')
                if audio_b64:
                    import base64
                    audio_bytes = base64.b64decode(audio_b64)
                    self.play_audio(audio_bytes)
                
            else:
                print(f"[ERROR] Compute Server Error: {response.text}")
                
        except Exception as e:
            print(f"[ERROR] Failed to process on compute: {e}")
        finally:
            self.is_processing = False
            update_dashboard_state("status", "listening")
            print("[LISTEN] Ready for next wake word...")
            resume_media()

if __name__ == "__main__":
    import io
    # Start the dashboard
    start_dashboard_thread(port=5000)
    
    assistant = EdgeAssistant()
    asyncio.run(assistant.run())
