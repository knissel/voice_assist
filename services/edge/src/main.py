
import asyncio
import base64
import collections
import contextlib
import io
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
from urllib.parse import urlparse, urlunparse
import websockets
from piper.voice import PiperVoice
from dotenv import load_dotenv

# Load environment variables FIRST
load_dotenv()

# Internal imports
from core.conversation import parse_clear_phrases
from core.event_bus import (
    EventBus, emit_state_changed, emit_transcript, 
    emit_assistant_text, emit_tool_call, emit_tool_result, emit_error
)
from core.assistant import Assistant
from tools.audio import pause_media, resume_media
from dashboard import update_state as update_dashboard_state
from tools.respeaker import RespeakerSettings, apply_settings as apply_respeaker_settings
from tools.respeaker_led import RespeakerLedConfig, RespeakerLedController

try:
    import openwakeword
    from openwakeword.model import Model as OpenWakeWordModel
except Exception:
    openwakeword = None
    OpenWakeWordModel = None

# === CONFIGURATION ===
WAKEWORD_MODELS = os.getenv("WAKEWORD_MODELS", "hey_jarvis").split(",")
WAKEWORD_THRESHOLD = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))
WAKEWORD_COOLDOWN_SECONDS = float(os.getenv("WAKEWORD_COOLDOWN_SECONDS", "1.5"))
WAKEWORD_SAMPLE_RATE = 16000
WAKEWORD_FRAME_LENGTH = 1280
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "5000"))
VAD_SILENCE_SECONDS = float(os.getenv("VAD_SILENCE_SECONDS", "0.8"))
VAD_GRACE_SECONDS = float(os.getenv("VAD_GRACE_SECONDS", "0.8"))
VAD_MAX_RECORD_SECONDS = float(os.getenv("VAD_MAX_RECORD_SECONDS", "12.0"))
VAD_MIN_RECORD_SECONDS = float(os.getenv("VAD_MIN_RECORD_SECONDS", "1.0")) # Minimum relevant audio

# === UTILS ===
def _resolve_wakeword_models(models: list[str], repo_root: str) -> list[str]:
    resolved = []
    pretrained_paths = None
    pretrained_map = None
    if openwakeword is not None:
        try:
            pretrained_paths = openwakeword.get_pretrained_model_paths()
            pretrained_map = openwakeword.models
        except Exception:
            pass
    for item in models:
        expanded = os.path.expanduser(item)
        if pretrained_map and expanded in pretrained_map:
            resolved.append(pretrained_map[expanded]["model_path"])
            continue
        if not os.path.isabs(expanded):
            candidate = os.path.join(repo_root, expanded)
            if os.path.exists(candidate):
                expanded = candidate
        if pretrained_paths and not os.path.isabs(expanded) and not expanded.endswith(".onnx"):
            matches = [p for p in pretrained_paths if os.path.basename(p).rsplit(".", 1)[0].startswith(expanded)]
            if matches:
                resolved.extend(matches)
                continue
        resolved.append(expanded)
    return resolved

def _ensure_openwakeword_models() -> None:
    if openwakeword is None:
        return
    resources_dir = os.path.join(os.path.dirname(openwakeword.__file__), "resources", "models")
    melspec_onnx = os.path.join(resources_dir, "melspectrogram.onnx")
    if os.path.exists(melspec_onnx):
        return
    try:
        from openwakeword.utils import download_models
        print("[WARN] OpenWakeWord model resources missing; downloading...")
        download_models(target_directory=resources_dir)
        print("[INFO] OpenWakeWord resources downloaded")
    except Exception as exc:
        print(f"[WARN] Failed to download OpenWakeWord resources: {exc}")

def _resolve_input_device_index(pa: pyaudio.PyAudio, preferred: str | None) -> int | None:
    if not preferred:
        return None
    preferred = str(preferred).strip()
    if preferred.isdigit():
        index = int(preferred)
        try:
            info = pa.get_device_info_by_index(index)
            print(f"[INFO] Using input device index {index}: {info.get('name')}")
            return index
        except Exception as exc:
            pass
            
    lowered = preferred.lower()
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        name = (info.get("name") or "").lower()
        if lowered in name and info.get("maxInputChannels", 0) > 0:
            print(f"[INFO] Using input device index {i}: {info.get('name')}")
            return i
    return None

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

        # Optional ReSpeaker LED/DSP
        self._setup_respeaker_led()
        self._setup_respeaker_dsp()
        
        # Wake Word
        self._setup_wakeword()
        
        # VAD
        self._setup_vad()
        
        # The Brain (Local Assistant)
        self.assistant = Assistant(self.bus)
        
        self.is_processing = False
        self._speech_end_time = None
        self._current_frames = None
        self._led_state = None
        
        # Wakeword state
        self._wakeword_debug = os.getenv("WAKEWORD_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
        self._wakeword_debug_last = 0.0
        buffer_frames = 5
        self._wakeword_audio_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=buffer_frames)
        self._suppress_wakeword_until = 0.0
        self._speaking = False


        # Event Bus Subscriptions
        self.bus.subscribe("state_changed", self._on_state_changed)
        self.bus.subscribe("transcript_final", self._on_transcript)
        self.bus.subscribe("assistant_text", self._on_assistant_text)
        
    def _on_state_changed(self, event):
        new_state = event.data["to_state"]
        self._set_status(new_state)

    def _on_transcript(self, event):
        update_dashboard_state("last_transcript", event.data["text"])

    def _on_assistant_text(self, event):
        update_dashboard_state("last_response", event.data["text"])

    def _set_status(self, status: str):
        update_dashboard_state("status", status)
        # Map specific states to LED/Status behavior
        if status == "thinking_local":
            status = "processing" # Use processing color for local thinking

        if self._led_controller:
            self._led_controller.set_state(status)
        self._led_state = status

    def _suppress_wakeword(self, seconds: float) -> None:
        if seconds <= 0: return
        self._suppress_wakeword_until = max(self._suppress_wakeword_until, time.time() + seconds)

    def _wakeword_suppressed(self) -> bool:
        return self._speaking or time.time() < self._suppress_wakeword_until

    def _setup_speaker(self):
        self.sample_rate = int(os.getenv("TTS_OUTPUT_SAMPLE_RATE", "24000"))
        self.out_stream = None
        try:
            self.out_stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16'
            )
            self.out_stream.start()
        except:
             # Fallback
            try:
                self.out_stream = sd.OutputStream(channels=1, dtype='int16')
                self.out_stream.start()
                self.sample_rate = self.out_stream.samplerate
            except Exception as e:
                print(f"[WARN] Failed to open audio output: {e}")

    def _setup_mic(self):
        self.pa = pyaudio.PyAudio()
        input_device = _resolve_input_device_index(self.pa, os.getenv("MIC_DEVICE_INDEX"))
        self.input_channels = int(os.getenv("MIC_CHANNELS", "1"))
        
        kwargs = {
            "rate": WAKEWORD_SAMPLE_RATE,
            "channels": self.input_channels,
            "format": pyaudio.paInt16,
            "input": True,
            "frames_per_buffer": WAKEWORD_FRAME_LENGTH,
        }
        if input_device is not None:
            kwargs["input_device_index"] = input_device
            
        try:
            self.in_stream = self.pa.open(**kwargs)
        except Exception:
             # Fallback mono
             kwargs["channels"] = 1
             self.input_channels = 1
             self.in_stream = self.pa.open(**kwargs)

    def _setup_respeaker_dsp(self):
        if os.getenv("RESPEAKER_DSP_ENABLED", "true") == "false": return
        # Basic settings apply
        apply_respeaker_settings(RespeakerSettings(aec_enabled=True, ns_enabled=True, agc_enabled=True))

    def _setup_respeaker_led(self):
        self._led_controller = None
        if os.getenv("RESPEAKER_LED_ENABLED", "true") == "false": return
        try:
            config = RespeakerLedConfig(brightness=int(os.getenv("RESPEAKER_LED_BRIGHTNESS", "10")))
            self._led_controller = RespeakerLedController(config=config)
        except Exception:
            pass

    def _setup_wakeword(self):
        if OpenWakeWordModel is None:
            raise RuntimeError("openwakeword not installed")
        _ensure_openwakeword_models()
        resolved = _resolve_wakeword_models(WAKEWORD_MODELS, self.repo_root)
        self.wakeword_detector = OpenWakeWordModel(wakeword_models=resolved, inference_framework="onnx")

    def _setup_vad(self):
        vad_path = os.path.join(self.repo_root, "models", "silero_vad.jit")
        if os.path.exists(vad_path):
            self.vad_model = torch.jit.load(vad_path)
            self.vad_model.eval()
            return
        # Download if needed
        model, _ = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', onnx=False)
        self.vad_model = model

    def play_audio(self, audio_bytes: bytes):
        if not self.out_stream or not audio_bytes: return
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        with wave.open(tmp_path, 'rb') as wf:
            data = wf.readframes(wf.getnframes())
            src_rate = wf.getframerate()
            audio_np = np.frombuffer(data, dtype=np.int16)
            
            # Resample if needed
            if src_rate != self.sample_rate:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(src_rate, self.sample_rate)
                audio_np = resample_poly(audio_np.astype(np.float32), self.sample_rate // g, src_rate // g)
                audio_np = np.clip(audio_np, -32768, 32767).astype(np.int16)
                
            self.out_stream.write(audio_np)
        os.remove(tmp_path)

    async def _process_locally(self, audio_data: bytes):
        """Process collected audio locally via Assistant class."""
        # Status is now handled via events from Assistant
        # self._set_status("processing") 
        
        # Save to temp WAV
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            with wave.open(tmp, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(WAKEWORD_SAMPLE_RATE)
                wf.writeframes(audio_data)
            audio_path = tmp.name

        try:
             # Run sync assistant logic in thread
            text, audio_bytes, lat = await asyncio.to_thread(
                self.assistant.process_voice_command, audio_path
            )
            
            update_dashboard_state("last_response", text)
            update_dashboard_state("end_to_final_ms", lat)
            
            if audio_bytes:
                self._speaking = True
                # self._set_status("speaking") # Event bus handles this
                await asyncio.to_thread(self.play_audio, audio_bytes)
                self._speaking = False
                self._suppress_wakeword(WAKEWORD_COOLDOWN_SECONDS)
                
                # Signal idle after speaking
                emit_state_changed(self.bus, "speaking", "idle")
                
        except Exception as e:
            print(f"[ERROR] Processing failed: {e}")
            emit_error(self.bus, "PROCESSING_ERROR", str(e))
        finally:
            if os.path.exists(audio_path):
                os.remove(audio_path)
            self.state = AudioRecorderState.LISTENING
            # self._set_status("listening") # Do not force; let events or loop handle
            resume_media()

    async def run(self):
        print("[LISTEN] Edge Assistant (Unified) Listening...")
        self.assistant.bus = self.bus # Ensure bus is linked
        emit_state_changed(self.bus, "initializing", "listening")
        
        pre_roll_buffer = collections.deque(maxlen=20) # ~1.5s
        recording_frames = []
        silence_counter = 0
        recording_max_frames = int(VAD_MAX_RECORD_SECONDS * WAKEWORD_SAMPLE_RATE / WAKEWORD_FRAME_LENGTH)
        silence_limit = int(VAD_SILENCE_SECONDS * WAKEWORD_SAMPLE_RATE / WAKEWORD_FRAME_LENGTH)
        
        # VAD helper
        def is_speech(pcm):
            # Normalize to -1..1
            float_pcm = torch.from_numpy(pcm.astype(np.float32) / 32768.0)
            return self.vad_model(float_pcm, WAKEWORD_SAMPLE_RATE).item() > 0.5

        while True:
            # Non-blocking audio read?
            # PyAudio read is blocking. We can run it in a thread or check self.in_stream.get_read_available()
            # But simple read is usually fine if buffer is small.
            if self.in_stream.get_read_available() < WAKEWORD_FRAME_LENGTH:
                await asyncio.sleep(0.01)
                continue
                
            pcm_bytes = self.in_stream.read(WAKEWORD_FRAME_LENGTH, exception_on_overflow=False)
            pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
            
            # Downmix if needed
            if self.input_channels > 1:
                pcm = pcm.reshape(-1, self.input_channels).mean(axis=1).astype(np.int16)
                pcm_bytes = pcm.tobytes()

            if self.state == AudioRecorderState.LISTENING:
                if self._wakeword_suppressed():
                    self._wakeword_audio_buffer.clear()
                    continue
                
                pre_roll_buffer.append(pcm_bytes)
                self._wakeword_audio_buffer.append(pcm)
                
                # Predict
                feed = pcm
                if len(self._wakeword_audio_buffer) >= 3:
                     feed = np.concatenate(list(self._wakeword_audio_buffer)[-3:])
                
                scores = self.wakeword_detector.predict(feed)
                found = scores and max(scores.values()) >= WAKEWORD_THRESHOLD
                
                if found and not self.is_processing:
                    print(f"[WAKE] Wake Word Detected!")
                    pause_media()
                    self.state = AudioRecorderState.RECORDING
                    self._set_status("recording")
                    # Prepend pre-roll
                    recording_frames = list(pre_roll_buffer)
                    silence_counter = 0

            elif self.state == AudioRecorderState.RECORDING:
                recording_frames.append(pcm_bytes)
                
                if is_speech(pcm):
                    silence_counter = 0
                else:
                    silence_counter += 1
                
                # Check end conditions
                is_silence_timeout = silence_counter > silence_limit
                is_max_len = len(recording_frames) > recording_max_frames
                
                if is_silence_timeout or is_max_len:
                    print(f"[RECORD] Finished recording ({len(recording_frames)} frames)")
                    self.state = AudioRecorderState.PROCESSING 
                    # Trigger processing
                    full_audio = b"".join(recording_frames)
                    asyncio.create_task(self._process_locally(full_audio))

            elif self.state == AudioRecorderState.PROCESSING:
                # Just drain audio while processing to keep buffer clean
                pass

            await asyncio.sleep(0.001)

if __name__ == "__main__":
    from dashboard import start_dashboard_thread
    start_dashboard_thread(port=DASHBOARD_PORT)
    
    assistant = EdgeAssistant()
    try:
        asyncio.run(assistant.run())
    except KeyboardInterrupt:
        print("\n[STOP] Stopping...")
