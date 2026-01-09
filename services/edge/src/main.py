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
from tools.respeaker import RespeakerSettings, apply_settings as apply_respeaker_settings
from tools.respeaker_led import RespeakerLedConfig, RespeakerLedController

try:
    import openwakeword
    from openwakeword.model import Model as OpenWakeWordModel
except Exception:
    openwakeword = None
    OpenWakeWordModel = None

# === CONFIGURATION ===
COMPUTE_SERVER_URL = os.getenv("COMPUTE_SERVER_URL", "http://localhost:8000")
def _compute_ws_url(http_url: str) -> str:
    parsed = urlparse(http_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    netloc = parsed.netloc or parsed.path
    return urlunparse((scheme, netloc, "/ws/audio", "", "", ""))

COMPUTE_WS_URL = os.getenv("COMPUTE_WS_URL", _compute_ws_url(COMPUTE_SERVER_URL))
WAKEWORD_MODELS = os.getenv("WAKEWORD_MODELS", "hey_jarvis").split(",")
WAKEWORD_THRESHOLD = float(os.getenv("WAKEWORD_THRESHOLD", "0.5"))
WAKEWORD_SAMPLE_RATE = 16000
WAKEWORD_FRAME_LENGTH = 1280
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "5000"))
STREAM_CHUNK_MS = int(os.getenv("STREAM_CHUNK_MS", "40"))
STREAM_QUEUE_MAX = int(os.getenv("STREAM_QUEUE_MAX", "200"))
VAD_PRE_ROLL_MS = int(os.getenv("VAD_PRE_ROLL_MS", "200"))
STREAM_CHUNK_SAMPLES = max(1, int(WAKEWORD_SAMPLE_RATE * STREAM_CHUNK_MS / 1000))
STREAM_CHUNK_BYTES = STREAM_CHUNK_SAMPLES * 2
STREAM_PRE_ROLL_CHUNKS = max(1, int(VAD_PRE_ROLL_MS / STREAM_CHUNK_MS))

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

def _ensure_openwakeword_models() -> None:
    """Download OpenWakeWord resources if they are missing."""
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
    """Resolve an input device index from an override (index or name substring)."""
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
            print(f"[WARN] Failed to read device index {preferred!r}: {exc}")
            return None

    lowered = preferred.lower()
    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        name = (info.get("name") or "").lower()
        if lowered in name and info.get("maxInputChannels", 0) > 0:
            print(f"[INFO] Using input device index {i}: {info.get('name')}")
            return i

    print(f"[WARN] No input device matched {preferred!r}; using default")
    return None

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

        # Optional ReSpeaker DSP control
        self._setup_respeaker_dsp()

        # Optional ReSpeaker LED ring
        self._setup_respeaker_led()
        
        # Wake Word
        self._setup_wakeword()
        
        # VAD
        self._setup_vad()
        
        self.is_processing = False
        self._stream_queue = None
        self._stream_task = None
        self._stream_session_id = None
        self._speech_end_time = None
        self._processing_done = False
        self._current_frames = None
        self._led_state = None

    def _set_status(self, status: str):
        update_dashboard_state("status", status)
        if self._led_controller:
            self._led_controller.set_state(status)
        self._led_state = status

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
        input_device = _resolve_input_device_index(self.pa, os.getenv("MIC_DEVICE_INDEX"))
        kwargs = {
            "rate": WAKEWORD_SAMPLE_RATE,
            "channels": 1,
            "format": pyaudio.paInt16,
            "input": True,
            "frames_per_buffer": WAKEWORD_FRAME_LENGTH,
        }
        if input_device is not None:
            kwargs["input_device_index"] = input_device
        self.in_stream = self.pa.open(
            **kwargs
        )

    def _setup_respeaker_dsp(self):
        def _env_truthy(value: str | None, default: bool) -> bool:
            if value is None:
                return default
            return value.strip().lower() in {"1", "true", "yes", "on"}

        if os.getenv("RESPEAKER_DSP_ENABLED", "true").strip().lower() in {"0", "false", "no", "off"}:
            print("[INFO] ReSpeaker DSP control disabled via RESPEAKER_DSP_ENABLED")
            return

        settings = RespeakerSettings(
            aec_enabled=_env_truthy(os.getenv("RESPEAKER_AEC"), True),
            ns_enabled=_env_truthy(os.getenv("RESPEAKER_NS"), True),
            agc_enabled=_env_truthy(os.getenv("RESPEAKER_AGC"), True),
        )

        def _parse_usb_id(value: str | None, default: int) -> int:
            if not value:
                return default
            try:
                return int(value, 0)
            except ValueError:
                print(f"[WARN] Invalid USB id {value!r}; using default {hex(default)}")
                return default

        vid = _parse_usb_id(os.getenv("RESPEAKER_USB_VID"), 0x2886)
        pid = _parse_usb_id(os.getenv("RESPEAKER_USB_PID"), 0x0018)

        try:
            applied = apply_respeaker_settings(settings, vid=vid, pid=pid)
        except Exception as exc:
            print(f"[WARN] Failed to configure ReSpeaker DSP: {exc}")
            return

        if applied:
            print(f"[INFO] ReSpeaker DSP configured (AEC={settings.aec_enabled}, NS={settings.ns_enabled}, AGC={settings.agc_enabled})")
        else:
            print("[INFO] ReSpeaker DSP device not found; skipping DSP configuration")

    def _setup_respeaker_led(self):
        def _env_truthy(value: str | None, default: bool) -> bool:
            if value is None:
                return default
            return value.strip().lower() in {"1", "true", "yes", "on"}

        self._led_controller = None
        if not _env_truthy(os.getenv("RESPEAKER_LED_ENABLED"), True):
            print("[INFO] ReSpeaker LED control disabled via RESPEAKER_LED_ENABLED")
            return

        def _parse_usb_id(value: str | None, default: int) -> int:
            if not value:
                return default
            try:
                return int(value, 0)
            except ValueError:
                print(f"[WARN] Invalid USB id {value!r}; using default {hex(default)}")
                return default

        def _parse_brightness(value: str | None) -> int | None:
            if value is None or value.strip() == "":
                return None
            try:
                brightness = int(value)
            except ValueError:
                print(f"[WARN] Invalid RESPEAKER_LED_BRIGHTNESS {value!r}; ignoring")
                return None
            return max(0, min(255, brightness))

        vid = _parse_usb_id(os.getenv("RESPEAKER_USB_VID"), 0x2886)
        pid = _parse_usb_id(os.getenv("RESPEAKER_USB_PID"), 0x0018)
        config = RespeakerLedConfig(brightness=_parse_brightness(os.getenv("RESPEAKER_LED_BRIGHTNESS")))
        try:
            controller = RespeakerLedController(vid=vid, pid=pid, config=config)
        except Exception as exc:
            print(f"[WARN] Failed to initialize ReSpeaker LED ring: {exc}")
            return

        if controller.available:
            self._led_controller = controller
            print("[INFO] ReSpeaker LED ring connected")
        else:
            print("[INFO] ReSpeaker LED ring not found or unavailable; skipping LED control")

    def _setup_wakeword(self):
        if OpenWakeWordModel is None:
            raise RuntimeError("openwakeword not installed")

        _ensure_openwakeword_models()
        
        resolved_models = _resolve_wakeword_models(WAKEWORD_MODELS, self.repo_root)
        print(f"[INFO] Loading wakeword models: {resolved_models}")

        # openwakeword API varies across versions (wakeword_models vs wakeword_model_paths).
        import inspect
        params = inspect.signature(OpenWakeWordModel).parameters
        kwargs = {}
        if "wakeword_model_paths" in params:
            kwargs["wakeword_model_paths"] = resolved_models
        else:
            kwargs["wakeword_models"] = resolved_models

        # Only pass inference_framework if supported; older versions forward kwargs to AudioFeatures.
        if "inference_framework" in params:
            kwargs["inference_framework"] = "onnx"

        self.wakeword_detector = OpenWakeWordModel(**kwargs)

    def _setup_vad(self):
        # We still need VAD on edge to know when to stop recording.
        # Prefer the bundled JIT model to avoid torch.hub downloads.
        vad_path = os.path.join(self.repo_root, "models", "silero_vad.jit")
        if os.path.exists(vad_path):
            self.vad_model = torch.jit.load(vad_path)
            self.vad_model.eval()
            self.vad_available = True
            return

        # Fallback to torch.hub if the local model is missing.
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

    def _finish_processing(self):
        if self._processing_done:
            return
        self._processing_done = True
        self.is_processing = False
        self._stream_queue = None
        self._stream_task = None
        self._stream_session_id = None
        self._speech_end_time = None
        self._current_frames = None
        self._set_status("listening")
        print("[LISTEN] Ready for next wake word...")
        resume_media()

    async def _enqueue_audio_chunk(self, chunk: bytes) -> None:
        if not self._stream_queue:
            return
        await self._stream_queue.put(chunk)

    async def _receive_compute(self, websocket, done_event: asyncio.Event):
        """Receive and process messages from compute server."""
        audio_chunks = []  # Buffer for streaming audio
        sample_rate = 24000  # Default XTTS sample rate
        is_streaming_audio = False
        
        try:
            async for message in websocket:
                # Handle binary audio chunks
                if isinstance(message, bytes):
                    if is_streaming_audio:
                        audio_chunks.append(message)
                        # Play chunk immediately for low latency
                        try:
                            audio_np = np.frombuffer(message, dtype=np.int16)
                            # Resample from 24kHz to output sample rate if needed
                            if self.sample_rate != sample_rate:
                                from scipy.signal import resample_poly
                                from math import gcd
                                g = gcd(sample_rate, self.sample_rate)
                                audio_np = resample_poly(audio_np.astype(np.float32), 
                                                        self.sample_rate // g, 
                                                        sample_rate // g)
                                audio_np = np.clip(audio_np, -32768, 32767).astype(np.int16)
                            self.out_stream.write(audio_np)
                        except Exception as e:
                            print(f"[WARN] Audio chunk playback error: {e}")
                    continue
                
                # Handle JSON messages
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    continue

                msg_type = data.get("type")
                if msg_type == "partial_transcript":
                    text = data.get("text", "")
                    if text:
                        emit_transcript(self.bus, text, is_final=False)
                        update_dashboard_state("last_transcript", text)
                elif msg_type == "final_transcript":
                    text = data.get("text", "")
                    emit_transcript(self.bus, text, is_final=True)
                    update_dashboard_state("last_transcript", text or "No speech detected")
                    if self._speech_end_time:
                        end_to_final_ms = int((time.time() - self._speech_end_time) * 1000)
                        update_dashboard_state("end_to_final_ms", end_to_final_ms)
                        print(f"[LATENCY] End-to-final transcript: {end_to_final_ms}ms")
                    if not text:
                        done_event.set()
                elif msg_type == "assistant_response":
                    response_text = data.get("response_text", "")
                    emit_assistant_text(self.bus, response_text, is_partial=False)
                    update_dashboard_state("last_response", response_text)
                    
                    # Check if audio is streaming or base64 encoded
                    if data.get("audio_streaming"):
                        is_streaming_audio = True
                        print(f"[TTS] Streaming audio...")
                        self._set_status("speaking")
                    elif data.get("audio_base64"):
                        # Fallback to base64 audio (non-streaming)
                        self._set_status("speaking")
                        audio_bytes = base64.b64decode(data["audio_base64"])
                        self.play_audio(audio_bytes)
                        done_event.set()
                    elif not data.get("audio_streaming"):
                        # No audio at all
                        done_event.set()
                elif msg_type == "audio_stream_end":
                    chunks_sent = data.get("chunks_sent", 0)
                    tts_latency = data.get("tts_latency_ms", 0)
                    print(f"[TTS] Stream complete: {chunks_sent} chunks, {tts_latency}ms")
                    done_event.set()
                elif msg_type == "error":
                    print(f"[ERROR] Compute WS error: {data.get('error')}")
                    done_event.set()
        except Exception as exc:
            print(f"[ERROR] Compute WS receive failed: {exc}")
            done_event.set()

    async def _stream_to_compute(self):
        session_id = self._stream_session_id or str(uuid.uuid4())[:8]
        self._stream_session_id = session_id
        done_event = asyncio.Event()

        print(f"[WS] Connecting to {COMPUTE_WS_URL}...")
        try:
            async with websockets.connect(COMPUTE_WS_URL, max_size=None, ping_interval=20) as websocket:
                print(f"[WS] Connected! Sending start message...")
                await websocket.send(json.dumps({
                    "type": "start",
                    "session_id": session_id,
                    "sample_rate": WAKEWORD_SAMPLE_RATE,
                    "sample_width": 2,
                    "channels": 1,
                    "chunk_ms": STREAM_CHUNK_MS,
                    "audio_format": "pcm_s16le"
                }))

                receiver_task = asyncio.create_task(self._receive_compute(websocket, done_event))

                chunks_sent = 0
                while True:
                    chunk = await self._stream_queue.get()
                    if chunk is None:
                        print(f"[WS] Queue signaled end. Sent {chunks_sent} chunks.")
                        break
                    await websocket.send(chunk)
                    chunks_sent += 1

                print(f"[WS] Sending stop message...")
                stop_payload = {
                    "type": "stop",
                    "session_id": session_id,
                    "speech_end_ts": self._speech_end_time
                }
                await websocket.send(json.dumps(stop_payload))

                print(f"[WS] Waiting for response...")
                try:
                    await asyncio.wait_for(done_event.wait(), timeout=40)
                    print(f"[WS] Response received!")
                except asyncio.TimeoutError:
                    print("[WARN] Timed out waiting for compute response")
                finally:
                    receiver_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await receiver_task

        except Exception as exc:
            print(f"[ERROR] WebSocket stream failed, falling back to HTTP: {exc}")
            # Use self._current_frames which has the actual recorded audio
            if self._current_frames:
                print(f"[HTTP-FALLBACK] Sending {len(self._current_frames)} frames via HTTP...")
                await asyncio.to_thread(self._process_on_compute, list(self._current_frames))
            else:
                print("[ERROR] No frames available for HTTP fallback")
        finally:
            self._finish_processing()

    async def run(self):
        print("[LISTEN] Edge Assistant Listening...")
        self._set_status("listening")
        pre_roll_buffer = collections.deque(maxlen=20) # ~1.5s
        stream_pre_roll = collections.deque(maxlen=STREAM_PRE_ROLL_CHUNKS)
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
                        self._set_status("recording")
                        
                        # Clear pre-roll buffer - don't include wake word audio
                        pre_roll_buffer.clear()
                        
                        # Small delay to let wake word audio finish (prevents "Oogway" in transcript)
                        await asyncio.sleep(0.3)
                        
                        self.state = AudioRecorderState.RECORDING
                        current_command_frames = []  # Start fresh, no pre-roll
                        stream_pre_roll = collections.deque(maxlen=STREAM_PRE_ROLL_CHUNKS)
                        speech_detected = False
                        silence_counter = 0
                        self._processing_done = False
                        self._speech_end_time = None
                        self._stream_queue = asyncio.Queue(maxsize=STREAM_QUEUE_MAX)
                        # Store reference to frames for potential HTTP fallback
                        self._current_frames = current_command_frames
                        self._stream_task = asyncio.create_task(
                            self._stream_to_compute()
                        )

                elif self.state == AudioRecorderState.RECORDING:
                    current_command_frames.append(pcm_bytes)
                    
                    # VAD Check - Silero VAD needs exactly 512 samples at 16kHz
                    audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
                    # Take only the first 512 samples for VAD (Silero requirement)
                    vad_chunk = audio_int16[:512] if len(audio_int16) >= 512 else audio_int16
                    vad_tensor = torch.from_numpy(vad_chunk.astype(np.float32) * VAD_NORMALIZE)
                    prob = self.vad_model(vad_tensor, WAKEWORD_SAMPLE_RATE).item()
                    
                    if prob >= 0.5:
                        if not speech_detected:
                            speech_detected = True
                            print(f"[VAD] Speech detected")
                        silence_counter = 0
                    else:
                        if speech_detected:
                            silence_counter += 1

                    # Always send audio chunks during recording (don't gate on speech_detected)
                    # The server's VAD will also filter, but we want to capture everything
                    for offset in range(0, len(pcm_bytes), STREAM_CHUNK_BYTES):
                        chunk = pcm_bytes[offset:offset + STREAM_CHUNK_BYTES]
                        await self._enqueue_audio_chunk(chunk)
                    
                    # End recording conditions - require minimum frames (~1 sec) before stopping
                    min_frames = 12  # ~1 second of audio at 80ms per frame
                    has_enough_audio = len(current_command_frames) >= min_frames
                    if (has_enough_audio and speech_detected and silence_counter > 10) or len(current_command_frames) > 150:
                        print("[STOP] Recording finished")
                        self._set_status("processing")
                        self.state = AudioRecorderState.PROCESSING
                        self.is_processing = True
                        self._speech_end_time = time.time()
                        
                        # Signal end of audio stream
                        if self._stream_queue:
                            await self._stream_queue.put(None)
                        
                        # Wait for stream task to complete before going back to listening
                        if self._stream_task:
                            try:
                                await asyncio.wait_for(self._stream_task, timeout=45)
                            except asyncio.TimeoutError:
                                print("[WARN] Stream task timed out")
                            except Exception as e:
                                print(f"[ERROR] Stream task error: {e}")
                        
                        # Now safe to reset state
                        self.state = AudioRecorderState.LISTENING
                        current_command_frames = []
                        pre_roll_buffer.clear()
                        self.wakeword_detector.reset()
                        self._current_frames = None

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
                emit_transcript(self.bus, transcript, is_final=True)
                emit_assistant_text(self.bus, response_text, is_partial=False)
                if self._speech_end_time:
                    end_to_final_ms = int((time.time() - self._speech_end_time) * 1000)
                    update_dashboard_state("end_to_final_ms", end_to_final_ms)
                    print(f"[LATENCY] End-to-final transcript: {end_to_final_ms}ms")
                
                # 3. Handle Audio Playback
                audio_b64 = data.get('audio_base64')
                if audio_b64:
                    self._set_status("speaking")
                    audio_bytes = base64.b64decode(audio_b64)
                    self.play_audio(audio_bytes)
                
            else:
                print(f"[ERROR] Compute Server Error: {response.text}")
                
        except Exception as e:
            print(f"[ERROR] Failed to process on compute: {e}")
        finally:
            self._finish_processing()

if __name__ == "__main__":
    # Start the dashboard
    start_dashboard_thread(port=DASHBOARD_PORT)
    
    assistant = EdgeAssistant()
    asyncio.run(assistant.run())
