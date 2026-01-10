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

try:
    import alsaaudio
except Exception:
    alsaaudio = None

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
WAKEWORD_COOLDOWN_SECONDS = float(os.getenv("WAKEWORD_COOLDOWN_SECONDS", "1.5"))
WAKEWORD_SAMPLE_RATE = 16000
WAKEWORD_FRAME_LENGTH = 1280
DASHBOARD_PORT = int(os.getenv("DASHBOARD_PORT", "5000"))
STREAM_CHUNK_MS = int(os.getenv("STREAM_CHUNK_MS", "40"))
STREAM_QUEUE_MAX = int(os.getenv("STREAM_QUEUE_MAX", "200"))
VAD_PRE_ROLL_MS = int(os.getenv("VAD_PRE_ROLL_MS", "200"))
VAD_SPEECH_THRESHOLD = float(os.getenv("VAD_SPEECH_THRESHOLD", "0.5"))
VAD_SILENCE_SECONDS = float(os.getenv("VAD_SILENCE_SECONDS", os.getenv("WAKEWORD_SILENCE_SECONDS", "0.8")))
VAD_GRACE_SECONDS = float(os.getenv("VAD_GRACE_SECONDS", os.getenv("WAKEWORD_GRACE_SECONDS", "0.8")))
VAD_NO_SPEECH_SECONDS = float(os.getenv("VAD_NO_SPEECH_SECONDS", str(VAD_GRACE_SECONDS * 2)))
VAD_MAX_RECORD_SECONDS = float(os.getenv("VAD_MAX_RECORD_SECONDS", os.getenv("WAKEWORD_MAX_RECORD_SECONDS", "12.0")))
VAD_MIN_RECORD_SECONDS = float(os.getenv("VAD_MIN_RECORD_SECONDS", "1.0"))
STREAM_CHUNK_SAMPLES = max(1, int(WAKEWORD_SAMPLE_RATE * STREAM_CHUNK_MS / 1000))
STREAM_CHUNK_BYTES = STREAM_CHUNK_SAMPLES * 2
STREAM_PRE_ROLL_CHUNKS = max(1, int(VAD_PRE_ROLL_MS / STREAM_CHUNK_MS))

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
            pretrained_paths = None
            pretrained_map = None
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

class AlsaInputStream:
    def __init__(self, device: str, rate: int, channels: int, frame_length: int, channel_index: int | None = None):
        if alsaaudio is None:
            raise RuntimeError("pyalsaaudio not installed")
        self.channels = channels
        self.frame_length = frame_length
        self.channel_index = channel_index
        self.pcm = alsaaudio.PCM(
            alsaaudio.PCM_CAPTURE,
            alsaaudio.PCM_NORMAL,
            device=device
        )
        self.pcm.setchannels(channels)
        self.pcm.setrate(rate)
        self.pcm.setformat(alsaaudio.PCM_FORMAT_S16_LE)
        self.pcm.setperiodsize(frame_length)

    def read(self, frame_length: int, exception_on_overflow: bool = False) -> bytes:
        length, data = self.pcm.read()
        if not data:
            return b""
        if self.channels == 1:
            return data
        audio = np.frombuffer(data, dtype=np.int16)
        if audio.size % self.channels != 0:
            return data
        audio = audio.reshape(-1, self.channels)
        if self.channel_index is not None and 0 <= self.channel_index < self.channels:
            mono = audio[:, self.channel_index]
        else:
            mono = audio.mean(axis=1).astype(np.int16)
        return mono.tobytes()

    def stop_stream(self):
        return

    def close(self):
        self.pcm.close()

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

        # Optional ReSpeaker LED ring
        self._setup_respeaker_led()

        # Optional ReSpeaker DSP control
        self._setup_respeaker_dsp()
        
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
        self.input_channels = 1
        channel_env = os.getenv("MIC_CHANNEL_INDEX")
        try:
            self._mic_channel_index = int(channel_env) if channel_env is not None else None
        except ValueError:
            self._mic_channel_index = None
        gain_env = os.getenv("MIC_GAIN", "1.0")
        try:
            self._mic_gain = float(gain_env)
        except ValueError:
            self._mic_gain = 1.0
        self._wakeword_debug = os.getenv("WAKEWORD_DEBUG", "false").strip().lower() in {"1", "true", "yes", "on"}
        self._wakeword_debug_last = 0.0
        buffer_frames = int(os.getenv("WAKEWORD_BUFFER_FRAMES", "25"))
        self._wakeword_audio_buffer: collections.deque[np.ndarray] = collections.deque(maxlen=buffer_frames)
        self._suppress_wakeword_until = 0.0
        self._speaking = False

    def _set_status(self, status: str):
        update_dashboard_state("status", status)
        if self._led_controller:
            self._led_controller.set_state(status)
        self._led_state = status

    def _suppress_wakeword(self, seconds: float) -> None:
        if seconds <= 0:
            return
        self._suppress_wakeword_until = max(self._suppress_wakeword_until, time.time() + seconds)

    def _wakeword_suppressed(self) -> bool:
        return self._speaking or time.time() < self._suppress_wakeword_until

    def _setup_speaker(self):
        # We'll use sounddevice for persistent output
        self.sample_rate = int(os.getenv("TTS_OUTPUT_SAMPLE_RATE", "22050"))
        self.out_stream = None

        def _open_output(sample_rate: int) -> sd.OutputStream:
            stream = sd.OutputStream(
                samplerate=sample_rate,
                channels=1,
                dtype='int16'
            )
            stream.start()
            return stream

        try:
            self.out_stream = _open_output(self.sample_rate)
        except Exception as exc:
            try:
                default_rate = int(sd.query_devices(None, 'output')["default_samplerate"])
            except Exception:
                default_rate = self.sample_rate
            if default_rate != self.sample_rate:
                print(f"[WARN] Output sample rate {self.sample_rate} unsupported; falling back to {default_rate}")
                self.sample_rate = default_rate
            try:
                self.out_stream = _open_output(self.sample_rate)
            except Exception as exc2:
                print(f"[WARN] Failed to open audio output: {exc2}")
                self.out_stream = None

    def _setup_mic(self):
        alsa_device = os.getenv("ALSA_INPUT_DEVICE")
        if alsa_device:
            channels = int(os.getenv("ALSA_INPUT_CHANNELS", "6"))
            self.pa = None
            self.in_stream = AlsaInputStream(
                device=alsa_device,
                rate=WAKEWORD_SAMPLE_RATE,
                channels=channels,
                frame_length=WAKEWORD_FRAME_LENGTH,
                channel_index=self._mic_channel_index,
            )
            self.input_channels = 1
            print(f"[INFO] Using ALSA input device {alsa_device} ({channels}ch)")
            return

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
        except Exception as exc:
            if self.input_channels > 1:
                print(f"[WARN] Failed to open mic with {self.input_channels}ch ({exc}); falling back to mono")
                self.input_channels = 1
                kwargs["channels"] = 1
                self.in_stream = self.pa.open(**kwargs)
            else:
                raise

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
            return max(0, min(31, brightness))

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
        if openwakeword is not None:
            repaired = []
            for model in resolved_models:
                if (not os.path.isabs(model) or not os.path.exists(model)) and model in getattr(openwakeword, "models", {}):
                    repaired.append(openwakeword.models[model]["model_path"])
                else:
                    repaired.append(model)
            resolved_models = repaired
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
        if not self.out_stream:
            print("[WARN] No audio output stream available; skipping playback")
            return
        # For simplicity, save to temp and play or use sounddevice
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(audio_bytes)
            tmp_path = tmp.name
        
        # Read WAV
        with wave.open(tmp_path, 'rb') as wf:
            data = wf.readframes(wf.getnframes())
            src_rate = wf.getframerate()
            # Assume 16-bit PCM for now
            audio_np = np.frombuffer(data, dtype=np.int16)
            if src_rate != self.sample_rate:
                from scipy.signal import resample_poly
                from math import gcd
                g = gcd(src_rate, self.sample_rate)
                audio_np = resample_poly(
                    audio_np.astype(np.float32),
                    self.sample_rate // g,
                    src_rate // g
                )
                audio_np = np.clip(audio_np, -32768, 32767).astype(np.int16)
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
        had_tts_audio = False
        
        try:
            async for message in websocket:
                # Handle binary audio chunks
                if isinstance(message, bytes):
                    if is_streaming_audio:
                        had_tts_audio = True
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
                        self._speaking = True
                        print(f"[TTS] Streaming audio...")
                        self._set_status("speaking")
                    elif data.get("audio_base64"):
                        # Fallback to base64 audio (non-streaming)
                        self._speaking = True
                        had_tts_audio = True
                        self._set_status("speaking")
                        audio_bytes = base64.b64decode(data["audio_base64"])
                        self.play_audio(audio_bytes)
                        self._speaking = False
                        self._suppress_wakeword(WAKEWORD_COOLDOWN_SECONDS)
                        done_event.set()
                    elif not data.get("audio_streaming"):
                        # No audio at all
                        done_event.set()
                elif msg_type == "audio_stream_end":
                    chunks_sent = data.get("chunks_sent", 0)
                    tts_latency = data.get("tts_latency_ms", 0)
                    print(f"[TTS] Stream complete: {chunks_sent} chunks, {tts_latency}ms")
                    self._speaking = False
                    if had_tts_audio:
                        self._suppress_wakeword(WAKEWORD_COOLDOWN_SECONDS)
                    done_event.set()
                elif msg_type == "error":
                    print(f"[ERROR] Compute WS error: {data.get('error')}")
                    done_event.set()
        except Exception as exc:
            print(f"[ERROR] Compute WS receive failed: {exc}")
            done_event.set()
        finally:
            self._speaking = False

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
        recording_frames = 0
        silence_limit_frames = max(1, int(VAD_SILENCE_SECONDS * WAKEWORD_SAMPLE_RATE / WAKEWORD_FRAME_LENGTH))
        max_record_frames = max(1, int(VAD_MAX_RECORD_SECONDS * WAKEWORD_SAMPLE_RATE / WAKEWORD_FRAME_LENGTH))
        grace_frames = max(1, int(VAD_GRACE_SECONDS * WAKEWORD_SAMPLE_RATE / WAKEWORD_FRAME_LENGTH))
        no_speech_frames = max(1, int(VAD_NO_SPEECH_SECONDS * WAKEWORD_SAMPLE_RATE / WAKEWORD_FRAME_LENGTH))
        min_record_frames = max(1, int(VAD_MIN_RECORD_SECONDS * WAKEWORD_SAMPLE_RATE / WAKEWORD_FRAME_LENGTH))
        
        VAD_NORMALIZE = 1.0 / 32768.0

        try:
            while True:
                pcm_bytes = self.in_stream.read(WAKEWORD_FRAME_LENGTH, exception_on_overflow=False)
                if self.input_channels > 1 and pcm_bytes:
                    audio = np.frombuffer(pcm_bytes, dtype=np.int16)
                    if audio.size % self.input_channels == 0:
                        audio = audio.reshape(-1, self.input_channels)
                        if self._mic_channel_index is not None and 0 <= self._mic_channel_index < self.input_channels:
                            mono = audio[:, self._mic_channel_index]
                        else:
                            mono = audio.mean(axis=1).astype(np.int16)
                        pcm_bytes = mono.tobytes()
                if not pcm_bytes:
                    await asyncio.sleep(0.01)
                    continue
                pcm = np.frombuffer(pcm_bytes, dtype=np.int16)
                if pcm.size < 400:
                    await asyncio.sleep(0.01)
                    continue
                if pcm.size != WAKEWORD_FRAME_LENGTH:
                    if pcm.size < WAKEWORD_FRAME_LENGTH:
                        pcm = np.pad(pcm, (0, WAKEWORD_FRAME_LENGTH - pcm.size))
                    else:
                        pcm = pcm[:WAKEWORD_FRAME_LENGTH]
                    pcm_bytes = pcm.tobytes()
                
                if self.state == AudioRecorderState.LISTENING:
                    pre_roll_buffer.append(pcm_bytes)
                    if self._wakeword_suppressed():
                        self._wakeword_audio_buffer.clear()
                        await asyncio.sleep(0.01)
                        continue
                    pcm_for_ww = pcm
                    if self._mic_gain != 1.0 and pcm.size:
                        pcm_for_ww = np.clip(
                            pcm.astype(np.float32) * self._mic_gain,
                            -32768,
                            32767
                        ).astype(np.int16)
                    self._wakeword_audio_buffer.append(pcm_for_ww)
                    if len(self._wakeword_audio_buffer) >= 5:
                        merged = np.concatenate(list(self._wakeword_audio_buffer), axis=0)
                        scores = self.wakeword_detector.predict(merged)
                    else:
                        scores = self.wakeword_detector.predict(pcm_for_ww)
                    max_name = None
                    max_score = 0.0
                    if scores:
                        max_name, max_score = max(scores.items(), key=lambda item: item[1])
                    if self._wakeword_debug and scores:
                        now = time.time()
                        if now - self._wakeword_debug_last >= 1.0:
                            rms = float(np.sqrt(np.mean(pcm.astype(np.float32) ** 2))) if pcm.size else 0.0
                            print(f"[WAKE-DEBUG] rms={rms:.1f} curr={max_name}:{max_score:.3f} buf={len(self._wakeword_audio_buffer)}")
                            self._wakeword_debug_last = now
                    
                    found = False
                    if max_name is not None and max_score >= WAKEWORD_THRESHOLD:
                        print(f"[WAKE] Wake Word: {max_name}")
                        found = True
                    
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
                        recording_frames = 0
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
                    recording_frames += 1
                    
                    # VAD Check - Silero VAD needs exactly 512 samples at 16kHz
                    prob = 0.0
                    if self.vad_available:
                        audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
                        try:
                            for start in range(0, len(audio_int16) - 511, 512):
                                vad_chunk = audio_int16[start:start + 512]
                                vad_tensor = torch.from_numpy(vad_chunk.astype(np.float32) * VAD_NORMALIZE)
                                prob = max(prob, self.vad_model(vad_tensor, WAKEWORD_SAMPLE_RATE).item())
                                if prob >= VAD_SPEECH_THRESHOLD:
                                    break
                        except Exception as exc:
                            print(f"[WARN] VAD error: {exc}")
                    
                    if prob >= VAD_SPEECH_THRESHOLD:
                        if not speech_detected:
                            speech_detected = True
                            print(f"[VAD] Speech detected")
                        silence_counter = 0
                    else:
                        if speech_detected or recording_frames >= grace_frames:
                            silence_counter += 1

                    # Always send audio chunks during recording (don't gate on speech_detected)
                    # The server's VAD will also filter, but we want to capture everything
                    for offset in range(0, len(pcm_bytes), STREAM_CHUNK_BYTES):
                        chunk = pcm_bytes[offset:offset + STREAM_CHUNK_BYTES]
                        await self._enqueue_audio_chunk(chunk)
                    
                    # End recording conditions - require minimum frames (~1 sec) before stopping
                    has_enough_audio = recording_frames >= min_record_frames
                    silence_timeout = speech_detected and silence_counter >= silence_limit_frames
                    no_speech_timeout = (not speech_detected) and recording_frames >= no_speech_frames
                    max_length = recording_frames >= max_record_frames
                    if (has_enough_audio and silence_timeout) or no_speech_timeout or max_length:
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
            if hasattr(self.in_stream, "stop_stream"):
                self.in_stream.stop_stream()
            if hasattr(self.in_stream, "close"):
                self.in_stream.close()
            if self.pa is not None:
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
