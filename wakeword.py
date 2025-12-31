import asyncio
import collections
import json
import pyaudio
import struct
import pvporcupine
import wave
import subprocess
import os
import shutil
import threading
import queue
import time
import numpy as np
import sounddevice as sd
import torch
from google import genai
from google.genai import types
from piper.voice import PiperVoice
from tools.registry import GEMINI_TOOLS, dispatch_tool
from tools.transcription import create_transcription_service
from tools.audio import pause_media, resume_media, route_to_bluetooth
from core.tts_preprocessing import preprocess_for_tts
from core.conversation import ConversationMemory, parse_clear_phrases, should_clear_history
from adapters.gpu_tts_client import GPUTTSClient
from core.event_bus import (
    EventBus, emit_state_changed, emit_transcript, 
    emit_assistant_text, emit_tool_call, emit_tool_result, emit_error
)
from dotenv import load_dotenv

try:
    import websockets
except Exception:
    websockets = None

load_dotenv()

AUTO_ROUTE_BT_SINK = os.getenv("AUTO_ROUTE_BT_SINK", "true").lower() == "true"
BT_AUDIO_DEVICE_NAME = os.getenv("BT_AUDIO_DEVICE_NAME")

if AUTO_ROUTE_BT_SINK:
    try:
        print(f"🔊 {route_to_bluetooth(BT_AUDIO_DEVICE_NAME)}")
    except Exception as e:
        print(f"⚠️  Bluetooth routing failed: {e}")

def _get_env_float(name: str, default: float) -> float:
    """Parse a float env var with a safe fallback."""
    value = os.getenv(name)
    if not value:
        return default
    try:
        return float(value)
    except ValueError:
        print(f"Invalid {name}={value!r}; using default {default}")
        return default

def _get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default
    try:
        return int(value)
    except ValueError:
        print(f"Invalid {name}={value!r}; using default {default}")
        return default

def _select_input_device_index(pa: pyaudio.PyAudio, preferred: str | None) -> int | None:
    """Resolve an input device index from an env override (index or name substring)."""
    if not preferred:
        return None
    try:
        index = int(preferred)
        info = pa.get_device_info_by_index(index)
        if info.get("maxInputChannels", 0) > 0:
            print(f"🎤 Using input device index {index}: {info.get('name')}")
            return index
        print(f"⚠️  Device index {index} has no input channels")
        return None
    except ValueError:
        needle = preferred.lower()
    except Exception as exc:
        print(f"⚠️  Failed to read device index {preferred!r}: {exc}")
        return None

    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        name = info.get("name", "")
        if info.get("maxInputChannels", 0) > 0 and needle in name.lower():
            print(f"🎤 Using input device index {i}: {name}")
            return i

    print(f"⚠️  No input device matched WAKEWORD_INPUT_DEVICE={preferred!r}")
    return None

def _open_input_stream(pa: pyaudio.PyAudio, rate: int, frames_per_buffer: int, device_index: int | None):
    kwargs = dict(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        input=True,
        frames_per_buffer=frames_per_buffer,
    )
    if device_index is not None:
        kwargs["input_device_index"] = device_index
    return pa.open(**kwargs)

def _open_wakeword_stream(pa: pyaudio.PyAudio, device_index: int | None):
    return _open_input_stream(
        pa,
        porcupine.sample_rate,
        porcupine.frame_length,
        device_index,
    )

CONVERSATION_ENABLED = os.getenv("CONVERSATION_ENABLED", "true").lower() == "true"
CONVERSATION_MAX_TURNS = _get_env_int("CONVERSATION_MAX_TURNS", 6)
CONVERSATION_TTL_SECONDS = _get_env_float("CONVERSATION_TTL_SECONDS", 600.0)
CONVERSATION_CLEAR_PHRASES = parse_clear_phrases(os.getenv("CONVERSATION_CLEAR_PHRASES"))
CONVERSATION_RESET_ON_TOOL_CALL = (
    os.getenv("CONVERSATION_RESET_ON_TOOL_CALL", "true").lower() == "true"
)
conversation_memory = (
    ConversationMemory(CONVERSATION_MAX_TURNS, CONVERSATION_TTL_SECONDS)
    if CONVERSATION_ENABLED
    else None
)

# === Global Event Bus ===
# UI clients can subscribe to receive real-time updates
event_bus = EventBus()
event_bus.start()

# === UI Event Bridge ===
# Sends EventBus updates to the UI server via WebSocket.
UI_WS_URL = os.getenv("UI_WS_URL", "ws://localhost:8766")


class UIEventBridge:
    """Bridge EventBus events to the UI WebSocket server."""

    def __init__(self, bus: EventBus, ws_url: str):
        self.bus = bus
        self.ws_url = ws_url
        self._queue = queue.Queue(maxsize=500)
        self._running = False
        self._thread = None

    def start(self):
        if not websockets:
            print("⚠️  websockets not installed; UI event bridge disabled")
            return
        self._running = True
        self.bus.subscribe("*", self._on_event)
        self._thread = threading.Thread(target=self._run, daemon=True, name="UIEventBridge")
        self._thread.start()

    def stop(self):
        self._running = False
        try:
            self._queue.put_nowait(None)
        except queue.Full:
            pass

    def _on_event(self, event):
        if not self._running:
            return
        try:
            self._queue.put_nowait(event.to_dict())
        except queue.Full:
            pass

    def _run(self):
        asyncio.run(self._async_loop())

    async def _async_loop(self):
        while self._running:
            try:
                async with websockets.connect(self.ws_url) as ws:
                    sender = asyncio.create_task(self._send_loop(ws))
                    receiver = asyncio.create_task(self._receive_loop(ws))
                    done, pending = await asyncio.wait(
                        [sender, receiver],
                        return_when=asyncio.FIRST_EXCEPTION
                    )
                    for task in pending:
                        task.cancel()
            except Exception as e:
                print(f"⚠️  UI bridge connection failed: {e}")
                await asyncio.sleep(2)

    async def _send_loop(self, ws):
        while self._running:
            loop = asyncio.get_running_loop()
            event = await loop.run_in_executor(None, self._queue.get)
            if event is None:
                return
            await ws.send(json.dumps({"type": "event", "data": event}))

    async def _receive_loop(self, ws):
        while self._running:
            raw = await ws.recv()
            try:
                message = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if not isinstance(message, dict):
                continue

            if message.get("type") == "tool_call":
                data = message.get("data", {})
                if data.get("origin") != "ui":
                    continue

                tool_name = data.get("tool_name")
                args = data.get("arguments", {})
                if not tool_name:
                    continue

                emit_tool_call(event_bus, tool_name, args)
                try:
                    result = dispatch_tool(tool_name, args)
                    emit_tool_result(event_bus, tool_name, True, result)
                except Exception as e:
                    emit_tool_result(event_bus, tool_name, False, str(e))


ui_bridge = UIEventBridge(event_bus, UI_WS_URL)
ui_bridge.start()

# === Pi 5 Optimizations ===
# Limit torch threads to reduce CPU contention on Pi
torch.set_num_threads(2)
torch.set_grad_enabled(False)  # Disable autograd (not needed for inference)

# Initialize transcription service with GPU offloading and fallback
transcription_service = create_transcription_service()

def _build_llm_contents(user_text: str, use_history: bool):
    if not use_history or not conversation_memory:
        return user_text
    history = conversation_memory.get_messages()
    if not history:
        return user_text
    contents = []
    for message in history:
        contents.append(
            types.Content(
                role=message["role"],
                parts=[types.Part(text=message["text"])]
            )
        )
    contents.append(types.Content(role="user", parts=[types.Part(text=user_text)]))
    return contents

# Initialize Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize Silero VAD with local caching
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

def load_vad_model():
    """Load VAD model from local cache or download once."""
    model_dir = os.path.join(REPO_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "silero_vad.jit")
    
    # Try loading from local cache first
    if os.path.exists(model_path):
        try:
            model = torch.jit.load(model_path)
            print("✅ VAD loaded from local cache")
            return model, True
        except Exception as e:
            print(f"⚠️  Failed to load cached VAD: {e}")
    
    # Download from hub and cache locally
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', 
            model='silero_vad', 
            force_reload=False, 
            onnx=False
        )
        # Save to local cache for future runs
        try:
            torch.jit.save(model, model_path)
            print(f"✅ VAD cached to {model_path}")
        except Exception:
            pass  # Caching failed, but model works
        return model, True
    except Exception as e:
        print(f"⚠️  VAD initialization failed: {e}. Using fixed recording duration.")
        return None, False

vad_model, vad_available = load_vad_model()

# Initialize Piper TTS
DEFAULT_PIPER_MODEL = os.path.join(REPO_ROOT, "piper_models", "en_US-lessac-medium.onnx")
PIPER_MODEL = os.getenv("PIPER_MODEL", DEFAULT_PIPER_MODEL)
piper_voice = None
if os.path.exists(PIPER_MODEL):
    piper_voice = PiperVoice.load(PIPER_MODEL)
    # Pre-warm the model for faster first response
    if piper_voice:
        try:
            _ = list(piper_voice.synthesize("Ready"))
            print("✅ Piper TTS pre-warmed")
        except Exception:
            pass
else:
    print(f"⚠️  Piper model not found at {PIPER_MODEL}. TTS will not work until model is downloaded.")


# === Persistent Audio Output Stream (Pi5 Optimization) ===
class PersistentAudioOutput:
    """
    Manages a persistent audio output stream to avoid per-utterance overhead.
    Creating/destroying streams costs ~50-100ms each time on Pi.
    """
    
    def __init__(self, sample_rate: int = 22050, channels: int = 1, device: int | str | None = None, allow_stereo_fallback: bool = True):
        self.sample_rate = sample_rate
        self.channels = channels
        self.device = device
        self.allow_stereo_fallback = allow_stereo_fallback
        self._stream = None
        self._lock = threading.Lock()
    
    def _ensure_stream(self):
        """Create stream if not exists or if closed."""
        if self._stream is None or not self._stream.active:
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
            try:
                self._stream = sd.OutputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype='int16',
                    device=self.device
                )
                self._stream.start()
            except Exception as e:
                # Fallback to stereo if device rejects mono.
                if self.channels == 1 and self.allow_stereo_fallback:
                    print(f"⚠️  OutputStream failed (mono). Retrying stereo: {e}")
                    self.channels = 2
                    self._stream = sd.OutputStream(
                        samplerate=self.sample_rate,
                        channels=self.channels,
                        dtype='int16',
                        device=self.device
                    )
                    self._stream.start()
                else:
                    raise
    
    def write(self, audio_data: np.ndarray):
        """Write audio data to the persistent stream."""
        with self._lock:
            self._ensure_stream()
            try:
                self._stream.write(audio_data)
            except Exception as e:
                print(f"⚠️  Audio write error: {e}")
                # Try to recover by recreating stream
                self._stream = None
                self._ensure_stream()
                self._stream.write(audio_data)
    
    def close(self):
        """Close the stream (call on shutdown)."""
        with self._lock:
            if self._stream is not None:
                try:
                    self._stream.stop()
                    self._stream.close()
                except Exception:
                    pass
                self._stream = None


# Global persistent audio output (initialized after piper_voice is loaded)
tts_audio_output = None
if piper_voice:
    output_device = os.getenv("TTS_OUTPUT_DEVICE")
    output_channels = _get_env_int("TTS_OUTPUT_CHANNELS", 1)
    allow_stereo_fallback = os.getenv("TTS_OUTPUT_ALLOW_STEREO_FALLBACK", "true").lower() == "true"
    if output_device is not None:
        try:
            output_device = int(output_device)
        except ValueError:
            pass
    tts_audio_output = PersistentAudioOutput(
        sample_rate=piper_voice.config.sample_rate,
        channels=output_channels,
        device=output_device,
        allow_stereo_fallback=allow_stereo_fallback
    )

# Initialize GPU TTS client with Piper fallback
XTTS_SERVER_URL = os.getenv("XTTS_SERVER_URL", "http://localhost:5001")
USE_GPU_TTS = os.getenv("USE_GPU_TTS", "true").lower() == "true"
XTTS_STREAM_TIMEOUT = float(os.getenv("XTTS_STREAM_TIMEOUT", "30"))
XTTS_STREAM_CHUNK_SIZE = int(os.getenv("XTTS_STREAM_CHUNK_SIZE", "15"))
XTTS_STREAM_READ_CHUNK_BYTES = int(os.getenv("XTTS_STREAM_READ_CHUNK_BYTES", "2400"))

gpu_tts_client = None
if USE_GPU_TTS:
    gpu_tts_client = GPUTTSClient(
        server_url=XTTS_SERVER_URL,
        piper_voice=piper_voice,
        piper_sample_rate=piper_voice.config.sample_rate if piper_voice else 22050,
        timeout_seconds=3.0,
        stream_timeout_seconds=XTTS_STREAM_TIMEOUT,
        stream_chunk_size=XTTS_STREAM_CHUNK_SIZE,
        stream_chunk_bytes=XTTS_STREAM_READ_CHUNK_BYTES
    )
    print(f"🔊 GPU TTS enabled: {XTTS_SERVER_URL}")
else:
    print("🔊 Using local Piper TTS only")

# === Worker Thread for Non-Blocking Processing ===
# This allows wakeword detection to continue while processing commands

class AssistantWorker:
    """Background worker thread for processing voice commands."""
    
    def __init__(self, bus: EventBus):
        self.command_queue = queue.Queue()
        self.is_processing = False
        self._running = False
        self._thread = None
        self.bus = bus
    
    def start(self):
        """Start the worker thread."""
        self._running = True
        self._thread = threading.Thread(target=self._worker_loop, daemon=True, name="AssistantWorker")
        self._thread.start()
        print("✅ Worker thread started")
    
    def stop(self):
        """Stop the worker thread."""
        self._running = False
        self.command_queue.put(None)  # Sentinel to unblock
        if self._thread:
            self._thread.join(timeout=2.0)
    
    def submit(self, audio_path: str):
        """Submit an audio file for processing."""
        if self.is_processing:
            print("⚠️  Already processing a command, ignoring...")
            return False
        self.command_queue.put(audio_path)
        return True
    
    def _worker_loop(self):
        """Main worker loop - processes commands from queue."""
        while self._running:
            try:
                audio_path = self.command_queue.get(timeout=0.5)
                if audio_path is None:  # Sentinel
                    continue
                
                self.is_processing = True
                try:
                    self._process_command(audio_path)
                except Exception as e:
                    print(f"❌ Worker error: {e}")
                finally:
                    self.is_processing = False
                    print("👂 Listening for wake word again...")
                    
            except queue.Empty:
                continue
    
    def _process_command(self, audio_path: str):
        """Process a single command (transcribe -> LLM -> TTS)."""
        # Start a new conversation turn
        self.bus.new_correlation_id()
        
        # 1. Transcribe
        emit_state_changed(self.bus, "listening", "transcribing")
        print("🎧 Transcribing...")
        
        start_time = time.time()
        try:
            user_command = transcription_service.transcribe(audio_path)
        finally:
            # Clean up temp audio file to avoid /tmp accumulation
            try:
                if audio_path and os.path.exists(audio_path):
                    os.remove(audio_path)
            except OSError:
                pass
        transcribe_ms = int((time.time() - start_time) * 1000)
        
        if not user_command:
            print("❌ No speech detected")
            emit_error(self.bus, "NO_SPEECH", "No speech detected in audio")
            emit_state_changed(self.bus, "transcribing", "idle")
            resume_media()  # Resume music if we paused it
            return
        
        print(f"📝 You said: {user_command}")
        emit_transcript(self.bus, user_command, is_final=True, duration_ms=transcribe_ms)
        
        # 2. Send to Gemini
        emit_state_changed(self.bus, "transcribing", "thinking")
        print("🧠 Consulting Gemini...")
        
        try:
            model_name = os.getenv("MODEL_NAME", "gemini-3-flash-preview")
            
            # Detect if this needs real-time info (weather, stocks, news, sports, current events)
            realtime_keywords = ['weather', 'temperature', 'forecast', 'stock', 'price', 'market',
                                'news', 'score', 'game', 'playing', 'today', 'tonight', 'current',
                                'right now', 'latest', 'recent', 'who won', 'what time']
            needs_search = any(kw in user_command.lower() for kw in realtime_keywords)
            
            # Detect if this is a smart home/tool command
            home_keywords = ['light', 'lights', 'lamp', 'brightness', 'dim', 'bright', 'turn on',
                           'turn off', 'kitchen', 'family room', 'foyer', 'stairs', 'island',
                           'bluetooth', 'connect', 'disconnect', 'volume', 'music', 'play', 'stop',
                           'timer', 'alarm', 'remind', 'minutes', 'seconds', 'hours', 'cancel timer']
            needs_tools = any(kw in user_command.lower() for kw in home_keywords)
            
            if conversation_memory:
                conversation_memory.maybe_expire()
                if should_clear_history(user_command, CONVERSATION_CLEAR_PHRASES):
                    conversation_memory.reset()

            use_history = conversation_memory is not None and not needs_tools
            contents = _build_llm_contents(user_command, use_history)

            # Location context
            location_context = "User is located in Charlotte, NC (zip code 28211)."
            
            if needs_search and not needs_tools:
                # Use Google Search for real-time info
                system_instruction = f"""You are Computer, a helpful voice assistant. {location_context}

IMPORTANT: Your responses will be spoken aloud via text-to-speech. Format for natural speech:
- Say "high of 58" not "58°F" or "58 degrees F"
- Say "around 3 PM" not "3:00 PM"
- Use conversational language, avoid abbreviations
- Keep responses to 1-2 sentences max"""
                google_search_tool = types.Tool(google_search=types.GoogleSearch())
                
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        tools=[google_search_tool],
                        system_instruction=system_instruction,
                        temperature=0.3
                    )
                )
            else:
                # Use function calling for smart home and general questions
                system_instruction = f"""You are Computer, a helpful voice assistant. {location_context}

For lighting commands, IMMEDIATELY call control_home_lighting function with NO explanation.
Device IDs: Kitchen Cans=85, Foyer=87, Stairs=89, Upstairs Hall=91, Front Door=93, Kitchen Island=95, Downstairs Hallway=97, Upstairs Deck=99, Family Room=204, Breakfast=206.
For ALL lights: use device_id=999 with brightness=100 (ON) or brightness=0 (OFF).
For pizza dough recipes or hydration adjustments, call pizza_dough_recipe with the requested parameters.

IMPORTANT: Your responses will be spoken aloud via text-to-speech. Format for natural speech:
- Say "high of 58" not "58°F" or "58 degrees F"
- Use conversational language, avoid abbreviations
- Keep responses to 1-2 sentences max"""
                
                response = client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=types.GenerateContentConfig(
                        tools=GEMINI_TOOLS,
                        system_instruction=system_instruction,
                        temperature=0.3
                    )
                )
            
            # 3. Execute Tool Call or Respond
            if response.candidates and response.candidates[0].content.parts:
                has_tool_call = False
                suppress_auto_resume = False
                suppress_resume_tools = {"play_youtube_music", "pause_audio", "stop_music"}
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        has_tool_call = True
                        tool_name = part.function_call.name
                        args = dict(part.function_call.args)
                        if tool_name in suppress_resume_tools:
                            suppress_auto_resume = True
                        
                        print(f"✅ Action: {tool_name}")
                        emit_tool_call(self.bus, tool_name, args)
                        emit_state_changed(self.bus, "thinking", "executing")
                        
                        tool_start = time.time()
                        result = dispatch_tool(tool_name, args)
                        tool_ms = int((time.time() - tool_start) * 1000)
                        
                        emit_tool_result(self.bus, tool_name, success=True, result=str(result), duration_ms=tool_ms)
                
                if has_tool_call:
                    if conversation_memory and CONVERSATION_RESET_ON_TOOL_CALL:
                        conversation_memory.reset()
                    emit_state_changed(self.bus, "executing", "speaking")
                    speak_tts("Done")
                    emit_state_changed(self.bus, "speaking", "idle")
                    if not suppress_auto_resume:
                        resume_media()  # Resume music if we paused it
                elif response.text:
                    if conversation_memory and use_history:
                        conversation_memory.add("user", user_command)
                        conversation_memory.add("model", response.text)
                    print(f"💬 Computer: {response.text}")
                    emit_assistant_text(self.bus, response.text)
                    emit_state_changed(self.bus, "thinking", "speaking")
                    speak_tts(response.text)
                    emit_state_changed(self.bus, "speaking", "idle")
                    resume_media()  # Resume music if we paused it
        
        except Exception as e:
            print(f"❌ Gemini API failed: {e}")
            emit_error(self.bus, "LLM_ERROR", str(e), recoverable=True)
            emit_state_changed(self.bus, "thinking", "speaking")
            speak_tts("I'm having trouble connecting to my brain.")
            emit_state_changed(self.bus, "speaking", "idle")
            resume_media()  # Resume music if we paused it

# Global worker instance
assistant_worker = AssistantWorker(event_bus)

def _to_stereo(audio_data: np.ndarray) -> np.ndarray:
    if audio_data.ndim == 1:
        return np.repeat(audio_data[:, None], 2, axis=1)
    if audio_data.shape[1] == 1:
        return np.repeat(audio_data, 2, axis=1)
    return audio_data

def speak_tts(text):
    """
    Speak text using GPU TTS (XTTS) with Piper fallback.
    
    Tries GPU streaming first for lowest latency, falls back to non-streaming,
    then to local Piper if server is unavailable.
    """
    # Preprocess text for more natural TTS
    text = preprocess_for_tts(text)
    
    if not text:
        return
    
    # Try GPU TTS streaming first (lowest latency - plays audio as it's generated)
    if gpu_tts_client and tts_audio_output:
        if gpu_tts_client.synthesize_stream(text, tts_audio_output):
            return
        
        # Fall back to non-streaming if streaming fails
        result = gpu_tts_client.synthesize(text, prefer_gpu=True)
        if result is not None:
            audio_data, sample_rate = result
            # Resample if needed (GPU outputs 24kHz, Piper outputs 22050Hz)
            if sample_rate != tts_audio_output.sample_rate:
                audio_data = gpu_tts_client._resample(
                    audio_data, sample_rate, tts_audio_output.sample_rate
                )
            if tts_audio_output.channels == 2:
                audio_data = _to_stereo(audio_data)
            tts_audio_output.write(audio_data)
            return
    
    # Fallback to direct Piper if GPU client not configured or failed
    if piper_voice and tts_audio_output:
        for audio_chunk in piper_voice.synthesize(text):
            int_data = np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16)
            if tts_audio_output.channels == 2:
                int_data = _to_stereo(int_data)
            tts_audio_output.write(int_data)
    elif piper_voice:
        # Fallback to per-call stream if persistent output not available
        stream = sd.OutputStream(
            samplerate=piper_voice.config.sample_rate,
            channels=1,
            dtype='int16'
        )
        stream.start()
        for audio_chunk in piper_voice.synthesize(text):
            int_data = np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16)
            stream.write(int_data)
        stream.stop()
        stream.close()
    else:
        print(f"⚠️  Cannot speak: '{text}' - No TTS available")

def play_audio(audio_path: str) -> None:
    """Play an audio file using the first available system player."""
    players = [
        ("afplay", ["afplay", audio_path]),
        ("ffplay", ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", audio_path]),
        ("aplay", ["aplay", "-q", audio_path]),
    ]
    for player, cmd in players:
        if shutil.which(player):
            subprocess.run(cmd, check=False)
            return
    print("⚠️  No audio player found. Install ffmpeg or use a system with audio support.")

# ==========================================
# ⚡️ ZERO-STOP AUDIO LOOP
# ==========================================

class AudioRecorderState:
    LISTENING = 0
    RECORDING = 1
    PROCESSING = 2

# 1. Setup the Engine
# 'keywords' can be standard ones like ['computer']
# or a path to your custom 'Gemini.ppn' file.
access_key = os.getenv("PORCUPINE_ACCESS_KEY")
porcupine = pvporcupine.create(
    access_key=access_key,
    keywords=['computer']
)

# 2. Setup the Microphone Stream (Open once, never close during runtime)
pa = pyaudio.PyAudio()
INPUT_DEVICE_INDEX = _select_input_device_index(pa, os.getenv("WAKEWORD_INPUT_DEVICE"))
audio_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length,
    input_device_index=INPUT_DEVICE_INDEX
)

FRAME_LENGTH = porcupine.frame_length
SAMPLE_RATE = porcupine.sample_rate
PRE_ROLL_SECONDS = _get_env_float("WAKEWORD_PRE_ROLL_SECONDS", 1.5)
MAX_PRE_ROLL_FRAMES = max(1, int(SAMPLE_RATE * PRE_ROLL_SECONDS / FRAME_LENGTH))

SILENCE_SECONDS = _get_env_float("WAKEWORD_SILENCE_SECONDS", 0.8)
MAX_RECORD_SECONDS = _get_env_float("WAKEWORD_MAX_RECORD_SECONDS", 15.0)
GRACE_SECONDS = _get_env_float("WAKEWORD_GRACE_SECONDS", 0.8)
FIXED_RECORD_SECONDS = _get_env_float("WAKEWORD_FIXED_RECORD_SECONDS", 4.0)

SILENCE_LIMIT_FRAMES = max(1, int(SILENCE_SECONDS * SAMPLE_RATE / FRAME_LENGTH))
MAX_RECORD_FRAMES = max(1, int(MAX_RECORD_SECONDS * SAMPLE_RATE / FRAME_LENGTH))
GRACE_FRAMES = max(1, int(GRACE_SECONDS * SAMPLE_RATE / FRAME_LENGTH))
NO_SPEECH_FRAMES = max(1, GRACE_FRAMES * 2)
FIXED_RECORD_FRAMES = max(1, int(FIXED_RECORD_SECONDS * SAMPLE_RATE / FRAME_LENGTH))
VAD_NORMALIZE = 1.0 / 32768.0

print(f"👂 High-Performance Loop Started. Pre-roll buffer: {MAX_PRE_ROLL_FRAMES} frames")

# 3. State Variables
state = AudioRecorderState.LISTENING
pre_roll_buffer = collections.deque(maxlen=MAX_PRE_ROLL_FRAMES)
current_command_frames = []
vad_silence_counter = 0
recording_frames = 0
speech_detected = False
vad_tensor = torch.zeros(FRAME_LENGTH, dtype=torch.float32)

# Start the worker thread
assistant_worker.start()

# 4. The "Infinite Ear" Loop
try:
    while True:
        # -----------------------------------------------------
        # 1. READ AUDIO (Non-blocking / Low Latency)
        # -----------------------------------------------------
        try:
            pcm_bytes = audio_stream.read(FRAME_LENGTH, exception_on_overflow=False)
        except IOError:
            continue

        pre_roll_buffer.append(pcm_bytes)

        # -----------------------------------------------------
        # 2. STATE MACHINE
        # -----------------------------------------------------
        if state == AudioRecorderState.LISTENING:
            pcm = struct.unpack_from("h" * FRAME_LENGTH, pcm_bytes)
            keyword_index = porcupine.process(pcm)

            if keyword_index >= 0:
                if assistant_worker.is_processing:
                    print("⚠️  Still processing previous command...")
                    continue

                print("🔔 Wake word detected! (Instant Switch)")

                if pause_media():
                    print("⏸️  Paused media playback")

                event_bus.emit("wakeword_detected", {"keyword_index": keyword_index})
                emit_state_changed(event_bus, "idle", "listening")

                current_command_frames = list(pre_roll_buffer)
                vad_silence_counter = 0
                recording_frames = 0
                speech_detected = False
                state = AudioRecorderState.RECORDING

        elif state == AudioRecorderState.RECORDING:
            current_command_frames.append(pcm_bytes)
            recording_frames += 1

            if vad_available:
                audio_int16 = np.frombuffer(pcm_bytes, dtype=np.int16)
                vad_tensor[:] = torch.from_numpy(audio_int16.astype(np.float32) * VAD_NORMALIZE)
                try:
                    speech_prob = vad_model(vad_tensor, SAMPLE_RATE).item()
                    if speech_prob >= 0.5:
                        speech_detected = True
                        vad_silence_counter = 0
                    else:
                        if speech_detected or recording_frames >= GRACE_FRAMES:
                            vad_silence_counter += 1
                except Exception:
                    if speech_detected or recording_frames >= GRACE_FRAMES:
                        vad_silence_counter += 1

                is_silence_timeout = speech_detected and vad_silence_counter >= SILENCE_LIMIT_FRAMES
                is_max_length = recording_frames >= MAX_RECORD_FRAMES
            else:
                is_silence_timeout = False
                is_max_length = recording_frames >= FIXED_RECORD_FRAMES

            is_no_speech_timeout = not speech_detected and recording_frames >= NO_SPEECH_FRAMES

            if is_silence_timeout or is_max_length or is_no_speech_timeout:
                print(f"🛑 Capture complete. Frames: {len(current_command_frames)}")

                temp_audio_path = f"/tmp/cmd_{int(time.time())}.wav"
                with wave.open(temp_audio_path, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
                    wf.setframerate(SAMPLE_RATE)
                    wf.writeframes(b''.join(current_command_frames))

                assistant_worker.submit(temp_audio_path)

                current_command_frames = []
                pre_roll_buffer.clear()
                state = AudioRecorderState.LISTENING

except KeyboardInterrupt:
    print("\n👋 Shutting down...")
    ui_bridge.stop()
    event_bus.stop()
    assistant_worker.stop()
    if tts_audio_output is not None:
        tts_audio_output.close()
    if porcupine is not None:
        porcupine.delete()
    if audio_stream is not None:
        audio_stream.close()
    if pa is not None:
        pa.terminate()
