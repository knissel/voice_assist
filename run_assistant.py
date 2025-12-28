#!/usr/bin/env python3
"""
Minimal push-to-talk voice assistant runner.
Press SPACE to record -> STT -> LLM -> TTS -> play
Press ESC to quit.
"""
import argparse
import pyaudio
import wave
import subprocess
import json
import os
import numpy as np
import sounddevice as sd
import torch
from typing import Optional
from google import genai
from google.genai import types
from piper.voice import PiperVoice
from tools.registry import GEMINI_TOOLS, dispatch_tool
from tools.transcription import create_transcription_service
from dotenv import load_dotenv

load_dotenv()

# === Pi 5 Optimizations ===
torch.set_num_threads(2)
torch.set_grad_enabled(False)

# === CONFIGURATION ===
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-flash-lite-latest")
RECORD_SECONDS = 4
RATE = int(os.getenv("MIC_RATE", "16000"))
CHUNK = 1024
MIC_DEVICE_INDEX = os.getenv("MIC_DEVICE_INDEX")

def _parse_mic_device_index(value: Optional[str]) -> Optional[int]:
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        print(f"⚠️  MIC_DEVICE_INDEX must be an integer, got: {value!r}. Using default input.")
        return None

# === INITIALIZE ===
client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize transcription service with GPU offloading and fallback
transcription_service = create_transcription_service()

# Initialize Silero VAD with local caching
def load_vad_model():
    """Load VAD model from local cache or download once."""
    model_dir = os.path.join(REPO_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "silero_vad.jit")
    
    if os.path.exists(model_path):
        try:
            model = torch.jit.load(model_path)
            print("✅ VAD loaded from local cache")
            return model, True
        except Exception as e:
            print(f"⚠️  Failed to load cached VAD: {e}")
    
    try:
        model, utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad', 
            model='silero_vad', 
            force_reload=False, 
            onnx=False
        )
        try:
            torch.jit.save(model, model_path)
            print(f"✅ VAD cached to {model_path}")
        except Exception:
            pass
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
    
    def __init__(self, sample_rate: int = 22050, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self._stream = None
    
    def _ensure_stream(self):
        """Create stream if not exists or if closed."""
        if self._stream is None or not self._stream.active:
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception:
                    pass
            self._stream = sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype='int16'
            )
            self._stream.start()
    
    def write(self, audio_data: np.ndarray):
        """Write audio data to the persistent stream."""
        self._ensure_stream()
        try:
            self._stream.write(audio_data)
        except Exception as e:
            print(f"⚠️  Audio write error: {e}")
            self._stream = None
            self._ensure_stream()
            self._stream.write(audio_data)
    
    def close(self):
        """Close the stream (call on shutdown)."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None


# Global persistent audio output
tts_audio_output = None
if piper_voice:
    tts_audio_output = PersistentAudioOutput(
        sample_rate=piper_voice.config.sample_rate,
        channels=1
    )

# === PIPELINE FUNCTIONS ===
def record_audio():
    """
    Record audio from microphone with VAD.
    
    Optimized for Raspberry Pi 5:
    - Ring buffer instead of np.concatenate (O(1) vs O(n))
    - Pre-allocated tensor buffer to avoid allocations in hot loop
    - Grace period before checking silence (time to collect thoughts)
    """
    print("🎤 Recording...")
    pa = pyaudio.PyAudio()
    device_index = _parse_mic_device_index(MIC_DEVICE_INDEX)
    try:
        stream = pa.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=RATE,
            input=True,
            frames_per_buffer=CHUNK,
            input_device_index=device_index,
        )
    except Exception as exc:
        if device_index is not None:
            print(f"⚠️  Failed to open MIC_DEVICE_INDEX={device_index}. Falling back to default input.")
            stream = pa.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK,
            )
        else:
            raise exc
    
    frames = []
    
    if vad_available:
        # VAD-based recording with optimized ring buffer
        MAX_RECORD_SECONDS = 15      # Extended max recording time
        GRACE_PERIOD_SECONDS = 1.5   # Wait this long before checking for silence
        SILENCE_DURATION = 2.0       # Require 2s of silence to stop (was 1.5s)
        VAD_CHUNK = 512  # Silero VAD requires exactly 512 samples for 16kHz
        
        silence_chunks = 0
        silence_threshold = int(SILENCE_DURATION * RATE / VAD_CHUNK)
        max_chunks = int(MAX_RECORD_SECONDS * RATE / CHUNK)
        grace_chunks = int(GRACE_PERIOD_SECONDS * RATE / CHUNK)
        
        # Pre-allocate ring buffer for VAD (avoids O(n) concatenation)
        RING_SIZE = VAD_CHUNK * 4  # Hold ~4 VAD windows
        ring_buffer = np.zeros(RING_SIZE, dtype=np.int16)
        ring_write_pos = 0
        ring_read_pos = 0
        ring_available = 0
        
        # Pre-allocate tensor buffer (reused each iteration)
        vad_tensor_buffer = torch.zeros(VAD_CHUNK, dtype=torch.float32)
        
        speech_detected = False  # Track if we've heard any speech yet
        
        for i in range(max_chunks):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except Exception as e:
                print(f"⚠️  Audio read error: {e}")
                break
            
            frames.append(data)
            
            # Write to ring buffer (O(1) operation)
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            chunk_len = len(audio_int16)
            
            # Handle wrap-around in ring buffer
            end_pos = ring_write_pos + chunk_len
            if end_pos <= RING_SIZE:
                ring_buffer[ring_write_pos:end_pos] = audio_int16
            else:
                first_part = RING_SIZE - ring_write_pos
                ring_buffer[ring_write_pos:] = audio_int16[:first_part]
                ring_buffer[:chunk_len - first_part] = audio_int16[first_part:]
            
            ring_write_pos = end_pos % RING_SIZE
            ring_available += chunk_len
            
            # Process VAD in 512-sample chunks from ring buffer
            while ring_available >= VAD_CHUNK:
                # Read from ring buffer (O(1) operation)
                if ring_read_pos + VAD_CHUNK <= RING_SIZE:
                    vad_chunk = ring_buffer[ring_read_pos:ring_read_pos + VAD_CHUNK]
                else:
                    first_part = RING_SIZE - ring_read_pos
                    vad_chunk = np.concatenate([
                        ring_buffer[ring_read_pos:],
                        ring_buffer[:VAD_CHUNK - first_part]
                    ])
                
                ring_read_pos = (ring_read_pos + VAD_CHUNK) % RING_SIZE
                ring_available -= VAD_CHUNK
                
                # Convert to tensor in-place (reuse buffer)
                vad_tensor_buffer[:] = torch.from_numpy(vad_chunk.astype(np.float32) / 32768.0)
                
                # Check for speech
                try:
                    speech_prob = vad_model(vad_tensor_buffer, RATE).item()
                    
                    if speech_prob >= 0.5:
                        speech_detected = True
                        silence_chunks = 0
                    else:
                        silence_chunks += 1
                except Exception:
                    # If VAD fails, just continue recording
                    pass
            
            # Only check for end-of-speech after grace period
            # AND only if we've detected speech at some point
            if i >= grace_chunks and speech_detected and silence_chunks >= silence_threshold:
                print(f"🛑 Speech ended (recorded {len(frames) * CHUNK / RATE:.1f}s)")
                break
            
            # Also stop if we've been in grace period and heard nothing at all
            # (prevents hanging if user doesn't speak)
            if i >= grace_chunks * 2 and not speech_detected:
                print(f"🛑 No speech detected (waited {len(frames) * CHUNK / RATE:.1f}s)")
                break
    else:
        # Fallback to fixed duration
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = stream.read(CHUNK)
            frames.append(data)
    
    stream.stop_stream()
    stream.close()
    
    temp_path = "/tmp/assistant_command.wav"
    with wave.open(temp_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    pa.terminate()
    return temp_path

def transcribe(audio_path):
    """Transcribe audio using GPU offloading with local fallback."""
    print("🎧 Transcribing...")
    return transcription_service.transcribe(audio_path)

def llm_respond_or_tool_call(user_text):
    """Send to LLM, handle tool calls or text response."""
    print(f"🧠 Processing: {user_text}")
    
    # Detect if this needs real-time info (weather, stocks, news, sports, current events)
    realtime_keywords = ['weather', 'temperature', 'forecast', 'stock', 'price', 'market',
                        'news', 'score', 'game', 'playing', 'today', 'tonight', 'current',
                        'right now', 'latest', 'recent', 'who won', 'what time']
    needs_search = any(kw in user_text.lower() for kw in realtime_keywords)
    
    # Detect if this is a smart home command
    home_keywords = ['light', 'lights', 'lamp', 'brightness', 'dim', 'bright', 'turn on',
                   'turn off', 'kitchen', 'family room', 'foyer', 'stairs', 'island',
                   'bluetooth', 'connect', 'disconnect', 'volume', 'music', 'play', 'stop']
    needs_tools = any(kw in user_text.lower() for kw in home_keywords)
    
    # Location context
    location_context = "User is located in Charlotte, NC (zip code 28211)."
    
    if needs_search and not needs_tools:
        # Use Google Search for real-time info
        system_instruction = f"You are Jarvis, a helpful voice assistant. {location_context} Answer concisely in 1-2 sentences."
        google_search_tool = types.Tool(google_search=types.GoogleSearch())
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_text,
            config=types.GenerateContentConfig(
                tools=[google_search_tool],
                system_instruction=system_instruction,
                temperature=0.3
            )
        )
    else:
        # Use function calling for smart home and general questions
        system_instruction = f"""You are Jarvis, a helpful voice assistant. {location_context}

For lighting commands, IMMEDIATELY call control_home_lighting function with NO explanation.
Device IDs: Kitchen Cans=85, Kitchen Island=95, Family Room=204, Foyer=87, Stairs=89.
For ALL lights: use device_id=999 with brightness=100 (ON) or brightness=0 (OFF).

For all other questions, answer concisely in 1-2 sentences max."""
        
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=user_text,
            config=types.GenerateContentConfig(
                tools=GEMINI_TOOLS,
                system_instruction=system_instruction,
                temperature=0.3
            )
        )
    
    if response.candidates and response.candidates[0].content.parts:
        for part in response.candidates[0].content.parts:
            if part.function_call:
                args = dict(part.function_call.args)
                result = dispatch_tool(part.function_call.name, args)
        if any(part.function_call for part in response.candidates[0].content.parts):
            return "Done"
        elif response.text:
            return response.text
    return ""

def speak(text):
    """
    Speak text using Piper TTS with persistent audio stream.
    
    Optimized for Pi5: Uses persistent audio output stream to avoid
    ~50-100ms overhead of creating/destroying streams per utterance.
    """
    if text and piper_voice and tts_audio_output:
        print(f"💬 {text}")
        for audio_chunk in piper_voice.synthesize(text):
            int_data = np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16)
            tts_audio_output.write(int_data)
    elif text and piper_voice:
        # Fallback to per-call stream if persistent output not available
        print(f"💬 {text}")
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
    elif text and not piper_voice:
        print(f"⚠️  Cannot speak: '{text}' - Piper model not loaded")

# === MAIN LOOP ===
def on_press(key):
    if key == keyboard.Key.space:
        try:
            audio_file = record_audio()
            transcript = transcribe(audio_file)
            if transcript:
                print(f"📝 You: {transcript}")
                response = llm_respond_or_tool_call(transcript)
                speak(response)
            else:
                print("❌ No speech detected")
        except Exception as e:
            print(f"❌ Error: {e}")
    elif key == keyboard.Key.esc:
        print("\n👋 Goodbye!")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Push-to-talk voice assistant")
    parser.add_argument(
        "--once",
        action="store_true",
        help="Record a single command and exit (no keyboard listener).",
    )
    args = parser.parse_args()

    if args.once:
        audio_file = record_audio()
        transcript = transcribe(audio_file)
        if transcript:
            print(f"📝 You: {transcript}")
            response = llm_respond_or_tool_call(transcript)
            speak(response)
        else:
            print("❌ No speech detected")
    else:
        from pynput import keyboard

        print("🎙️  Push-to-Talk Assistant Ready")
        print("   Press SPACE to record")
        print("   Press ESC to quit\n")
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()
