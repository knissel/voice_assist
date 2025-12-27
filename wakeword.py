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
from dotenv import load_dotenv

load_dotenv()

# === Pi 5 Optimizations ===
# Limit torch threads to reduce CPU contention on Pi
torch.set_num_threads(2)
torch.set_grad_enabled(False)  # Disable autograd (not needed for inference)

# Initialize transcription service with GPU offloading and fallback
transcription_service = create_transcription_service()

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

# === Worker Thread for Non-Blocking Processing ===
# This allows wakeword detection to continue while processing commands

class AssistantWorker:
    """Background worker that processes commands without blocking wakeword detection."""
    
    def __init__(self):
        self.command_queue = queue.Queue()
        self.is_processing = False
        self._running = False
        self._thread = None
    
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
        # 1. Transcribe
        print("🎧 Transcribing...")
        user_command = transcription_service.transcribe(audio_path)
        
        if not user_command:
            print("❌ No speech detected")
            return
        
        print(f"📝 You said: {user_command}")
        
        # 2. Send to Gemini
        print("🧠 Consulting Gemini Flash...")
        
        try:
            system_instruction = "You are Jarvis. For lighting commands, IMMEDIATELY call control_home_lighting function with NO explanation. Kitchen Cans=85, Kitchen Island=95, Family Room=204, Foyer=87, Stairs=89. For non-lighting questions, answer in 1-2 sentences max."
            
            model_name = os.getenv("MODEL_NAME", "gemini-flash-lite-latest")
            response = client.models.generate_content(
                model=model_name,
                contents=user_command,
                config=types.GenerateContentConfig(
                    tools=GEMINI_TOOLS,
                    system_instruction=system_instruction,
                    temperature=0.1
                )
            )
            
            # 3. Execute Tool Call or Respond
            if response.candidates and response.candidates[0].content.parts:
                has_tool_call = False
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        has_tool_call = True
                        print(f"✅ Action: {part.function_call.name}")
                        args = dict(part.function_call.args)
                        result = dispatch_tool(part.function_call.name, args)
                
                if has_tool_call:
                    speak_tts("Done")
                elif response.text:
                    print(f"💬 Jarvis: {response.text}")
                    speak_tts(response.text)
        
        except Exception as e:
            print(f"❌ Gemini API failed: {e}")
            speak_tts("I'm having trouble connecting to my brain.")

# Global worker instance
assistant_worker = AssistantWorker()

def capture_audio_only():
    """
    Record audio with VAD and return the path to the saved file.
    This runs in the main thread to keep audio capture real-time safe.
    Returns the path to the audio file, or None if recording failed.
    """
    print("🎤 Jarvis is listening...")

    CHUNK = 1024
    RATE = 16000
    MAX_RECORD_SECONDS = 10
    MIN_RECORD_SECONDS = 0.5
    SILENCE_DURATION = 1.5
    
    frames = []
    temp_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    if vad_available:
        VAD_CHUNK = 512
        silence_chunks = 0
        silence_threshold = int(SILENCE_DURATION * RATE / VAD_CHUNK)
        max_chunks = int(MAX_RECORD_SECONDS * RATE / CHUNK)
        min_chunks = int(MIN_RECORD_SECONDS * RATE / CHUNK)
        
        vad_buffer = np.array([], dtype=np.int16)
        
        for i in range(max_chunks):
            try:
                data = temp_stream.read(CHUNK, exception_on_overflow=False)
            except Exception as e:
                print(f"⚠️  Audio read error: {e}")
                break
            
            frames.append(data)
            
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            vad_buffer = np.concatenate([vad_buffer, audio_int16])
            
            while len(vad_buffer) >= VAD_CHUNK:
                vad_chunk = vad_buffer[:VAD_CHUNK]
                vad_buffer = vad_buffer[VAD_CHUNK:]
                
                audio_float32 = vad_chunk.astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(audio_float32)
                
                try:
                    speech_prob = vad_model(audio_tensor, RATE).item()
                    if speech_prob < 0.5:
                        silence_chunks += 1
                    else:
                        silence_chunks = 0
                except Exception:
                    pass
            
            if i >= min_chunks and silence_chunks >= silence_threshold:
                print(f"🛑 Speech ended (recorded {len(frames) * CHUNK / RATE:.1f}s)")
                break
    else:
        RECORD_SECONDS = 4
        for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
            data = temp_stream.read(CHUNK)
            frames.append(data)
    
    temp_stream.stop_stream()
    temp_stream.close()
    
    if not frames:
        return None

    # Save to temp file with unique timestamp
    temp_audio_path = f"/tmp/jarvis_command_{int(time.time())}.wav"
    with wave.open(temp_audio_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    return temp_audio_path

def capture_and_process():
    """
    Capture audio and submit to worker thread for async processing.
    Audio capture happens in main thread (real-time safe).
    Processing happens in background (non-blocking).
    """
    audio_path = capture_audio_only()
    if audio_path:
        assistant_worker.submit(audio_path)

def speak_tts(text):
    """Speak text using Piper TTS with direct audio playback."""
    if text and piper_voice:
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

# 1. Setup the Engine
# 'keywords' can be standard ones like ['jarvis', 'computer']
# or a path to your custom 'Gemini.ppn' file.
access_key = os.getenv("PORCUPINE_ACCESS_KEY")
porcupine = pvporcupine.create(
    access_key=access_key,
    keywords=['americano', 'computer'] 
)

# 2. Setup the Microphone Stream
pa = pyaudio.PyAudio()
audio_stream = pa.open(
    rate=porcupine.sample_rate,
    channels=1,
    format=pyaudio.paInt16,
    input=True,
    frames_per_buffer=porcupine.frame_length
)

print("👂 Listening for wake word... (Press Ctrl+C to exit)")

# Start the worker thread
assistant_worker.start()

# 3. The "Infinite Ear" Loop
try:
    while True:
        # Read a tiny chunk of audio (approx 0.03 seconds worth)
        pcm = audio_stream.read(porcupine.frame_length, exception_on_overflow=False)
        
        # Unpack bits to match Porcupine's expected format
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

        # Process the chunk
        keyword_index = porcupine.process(pcm)

        # 4. Wake Word Detected!
        if keyword_index >= 0:
            # Skip if already processing a command
            if assistant_worker.is_processing:
                print("⚠️  Still processing previous command...")
                continue
            
            print("🔔 Wake word detected!")
            
            # Capture audio (blocking, but fast ~1-10s)
            # Then submit to worker for async processing
            capture_and_process()
            
            # Main loop immediately returns to listening for wake word
            # Worker thread handles transcription/LLM/TTS in background

except KeyboardInterrupt:
    print("\n👋 Shutting down...")
    assistant_worker.stop()
    if porcupine is not None:
        porcupine.delete()
    if audio_stream is not None:
        audio_stream.close()
    if pa is not None:
        pa.terminate()
