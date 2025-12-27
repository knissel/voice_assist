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

# === PIPELINE FUNCTIONS ===
def record_audio():
    """Record audio from microphone with VAD."""
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
        # VAD-based recording
        MAX_RECORD_SECONDS = 10
        MIN_RECORD_SECONDS = 0.5
        SILENCE_DURATION = 1.5
        VAD_CHUNK = 512  # Silero VAD requires exactly 512 samples for 16kHz
        
        silence_chunks = 0
        silence_threshold = int(SILENCE_DURATION * RATE / VAD_CHUNK)
        max_chunks = int(MAX_RECORD_SECONDS * RATE / CHUNK)
        min_chunks = int(MIN_RECORD_SECONDS * RATE / CHUNK)
        
        vad_buffer = np.array([], dtype=np.int16)
        
        for i in range(max_chunks):
            try:
                data = stream.read(CHUNK, exception_on_overflow=False)
            except Exception as e:
                print(f"⚠️  Audio read error: {e}")
                break
            
            frames.append(data)
            
            # Add to VAD buffer
            audio_int16 = np.frombuffer(data, dtype=np.int16)
            vad_buffer = np.concatenate([vad_buffer, audio_int16])
            
            # Process VAD in 512-sample chunks
            while len(vad_buffer) >= VAD_CHUNK:
                vad_chunk = vad_buffer[:VAD_CHUNK]
                vad_buffer = vad_buffer[VAD_CHUNK:]
                
                # Convert to tensor for VAD
                audio_float32 = vad_chunk.astype(np.float32) / 32768.0
                audio_tensor = torch.from_numpy(audio_float32)
                
                # Check for speech
                try:
                    speech_prob = vad_model(audio_tensor, RATE).item()
                    
                    if speech_prob < 0.5:
                        silence_chunks += 1
                    else:
                        silence_chunks = 0
                except Exception:
                    # If VAD fails, just continue recording
                    pass
            
            # Stop if we've had enough silence after minimum recording time
            if i >= min_chunks and silence_chunks >= silence_threshold:
                print(f"🛑 Speech ended (recorded {len(frames) * CHUNK / RATE:.1f}s)")
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
    lighting_keywords = ['light', 'lights', 'lamp', 'brightness', 'dim', 'bright', 'kitchen', 'family room', 'foyer', 'stairs', 'island']
    is_lighting = any(kw in user_text.lower() for kw in lighting_keywords)
    
    system_instruction = "You are Jarvis. For lighting commands, IMMEDIATELY call control_home_lighting function with NO explanation. Kitchen Cans=85, Kitchen Island=95, Family Room=204, Foyer=87, Stairs=89. For non-lighting questions, answer in 1-2 sentences max."
    
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_text,
        config=types.GenerateContentConfig(
            tools=GEMINI_TOOLS,
            system_instruction=system_instruction,
            temperature=0.1
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
    """Speak text using Piper TTS with direct audio playback."""
    if text and piper_voice:
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
