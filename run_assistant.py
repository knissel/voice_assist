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
from typing import Optional
from google import genai
from google.genai import types
from tools.registry import GEMINI_TOOLS, dispatch_tool
from dotenv import load_dotenv

load_dotenv()

# === CONFIGURATION ===
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WHISPER_PATH = os.path.join(REPO_ROOT, "whisper.cpp", "build", "bin", "whisper-cli")
DEFAULT_MODEL_PATH = os.path.join(REPO_ROOT, "whisper.cpp", "models", "ggml-tiny.bin")

def _resolve_path(env_value: Optional[str], default_path: str) -> str:
    """Prefer env path when it exists; otherwise fall back to repo default."""
    if env_value and os.path.exists(env_value):
        return env_value
    return default_path

WHISPER_PATH = _resolve_path(os.getenv("WHISPER_PATH"), DEFAULT_WHISPER_PATH)
MODEL_PATH = _resolve_path(os.getenv("MODEL_PATH"), DEFAULT_MODEL_PATH)
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

# === PIPELINE FUNCTIONS ===
def record_audio():
    """Record audio from microphone."""
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
    """Transcribe audio using Whisper.cpp."""
    print("🎧 Transcribing...")
    try:
        result = subprocess.run(
            [WHISPER_PATH, "-m", MODEL_PATH, "-f", audio_path, "-nt"],
            capture_output=True,
            text=True,
            timeout=30
        )
        text = result.stdout.strip()
        if text:
            lines = text.split('\n')
            text = next((line.strip() for line in reversed(lines) if line.strip() and not line.startswith('[')), "")
        return text
    except subprocess.TimeoutExpired:
        print("⚠️  Transcription timed out")
        return ""
    except Exception as e:
        print(f"⚠️  Transcription error: {e}")
        return ""

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
    """Speak text using Gemini 2.5 Flash TTS with direct playback."""
    if text:
        print(f"💬 {text}")
        response = client.models.generate_content(
            model="gemini-2.5-flash-preview-tts",
            contents=text,
            config=types.GenerateContentConfig(
                response_modalities=["AUDIO"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name="Kore"
                        )
                    )
                )
            )
        )
        
        audio_data = response.candidates[0].content.parts[0].inline_data.data
        
        p = pyaudio.PyAudio()
        stream = p.open(format=pyaudio.paInt16,
                       channels=1,
                       rate=24000,
                       output=True)
        stream.write(audio_data)
        stream.stop_stream()
        stream.close()
        p.terminate()

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
