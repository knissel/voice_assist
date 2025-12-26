#!/usr/bin/env python3
"""
Minimal push-to-talk voice assistant runner.
Press SPACE to record -> STT -> LLM -> TTS -> play
Press ESC to quit.
"""
import pyaudio
import wave
import subprocess
import json
import os
from pynput import keyboard
import pyttsx3
from google import genai
from google.genai import types
from tools.registry import GEMINI_TOOLS, dispatch_tool
from dotenv import load_dotenv

load_dotenv()

# === CONFIGURATION ===
WHISPER_PATH = os.getenv("WHISPER_PATH", "/Users/kennynissel/voice_assist/whisper.cpp/build/bin/whisper-cli")
MODEL_PATH = os.getenv("MODEL_PATH", "/Users/kennynissel/voice_assist/whisper.cpp/models/ggml-tiny.bin")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
RECORD_SECONDS = 4
RATE = 16000
CHUNK = 1024

# === INITIALIZE ===
tts_engine = pyttsx3.init()
voices = tts_engine.getProperty('voices')
for voice in voices:
    if 'daniel' in voice.name.lower() or 'alex' in voice.name.lower():
        tts_engine.setProperty('voice', voice.id)
        break
tts_engine.setProperty('rate', 165)
tts_engine.setProperty('volume', 0.95)

client = genai.Client(api_key=GEMINI_API_KEY)

# === PIPELINE FUNCTIONS ===
def record_audio():
    """Record audio from microphone."""
    print("🎤 Recording...")
    pa = pyaudio.PyAudio()
    stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
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
    result = subprocess.run([WHISPER_PATH, "-m", MODEL_PATH, "-f", audio_path, "-nt"], capture_output=True, text=True)
    text = result.stdout.strip()
    if text:
        lines = text.split('\n')
        text = next((line.strip() for line in reversed(lines) if line.strip() and not line.startswith('[')), "")
    return text

def llm_respond_or_tool_call(user_text):
    """Send to LLM, handle tool calls or text response."""
    print(f"🧠 Processing: {user_text}")
    lighting_keywords = ['light', 'lights', 'lamp', 'brightness', 'dim', 'bright', 'kitchen', 'family room', 'foyer', 'stairs', 'island']
    is_lighting = any(kw in user_text.lower() for kw in lighting_keywords)
    
    system_instruction = "You are Jarvis. For lighting commands, IMMEDIATELY call control_home_lighting function with NO explanation. Kitchen Cans=85, Kitchen Island=95, Family Room=204, Foyer=87, Stairs=89. For non-lighting questions, answer in 1-2 sentences max."
    
    response = client.models.generate_content(
        model="gemini-2.0-flash-exp",
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
    """Speak text using TTS."""
    if text:
        print(f"💬 {text}")
        tts_engine.say(text)
        tts_engine.runAndWait()

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
    print("🎙️  Push-to-Talk Assistant Ready")
    print("   Press SPACE to record")
    print("   Press ESC to quit\n")
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()
