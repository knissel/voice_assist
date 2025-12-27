import pyaudio
import struct
import pvporcupine
import wave
import subprocess
import os
import shutil
from google import genai
from google.genai import types
from tools.registry import GEMINI_TOOLS, dispatch_tool
from dotenv import load_dotenv

load_dotenv()

# Initialize Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def capture_and_process():
    # 1. Feedback to user (Optional: Play a local 'ding' file here)
    print("🎤 Jarvis is listening...")

    # 2. Record the command (4 seconds is usually enough for a home command)
    RECORD_SECONDS = 4
    CHUNK = 1024
    RATE = 16000 # Gemini handles 16k perfectly
    
    frames = []
    # We use the existing 'pa' and 'audio_stream' if shared, 
    # but for simplicity, let's capture a fresh burst:
    temp_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = temp_stream.read(CHUNK)
        frames.append(data)
    
    temp_stream.stop_stream()
    temp_stream.close()

    # 3. Save audio to temporary WAV file for Whisper
    temp_audio_path = "/tmp/jarvis_command.wav"
    with wave.open(temp_audio_path, 'wb') as wf:
        wf.setnchannels(1)
        wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
    
    # 4. Transcribe with Whisper.cpp
    print("🎧 Transcribing with Whisper...")
    repo_root = os.path.dirname(os.path.abspath(__file__))
    default_whisper_path = os.path.join(repo_root, "whisper.cpp", "build", "bin", "whisper-cli")
    default_model_path = os.path.join(repo_root, "whisper.cpp", "models", "ggml-tiny.bin")

    def resolve_path(env_value: str | None, default_path: str) -> str:
        if env_value and os.path.exists(env_value):
            return env_value
        return default_path

    whisper_path = resolve_path(os.getenv("WHISPER_PATH"), default_whisper_path)
    model_path = resolve_path(os.getenv("MODEL_PATH"), default_model_path)
    
    result = subprocess.run(
        [whisper_path, "-m", model_path, "-f", temp_audio_path, "-nt"],
        capture_output=True,
        text=True
    )
    
    # Extract transcribed text from output
    user_command = result.stdout.strip()
    if user_command:
        # Clean up the output - whisper sometimes adds timestamps
        lines = user_command.split('\n')
        # Get the last non-empty line which is usually the transcription
        user_command = next((line.strip() for line in reversed(lines) if line.strip() and not line.startswith('[')), "")
    
    if not user_command:
        print("❌ No speech detected")
        return
    
    print(f"📝 You said: {user_command}")
    
    # 5. Send to Gemini
    print("🧠 Consulting Gemini Flash...")
    
    # Detect if this is a lighting command
    lighting_keywords = ['light', 'lights', 'lamp', 'brightness', 'dim', 'bright', 'kitchen', 'family room', 'foyer', 'stairs', 'island']
    is_lighting_command = any(keyword in user_command.lower() for keyword in lighting_keywords)
    
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
        
        # 6. Execute Tool Call or Respond
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
        error_msg = "I'm having trouble connecting to my brain."
        speak_tts(error_msg)

def speak_tts(text):
    """Speak text using Gemini 2.5 Flash TTS with direct playback."""
    if text:
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

print("Listening... (Press Ctrl+C to exit)")

# 3. The "Infinite Ear" Loop
try:
    while True:
        # Read a tiny chunk of audio (approx 0.03 seconds worth)
        pcm = audio_stream.read(porcupine.frame_length)
        
        # Unpack bits to match Porcupine's expected format
        pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)

        # Process the chunk
        keyword_index = porcupine.process(pcm)

        # 4. Wake Word Detected!
        if keyword_index >= 0:
            print("Wake word detected! (Robot)")
            capture_and_process()
            print("👂 Listening for wake word again...")
            # --- ACTION TAKEN HERE ---
            # 1. Pause any music playing
            # 2. Play a 'ding' sound (feedback)
            # 3. Start recording the NEXT 5 seconds of audio for Gemini
            # 4. Send that new audio to Gemini Flash 3
            # -------------------------

except KeyboardInterrupt:
    if porcupine is not None:
        porcupine.delete()
    if audio_stream is not None:
        audio_stream.close()
    if pa is not None:
        pa.terminate()
