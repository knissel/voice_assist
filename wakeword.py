import pyaudio
import struct
import pvporcupine
import wave
import subprocess
import os
import shutil
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

# Initialize transcription service with GPU offloading and fallback
transcription_service = create_transcription_service()

# Initialize Gemini client
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

# Initialize Silero VAD
try:
    vad_model, utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
    (get_speech_timestamps, _, read_audio, _, _) = utils
    vad_available = True
    print("✅ VAD initialized successfully")
except Exception as e:
    print(f"⚠️  VAD initialization failed: {e}. Using fixed recording duration.")
    vad_available = False

# Initialize Piper TTS
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_PIPER_MODEL = os.path.join(REPO_ROOT, "piper_models", "en_US-lessac-medium.onnx")
PIPER_MODEL = os.getenv("PIPER_MODEL", DEFAULT_PIPER_MODEL)
piper_voice = None
if os.path.exists(PIPER_MODEL):
    piper_voice = PiperVoice.load(PIPER_MODEL)
else:
    print(f"⚠️  Piper model not found at {PIPER_MODEL}. TTS will not work until model is downloaded.")

def capture_and_process():
    # 1. Feedback to user (Optional: Play a local 'ding' file here)
    print("🎤 Jarvis is listening...")

    # 2. Record with VAD or fixed duration
    CHUNK = 1024
    RATE = 16000
    MAX_RECORD_SECONDS = 10  # Maximum recording time
    MIN_RECORD_SECONDS = 0.5  # Minimum recording time
    SILENCE_DURATION = 1.5  # Seconds of silence to stop recording
    
    frames = []
    temp_stream = pa.open(format=pyaudio.paInt16, channels=1, rate=RATE, input=True, frames_per_buffer=CHUNK)
    
    if vad_available:
        # VAD-based recording
        VAD_CHUNK = 512  # Silero VAD requires exactly 512 samples for 16kHz
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
                    
                    if speech_prob < 0.5:  # No speech detected
                        silence_chunks += 1
                    else:
                        silence_chunks = 0  # Reset silence counter
                except Exception:
                    # If VAD fails, just continue recording
                    pass
            
            # Stop if we've had enough silence after minimum recording time
            if i >= min_chunks and silence_chunks >= silence_threshold:
                print(f"🛑 Speech ended (recorded {len(frames) * CHUNK / RATE:.1f}s)")
                break
    else:
        # Fallback to fixed duration
        RECORD_SECONDS = 4
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
    
    # 4. Transcribe with GPU offloading (fallback to local if unavailable)
    print("🎧 Transcribing...")
    user_command = transcription_service.transcribe(temp_audio_path)
    
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
