import os
import time
import logging
import tempfile
import json
import io
import wave
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel
from typing import Optional, Dict, Any, List

# Internal imports (relative to services/compute/src/)
from tools.transcription import create_transcription_service
from tools.registry import GEMINI_TOOLS, dispatch_tool
from core.conversation import ConversationMemory, parse_clear_phrases, should_clear_history
from google import genai
from google.genai import types
from adapters.gpu_tts_client import GPUTTSClient
from core.tts_preprocessing import preprocess_for_tts
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Assistant Compute Server")

# === INITIALIZE ===
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)
MODEL_NAME = os.getenv("MODEL_NAME", "gemini-2.5-flash-lite")

# Transcription
transcription_service = create_transcription_service()

# Memory
CONVERSATION_ENABLED = os.getenv("CONVERSATION_ENABLED", "true").lower() == "true"
conversation_memory = (
    ConversationMemory(
        int(os.getenv("CONVERSATION_MAX_TURNS", "6")), 
        float(os.getenv("CONVERSATION_TTL_SECONDS", "600.0"))
    )
    if CONVERSATION_ENABLED
    else None
)
CONVERSATION_CLEAR_PHRASES = parse_clear_phrases(os.getenv("CONVERSATION_CLEAR_PHRASES"))
CONVERSATION_RESET_ON_TOOL_CALL = (
    os.getenv("CONVERSATION_RESET_ON_TOOL_CALL", "true").lower() == "true"
)

# GPU TTS
XTTS_SERVER_URL = os.getenv("XTTS_SERVER_URL", "http://localhost:5001")
USE_GPU_TTS = os.getenv("USE_GPU_TTS", "true").lower() == "true"
gpu_tts_client = None
if USE_GPU_TTS:
    gpu_tts_client = GPUTTSClient(
        server_url=XTTS_SERVER_URL,
        piper_voice=None, # Piper not needed on compute server? Maybe as backup if XTTS fails
        piper_sample_rate=22050,
    )

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

@app.get("/health")
def health():
    return {"status": "ok", "gpu_tts": USE_GPU_TTS}

@app.post("/process")
async def process_audio(audio: UploadFile = File(...)):
    """
    Process audio: STT -> LLM -> Tool/Response -> TTS
    Returns JSON with transcript, response text, and metadata.
    Note: For Story 1, we return the text; the Edge will handle playback 
    (or we could return audio bytes). 
    The plan said "Compute handles TTS", so we will synthesize and return audio.
    """
    start_time = time.time()
    
    # Save uploaded audio to temp file
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(await audio.read())
        audio_path = tmp.name

    try:
        # 1. Transcribe
        user_command = transcription_service.transcribe(audio_path)
        if not user_command:
            return JSONResponse({"error": "No speech detected", "transcript": ""}, status_code=200)

        logger.info(f"🎤 User: {user_command}")

        # 2. LLM Routing & Processing
        realtime_keywords = ['weather', 'temperature', 'forecast', 'stock', 'price', 'market',
                            'news', 'score', 'game', 'playing', 'today', 'tonight', 'current',
                            'right now', 'latest', 'recent', 'who won', 'what time']
        needs_search = any(kw in user_command.lower() for kw in realtime_keywords)
        
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
        location_context = "User is located in Charlotte, NC (zip code 28211)."
        
        system_instruction = f"You are Computer, a helpful voice assistant. {location_context} Format for natural speech (1-2 sentences)."

        if needs_search and not needs_tools:
            google_search_tool = types.Tool(google_search=types.GoogleSearch())
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=[google_search_tool],
                    system_instruction=system_instruction,
                    temperature=0.3
                )
            )
        else:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=GEMINI_TOOLS,
                    system_instruction=system_instruction,
                    temperature=0.3
                )
            )

        response_text = ""
        has_tool_call = False
        if response.candidates and response.candidates[0].content.parts:
            for part in response.candidates[0].content.parts:
                if part.function_call:
                    has_tool_call = True
                    tool_name = part.function_call.name
                    args = dict(part.function_call.args)
                    logger.info(f"✅ Executing tool: {tool_name}")
                    result = dispatch_tool(tool_name, args)
            
            if has_tool_call:
                if conversation_memory and CONVERSATION_RESET_ON_TOOL_CALL:
                    conversation_memory.reset()
                response_text = "Done"
            else:
                response_text = response.text or ""
                if conversation_memory and use_history and response_text:
                    conversation_memory.add("user", user_command)
                    conversation_memory.add("model", response_text)

        logger.info(f"💬 Assistant: {response_text}")

        # 3. TTS Synthesis (Optional for Story 1, but requested)
        # We'll return the text and the audio in a multi-part response or just JSON for now.
        # Simplest: Return JSON including transcript and response. Edge will call TTS if it wants.
        # But wait, plan says "Compute handles TTS".
        # Let's return the audio bytes if possible.
        
        audio_payload = None
        if response_text and gpu_tts_client:
            # Preprocess
            clean_text = preprocess_for_tts(response_text)
            # Synthesize (non-streaming for simplicity in HTTP response)
            tts_result = gpu_tts_client.synthesize(clean_text)
            if tts_result:
                audio_data, sample_rate = tts_result
                # Convert to WAV bytes
                buffer = io.BytesIO()
                with wave.open(buffer, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2) # 16-bit
                    wf.setframerate(sample_rate)
                    wf.writeframes(audio_data.tobytes())
                audio_payload = buffer.getvalue()

        latency_ms = int((time.time() - start_time) * 1000)
        
        # If we have audio, we can return it. But we also want to return the text.
        # For MVP, let's return a JSON with everything, and base64 encoded audio? 
        # Or just return JSON and have a separate /tts endpoint?
        # Let's do JSON with base64 for now to keep it one round trip.
        
        import base64
        return {
            "transcript": user_command,
            "response_text": response_text,
            "audio_base64": base64.b64encode(audio_payload).decode('utf-8') if audio_payload else None,
            "latency_ms": latency_ms
        }

    except Exception as e:
        logger.error(f"Error processing command: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
