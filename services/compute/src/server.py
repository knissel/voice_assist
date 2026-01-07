import asyncio
import base64
import contextlib
import os
import time
import logging
import tempfile
import json
import io
import wave
import uuid
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException, WebSocket, WebSocketDisconnect
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

WS_PARTIAL_INTERVAL_MS = int(os.getenv("WS_PARTIAL_INTERVAL_MS", "500"))
WS_PARTIAL_MIN_AUDIO_MS = int(os.getenv("WS_PARTIAL_MIN_AUDIO_MS", "300"))
WS_PARTIAL_MAX_AUDIO_MS = int(os.getenv("WS_PARTIAL_MAX_AUDIO_MS", "8000"))

def _audio_duration_ms(byte_len: int, sample_rate: int, sample_width: int, channels: int) -> int:
    bytes_per_second = sample_rate * sample_width * channels
    if bytes_per_second <= 0:
        return 0
    return int((byte_len / bytes_per_second) * 1000)

def _transcribe_audio_bytes(audio_bytes: bytes, sample_rate: int, sample_width: int, channels: int) -> str:
    if not audio_bytes:
        return ""
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            temp_path = tmp.name
        with wave.open(temp_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sample_width)
            wf.setframerate(sample_rate)
            wf.writeframes(audio_bytes)
        return transcription_service.transcribe(temp_path)
    finally:
        if temp_path and os.path.exists(temp_path):
            with contextlib.suppress(OSError):
                os.remove(temp_path)

def _process_text(user_command: str) -> tuple[str, Optional[bytes]]:
    if not user_command:
        return "", None

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
                dispatch_tool(tool_name, args)
        
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

    audio_payload = None
    if response_text and gpu_tts_client:
        clean_text = preprocess_for_tts(response_text)
        tts_result = gpu_tts_client.synthesize(clean_text)
        if tts_result:
            audio_data, sample_rate = tts_result
            buffer = io.BytesIO()
            with wave.open(buffer, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data.tobytes())
            audio_payload = buffer.getvalue()

    return response_text, audio_payload

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

        response_text, audio_payload = _process_text(user_command)

        latency_ms = int((time.time() - start_time) * 1000)
        
        # If we have audio, we can return it. But we also want to return the text.
        # For MVP, let's return a JSON with everything, and base64 encoded audio? 
        # Or just return JSON and have a separate /tts endpoint?
        # Let's do JSON with base64 for now to keep it one round trip.
        
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

@app.websocket("/ws/audio")
async def websocket_audio_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = ""
    sample_rate = 16000
    sample_width = 2
    channels = 1
    chunk_ms = None
    audio_buffer = bytearray()
    last_partial_time = 0.0
    last_partial_text = ""
    partial_task = None
    partial_lock = asyncio.Lock()
    stop_received_at = None

    async def send_json(payload: Dict[str, Any]) -> None:
        await websocket.send_text(json.dumps(payload))

    async def send_partial(audio_bytes: bytes, audio_ms: int) -> None:
        nonlocal last_partial_text
        start = time.time()
        text = await asyncio.to_thread(
            _transcribe_audio_bytes, audio_bytes, sample_rate, sample_width, channels
        )
        latency_ms = int((time.time() - start) * 1000)
        if text and text != last_partial_text:
            last_partial_text = text
            await send_json({
                "type": "partial_transcript",
                "session_id": session_id,
                "text": text,
                "audio_ms": audio_ms,
                "latency_ms": latency_ms
            })

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                break

            if "text" in message:
                try:
                    data = json.loads(message["text"])
                except json.JSONDecodeError:
                    await send_json({"type": "error", "error": "invalid_json"})
                    continue

                msg_type = data.get("type")
                if msg_type == "start":
                    session_id = data.get("session_id") or uuid.uuid4().hex[:8]
                    sample_rate = int(data.get("sample_rate", sample_rate))
                    sample_width = int(data.get("sample_width", sample_width))
                    channels = int(data.get("channels", channels))
                    chunk_ms = data.get("chunk_ms")
                    audio_buffer.clear()
                    last_partial_time = 0.0
                    last_partial_text = ""
                    await send_json({
                        "type": "ready",
                        "session_id": session_id,
                        "sample_rate": sample_rate,
                        "sample_width": sample_width,
                        "channels": channels,
                        "chunk_ms": chunk_ms
                    })
                elif msg_type == "stop":
                    stop_received_at = time.time()
                    if partial_task and not partial_task.done():
                        partial_task.cancel()
                        with contextlib.suppress(asyncio.CancelledError):
                            await partial_task

                    final_start = time.time()
                    final_text = await asyncio.to_thread(
                        _transcribe_audio_bytes, bytes(audio_buffer), sample_rate, sample_width, channels
                    )
                    if final_text:
                        logger.info(f"🎤 User: {final_text}")
                    transcribe_ms = int((time.time() - final_start) * 1000)
                    audio_ms = _audio_duration_ms(len(audio_buffer), sample_rate, sample_width, channels)
                    end_to_final_ms = (
                        int((time.time() - stop_received_at) * 1000)
                        if stop_received_at else None
                    )
                    await send_json({
                        "type": "final_transcript",
                        "session_id": session_id,
                        "text": final_text,
                        "audio_ms": audio_ms,
                        "latency_ms": transcribe_ms,
                        "end_to_final_ms": end_to_final_ms
                    })

                    if final_text:
                        response_start = time.time()
                        response_text, audio_payload = await asyncio.to_thread(_process_text, final_text)
                        response_latency_ms = int((time.time() - response_start) * 1000)
                        await send_json({
                            "type": "assistant_response",
                            "session_id": session_id,
                            "response_text": response_text,
                            "audio_base64": base64.b64encode(audio_payload).decode("utf-8")
                            if audio_payload else None,
                            "latency_ms": response_latency_ms
                        })
                    break
                elif msg_type == "ping":
                    await send_json({"type": "pong", "session_id": session_id})
                else:
                    await send_json({"type": "error", "error": "unknown_message_type"})

            elif "bytes" in message:
                if not session_id:
                    await send_json({"type": "error", "error": "missing_start"})
                    continue

                audio_buffer.extend(message["bytes"])
                audio_ms = _audio_duration_ms(len(audio_buffer), sample_rate, sample_width, channels)
                now = time.time()

                if audio_ms >= WS_PARTIAL_MIN_AUDIO_MS and (now - last_partial_time) * 1000 >= WS_PARTIAL_INTERVAL_MS:
                    last_partial_time = now
                    max_bytes = int((WS_PARTIAL_MAX_AUDIO_MS / 1000.0) * sample_rate * sample_width * channels)
                    snapshot = bytes(audio_buffer[-max_bytes:]) if max_bytes > 0 else bytes(audio_buffer)
                    if partial_task is None or partial_task.done():
                        partial_task = asyncio.create_task(send_partial(snapshot, audio_ms))

    except WebSocketDisconnect:
        return
    except Exception as exc:
        logger.error(f"WebSocket audio error: {exc}")
        with contextlib.suppress(Exception):
            await send_json({"type": "error", "error": "server_error"})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
