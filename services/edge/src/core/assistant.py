
import os
import time
import logging
import json
import io
import wave
import contextlib
import tempfile
import threading
from typing import Optional, Tuple, Any

from google import genai
from google.genai import types

from core.event_bus import (
    EventBus, emit_state_changed, emit_transcript, 
    emit_assistant_text, emit_tool_call, emit_tool_result, emit_error
)
from core.conversation import ConversationMemory, parse_clear_phrases, should_clear_history
from core.tts_preprocessing import preprocess_for_tts
from tools.transcription import create_transcription_service
from tools.registry import GEMINI_TOOLS, dispatch_tool
from adapters.gpu_tts_client import GPUTTSClient

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Assistant:
    """
    Core Assistant Logic (The 'Brain').
    Handles Transcription -> Thinking (LLM) -> Tools -> Speaking (TTS).
    """

    def __init__(self, bus: EventBus):
        self.bus = bus
        
        # 1. Initialize Gemini
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            logger.warning("⚠️ GEMINI_API_KEY not found. LLM features will fail.")
        self.client = genai.Client(api_key=self.api_key)
        self.model_name = os.getenv("MODEL_NAME", "gemini-2.5-flash-lite")
        
        # 2. Transcription Service
        self.transcription = create_transcription_service()
        
        # 3. Memory
        self.conversation_enabled = os.getenv("CONVERSATION_ENABLED", "true").lower() == "true"
        self.memory = None
        if self.conversation_enabled:
            max_turns = int(os.getenv("CONVERSATION_MAX_TURNS", "6"))
            ttl = float(os.getenv("CONVERSATION_TTL_SECONDS", "600.0"))
            self.memory = ConversationMemory(max_turns, ttl)
        
        self.clear_phrases = parse_clear_phrases(os.getenv("CONVERSATION_CLEAR_PHRASES"))
        self.reset_on_tool = os.getenv("CONVERSATION_RESET_ON_TOOL_CALL", "true").lower() == "true"
        
        # 4. TTS (GPU with fallback)
        self.xtts_url = os.getenv("XTTS_SERVER_URL", "http://localhost:5001")
        self.gpu_tts = None
        if os.getenv("USE_GPU_TTS", "true").lower() == "true":
            # Pass piper_voice=None as we might not load local piper in this class
            # The main.py might handle simplified playback, or we handle it here.
            # We'll use GPUTTSClient for synthesis.
            self.gpu_tts = GPUTTSClient(
                server_url=self.xtts_url,
                piper_voice=None,
                piper_sample_rate=22050
            )

    def process_voice_command(self, audio_path: str) -> Tuple[str, Optional[bytes], int]:
        """
        Full pipeline: Audio File -> Transcript -> LLM/Tools -> TTS Audio.
        Returns: (response_text, audio_wav_bytes, latency_ms)
        """
        start_time = time.time()
        
        # 1. Transcribe
        emit_state_changed(self.bus, "listening", "transcribing")
        transcript = self.transcription.transcribe(audio_path)
        
        if not transcript:
            emit_state_changed(self.bus, "transcribing", "idle")
            return "", None, int((time.time() - start_time) * 1000)
            
        emit_transcript(self.bus, transcript, is_final=True)
        
        # 2. Think (LLM & Tools)
        emit_state_changed(self.bus, "transcribing", "thinking")
        response_text = self._process_llm(transcript)
        
        # 3. Speak (TTS)
        audio_bytes = None
        if response_text:
            emit_assistant_text(self.bus, response_text)
            emit_state_changed(self.bus, "thinking", "speaking")
            audio_bytes = self._synthesize_tts(response_text)
            # Note: The caller (main.py) is responsible for playing the audio
            # and setting state back to 'idle' after playback.
        else:
             emit_state_changed(self.bus, "thinking", "idle")

        latency_ms = int((time.time() - start_time) * 1000)
        return response_text, audio_bytes, latency_ms

    def _process_llm(self, user_text: str) -> str:
        """Run the LLM, execute tools, and return final response text."""
        try:
            # Memory Management
            if self.memory:
                self.memory.maybe_expire()
                if should_clear_history(user_text, self.clear_phrases):
                    self.memory.reset()
            
            use_history = self.memory is not None
            
            # Simple keyword check for tool-heavy requests to skip history
            # (Optimization from legacy code)
            tool_keywords = ['light', 'turn on', 'turn off', 'timer', 'volume', 'play', 'stop']
            if any(k in user_text.lower() for k in tool_keywords):
                use_history = False

            contents = self._build_contents(user_text, use_history)
            
            # Location Context (Hardcoded for now, can be env var later)
            location = "User is located in Charlotte, NC (zip code 28211)."
            system_inst = f"You are Computer, a helpful voice assistant. {location} Keep answers short (1-2 sentences)."

            # Configure Tools
            # For real-time info keywords, use Google Search
            search_kw = ['weather', 'stock', 'news', 'who won', 'what time', 'current']
            if any(k in user_text.lower() for k in search_kw) and not any(k in user_text.lower() for k in tool_keywords):
                 tools = [types.Tool(google_search=types.GoogleSearch())]
            else:
                 tools = GEMINI_TOOLS

            # Call Gemini
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=contents,
                config=types.GenerateContentConfig(
                    tools=tools,
                    system_instruction=system_inst,
                    temperature=0.3
                )
            )
            
            # Handle Response / Tool Calls
            final_text = ""
            has_tool = False
            
            if response.candidates and response.candidates[0].content.parts:
                for part in response.candidates[0].content.parts:
                    if part.function_call:
                        has_tool = True
                        t_name = part.function_call.name
                        t_args = dict(part.function_call.args)
                        
                        emit_tool_call(self.bus, t_name, t_args)
                        emit_state_changed(self.bus, "thinking", "executing")
                        
                        try:
                            t_start = time.time()
                            result = dispatch_tool(t_name, t_args)
                            dur = int((time.time() - t_start) * 1000)
                            emit_tool_result(self.bus, t_name, True, str(result), dur)
                        except Exception as e:
                            emit_tool_result(self.bus, t_name, False, str(e))
            
                if has_tool:
                    if self.memory and self.reset_on_tool:
                        self.memory.reset()
                    final_text = "Done" # Simple acknowledgement for actions
                else:
                    final_text = response.text or ""
                    if self.memory and use_history and final_text:
                        self.memory.add("user", user_text)
                        self.memory.add("model", final_text)
            
            return final_text

        except Exception as e:
            logger.error(f"LLM Error: {e}")
            emit_error(self.bus, "LLM_ERROR", str(e))
            return "I'm having trouble thinking right now."

    def _build_contents(self, text: str, use_history: bool):
        if not use_history or not self.memory:
            return text
        
        history = self.memory.get_messages()
        if not history:
            return text
            
        contents = []
        for msg in history:
            contents.append(types.Content(
                role=msg["role"],
                parts=[types.Part(text=msg["text"])]
            ))
        contents.append(types.Content(role="user", parts=[types.Part(text=text)]))
        return contents

    def _synthesize_tts(self, text: str) -> Optional[bytes]:
        """Convert text to WAV bytes using GPU TTS."""
        if not text or not self.gpu_tts:
            return None
            
        try:
            clean = preprocess_for_tts(text)
            # synthesize returns (audio_int16_numpy, sample_rate)
            result = self.gpu_tts.synthesize(clean)
            
            if result:
                data, rate = result
                # Convert numpy int16 array to WAV bytes
                buf = io.BytesIO()
                with wave.open(buf, 'wb') as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)
                    wf.setframerate(rate)
                    wf.writeframes(data.tobytes())
                return buf.getvalue()
        except Exception as e:
            logger.error(f"TTS Error: {e}")
            
        return None
