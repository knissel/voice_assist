
import os
import time
import logging
import json
import io
import wave
import contextlib
import tempfile
import threading
import requests
import numpy as np
import re
from typing import Optional, Tuple, Any, Dict, Callable

from google import genai
from google.genai import types

from core.event_bus import (
    EventBus, emit_state_changed, emit_transcript, 
    emit_assistant_text, emit_tool_call, emit_tool_result, emit_error
)
from core.conversation import ConversationMemory, parse_clear_phrases, should_clear_history
from core.runtime_mode import EDGE_MODE_LOCAL, resolve_edge_mode
from core.tts_preprocessing import preprocess_for_tts
from tools.transcription import create_transcription_service
from tools.registry import GEMINI_TOOLS, dispatch_tool
from adapters.gpu_tts_client import GPUTTSClient

try:
    from piper.voice import PiperVoice
except ImportError:
    PiperVoice = None
try:
    from pocket_tts import TTSModel
except ImportError:
    TTSModel = None

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
        self.edge_mode = resolve_edge_mode()
        
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

        # Wakeword phrases to strip from transcript before LLM
        wakeword_raw = os.getenv("WAKEWORD_PHRASES", "hey jarvis")
        self.wakeword_phrases = [p.strip().lower() for p in wakeword_raw.split(",") if p.strip()]
        

        # 4. TTS Configuration
        # Options: "gpu" (XTTS/VoxCPM via 5090), "local" (Piper), "pocket" (Pocket TTS)
        self.tts_provider = os.getenv("TTS_PROVIDER", "gpu").lower()
        if self.edge_mode == EDGE_MODE_LOCAL and self.tts_provider == "gpu":
            raise RuntimeError(
                "EDGE_MODE=local cannot use TTS_PROVIDER=gpu. Use TTS_PROVIDER=pocket or TTS_PROVIDER=local."
            )
        
        # GPU TTS
        self.xtts_url = os.getenv("XTTS_SERVER_URL", "http://localhost:5001")
        self.gpu_tts = None
        if self.tts_provider == "gpu":
            self.gpu_tts = GPUTTSClient(
                server_url=self.xtts_url,
                piper_voice=None,
                piper_sample_rate=22050
            )

        # Local TTS (Piper)
        self.piper_voice = None
        if self.tts_provider == "local":
            if PiperVoice:
                model_path = os.getenv("PIPER_MODEL_PATH")
                config_path = f"{model_path}.json" if model_path else None
                if model_path and os.path.exists(model_path):
                    try:
                        self.piper_voice = PiperVoice.load(model_path, config_path=config_path)
                        logger.info(f"Loaded Piper model: {model_path}")
                    except Exception as e:
                        logger.error(f"Failed to load Piper model: {e}")
                else:
                    logger.warning("PIPER_MODEL_PATH not set or invalid. Local TTS will fail.")
            else:
                 logger.warning("piper-tts not installed or load failed.")

        # Local TTS (Pocket TTS)
        self.pocket_tts_model = None
        self.pocket_tts_voice_state = None
        self.pocket_streaming = os.getenv("POCKET_TTS_STREAMING", "true").lower() == "true"
        if self.tts_provider == "pocket":
            if TTSModel:
                try:
                    variant = os.getenv("POCKET_TTS_VARIANT", "b6369a24")
                    temperature = float(os.getenv("POCKET_TTS_TEMPERATURE", "0.7"))
                    lsd_decode_steps = int(os.getenv("POCKET_TTS_LSD_DECODE_STEPS", "1"))
                    eos_threshold = float(os.getenv("POCKET_TTS_EOS_THRESHOLD", "-4.0"))
                    noise_clamp_raw = os.getenv("POCKET_TTS_NOISE_CLAMP", "").strip().lower()
                    noise_clamp = None if noise_clamp_raw in {"", "none", "null"} else float(noise_clamp_raw)
                    voice_prompt = os.getenv(
                        "POCKET_TTS_VOICE",
                        "alba"
                    )
                    self.pocket_tts_model = TTSModel.load_model(
                        variant=variant,
                        temp=temperature,
                        lsd_decode_steps=lsd_decode_steps,
                        noise_clamp=noise_clamp,
                        eos_threshold=eos_threshold
                    )
                    self.pocket_tts_voice_state = self.pocket_tts_model.get_state_for_audio_prompt(
                        voice_prompt
                    )
                    logger.info(
                        "Loaded Pocket TTS model (variant=%s, voice=%s)",
                        variant,
                        voice_prompt
                    )
                except Exception as e:
                    logger.error(f"Failed to load Pocket TTS model: {e}")
            else:
                logger.warning("pocket-tts not installed or load failed.")

        # 5. TTS Response Cache (pre-baked phrases)
        self.tts_cache_enabled = os.getenv("TTS_CACHE_ENABLED", "true").lower() == "true"
        self.tts_prebaked_phrases = self._parse_prebaked_phrases(
            os.getenv("TTS_PREBAKED_PHRASES", "done,ok,okay")
        )
        self.tts_prebake_on_start = os.getenv("TTS_PREBAKE_ON_START", "false").lower() == "true"
        self._tts_cache: Dict[str, bytes] = {}
        self._tts_cache_lock = threading.Lock()
        self._tts_prebaked_keys = {self._tts_cache_key(p) for p in self.tts_prebaked_phrases}
        if self.tts_cache_enabled and self.tts_prebake_on_start and self._tts_prebaked_keys:
            threading.Thread(target=self._warm_tts_cache, daemon=True).start()

        # 6. Local LLM Configuration (for Hybrid Routing)
        self.local_llm_url = os.getenv("LOCAL_LLM_URL", "http://localhost:8080/v1")
        self.use_local_llm = os.getenv("USE_LOCAL_LLM", "false").lower() == "true"
        self._log_runtime_routing()

    def _log_runtime_routing(self) -> None:
        if self.edge_mode == EDGE_MODE_LOCAL:
            stt_route = "local"
        elif getattr(self.transcription, "remote_url", None):
            stt_route = "remote->local fallback"
        else:
            stt_route = "local"

        if self.tts_provider == "gpu":
            tts_route = "gpu(remote XTTS)"
        elif self.tts_provider == "pocket":
            tts_route = "pocket(local)"
        elif self.tts_provider == "local":
            tts_route = "piper(local)"
        else:
            tts_route = self.tts_provider

        logger.info("🚦 EDGE_MODE=%s | STT=%s | TTS=%s", self.edge_mode, stt_route, tts_route)

    def process_voice_command(
        self,
        audio_path: str,
        stream_callback: Optional[Callable[[np.ndarray, int], None]] = None,
        on_tts_start: Optional[Callable[[], None]] = None,
        on_tts_end: Optional[Callable[[], None]] = None
    ) -> Tuple[str, Optional[bytes], int]:
        """
        Full pipeline: Audio File -> Transcript -> LLM/Tools -> TTS Audio.
        Returns: (response_text, audio_wav_bytes, latency_ms)
        """
        start_time = time.time()
        
        # 1. Transcribe
        emit_state_changed(self.bus, "listening", "transcribing")
        transcript = self.transcription.transcribe(audio_path)
        transcript = self._strip_wakeword_phrase(transcript)
        
        if not transcript:
            emit_state_changed(self.bus, "transcribing", "idle")
            return "", None, int((time.time() - start_time) * 1000)
            
        emit_transcript(self.bus, transcript, is_final=True)
        logger.info(f"📝 STT transcript: {transcript}")
        
        # 2. Think (Hybrid Routing)
        emit_state_changed(self.bus, "transcribing", "thinking")
        
        # Decide: Local vs Cloud
        use_local = self._should_use_local_llm(transcript)
        response_text = ""
        
        if use_local and self.use_local_llm:
            try:
                response_text = self._process_local_llm(transcript)
                # Fallback to cloud if local fails or returns empty?
                if not response_text:
                     logger.warning("Local LLM returned empty, falling back to Gemini.")
                     response_text = self._process_gemini(transcript)
            except Exception as e:
                logger.error(f"Local LLM failed: {e}. Falling back to Gemini.")
                # Fallback
                response_text = self._process_gemini(transcript)
        else:
            # Default to Cloud (Gemini)
            response_text = self._process_gemini(transcript)
        
        # 3. Speak (TTS)
        audio_bytes = None
        if response_text:
            emit_assistant_text(self.bus, response_text)
            emit_state_changed(self.bus, "thinking", "speaking")
            logger.info(f"🗣️ TTS response: {response_text}")

            # Select TTS Strategy (use cache for pre-baked phrases)
            is_prebaked = self._is_prebaked_phrase(response_text)
            if is_prebaked:
                audio_bytes = self._get_cached_tts(response_text)

            if audio_bytes is None:
                allow_streaming = not is_prebaked
                audio_bytes = self._synthesize_tts_bytes(
                    response_text,
                    stream_callback=stream_callback,
                    on_tts_start=on_tts_start,
                    on_tts_end=on_tts_end,
                    allow_streaming=allow_streaming
                )
                if is_prebaked and audio_bytes:
                    self._store_cached_tts(response_text, audio_bytes)
            # Note: The caller (main.py) is responsible for playing the audio
            # and setting state back to 'idle' after playback.
        else:
            emit_state_changed(self.bus, "thinking", "idle")

        latency_ms = int((time.time() - start_time) * 1000)
        return response_text, audio_bytes, latency_ms

    def _should_use_local_llm(self, text: str) -> bool:
        """Heuristic to decide if we can handle this locally."""
        if not self.use_local_llm:
            return False
            
        text = text.lower()
        
        # Simple keywords suitable for a small model
        local_intents = [
            'turn on', 'turn off', 'light', 'fan', 'switch', # Home control
            'time', 'date', 'timer', 'alarm',               # Time
            'repeat', 'say again',                          # Meta
            'who are you', 'what is your name'              # Identity
        ]
        
        # Check if it matches any local intent
        is_simple = any(x in text for x in local_intents)
        
        # If it looks like a complex query, force cloud
        complex_triggers = ['explain', 'why', 'how', 'search', 'news', 'weather', 'stock', 'code', 'write']
        if any(x in text for x in complex_triggers):
            return False
            
        return is_simple

    def _strip_wakeword_phrase(self, text: str) -> str:
        if not text:
            return text
        stripped = text.strip()
        for phrase in self.wakeword_phrases:
            pattern = rf"^{re.escape(phrase)}[\\s,;:!?.-]*"
            updated = re.sub(pattern, "", stripped, flags=re.IGNORECASE)
            if updated != stripped:
                return updated.strip()
        return stripped

    def _process_local_llm(self, user_text: str) -> str:
        """Call a local OpenAI-compatible endpoint (e.g. llama-server)."""
        logger.info(f"🧠 Routing to Local LLM: {user_text}")
        emit_state_changed(self.bus, "thinking", "thinking_local")
        
        try:
            # Assuming standard OpenAI format
            payload = {
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant. Keep answers very short and concise."},
                    {"role": "user", "content": user_text}
                ],
                "temperature": 0.3,
                "max_tokens": 150
            }
            res = requests.post(f"{self.local_llm_url}/chat/completions", json=payload, timeout=5)
            if res.status_code == 200:
                data = res.json()
                content = data['choices'][0]['message']['content']
                return content.strip()
            else:
                logger.error(f"Local LLM Error {res.status_code}: {res.text}")
                return ""
        except Exception as e:
            logger.error(f"Local LLM Request Failed: {e}")
            raise e

    def _process_gemini(self, user_text: str) -> str:
        """Run the Cloud LLM (Gemini), execute tools, and return final response text."""
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
            tool_outputs = []
            
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
                            result_text = str(result)
                            tool_outputs.append(result_text)
                            emit_tool_result(self.bus, t_name, True, result_text, dur)
                        except Exception as e:
                            tool_outputs.append(str(e))
                            emit_tool_result(self.bus, t_name, False, str(e))
            
                if has_tool:
                    if self.memory and self.reset_on_tool:
                        self.memory.reset()
                    if len(tool_outputs) == 1:
                        final_text = tool_outputs[0]
                    elif len(tool_outputs) > 1:
                        final_text = " ".join(tool_outputs)
                    else:
                        final_text = "Done"
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

    def _parse_prebaked_phrases(self, raw: Optional[str]) -> list:
        if not raw:
            return []
        return [p.strip() for p in raw.split(",") if p.strip()]

    def _normalize_tts_text(self, text: str) -> str:
        if not text:
            return ""
        cleaned = text
        if self.tts_provider in {"gpu", "pocket"}:
            cleaned = preprocess_for_tts(text)
        return " ".join(cleaned.strip().lower().split())

    def _tts_cache_key(self, text: str) -> str:
        normalized = self._normalize_tts_text(text)
        return f"{self.tts_provider}:{normalized}"

    def _is_prebaked_phrase(self, text: str) -> bool:
        if not self.tts_cache_enabled or not self._tts_prebaked_keys:
            return False
        return self._tts_cache_key(text) in self._tts_prebaked_keys

    def _get_cached_tts(self, text: str) -> Optional[bytes]:
        if not self.tts_cache_enabled:
            return None
        key = self._tts_cache_key(text)
        with self._tts_cache_lock:
            return self._tts_cache.get(key)

    def _store_cached_tts(self, text: str, audio_bytes: Optional[bytes]) -> None:
        if not self.tts_cache_enabled or not audio_bytes:
            return
        key = self._tts_cache_key(text)
        with self._tts_cache_lock:
            if key not in self._tts_cache:
                self._tts_cache[key] = audio_bytes

    def _warm_tts_cache(self) -> None:
        for phrase in self.tts_prebaked_phrases:
            if not phrase:
                continue
            if self._get_cached_tts(phrase):
                continue
            audio_bytes = self._synthesize_tts_bytes(phrase, allow_streaming=False)
            if audio_bytes:
                self._store_cached_tts(phrase, audio_bytes)

    def _synthesize_tts_bytes(
        self,
        text: str,
        stream_callback: Optional[Callable[[np.ndarray, int], None]] = None,
        on_tts_start: Optional[Callable[[], None]] = None,
        on_tts_end: Optional[Callable[[], None]] = None,
        allow_streaming: bool = True
    ) -> Optional[bytes]:
        if self.tts_provider == "local" and self.piper_voice:
            return self._synthesize_local_tts(text)
        if self.tts_provider == "pocket" and self.pocket_tts_model:
            if allow_streaming and self.pocket_streaming and stream_callback:
                streamed = self._stream_pocket_tts(
                    text,
                    stream_callback,
                    on_tts_start=on_tts_start,
                    on_tts_end=on_tts_end
                )
                if streamed:
                    return None
            return self._synthesize_pocket_tts(text)
        return self._synthesize_gpu_tts(text)

    def _synthesize_local_tts(self, text: str) -> Optional[bytes]:
        """Synthesize using local Piper model."""
        if not text or not self.piper_voice:
            return None
        
        try:
            start = time.time()
            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wav_file:
                # Piper writes directly to WAV file object
                self.piper_voice.synthesize(text, wav_file)
            
            dur = (time.time() - start) * 1000
            logger.info(f"🔊 Local Piper TTS: {dur:.0f}ms")
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Local TTS Error: {e}")
            return None

    def _synthesize_gpu_tts(self, text: str) -> Optional[bytes]:
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

    def _synthesize_pocket_tts(self, text: str) -> Optional[bytes]:
        """Synthesize using local Pocket TTS model."""
        if not text or not self.pocket_tts_model or not self.pocket_tts_voice_state:
            return None

        try:
            start = time.time()
            clean = preprocess_for_tts(text)
            audio = self.pocket_tts_model.generate_audio(self.pocket_tts_voice_state, clean)
            audio_np = audio.detach().cpu().numpy().squeeze()
            audio_np = np.clip(audio_np, -1.0, 1.0)
            audio_int16 = (audio_np * 32767.0).astype(np.int16)

            buf = io.BytesIO()
            with wave.open(buf, 'wb') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self.pocket_tts_model.sample_rate)
                wf.writeframes(audio_int16.tobytes())

            dur = (time.time() - start) * 1000
            logger.info(f"🔊 Pocket TTS: {dur:.0f}ms")
            return buf.getvalue()
        except Exception as e:
            logger.error(f"Pocket TTS Error: {e}")
            return None

    def _stream_pocket_tts(
        self,
        text: str,
        stream_callback: Callable[[np.ndarray, int], None],
        on_tts_start: Optional[Callable[[], None]] = None,
        on_tts_end: Optional[Callable[[], None]] = None
    ) -> bool:
        """Stream Pocket TTS audio chunks to a callback."""
        if not text or not self.pocket_tts_model or not self.pocket_tts_voice_state:
            return False

        try:
            if on_tts_start:
                on_tts_start()
            clean = preprocess_for_tts(text)
            for chunk in self.pocket_tts_model.generate_audio_stream(self.pocket_tts_voice_state, clean):
                if chunk is None:
                    continue
                chunk_np = chunk.detach().cpu().numpy().squeeze()
                if chunk_np.size == 0:
                    continue
                chunk_np = np.clip(chunk_np, -1.0, 1.0)
                chunk_int16 = (chunk_np * 32767.0).astype(np.int16)
                stream_callback(chunk_int16, self.pocket_tts_model.sample_rate)
            return True
        except Exception as e:
            logger.error(f"Pocket TTS Stream Error: {e}")
            return False
        finally:
            if on_tts_end:
                on_tts_end()
