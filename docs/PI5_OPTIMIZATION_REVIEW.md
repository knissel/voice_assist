# Raspberry Pi 5 Voice Assistant - Staff Engineer Review

**Date:** December 2024  
**Scope:** Performance, reliability, and UI-readiness audit for always-on Pi 5 deployment

---

## Executive Summary

The codebase is a functional voice assistant with good foundations (Porcupine wakeword, Silero VAD, remote Whisper, Gemini LLM, Piper TTS). However, it has several **Pi-hostile patterns** that will cause CPU spikes, audio glitches, and reliability issues in production. This document provides a prioritized remediation plan.

**Key Risks:**
1. **Blocking main thread** - All operations (network, TTS, tool calls) block the audio capture loop
2. **No state machine** - Race conditions possible, no barge-in support
3. **Synchronous HTTP** - Network calls block without timeouts/retries
4. **Model reloading** - Control4Manager reconnects on every call
5. **No event bus** - UI integration will require significant refactoring

---

## A) Architecture Analysis

### Current Control Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           wakeword.py (Main Entry)                       │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  STARTUP (Module-level, BLOCKING):                                       │
│  ├─ torch.hub.load('silero-vad')     ← Downloads model if missing!      │
│  ├─ PiperVoice.load(model)           ← ~200MB ONNX model load           │
│  ├─ create_transcription_service()   ← HTTP health check (blocking)     │
│  ├─ genai.Client()                   ← SDK init                         │
│  └─ pvporcupine.create()             ← Wakeword engine init             │
│                                                                          │
│  MAIN LOOP (Single-threaded, BLOCKING):                                  │
│  while True:                                                             │
│      pcm = audio_stream.read(512)    ← Blocking read (OK)               │
│      if porcupine.process(pcm) >= 0:                                    │
│          capture_and_process()        ← BLOCKS ENTIRE LOOP              │
│                                                                          │
│  capture_and_process():                                                  │
│  ├─ record_audio()                   ← 0.5-10s blocking                 │
│  │   └─ VAD runs per 512-sample chunk (CPU-bound)                       │
│  ├─ transcription_service.transcribe()  ← Sync HTTP POST (0.5-5s)       │
│  ├─ client.models.generate_content()    ← Sync HTTP (0.3-3s)            │
│  ├─ dispatch_tool()                     ← May call asyncio.run()!       │
│  │   └─ control4_tool creates new Control4Manager each call             │
│  │   └─ asyncio.run() inside sync function = event loop overhead        │
│  └─ speak_tts()                         ← Blocking audio playback       │
│                                                                          │
└─────────────────────────────────────────────────────────────────────────┘
```

### Thread/Process Model

| Component | Current | Risk |
|-----------|---------|------|
| Wakeword detection | Main thread | ✅ OK (lightweight) |
| VAD processing | Main thread | ⚠️ CPU spikes during recording |
| Audio recording | Main thread | ❌ Blocks wakeword detection |
| Transcription (remote) | Main thread, sync HTTP | ❌ Blocks everything |
| LLM call | Main thread, sync HTTP | ❌ Blocks everything |
| Tool execution | Main thread | ❌ asyncio.run() per call |
| TTS playback | Main thread | ❌ Blocks until complete |

### Files & Responsibilities

| File | Purpose | Lines | Issues |
|------|---------|-------|--------|
| `wakeword.py` | Main entry, wakeword loop | 263 | Monolithic, no state machine |
| `run_assistant.py` | Push-to-talk mode | 267 | Duplicates wakeword.py logic |
| `tools/transcription.py` | Remote/local ASR | 154 | Sync HTTP, no streaming |
| `tools/control4_tool.py` | Smart home | 62 | Reconnects every call |
| `tools/registry.py` | Tool dispatch | 298 | Good abstraction |
| `tools/youtube_music.py` | Music playback | 275 | subprocess.Popen (OK) |
| `tools/audio.py` | Volume/routing | 341 | subprocess.run (OK) |
| `whisper_server.py` | GPU server | 96 | Flask (not for Pi) |

---

## B) Pi 5 Performance Issues & Fixes

### 🔴 Critical Issues

#### 1. Blocking Main Loop During Processing
**Problem:** When wake word triggers, `capture_and_process()` blocks for 5-15 seconds. During this time:
- Wakeword detection stops
- Audio buffer may overflow
- User cannot cancel/interrupt

**Fix:** Move processing to a worker thread with queue-based communication.

```python
# BEFORE (wakeword.py:234-247)
while True:
    pcm = audio_stream.read(porcupine.frame_length)
    pcm = struct.unpack_from("h" * porcupine.frame_length, pcm)
    keyword_index = porcupine.process(pcm)
    if keyword_index >= 0:
        capture_and_process()  # BLOCKS 5-15 seconds!

# AFTER
import queue
import threading

wake_queue = queue.Queue()
result_queue = queue.Queue()

def worker():
    while True:
        audio_path = wake_queue.get()
        if audio_path is None:
            break
        try:
            process_command(audio_path)
        except Exception as e:
            result_queue.put(("error", str(e)))

worker_thread = threading.Thread(target=worker, daemon=True)
worker_thread.start()

while True:
    pcm = audio_stream.read(porcupine.frame_length)
    # ... wakeword detection continues even during processing
    if keyword_index >= 0 and wake_queue.empty():
        audio_path = record_audio()  # Still blocking, but shorter
        wake_queue.put(audio_path)
```

#### 2. Synchronous HTTP Without Timeouts/Retries
**Problem:** `transcription.py` and Gemini calls can hang indefinitely on network issues.

```python
# BEFORE (tools/transcription.py:62-66)
response = requests.post(
    f"{self.remote_url}/transcribe",
    files=files,
    timeout=30  # Only timeout, no retry
)

# AFTER - Add retry with exponential backoff
import httpx
from tenacity import retry, stop_after_attempt, wait_exponential

class TranscriptionService:
    def __init__(self, ...):
        self.client = httpx.Client(timeout=httpx.Timeout(10.0, connect=5.0))
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    def _transcribe_remote(self, audio_path: str) -> Optional[str]:
        with open(audio_path, 'rb') as f:
            response = self.client.post(
                f"{self.remote_url}/transcribe",
                files={'audio': f}
            )
        response.raise_for_status()
        return response.json().get('text', '').strip()
```

#### 3. Control4 Reconnects Every Call
**Problem:** `control4_tool.py` creates a new `Control4Manager` and calls `asyncio.run()` for every light command.

```python
# BEFORE (tools/control4_tool.py:44-62)
def control_home_lighting(device_id: int, brightness: int):
    manager = Control4Manager(...)  # New instance every call!
    return asyncio.run(manager.set_light(...))  # New event loop!

# AFTER - Singleton with connection pooling
_control4_manager = None
_control4_lock = threading.Lock()

def get_control4_manager():
    global _control4_manager
    with _control4_lock:
        if _control4_manager is None:
            _control4_manager = Control4Manager(...)
            asyncio.get_event_loop().run_until_complete(_control4_manager.connect())
        return _control4_manager

def control_home_lighting(device_id: int, brightness: int):
    manager = get_control4_manager()
    # Use existing event loop or run in thread pool
    loop = asyncio.get_event_loop()
    if loop.is_running():
        future = asyncio.run_coroutine_threadsafe(
            manager.set_light(device_id, brightness), loop
        )
        return future.result(timeout=10)
    return asyncio.run(manager.set_light(device_id, brightness))
```

#### 4. VAD Buffer Concatenation Creates Garbage
**Problem:** `np.concatenate` in the VAD loop creates new arrays every iteration.

```python
# BEFORE (wakeword.py:82)
vad_buffer = np.concatenate([vad_buffer, audio_int16])  # Allocates new array!

# AFTER - Use collections.deque or pre-allocated ring buffer
from collections import deque

class RingBuffer:
    def __init__(self, capacity):
        self.buffer = np.zeros(capacity, dtype=np.int16)
        self.write_pos = 0
        self.read_pos = 0
        self.count = 0
        self.capacity = capacity
    
    def write(self, data):
        n = len(data)
        # ... efficient circular write
    
    def read(self, n):
        # ... efficient circular read without allocation
```

### 🟡 Medium Issues

#### 5. Piper TTS Blocks Until Complete
**Problem:** Long responses block for several seconds during synthesis.

```python
# BEFORE (wakeword.py:179-194)
def speak_tts(text):
    stream = sd.OutputStream(...)
    stream.start()
    for audio_chunk in piper_voice.synthesize(text):  # Generates all at once
        stream.write(int_data)
    stream.stop()

# AFTER - Sentence splitting for faster time-to-first-audio
import re

def speak_tts(text, interrupt_event=None):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    stream = sd.OutputStream(...)
    stream.start()
    
    for sentence in sentences:
        if interrupt_event and interrupt_event.is_set():
            break
        for audio_chunk in piper_voice.synthesize(sentence):
            if interrupt_event and interrupt_event.is_set():
                break
            stream.write(np.frombuffer(audio_chunk.audio_int16_bytes, dtype=np.int16))
    
    stream.stop()
```

#### 6. Silero VAD Model Download at Runtime
**Problem:** `torch.hub.load()` may download the model on first run.

```python
# BEFORE (wakeword.py:28-29)
vad_model, utils = torch.hub.load(
    repo_or_dir='snakers4/silero-vad', 
    model='silero_vad', 
    force_reload=False  # Still checks network!
)

# AFTER - Use local model file
import torch

def load_vad_model():
    model_path = os.path.join(REPO_ROOT, "models", "silero_vad.jit")
    if os.path.exists(model_path):
        return torch.jit.load(model_path)
    # Fallback to hub, then save locally
    model, _ = torch.hub.load('snakers4/silero-vad', 'silero_vad')
    torch.jit.save(model, model_path)
    return model
```

#### 7. Duplicate Code Between wakeword.py and run_assistant.py
**Problem:** Recording, VAD, transcription, LLM, TTS logic duplicated.

**Fix:** Extract to shared `core/pipeline.py` module.

### 🟢 Quick Wins

#### 8. Add `exception_on_overflow=False` Consistently
Already done in some places, ensure everywhere:
```python
data = stream.read(CHUNK, exception_on_overflow=False)
```

#### 9. Reduce Torch Memory Usage
```python
# Add at startup
torch.set_num_threads(2)  # Limit CPU threads on Pi
torch.set_grad_enabled(False)  # Disable autograd
```

#### 10. Pre-warm Models at Startup
```python
# After loading Piper
dummy_audio = list(piper_voice.synthesize("Ready"))  # Warm JIT
```

---

## C) Reliability & State Machine

### Current State (Implicit)

The code has no explicit state management. The "state" is implicit in the call stack:
- Idle = waiting in `audio_stream.read()`
- Wake detected = entered `capture_and_process()`
- Listening = in `record_audio()` loop
- etc.

### Proposed State Machine

```python
from enum import Enum, auto
from dataclasses import dataclass
from typing import Optional
import threading
import time

class AssistantState(Enum):
    IDLE = auto()           # Listening for wake word
    WAKE_DETECTED = auto()  # Wake word heard, preparing to record
    LISTENING = auto()      # Recording user speech
    TRANSCRIBING = auto()   # Sending to ASR
    THINKING = auto()       # Waiting for LLM
    EXECUTING = auto()      # Running tool
    SPEAKING = auto()       # TTS playback
    ERROR = auto()          # Recoverable error
    CANCELLED = auto()      # User interrupted

@dataclass
class StateContext:
    state: AssistantState
    entered_at: float
    correlation_id: str
    transcript: Optional[str] = None
    response: Optional[str] = None
    error: Optional[str] = None

class StateMachine:
    TIMEOUTS = {
        AssistantState.WAKE_DETECTED: 1.0,
        AssistantState.LISTENING: 15.0,
        AssistantState.TRANSCRIBING: 10.0,
        AssistantState.THINKING: 15.0,
        AssistantState.EXECUTING: 30.0,
        AssistantState.SPEAKING: 60.0,
    }
    
    VALID_TRANSITIONS = {
        AssistantState.IDLE: {AssistantState.WAKE_DETECTED},
        AssistantState.WAKE_DETECTED: {AssistantState.LISTENING, AssistantState.IDLE},
        AssistantState.LISTENING: {AssistantState.TRANSCRIBING, AssistantState.CANCELLED, AssistantState.ERROR},
        AssistantState.TRANSCRIBING: {AssistantState.THINKING, AssistantState.ERROR},
        AssistantState.THINKING: {AssistantState.EXECUTING, AssistantState.SPEAKING, AssistantState.ERROR},
        AssistantState.EXECUTING: {AssistantState.SPEAKING, AssistantState.ERROR},
        AssistantState.SPEAKING: {AssistantState.IDLE, AssistantState.CANCELLED},
        AssistantState.ERROR: {AssistantState.IDLE},
        AssistantState.CANCELLED: {AssistantState.IDLE},
    }
    
    def __init__(self, event_bus):
        self.context = StateContext(
            state=AssistantState.IDLE,
            entered_at=time.time(),
            correlation_id=""
        )
        self.lock = threading.Lock()
        self.event_bus = event_bus
    
    def transition(self, new_state: AssistantState, **kwargs):
        with self.lock:
            if new_state not in self.VALID_TRANSITIONS.get(self.context.state, set()):
                raise ValueError(f"Invalid transition: {self.context.state} -> {new_state}")
            
            old_state = self.context.state
            self.context.state = new_state
            self.context.entered_at = time.time()
            
            for key, value in kwargs.items():
                setattr(self.context, key, value)
            
            self.event_bus.emit("state_changed", {
                "from": old_state.name,
                "to": new_state.name,
                "correlation_id": self.context.correlation_id
            })
    
    def check_timeout(self) -> bool:
        timeout = self.TIMEOUTS.get(self.context.state)
        if timeout and (time.time() - self.context.entered_at) > timeout:
            return True
        return False
```

### Barge-In Support

```python
class BargeInDetector:
    """Monitors audio during TTS for user interruption."""
    
    def __init__(self, vad_model, threshold=0.7):
        self.vad_model = vad_model
        self.threshold = threshold
        self.interrupt_event = threading.Event()
        self._running = False
    
    def start_monitoring(self, audio_stream):
        self._running = True
        self.interrupt_event.clear()
        
        def monitor():
            while self._running:
                try:
                    pcm = audio_stream.read(512, exception_on_overflow=False)
                    audio = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
                    prob = self.vad_model(torch.from_numpy(audio), 16000).item()
                    if prob > self.threshold:
                        self.interrupt_event.set()
                        break
                except Exception:
                    break
        
        self.thread = threading.Thread(target=monitor, daemon=True)
        self.thread.start()
    
    def stop_monitoring(self):
        self._running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)
```

### Watchdog & Self-Healing

```python
class Watchdog:
    def __init__(self, state_machine, restart_callback):
        self.state_machine = state_machine
        self.restart_callback = restart_callback
        self._running = False
    
    def start(self):
        self._running = True
        self.thread = threading.Thread(target=self._monitor, daemon=True)
        self.thread.start()
    
    def _monitor(self):
        while self._running:
            time.sleep(1.0)
            if self.state_machine.check_timeout():
                logging.error(f"Watchdog: State {self.state_machine.context.state} timed out")
                self.state_machine.transition(AssistantState.ERROR, 
                    error="Timeout in " + self.state_machine.context.state.name)
                self.restart_callback()
```

### Circuit Breaker for Remote Services

```python
from dataclasses import dataclass
from enum import Enum
import time

class CircuitState(Enum):
    CLOSED = auto()    # Normal operation
    OPEN = auto()      # Failing, reject calls
    HALF_OPEN = auto() # Testing if recovered

@dataclass
class CircuitBreaker:
    failure_threshold: int = 3
    recovery_timeout: float = 30.0
    
    def __post_init__(self):
        self.failures = 0
        self.state = CircuitState.CLOSED
        self.last_failure_time = 0
    
    def call(self, func, fallback=None):
        if self.state == CircuitState.OPEN:
            if time.time() - self.last_failure_time > self.recovery_timeout:
                self.state = CircuitState.HALF_OPEN
            else:
                if fallback:
                    return fallback()
                raise Exception("Circuit breaker open")
        
        try:
            result = func()
            if self.state == CircuitState.HALF_OPEN:
                self.state = CircuitState.CLOSED
                self.failures = 0
            return result
        except Exception as e:
            self.failures += 1
            self.last_failure_time = time.time()
            if self.failures >= self.failure_threshold:
                self.state = CircuitState.OPEN
            raise
```

---

## D) UI-Ready Architecture

### Proposed Directory Structure

```
voice_assist/
├── core/
│   ├── __init__.py
│   ├── state_machine.py      # State management
│   ├── event_bus.py          # Pub/sub for UI events
│   ├── pipeline.py           # Audio → ASR → LLM → TTS orchestration
│   ├── audio_capture.py      # Mic input, ring buffer
│   └── config.py             # Centralized configuration
├── adapters/
│   ├── __init__.py
│   ├── asr_client.py         # Remote Whisper client (async)
│   ├── llm_client.py         # Gemini client (async, streaming)
│   ├── tts_adapter.py        # Piper TTS wrapper
│   └── wakeword_adapter.py   # Porcupine wrapper
├── tools/
│   ├── __init__.py
│   ├── base.py               # Tool interface
│   ├── registry.py           # Tool registration
│   ├── control4.py           # Smart home
│   ├── youtube_music.py      # Music
│   └── ...
├── schemas/
│   ├── __init__.py
│   ├── events.py             # Pydantic event models
│   └── tools.py              # Tool input/output schemas
├── ui_gateway/
│   ├── __init__.py
│   ├── server.py             # FastAPI + WebSocket
│   └── routes.py             # REST endpoints
├── wakeword.py               # Entry point (thin wrapper)
├── run_assistant.py          # Push-to-talk entry
└── requirements.txt
```

### Event Bus Implementation

```python
# core/event_bus.py
from typing import Callable, Dict, List, Any
from dataclasses import dataclass, field
from datetime import datetime
import threading
import queue
import json
import logging

@dataclass
class Event:
    type: str
    data: Dict[str, Any]
    correlation_id: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_json(self) -> str:
        return json.dumps({
            "type": self.type,
            "data": self.data,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat()
        })

class EventBus:
    def __init__(self):
        self._subscribers: Dict[str, List[Callable]] = {}
        self._lock = threading.Lock()
        self._queue = queue.Queue()
        self._running = False
    
    def subscribe(self, event_type: str, callback: Callable[[Event], None]):
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
    
    def unsubscribe(self, event_type: str, callback: Callable):
        with self._lock:
            if event_type in self._subscribers:
                self._subscribers[event_type].remove(callback)
    
    def emit(self, event_type: str, data: Dict[str, Any], correlation_id: str = ""):
        event = Event(type=event_type, data=data, correlation_id=correlation_id)
        self._queue.put(event)
    
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._process_events, daemon=True)
        self._thread.start()
    
    def _process_events(self):
        while self._running:
            try:
                event = self._queue.get(timeout=0.1)
                with self._lock:
                    callbacks = self._subscribers.get(event.type, [])
                    callbacks += self._subscribers.get("*", [])  # Wildcard subscribers
                
                for callback in callbacks:
                    try:
                        callback(event)
                    except Exception as e:
                        logging.error(f"Event callback error: {e}")
            except queue.Empty:
                continue
    
    def stop(self):
        self._running = False
        self._thread.join(timeout=2.0)
```

### Pydantic Event Schemas

```python
# schemas/events.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from datetime import datetime
from enum import Enum

class EventType(str, Enum):
    STATE_CHANGED = "state_changed"
    TRANSCRIPT_PARTIAL = "transcript_partial"
    TRANSCRIPT_FINAL = "transcript_final"
    ASSISTANT_TEXT = "assistant_text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    TTS_START = "tts_start"
    TTS_END = "tts_end"
    ERROR = "error"
    UI_CARD = "ui_card"
    UI_ACTION = "ui_action"
    NOTIFICATION = "notification"

class BaseEvent(BaseModel):
    type: EventType
    correlation_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class StateChangedEvent(BaseEvent):
    type: Literal[EventType.STATE_CHANGED] = EventType.STATE_CHANGED
    from_state: str
    to_state: str

class TranscriptPartialEvent(BaseEvent):
    type: Literal[EventType.TRANSCRIPT_PARTIAL] = EventType.TRANSCRIPT_PARTIAL
    text: str
    confidence: Optional[float] = None

class TranscriptFinalEvent(BaseEvent):
    type: Literal[EventType.TRANSCRIPT_FINAL] = EventType.TRANSCRIPT_FINAL
    text: str
    duration_ms: int

class AssistantTextEvent(BaseEvent):
    type: Literal[EventType.ASSISTANT_TEXT] = EventType.ASSISTANT_TEXT
    text: str
    is_partial: bool = False

class ToolCallEvent(BaseEvent):
    type: Literal[EventType.TOOL_CALL] = EventType.TOOL_CALL
    tool_name: str
    arguments: Dict[str, Any]

class ToolResultEvent(BaseEvent):
    type: Literal[EventType.TOOL_RESULT] = EventType.TOOL_RESULT
    tool_name: str
    success: bool
    result: Any
    ui_card: Optional["UICard"] = None

class TTSStartEvent(BaseEvent):
    type: Literal[EventType.TTS_START] = EventType.TTS_START
    text: str
    estimated_duration_ms: Optional[int] = None

class TTSEndEvent(BaseEvent):
    type: Literal[EventType.TTS_END] = EventType.TTS_END
    was_interrupted: bool = False

class ErrorEvent(BaseEvent):
    type: Literal[EventType.ERROR] = EventType.ERROR
    error_code: str
    message: str
    recoverable: bool = True

# UI Cards for rich display
class CardType(str, Enum):
    RECIPE = "recipe"
    WEATHER = "weather"
    CALENDAR = "calendar"
    TIMER = "timer"
    NOTE = "note"
    GENERIC = "generic"

class UIAction(BaseModel):
    id: str
    label: str
    icon: Optional[str] = None
    action_type: Literal["navigate", "command", "dismiss"]
    payload: Optional[Dict[str, Any]] = None

class UICard(BaseModel):
    card_type: CardType
    title: str
    subtitle: Optional[str] = None
    body: Optional[str] = None
    image_url: Optional[str] = None
    actions: List[UIAction] = []
    data: Dict[str, Any] = {}  # Type-specific data

class RecipeCard(UICard):
    card_type: Literal[CardType.RECIPE] = CardType.RECIPE
    data: Dict[str, Any] = Field(default_factory=lambda: {
        "current_step": 0,
        "total_steps": 0,
        "steps": [],
        "ingredients": []
    })

class WeatherCard(UICard):
    card_type: Literal[CardType.WEATHER] = CardType.WEATHER
    data: Dict[str, Any] = Field(default_factory=lambda: {
        "temperature": 0,
        "condition": "",
        "forecast": []
    })

class CalendarCard(UICard):
    card_type: Literal[CardType.CALENDAR] = CardType.CALENDAR
    data: Dict[str, Any] = Field(default_factory=lambda: {
        "events": []
    })

class UICardEvent(BaseEvent):
    type: Literal[EventType.UI_CARD] = EventType.UI_CARD
    card: UICard

class UIActionEvent(BaseEvent):
    type: Literal[EventType.UI_ACTION] = EventType.UI_ACTION
    action: UIAction

class NotificationEvent(BaseEvent):
    type: Literal[EventType.NOTIFICATION] = EventType.NOTIFICATION
    level: Literal["info", "warning", "error", "success"]
    title: str
    message: str
    duration_ms: int = 5000
```

### Tool Interface

```python
# tools/base.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
from schemas.events import UICard

class ToolInput(BaseModel):
    """Base class for tool inputs - subclass for each tool."""
    pass

class ToolOutput(BaseModel):
    """Standardized tool output."""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
    ui_cards: List[UICard] = []
    speak: Optional[str] = None  # Text to speak

class BaseTool(ABC):
    """Base class for all tools/skills."""
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Tool name for LLM function calling."""
        pass
    
    @property
    @abstractmethod
    def description(self) -> str:
        """Tool description for LLM."""
        pass
    
    @property
    @abstractmethod
    def parameters_schema(self) -> Dict[str, Any]:
        """JSON Schema for parameters."""
        pass
    
    @abstractmethod
    async def execute(self, **kwargs) -> ToolOutput:
        """Execute the tool with given parameters."""
        pass
    
    def to_gemini_declaration(self):
        """Convert to Gemini FunctionDeclaration format."""
        from google.genai import types
        return types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=types.Schema(**self.parameters_schema)
        )

# Example tool implementation
class LightingTool(BaseTool):
    name = "control_home_lighting"
    description = "Controls home lights. Kitchen Cans=85, Kitchen Island=95..."
    
    parameters_schema = {
        "type": "OBJECT",
        "properties": {
            "device_id": {"type": "INTEGER", "description": "Light device ID"},
            "brightness": {"type": "INTEGER", "description": "0-100"}
        },
        "required": ["device_id", "brightness"]
    }
    
    def __init__(self, control4_manager):
        self.manager = control4_manager
    
    async def execute(self, device_id: int, brightness: int) -> ToolOutput:
        try:
            await self.manager.set_light(device_id, brightness)
            return ToolOutput(
                success=True,
                message=f"Set light {device_id} to {brightness}%",
                speak="Done"
            )
        except Exception as e:
            return ToolOutput(
                success=False,
                message=str(e),
                speak="I couldn't control that light"
            )
```

### UI Gateway (FastAPI + WebSocket)

```python
# ui_gateway/server.py
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from typing import List
import asyncio
import json

app = FastAPI(title="Voice Assistant UI Gateway")

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                pass

manager = ConnectionManager()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            # Handle incoming UI actions
            action = json.loads(data)
            if action.get("type") == "ui_action":
                # Dispatch to assistant
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.get("/health")
async def health():
    return {"status": "healthy"}

# Integration with EventBus
def setup_event_bridge(event_bus, connection_manager):
    """Bridge EventBus to WebSocket clients."""
    def on_event(event):
        asyncio.create_task(connection_manager.broadcast(event.to_json()))
    
    event_bus.subscribe("*", on_event)
```

---

## E) Optimization Plan

### Quick Wins (≤2 hours each)

| # | Task | Impact | Effort |
|---|------|--------|--------|
| 1 | Add `torch.set_num_threads(2)` at startup | Reduce CPU contention | 5 min |
| 2 | Pre-warm Piper with dummy synthesis | Faster first response | 10 min |
| 3 | Add `exception_on_overflow=False` everywhere | Prevent crashes | 15 min |
| 4 | Cache Silero VAD model locally | Eliminate network check | 30 min |
| 5 | Add timeouts to all HTTP calls | Prevent hangs | 30 min |
| 6 | Singleton Control4Manager | Reduce reconnects | 1 hour |
| 7 | Add structured JSON logging | Better debugging | 1 hour |

### Medium Tasks (≤2 days each)

| # | Task | Impact | Effort |
|---|------|--------|--------|
| 8 | Extract shared code to `core/pipeline.py` | Maintainability | 4 hours |
| 9 | Implement basic state machine | Reliability | 4 hours |
| 10 | Add worker thread for processing | Non-blocking wakeword | 6 hours |
| 11 | Implement ring buffer for VAD | Reduce allocations | 4 hours |
| 12 | Add retry/backoff to remote calls | Reliability | 4 hours |
| 13 | Sentence-split TTS for faster TTFA | Perceived latency | 4 hours |
| 14 | Add barge-in detection | UX improvement | 8 hours |

### Larger Refactors (≤2 weeks)

| # | Task | Impact | Effort |
|---|------|--------|--------|
| 15 | Full event bus implementation | UI-ready | 3 days |
| 16 | Pydantic schemas for all events | Type safety | 2 days |
| 17 | Tool interface refactor | Extensibility | 3 days |
| 18 | FastAPI UI gateway | UI integration | 3 days |
| 19 | Async Gemini client with streaming | Lower latency | 2 days |
| 20 | Comprehensive test suite | Reliability | 3 days |

---

## F) Sample Patches

### Patch 1: Quick Wins Bundle

```python
# Add to wakeword.py after imports
import torch
torch.set_num_threads(2)
torch.set_grad_enabled(False)

# Replace VAD loading (line 28-35)
def load_vad_model():
    """Load VAD model from local cache or download once."""
    model_dir = os.path.join(REPO_ROOT, "models")
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "silero_vad.jit")
    
    if os.path.exists(model_path):
        return torch.jit.load(model_path), True
    
    try:
        model, utils = torch.hub.load('snakers4/silero-vad', 'silero_vad', force_reload=False)
        torch.jit.save(model, model_path)
        return model, True
    except Exception as e:
        print(f"⚠️  VAD initialization failed: {e}")
        return None, False

vad_model, vad_available = load_vad_model()

# Add after Piper loading (line 45)
if piper_voice:
    # Pre-warm the model
    _ = list(piper_voice.synthesize("Ready"))
    print("✅ Piper TTS pre-warmed")
```

### Patch 2: Structured Logging

```python
# core/logging_config.py
import logging
import json
from datetime import datetime
from typing import Any, Dict

class JSONFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        
        # Add correlation_id if present
        if hasattr(record, 'correlation_id'):
            log_data["correlation_id"] = record.correlation_id
        
        # Add extra fields
        if hasattr(record, 'extra'):
            log_data.update(record.extra)
        
        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)

def setup_logging(level=logging.INFO):
    handler = logging.StreamHandler()
    handler.setFormatter(JSONFormatter())
    
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]
    
    return root

# Usage
logger = logging.getLogger(__name__)
logger.info("Processing command", extra={
    "correlation_id": "abc123",
    "extra": {"stage": "transcription", "duration_ms": 450}
})
```

### Patch 3: Async Transcription Client

```python
# adapters/asr_client.py
import httpx
import asyncio
from typing import Optional
from dataclasses import dataclass

@dataclass
class TranscriptionResult:
    text: str
    duration_ms: int
    language: str = "en"

class AsyncASRClient:
    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout, connect=5.0),
            limits=httpx.Limits(max_connections=2)
        )
        self._healthy = False
    
    async def health_check(self) -> bool:
        try:
            response = await self.client.get(f"{self.base_url}/health")
            self._healthy = response.status_code == 200
            return self._healthy
        except Exception:
            self._healthy = False
            return False
    
    async def transcribe(self, audio_path: str) -> Optional[TranscriptionResult]:
        if not self._healthy:
            await self.health_check()
            if not self._healthy:
                return None
        
        try:
            with open(audio_path, 'rb') as f:
                response = await self.client.post(
                    f"{self.base_url}/transcribe",
                    files={'audio': f}
                )
            
            if response.status_code == 200:
                data = response.json()
                return TranscriptionResult(
                    text=data.get('text', '').strip(),
                    duration_ms=int(data.get('duration', 0) * 1000),
                    language=data.get('language', 'en')
                )
        except Exception as e:
            self._healthy = False
            raise
        
        return None
    
    async def close(self):
        await self.client.aclose()
```

---

## G) Minimal UI MVP Plan

### Phase 1: WebSocket Event Stream (1 day)

1. Add FastAPI to requirements
2. Create `ui_gateway/server.py` with WebSocket endpoint
3. Bridge EventBus → WebSocket broadcast
4. Test with `websocat` CLI tool

### Phase 2: Basic Web UI (2 days)

```
ui/
├── index.html
├── styles.css
└── app.js
```

**Features:**
- Live transcript display (partial → final)
- Assistant response with typing animation
- Conversation history (last 10 turns)
- State indicator (listening/thinking/speaking)

### Phase 3: Cards & Actions (2 days)

- Recipe card with step navigation
- Weather summary card
- Calendar agenda card
- Touch-friendly action buttons

### Phase 4: Kiosk Mode (1 day)

- Chromium autostart in kiosk mode
- Disable screen blanking
- Auto-reconnect WebSocket

**Systemd service for kiosk:**
```ini
[Unit]
Description=Voice Assistant Kiosk
After=graphical.target

[Service]
User=pi
Environment=DISPLAY=:0
ExecStart=/usr/bin/chromium-browser --kiosk --disable-restore-session-state http://localhost:8000
Restart=always

[Install]
WantedBy=graphical.target
```

---

## H) README Updates

### Add to README.md

```markdown
## Profiling & Debugging

### CPU Profiling with py-spy
```bash
# Install
pip install py-spy

# Profile running assistant
sudo py-spy top --pid $(pgrep -f wakeword.py)

# Generate flame graph
sudo py-spy record -o profile.svg --pid $(pgrep -f wakeword.py)
```

### Memory Profiling
```bash
pip install memory_profiler
python -m memory_profiler wakeword.py
```

### Audio Debugging
```bash
# List audio devices
python -c "import pyaudio; p = pyaudio.PyAudio(); [print(p.get_device_info_by_index(i)) for i in range(p.get_device_count())]"

# Test recording
arecord -d 5 -f S16_LE -r 16000 test.wav
aplay test.wav
```

## Deployment as Service

### Systemd Unit File

Create `/etc/systemd/system/voice-assistant.service`:

```ini
[Unit]
Description=Voice Assistant
After=network.target sound.target
Wants=network-online.target

[Service]
Type=simple
User=pi
Group=pi
WorkingDirectory=/home/pi/voice_assist
Environment=PATH=/home/pi/voice_assist/.venv/bin:/usr/bin
ExecStart=/home/pi/voice_assist/.venv/bin/python wakeword.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

# Resource limits for Pi
CPUQuota=80%
MemoryMax=512M

[Install]
WantedBy=multi-user.target
```

### Enable and Start
```bash
sudo systemctl daemon-reload
sudo systemctl enable voice-assistant
sudo systemctl start voice-assistant

# View logs
journalctl -u voice-assistant -f
```

## Packaging for Pi

### Minimal requirements_pi.txt
```
google-genai>=0.3.0
pyaudio>=0.2.13
pvporcupine>=3.0.0
python-dotenv>=1.0.0
requests>=2.31.0
piper-tts>=1.2.0
sounddevice>=0.4.6
numpy>=1.24.0,<2.0.0
torch>=2.0.0  # CPU-only wheel for ARM
```

### Install on Pi
```bash
# Create venv
python3 -m venv .venv
source .venv/bin/activate

# Install PyTorch CPU-only for ARM
pip install torch --index-url https://download.pytorch.org/whl/cpu

# Install other deps
pip install -r requirements_pi.txt
```
```

---

## Summary

### Strengths
- ✅ Good tool abstraction in `registry.py`
- ✅ Remote Whisper offloading (correct for Pi)
- ✅ Piper TTS (fast, local, no API costs)
- ✅ Silero VAD (lightweight)
- ✅ Headless mode for Pi

### Critical Risks
- ❌ Blocking main thread during processing
- ❌ No state machine or error recovery
- ❌ Synchronous HTTP without retries
- ❌ Control4 reconnects every call
- ❌ No barge-in support
- ❌ No event bus for UI

### Recommended Priority
1. **Week 1:** Quick wins + worker thread + state machine
2. **Week 2:** Event bus + async clients + barge-in
3. **Week 3:** UI gateway + basic web UI
4. **Week 4:** Cards, actions, kiosk mode, polish

---

*Generated by Staff Engineer Review - December 2024*
