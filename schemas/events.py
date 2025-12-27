"""
Pydantic event schemas for UI-ready voice assistant.
These events are emitted by the core pipeline and consumed by UI clients via WebSocket.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal, Union
from datetime import datetime
from enum import Enum


class EventType(str, Enum):
    """All event types emitted by the assistant."""
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
    WAKE_WORD = "wake_word"
    LISTENING_START = "listening_start"
    LISTENING_END = "listening_end"


class AssistantState(str, Enum):
    """Assistant state machine states."""
    IDLE = "idle"
    WAKE_DETECTED = "wake_detected"
    LISTENING = "listening"
    TRANSCRIBING = "transcribing"
    THINKING = "thinking"
    EXECUTING = "executing"
    SPEAKING = "speaking"
    ERROR = "error"
    CANCELLED = "cancelled"


# =============================================================================
# Base Event
# =============================================================================

class BaseEvent(BaseModel):
    """Base class for all events."""
    type: EventType
    correlation_id: str = Field(default="", description="Unique ID for this conversation turn")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        use_enum_values = True
    
    def to_json(self) -> str:
        return self.model_dump_json()


# =============================================================================
# State Events
# =============================================================================

class StateChangedEvent(BaseEvent):
    """Emitted when assistant state changes."""
    type: Literal[EventType.STATE_CHANGED] = EventType.STATE_CHANGED
    from_state: AssistantState
    to_state: AssistantState
    reason: Optional[str] = None


class WakeWordEvent(BaseEvent):
    """Emitted when wake word is detected."""
    type: Literal[EventType.WAKE_WORD] = EventType.WAKE_WORD
    keyword: str = Field(description="Which wake word was detected")
    confidence: Optional[float] = None


# =============================================================================
# Transcript Events
# =============================================================================

class TranscriptPartialEvent(BaseEvent):
    """Emitted during streaming ASR with partial results."""
    type: Literal[EventType.TRANSCRIPT_PARTIAL] = EventType.TRANSCRIPT_PARTIAL
    text: str
    confidence: Optional[float] = None
    is_final: bool = False


class TranscriptFinalEvent(BaseEvent):
    """Emitted when transcription is complete."""
    type: Literal[EventType.TRANSCRIPT_FINAL] = EventType.TRANSCRIPT_FINAL
    text: str
    duration_ms: int = Field(description="Time taken for transcription")
    audio_duration_ms: Optional[int] = Field(default=None, description="Duration of recorded audio")
    language: str = "en"
    confidence: Optional[float] = None


# =============================================================================
# Assistant Response Events
# =============================================================================

class AssistantTextEvent(BaseEvent):
    """Emitted when assistant generates text response."""
    type: Literal[EventType.ASSISTANT_TEXT] = EventType.ASSISTANT_TEXT
    text: str
    is_partial: bool = Field(default=False, description="True if streaming partial response")
    is_final: bool = Field(default=True, description="True if this is the final chunk")


class ToolCallEvent(BaseEvent):
    """Emitted when LLM decides to call a tool."""
    type: Literal[EventType.TOOL_CALL] = EventType.TOOL_CALL
    tool_name: str
    arguments: Dict[str, Any]


class ToolResultEvent(BaseEvent):
    """Emitted after tool execution completes."""
    type: Literal[EventType.TOOL_RESULT] = EventType.TOOL_RESULT
    tool_name: str
    success: bool
    result: Any
    duration_ms: int = 0
    ui_card: Optional["UICard"] = None


# =============================================================================
# TTS Events
# =============================================================================

class TTSStartEvent(BaseEvent):
    """Emitted when TTS begins speaking."""
    type: Literal[EventType.TTS_START] = EventType.TTS_START
    text: str
    estimated_duration_ms: Optional[int] = None


class TTSEndEvent(BaseEvent):
    """Emitted when TTS finishes speaking."""
    type: Literal[EventType.TTS_END] = EventType.TTS_END
    was_interrupted: bool = Field(default=False, description="True if user barged in")
    actual_duration_ms: Optional[int] = None


# =============================================================================
# Error Events
# =============================================================================

class ErrorEvent(BaseEvent):
    """Emitted when an error occurs."""
    type: Literal[EventType.ERROR] = EventType.ERROR
    error_code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error message")
    recoverable: bool = Field(default=True, description="Can the assistant recover?")
    details: Optional[Dict[str, Any]] = None


# =============================================================================
# UI Card System
# =============================================================================

class CardType(str, Enum):
    """Types of UI cards that can be displayed."""
    RECIPE = "recipe"
    WEATHER = "weather"
    CALENDAR = "calendar"
    TIMER = "timer"
    NOTE = "note"
    MUSIC = "music"
    LIGHT_STATUS = "light_status"
    GENERIC = "generic"


class UIAction(BaseModel):
    """An action button that can be displayed on a card."""
    id: str = Field(description="Unique action identifier")
    label: str = Field(description="Button text")
    icon: Optional[str] = Field(default=None, description="Icon name (e.g., 'play', 'next')")
    action_type: Literal["navigate", "command", "dismiss"] = "command"
    payload: Optional[Dict[str, Any]] = Field(default=None, description="Data to send when action triggered")
    primary: bool = Field(default=False, description="Is this the primary action?")


class UICard(BaseModel):
    """A rich UI card for displaying structured information."""
    card_type: CardType
    title: str
    subtitle: Optional[str] = None
    body: Optional[str] = None
    image_url: Optional[str] = None
    actions: List[UIAction] = Field(default_factory=list)
    data: Dict[str, Any] = Field(default_factory=dict, description="Type-specific data")
    expires_at: Optional[datetime] = Field(default=None, description="When this card should be dismissed")
    
    class Config:
        use_enum_values = True


# =============================================================================
# Specialized Cards
# =============================================================================

class RecipeStep(BaseModel):
    """A single step in a recipe."""
    number: int
    instruction: str
    duration_minutes: Optional[int] = None
    tip: Optional[str] = None


class RecipeCard(UICard):
    """Card for displaying recipe information."""
    card_type: Literal[CardType.RECIPE] = CardType.RECIPE
    data: Dict[str, Any] = Field(default_factory=lambda: {
        "current_step": 0,
        "total_steps": 0,
        "steps": [],
        "ingredients": [],
        "prep_time_minutes": None,
        "cook_time_minutes": None,
        "servings": None
    })


class WeatherForecast(BaseModel):
    """Weather forecast for a single period."""
    time: str
    temperature: int
    condition: str
    icon: str
    precipitation_chance: Optional[int] = None


class WeatherCard(UICard):
    """Card for displaying weather information."""
    card_type: Literal[CardType.WEATHER] = CardType.WEATHER
    data: Dict[str, Any] = Field(default_factory=lambda: {
        "current_temperature": 0,
        "feels_like": 0,
        "condition": "",
        "humidity": 0,
        "wind_speed": 0,
        "forecast": []
    })


class CalendarEvent(BaseModel):
    """A single calendar event."""
    title: str
    start_time: datetime
    end_time: Optional[datetime] = None
    location: Optional[str] = None
    is_all_day: bool = False


class CalendarCard(UICard):
    """Card for displaying calendar events."""
    card_type: Literal[CardType.CALENDAR] = CardType.CALENDAR
    data: Dict[str, Any] = Field(default_factory=lambda: {
        "date": None,
        "events": []
    })


class TimerCard(UICard):
    """Card for displaying active timers."""
    card_type: Literal[CardType.TIMER] = CardType.TIMER
    data: Dict[str, Any] = Field(default_factory=lambda: {
        "name": "",
        "duration_seconds": 0,
        "remaining_seconds": 0,
        "started_at": None,
        "is_paused": False
    })


class MusicCard(UICard):
    """Card for displaying now playing information."""
    card_type: Literal[CardType.MUSIC] = CardType.MUSIC
    data: Dict[str, Any] = Field(default_factory=lambda: {
        "title": "",
        "artist": "",
        "album": "",
        "album_art_url": None,
        "is_playing": False,
        "progress_seconds": 0,
        "duration_seconds": 0
    })


class LightStatusCard(UICard):
    """Card for displaying light status."""
    card_type: Literal[CardType.LIGHT_STATUS] = CardType.LIGHT_STATUS
    data: Dict[str, Any] = Field(default_factory=lambda: {
        "lights": []  # List of {name, brightness, is_on}
    })


# =============================================================================
# UI Events
# =============================================================================

class UICardEvent(BaseEvent):
    """Emitted when a UI card should be displayed."""
    type: Literal[EventType.UI_CARD] = EventType.UI_CARD
    card: UICard
    replace_existing: bool = Field(default=False, description="Replace card of same type?")


class UIActionEvent(BaseEvent):
    """Emitted when user triggers a UI action (from touchscreen)."""
    type: Literal[EventType.UI_ACTION] = EventType.UI_ACTION
    action_id: str
    action_type: str
    payload: Optional[Dict[str, Any]] = None


class NotificationEvent(BaseEvent):
    """Emitted for toast-style notifications."""
    type: Literal[EventType.NOTIFICATION] = EventType.NOTIFICATION
    level: Literal["info", "warning", "error", "success"]
    title: str
    message: str
    duration_ms: int = Field(default=5000, description="How long to show notification")
    action: Optional[UIAction] = None


# =============================================================================
# Listening Events
# =============================================================================

class ListeningStartEvent(BaseEvent):
    """Emitted when assistant starts listening for user speech."""
    type: Literal[EventType.LISTENING_START] = EventType.LISTENING_START
    max_duration_seconds: float = 10.0


class ListeningEndEvent(BaseEvent):
    """Emitted when assistant stops listening."""
    type: Literal[EventType.LISTENING_END] = EventType.LISTENING_END
    reason: Literal["silence", "max_duration", "cancelled", "error"]
    audio_duration_ms: int = 0


# =============================================================================
# Event Union Type
# =============================================================================

AnyEvent = Union[
    StateChangedEvent,
    WakeWordEvent,
    TranscriptPartialEvent,
    TranscriptFinalEvent,
    AssistantTextEvent,
    ToolCallEvent,
    ToolResultEvent,
    TTSStartEvent,
    TTSEndEvent,
    ErrorEvent,
    UICardEvent,
    UIActionEvent,
    NotificationEvent,
    ListeningStartEvent,
    ListeningEndEvent,
]
