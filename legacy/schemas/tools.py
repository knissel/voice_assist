"""
Pydantic schemas for tool inputs and outputs.
Provides type safety and validation for all tool interactions.
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Literal
from enum import Enum


# =============================================================================
# Tool Output Schema
# =============================================================================

class ToolOutput(BaseModel):
    """Standardized output from all tools."""
    success: bool = Field(description="Whether the tool executed successfully")
    message: str = Field(description="Human-readable result message")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Structured result data")
    speak: Optional[str] = Field(default=None, description="Text for TTS to speak")
    ui_card: Optional[Any] = Field(default=None, description="Optional UI card to display")
    
    class Config:
        extra = "allow"


# =============================================================================
# Lighting Tool Schemas
# =============================================================================

class LightDevice(BaseModel):
    """A controllable light device."""
    device_id: int
    name: str
    brightness: int = Field(ge=0, le=100)
    is_on: bool


class LightingInput(BaseModel):
    """Input for control_home_lighting tool."""
    device_id: int = Field(description="Device ID (999 for all lights)")
    brightness: int = Field(ge=0, le=100, description="Brightness 0-100")


class LightingOutput(ToolOutput):
    """Output from control_home_lighting tool."""
    data: Optional[Dict[str, Any]] = Field(default_factory=lambda: {
        "device_id": 0,
        "brightness": 0,
        "previous_brightness": None
    })


# =============================================================================
# Bluetooth Tool Schemas
# =============================================================================

class BluetoothDevice(BaseModel):
    """A Bluetooth device."""
    mac: str = Field(description="MAC address")
    name: str
    is_connected: bool = False
    device_type: Optional[str] = None


class BluetoothConnectInput(BaseModel):
    """Input for connect_bluetooth_device tool."""
    device_mac: str = Field(description="MAC address (e.g., AA:BB:CC:DD:EE:FF)")


class BluetoothDisconnectInput(BaseModel):
    """Input for disconnect_bluetooth_device tool."""
    device_mac: str


class BluetoothOutput(ToolOutput):
    """Output from Bluetooth tools."""
    data: Optional[Dict[str, Any]] = Field(default_factory=lambda: {
        "device_mac": "",
        "device_name": None,
        "is_connected": False
    })


# =============================================================================
# Audio Tool Schemas
# =============================================================================

class AudioSink(BaseModel):
    """An audio output device."""
    sink_id: str
    name: str
    is_default: bool = False
    volume: Optional[int] = None


class VolumeAction(str, Enum):
    """Volume control actions."""
    UP = "up"
    DOWN = "down"
    SET = "set"
    MUTE = "mute"
    UNMUTE = "unmute"


class VolumeControlInput(BaseModel):
    """Input for control_volume tool."""
    action: VolumeAction
    level: Optional[int] = Field(default=None, ge=0, le=100, description="Volume level for 'set' action")


class VolumeOutput(ToolOutput):
    """Output from volume control tool."""
    data: Optional[Dict[str, Any]] = Field(default_factory=lambda: {
        "previous_volume": 0,
        "current_volume": 0,
        "is_muted": False
    })


class AudioRoutingInput(BaseModel):
    """Input for route_to_bluetooth tool."""
    device_name: Optional[str] = Field(default=None, description="Specific device name")


class SetAudioSinkInput(BaseModel):
    """Input for set_audio_sink tool."""
    sink_id: str


# =============================================================================
# YouTube Music Tool Schemas
# =============================================================================

class ContentType(str, Enum):
    """Types of content to play."""
    SONG = "song"
    VIDEO = "video"
    ALBUM = "album"
    ARTIST = "artist"
    PLAYLIST = "playlist"


class PlayMusicInput(BaseModel):
    """Input for play_youtube_music tool."""
    query: str = Field(description="What to play")
    content_type: ContentType = Field(default=ContentType.SONG)


class PlayMusicOutput(ToolOutput):
    """Output from play_youtube_music tool."""
    data: Optional[Dict[str, Any]] = Field(default_factory=lambda: {
        "title": "",
        "artist": "",
        "url": "",
        "video_id": None,
        "is_playing": False
    })


class StopMusicInput(BaseModel):
    """Input for stop_music tool (no parameters)."""
    pass


# =============================================================================
# Recipe Tool Schemas (Future)
# =============================================================================

class RecipeSearchInput(BaseModel):
    """Input for searching recipes."""
    query: str
    cuisine: Optional[str] = None
    max_time_minutes: Optional[int] = None
    dietary_restrictions: List[str] = Field(default_factory=list)


class RecipeOutput(ToolOutput):
    """Output from recipe tools."""
    data: Optional[Dict[str, Any]] = Field(default_factory=lambda: {
        "recipe_id": "",
        "title": "",
        "ingredients": [],
        "steps": [],
        "prep_time_minutes": 0,
        "cook_time_minutes": 0,
        "servings": 0,
        "source_url": None
    })


# =============================================================================
# Weather Tool Schemas (Future)
# =============================================================================

class WeatherInput(BaseModel):
    """Input for weather tool."""
    location: Optional[str] = Field(default=None, description="Location (default: current)")
    days: int = Field(default=1, ge=1, le=7, description="Forecast days")


class WeatherOutput(ToolOutput):
    """Output from weather tool."""
    data: Optional[Dict[str, Any]] = Field(default_factory=lambda: {
        "location": "",
        "current": {
            "temperature": 0,
            "feels_like": 0,
            "condition": "",
            "humidity": 0,
            "wind_speed": 0
        },
        "forecast": []
    })


# =============================================================================
# Calendar Tool Schemas (Future)
# =============================================================================

class CalendarQueryInput(BaseModel):
    """Input for querying calendar."""
    date: Optional[str] = Field(default=None, description="Date (YYYY-MM-DD) or 'today'/'tomorrow'")
    days: int = Field(default=1, ge=1, le=7)


class CalendarAddInput(BaseModel):
    """Input for adding calendar event."""
    title: str
    start_time: str = Field(description="ISO datetime or natural language")
    end_time: Optional[str] = None
    location: Optional[str] = None
    description: Optional[str] = None


class CalendarOutput(ToolOutput):
    """Output from calendar tools."""
    data: Optional[Dict[str, Any]] = Field(default_factory=lambda: {
        "events": [],
        "date_range": {
            "start": None,
            "end": None
        }
    })


# =============================================================================
# Timer Tool Schemas (Future)
# =============================================================================

class TimerSetInput(BaseModel):
    """Input for setting a timer."""
    duration: str = Field(description="Duration (e.g., '5 minutes', '1 hour 30 minutes')")
    name: Optional[str] = Field(default=None, description="Timer name")


class TimerCancelInput(BaseModel):
    """Input for cancelling a timer."""
    name: Optional[str] = Field(default=None, description="Timer name (None = cancel all)")


class TimerOutput(ToolOutput):
    """Output from timer tools."""
    data: Optional[Dict[str, Any]] = Field(default_factory=lambda: {
        "timer_id": "",
        "name": "",
        "duration_seconds": 0,
        "remaining_seconds": 0,
        "is_active": False
    })


# =============================================================================
# Note/Memory Tool Schemas (Future)
# =============================================================================

class NoteAddInput(BaseModel):
    """Input for adding a note."""
    content: str
    category: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


class NoteSearchInput(BaseModel):
    """Input for searching notes."""
    query: str
    category: Optional[str] = None
    limit: int = Field(default=5, ge=1, le=20)


class NoteOutput(ToolOutput):
    """Output from note tools."""
    data: Optional[Dict[str, Any]] = Field(default_factory=lambda: {
        "note_id": "",
        "content": "",
        "category": None,
        "tags": [],
        "created_at": None
    })


# =============================================================================
# Tool Registry Schema
# =============================================================================

class ToolDefinition(BaseModel):
    """Definition of a tool for registration."""
    name: str
    description: str
    parameters_schema: Dict[str, Any]
    input_model: Optional[str] = Field(default=None, description="Pydantic model class name")
    output_model: Optional[str] = Field(default=None, description="Pydantic model class name")
    category: str = "general"
    requires_confirmation: bool = False
    is_async: bool = False
