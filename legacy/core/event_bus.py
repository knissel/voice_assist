"""
Event bus for UI-agnostic event distribution.
Allows the core assistant to emit events that any UI can subscribe to.
"""
from typing import Callable, Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import threading
import queue
import json
import logging
import uuid

logger = logging.getLogger(__name__)


@dataclass
class Event:
    """A single event emitted by the assistant."""
    type: str
    data: Dict[str, Any]
    correlation_id: str = ""
    timestamp: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.type,
            "data": self.data,
            "correlation_id": self.correlation_id,
            "timestamp": self.timestamp.isoformat()
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())


class EventBus:
    """
    Thread-safe event bus for pub/sub communication.
    
    Usage:
        bus = EventBus()
        bus.start()
        
        # Subscribe to specific events
        bus.subscribe("transcript_final", lambda e: print(e.data))
        
        # Subscribe to all events (for UI streaming)
        bus.subscribe("*", lambda e: websocket.send(e.to_json()))
        
        # Emit events
        bus.emit("transcript_final", {"text": "Hello"}, correlation_id="abc123")
    """
    
    def __init__(self, max_queue_size: int = 1000):
        self._subscribers: Dict[str, List[Callable[[Event], None]]] = {}
        self._lock = threading.RLock()
        self._queue: queue.Queue = queue.Queue(maxsize=max_queue_size)
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._correlation_id: str = ""
    
    def new_correlation_id(self) -> str:
        """Generate a new correlation ID for a conversation turn."""
        self._correlation_id = str(uuid.uuid4())[:8]
        return self._correlation_id
    
    @property
    def correlation_id(self) -> str:
        """Get current correlation ID."""
        return self._correlation_id
    
    def subscribe(self, event_type: str, callback: Callable[[Event], None]) -> None:
        """
        Subscribe to events of a specific type.
        Use "*" to subscribe to all events.
        """
        with self._lock:
            if event_type not in self._subscribers:
                self._subscribers[event_type] = []
            self._subscribers[event_type].append(callback)
            logger.debug(f"Subscribed to '{event_type}': {callback.__name__}")
    
    def unsubscribe(self, event_type: str, callback: Callable[[Event], None]) -> bool:
        """Unsubscribe a callback from an event type."""
        with self._lock:
            if event_type in self._subscribers:
                try:
                    self._subscribers[event_type].remove(callback)
                    return True
                except ValueError:
                    pass
        return False
    
    def emit(self, event_type: str, data: Dict[str, Any], correlation_id: Optional[str] = None) -> None:
        """
        Emit an event to all subscribers.
        Events are queued and processed asynchronously.
        """
        event = Event(
            type=event_type,
            data=data,
            correlation_id=correlation_id or self._correlation_id
        )
        
        try:
            self._queue.put_nowait(event)
        except queue.Full:
            logger.warning(f"Event queue full, dropping event: {event_type}")
    
    def emit_sync(self, event_type: str, data: Dict[str, Any], correlation_id: Optional[str] = None) -> None:
        """
        Emit an event synchronously (blocks until all callbacks complete).
        Use sparingly - prefer emit() for non-blocking behavior.
        """
        event = Event(
            type=event_type,
            data=data,
            correlation_id=correlation_id or self._correlation_id
        )
        self._dispatch_event(event)
    
    def start(self) -> None:
        """Start the event processing thread."""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._process_events, daemon=True, name="EventBus")
        self._thread.start()
        logger.info("EventBus started")
    
    def stop(self, timeout: float = 2.0) -> None:
        """Stop the event processing thread."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("EventBus stopped")
    
    def _process_events(self) -> None:
        """Main event processing loop."""
        while self._running:
            try:
                event = self._queue.get(timeout=0.1)
                self._dispatch_event(event)
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error processing event: {e}")
    
    def _dispatch_event(self, event: Event) -> None:
        """Dispatch event to all matching subscribers."""
        with self._lock:
            # Get specific subscribers
            callbacks = list(self._subscribers.get(event.type, []))
            # Add wildcard subscribers
            callbacks.extend(self._subscribers.get("*", []))
        
        for callback in callbacks:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error for '{event.type}': {e}")
    
    def clear_subscribers(self, event_type: Optional[str] = None) -> None:
        """Clear subscribers for a specific event type or all."""
        with self._lock:
            if event_type:
                self._subscribers.pop(event_type, None)
            else:
                self._subscribers.clear()


# Convenience functions for common events
def emit_state_changed(bus: EventBus, from_state: str, to_state: str, reason: str = None):
    """Emit a state change event."""
    bus.emit("state_changed", {
        "from_state": from_state,
        "to_state": to_state,
        "reason": reason
    })


def emit_transcript(bus: EventBus, text: str, is_final: bool = True, duration_ms: int = 0):
    """Emit a transcript event."""
    event_type = "transcript_final" if is_final else "transcript_partial"
    bus.emit(event_type, {
        "text": text,
        "is_final": is_final,
        "duration_ms": duration_ms
    })


def emit_assistant_text(bus: EventBus, text: str, is_partial: bool = False):
    """Emit assistant response text."""
    bus.emit("assistant_text", {
        "text": text,
        "is_partial": is_partial
    })


def emit_tool_call(bus: EventBus, tool_name: str, arguments: Dict[str, Any]):
    """Emit a tool call event."""
    bus.emit("tool_call", {
        "tool_name": tool_name,
        "arguments": arguments
    })


def emit_tool_result(bus: EventBus, tool_name: str, success: bool, result: Any, duration_ms: int = 0):
    """Emit a tool result event."""
    bus.emit("tool_result", {
        "tool_name": tool_name,
        "success": success,
        "result": result,
        "duration_ms": duration_ms
    })


def emit_error(bus: EventBus, error_code: str, message: str, recoverable: bool = True):
    """Emit an error event."""
    bus.emit("error", {
        "error_code": error_code,
        "message": message,
        "recoverable": recoverable
    })


def emit_notification(bus: EventBus, level: str, title: str, message: str, duration_ms: int = 5000):
    """Emit a notification event."""
    bus.emit("notification", {
        "level": level,
        "title": title,
        "message": message,
        "duration_ms": duration_ms
    })
