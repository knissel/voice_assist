"""
State machine for voice assistant.
Provides explicit state transitions, timeout handling, and watchdog functionality.
"""
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Optional, Set, Dict, Callable, Any
import threading
import time
import logging

logger = logging.getLogger(__name__)


class AssistantState(Enum):
    """All possible states of the voice assistant."""
    IDLE = auto()           # Listening for wake word
    WAKE_DETECTED = auto()  # Wake word heard, preparing to record
    LISTENING = auto()      # Recording user speech
    TRANSCRIBING = auto()   # Sending audio to ASR
    THINKING = auto()       # Waiting for LLM response
    EXECUTING = auto()      # Running a tool
    SPEAKING = auto()       # TTS playback
    ERROR = auto()          # Recoverable error state
    CANCELLED = auto()      # User cancelled (barge-in)


@dataclass
class StateContext:
    """Context data for current state."""
    state: AssistantState
    entered_at: float = field(default_factory=time.time)
    correlation_id: str = ""
    transcript: Optional[str] = None
    response: Optional[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class StateMachine:
    """
    State machine with timeout handling and event emission.
    
    Usage:
        from core.event_bus import EventBus
        
        bus = EventBus()
        sm = StateMachine(bus)
        
        sm.transition(AssistantState.WAKE_DETECTED)
        sm.transition(AssistantState.LISTENING)
        # ...
    """
    
    # Timeout in seconds for each state (None = no timeout)
    TIMEOUTS: Dict[AssistantState, Optional[float]] = {
        AssistantState.IDLE: None,
        AssistantState.WAKE_DETECTED: 2.0,
        AssistantState.LISTENING: 15.0,
        AssistantState.TRANSCRIBING: 30.0,
        AssistantState.THINKING: 30.0,
        AssistantState.EXECUTING: 60.0,
        AssistantState.SPEAKING: 120.0,
        AssistantState.ERROR: 5.0,
        AssistantState.CANCELLED: 1.0,
    }
    
    # Valid state transitions
    VALID_TRANSITIONS: Dict[AssistantState, Set[AssistantState]] = {
        AssistantState.IDLE: {
            AssistantState.WAKE_DETECTED,
            AssistantState.ERROR,
        },
        AssistantState.WAKE_DETECTED: {
            AssistantState.LISTENING,
            AssistantState.IDLE,
            AssistantState.ERROR,
        },
        AssistantState.LISTENING: {
            AssistantState.TRANSCRIBING,
            AssistantState.CANCELLED,
            AssistantState.ERROR,
            AssistantState.IDLE,
        },
        AssistantState.TRANSCRIBING: {
            AssistantState.THINKING,
            AssistantState.ERROR,
            AssistantState.IDLE,
        },
        AssistantState.THINKING: {
            AssistantState.EXECUTING,
            AssistantState.SPEAKING,
            AssistantState.ERROR,
            AssistantState.IDLE,
        },
        AssistantState.EXECUTING: {
            AssistantState.SPEAKING,
            AssistantState.THINKING,  # For multi-turn tool use
            AssistantState.ERROR,
            AssistantState.IDLE,
        },
        AssistantState.SPEAKING: {
            AssistantState.IDLE,
            AssistantState.CANCELLED,
            AssistantState.LISTENING,  # For follow-up questions
            AssistantState.ERROR,
        },
        AssistantState.ERROR: {
            AssistantState.IDLE,
        },
        AssistantState.CANCELLED: {
            AssistantState.IDLE,
            AssistantState.LISTENING,  # Resume listening after barge-in
        },
    }
    
    def __init__(self, event_bus=None):
        """
        Initialize state machine.
        
        Args:
            event_bus: Optional EventBus for emitting state change events
        """
        self.context = StateContext(state=AssistantState.IDLE)
        self._lock = threading.RLock()
        self._event_bus = event_bus
        self._on_timeout_callback: Optional[Callable[[AssistantState], None]] = None
        self._watchdog_thread: Optional[threading.Thread] = None
        self._watchdog_running = False
    
    @property
    def state(self) -> AssistantState:
        """Get current state."""
        with self._lock:
            return self.context.state
    
    @property
    def time_in_state(self) -> float:
        """Get seconds spent in current state."""
        with self._lock:
            return time.time() - self.context.entered_at
    
    def can_transition(self, new_state: AssistantState) -> bool:
        """Check if transition to new_state is valid."""
        with self._lock:
            valid = self.VALID_TRANSITIONS.get(self.context.state, set())
            return new_state in valid
    
    def transition(
        self,
        new_state: AssistantState,
        correlation_id: Optional[str] = None,
        **kwargs
    ) -> bool:
        """
        Transition to a new state.
        
        Args:
            new_state: Target state
            correlation_id: Optional correlation ID (uses existing if not provided)
            **kwargs: Additional context data (transcript, response, error, etc.)
        
        Returns:
            True if transition succeeded, False if invalid
        
        Raises:
            ValueError: If transition is invalid and strict mode is enabled
        """
        with self._lock:
            old_state = self.context.state
            
            if not self.can_transition(new_state):
                logger.warning(
                    f"Invalid transition: {old_state.name} -> {new_state.name}"
                )
                return False
            
            # Update context
            self.context.state = new_state
            self.context.entered_at = time.time()
            
            if correlation_id:
                self.context.correlation_id = correlation_id
            
            # Update optional fields
            for key in ['transcript', 'response', 'error']:
                if key in kwargs:
                    setattr(self.context, key, kwargs[key])
            
            # Update metadata
            if 'metadata' in kwargs:
                self.context.metadata.update(kwargs['metadata'])
            
            logger.info(
                f"State: {old_state.name} -> {new_state.name} "
                f"[{self.context.correlation_id}]"
            )
            
            # Emit event
            if self._event_bus:
                self._event_bus.emit("state_changed", {
                    "from_state": old_state.name,
                    "to_state": new_state.name,
                    "correlation_id": self.context.correlation_id
                })
            
            return True
    
    def reset(self) -> None:
        """Reset to IDLE state."""
        with self._lock:
            old_state = self.context.state
            self.context = StateContext(state=AssistantState.IDLE)
            
            if self._event_bus and old_state != AssistantState.IDLE:
                self._event_bus.emit("state_changed", {
                    "from_state": old_state.name,
                    "to_state": AssistantState.IDLE.name,
                    "reason": "reset"
                })
    
    def check_timeout(self) -> bool:
        """
        Check if current state has timed out.
        
        Returns:
            True if timed out, False otherwise
        """
        with self._lock:
            timeout = self.TIMEOUTS.get(self.context.state)
            if timeout is None:
                return False
            return self.time_in_state > timeout
    
    def set_timeout_callback(self, callback: Callable[[AssistantState], None]) -> None:
        """Set callback to be called when a state times out."""
        self._on_timeout_callback = callback
    
    def start_watchdog(self, check_interval: float = 1.0) -> None:
        """
        Start watchdog thread that monitors for timeouts.
        
        Args:
            check_interval: How often to check for timeouts (seconds)
        """
        if self._watchdog_running:
            return
        
        self._watchdog_running = True
        self._watchdog_thread = threading.Thread(
            target=self._watchdog_loop,
            args=(check_interval,),
            daemon=True,
            name="StateWatchdog"
        )
        self._watchdog_thread.start()
        logger.info("State watchdog started")
    
    def stop_watchdog(self) -> None:
        """Stop the watchdog thread."""
        self._watchdog_running = False
        if self._watchdog_thread:
            self._watchdog_thread.join(timeout=2.0)
            self._watchdog_thread = None
        logger.info("State watchdog stopped")
    
    def _watchdog_loop(self, check_interval: float) -> None:
        """Watchdog monitoring loop."""
        while self._watchdog_running:
            time.sleep(check_interval)
            
            if self.check_timeout():
                state = self.state
                logger.warning(f"State timeout: {state.name}")
                
                # Emit error event
                if self._event_bus:
                    self._event_bus.emit("error", {
                        "error_code": "STATE_TIMEOUT",
                        "message": f"Timeout in state: {state.name}",
                        "recoverable": True
                    })
                
                # Call timeout callback
                if self._on_timeout_callback:
                    try:
                        self._on_timeout_callback(state)
                    except Exception as e:
                        logger.error(f"Timeout callback error: {e}")
                
                # Auto-transition to ERROR then IDLE
                self.transition(AssistantState.ERROR, error=f"Timeout in {state.name}")
                time.sleep(0.5)
                self.transition(AssistantState.IDLE)


class CircuitBreaker:
    """
    Circuit breaker for remote service calls.
    Prevents cascading failures by failing fast when a service is down.
    
    Usage:
        breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
        
        try:
            result = breaker.call(lambda: remote_api_call())
        except CircuitBreakerOpen:
            # Service is down, use fallback
            result = fallback_response()
    """
    
    class State(Enum):
        CLOSED = auto()     # Normal operation
        OPEN = auto()       # Failing, reject calls
        HALF_OPEN = auto()  # Testing if recovered
    
    def __init__(
        self,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        name: str = "default"
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.name = name
        
        self._failures = 0
        self._state = self.State.CLOSED
        self._last_failure_time = 0.0
        self._lock = threading.Lock()
    
    @property
    def state(self) -> 'CircuitBreaker.State':
        with self._lock:
            return self._state
    
    @property
    def is_open(self) -> bool:
        with self._lock:
            if self._state == self.State.OPEN:
                # Check if recovery timeout has passed
                if time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = self.State.HALF_OPEN
                    return False
                return True
            return False
    
    def call(self, func: Callable, fallback: Optional[Callable] = None):
        """
        Execute function with circuit breaker protection.
        
        Args:
            func: Function to call
            fallback: Optional fallback function if circuit is open
        
        Returns:
            Result from func or fallback
        
        Raises:
            CircuitBreakerOpen: If circuit is open and no fallback provided
        """
        with self._lock:
            if self._state == self.State.OPEN:
                if time.time() - self._last_failure_time > self.recovery_timeout:
                    self._state = self.State.HALF_OPEN
                    logger.info(f"Circuit breaker '{self.name}' half-open, testing...")
                else:
                    logger.debug(f"Circuit breaker '{self.name}' open, rejecting call")
                    if fallback:
                        return fallback()
                    raise CircuitBreakerOpen(f"Circuit breaker '{self.name}' is open")
        
        try:
            result = func()
            
            with self._lock:
                if self._state == self.State.HALF_OPEN:
                    logger.info(f"Circuit breaker '{self.name}' recovered, closing")
                    self._state = self.State.CLOSED
                    self._failures = 0
            
            return result
            
        except Exception as e:
            with self._lock:
                self._failures += 1
                self._last_failure_time = time.time()
                
                if self._failures >= self.failure_threshold:
                    self._state = self.State.OPEN
                    logger.warning(
                        f"Circuit breaker '{self.name}' opened after "
                        f"{self._failures} failures"
                    )
            
            raise
    
    def reset(self) -> None:
        """Manually reset the circuit breaker."""
        with self._lock:
            self._state = self.State.CLOSED
            self._failures = 0
            self._last_failure_time = 0.0


class CircuitBreakerOpen(Exception):
    """Raised when circuit breaker is open."""
    pass
