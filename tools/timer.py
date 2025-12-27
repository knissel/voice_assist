"""
Timer tool for voice assistant.
Supports setting, canceling, and listing timers with audio alerts.
"""
import threading
import time
import subprocess
import shutil
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import os

# Try to import event bus for UI notifications
try:
    from core.event_bus import EventBus
    EVENT_BUS_AVAILABLE = True
except ImportError:
    EVENT_BUS_AVAILABLE = False


@dataclass
class Timer:
    """Represents a single timer."""
    id: str
    name: str
    duration_seconds: int
    created_at: datetime = field(default_factory=datetime.now)
    end_time: datetime = field(init=False)
    cancelled: bool = False
    _thread: Optional[threading.Thread] = field(default=None, repr=False)
    
    def __post_init__(self):
        self.end_time = self.created_at + timedelta(seconds=self.duration_seconds)
    
    @property
    def remaining_seconds(self) -> int:
        """Get remaining seconds on timer."""
        if self.cancelled:
            return 0
        remaining = (self.end_time - datetime.now()).total_seconds()
        return max(0, int(remaining))
    
    @property
    def is_active(self) -> bool:
        """Check if timer is still running."""
        return not self.cancelled and self.remaining_seconds > 0
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "duration_seconds": self.duration_seconds,
            "remaining_seconds": self.remaining_seconds,
            "end_time": self.end_time.isoformat(),
            "is_active": self.is_active,
            "cancelled": self.cancelled
        }


class TimerManager:
    """Manages multiple timers with thread-safe operations."""
    
    def __init__(self, event_bus: Optional['EventBus'] = None):
        self._timers: Dict[str, Timer] = {}
        self._lock = threading.Lock()
        self._counter = 0
        self.event_bus = event_bus
    
    def _generate_id(self) -> str:
        """Generate a unique timer ID."""
        self._counter += 1
        return f"timer_{self._counter}"
    
    def _emit_event(self, event_type: str, data: dict):
        """Emit event to event bus if available."""
        if self.event_bus:
            self.event_bus.emit(event_type, data)
    
    def _play_alarm(self, timer: Timer):
        """Play alarm sound when timer completes."""
        print(f"🔔 TIMER COMPLETE: {timer.name}!")
        
        # Emit event for UI
        self._emit_event("timer_complete", {
            "id": timer.id,
            "name": timer.name
        })
        
        # Try to play alarm sound
        # First try system sounds, then fall back to beep
        alarm_played = False
        
        # macOS system sound
        if shutil.which("afplay"):
            sound_paths = [
                "/System/Library/Sounds/Glass.aiff",
                "/System/Library/Sounds/Ping.aiff",
                "/System/Library/Sounds/Hero.aiff"
            ]
            for sound in sound_paths:
                if os.path.exists(sound):
                    # Play 3 times for attention
                    for _ in range(3):
                        subprocess.run(["afplay", sound], check=False)
                        time.sleep(0.3)
                    alarm_played = True
                    break
        
        # Linux - try paplay or aplay
        if not alarm_played and shutil.which("paplay"):
            sound_paths = [
                "/usr/share/sounds/freedesktop/stereo/alarm-clock-elapsed.oga",
                "/usr/share/sounds/freedesktop/stereo/complete.oga"
            ]
            for sound in sound_paths:
                if os.path.exists(sound):
                    for _ in range(3):
                        subprocess.run(["paplay", sound], check=False)
                        time.sleep(0.3)
                    alarm_played = True
                    break
        
        # Fallback: use speaker-test for beep on Linux
        if not alarm_played and shutil.which("speaker-test"):
            subprocess.run(
                ["speaker-test", "-t", "sine", "-f", "1000", "-l", "1"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=2
            )
        
        # Try TTS announcement
        try:
            # Import here to avoid circular dependency
            from tools.youtube_music import stop_music
            # Don't stop music, just announce over it
        except:
            pass
    
    def _timer_thread(self, timer: Timer):
        """Background thread that waits for timer completion."""
        while timer.remaining_seconds > 0 and not timer.cancelled:
            time.sleep(1)
            
            # Emit periodic updates for UI (every 5 seconds or last 10 seconds)
            remaining = timer.remaining_seconds
            if remaining <= 10 or remaining % 5 == 0:
                self._emit_event("timer_tick", {
                    "id": timer.id,
                    "name": timer.name,
                    "remaining_seconds": remaining
                })
        
        if not timer.cancelled:
            self._play_alarm(timer)
        
        # Clean up completed timer after a delay
        time.sleep(5)
        with self._lock:
            if timer.id in self._timers and not self._timers[timer.id].is_active:
                del self._timers[timer.id]
    
    def set_timer(self, duration_seconds: int, name: str = None) -> Timer:
        """
        Set a new timer.
        
        Args:
            duration_seconds: How long the timer should run
            name: Optional name for the timer (e.g., "pasta", "laundry")
        
        Returns:
            The created Timer object
        """
        with self._lock:
            timer_id = self._generate_id()
            
            # Generate name if not provided
            if not name:
                name = f"Timer {self._counter}"
            
            timer = Timer(
                id=timer_id,
                name=name,
                duration_seconds=duration_seconds
            )
            
            # Start background thread
            thread = threading.Thread(
                target=self._timer_thread,
                args=(timer,),
                daemon=True,
                name=f"Timer-{timer_id}"
            )
            timer._thread = thread
            thread.start()
            
            self._timers[timer_id] = timer
            
            # Emit event
            self._emit_event("timer_started", timer.to_dict())
            
            print(f"⏱️  Timer set: {name} for {self._format_duration(duration_seconds)}")
            return timer
    
    def cancel_timer(self, timer_id: str = None, name: str = None) -> bool:
        """
        Cancel a timer by ID or name.
        If neither provided, cancels the most recent timer.
        
        Returns:
            True if timer was cancelled, False if not found
        """
        with self._lock:
            timer = None
            
            if timer_id and timer_id in self._timers:
                timer = self._timers[timer_id]
            elif name:
                # Find by name (case-insensitive)
                for t in self._timers.values():
                    if t.name.lower() == name.lower() and t.is_active:
                        timer = t
                        break
            else:
                # Cancel most recent active timer
                active = [t for t in self._timers.values() if t.is_active]
                if active:
                    timer = max(active, key=lambda t: t.created_at)
            
            if timer and timer.is_active:
                timer.cancelled = True
                self._emit_event("timer_cancelled", {"id": timer.id, "name": timer.name})
                print(f"❌ Timer cancelled: {timer.name}")
                return True
            
            return False
    
    def list_timers(self) -> List[Timer]:
        """Get all active timers."""
        with self._lock:
            return [t for t in self._timers.values() if t.is_active]
    
    def get_timer(self, timer_id: str) -> Optional[Timer]:
        """Get a specific timer by ID."""
        with self._lock:
            return self._timers.get(timer_id)
    
    @staticmethod
    def _format_duration(seconds: int) -> str:
        """Format seconds as human-readable duration."""
        if seconds < 60:
            return f"{seconds} seconds"
        elif seconds < 3600:
            mins = seconds // 60
            secs = seconds % 60
            if secs:
                return f"{mins} minutes and {secs} seconds"
            return f"{mins} minutes"
        else:
            hours = seconds // 3600
            mins = (seconds % 3600) // 60
            if mins:
                return f"{hours} hours and {mins} minutes"
            return f"{hours} hours"
    
    @staticmethod
    def parse_duration(duration_str: str) -> int:
        """
        Parse a duration string into seconds.
        Supports: "5 minutes", "1 hour 30 minutes", "90 seconds", "1h30m", etc.
        """
        duration_str = duration_str.lower().strip()
        total_seconds = 0
        
        # Handle formats like "1h30m" or "5m"
        import re
        
        # Try parsing "Xh Ym Zs" format
        hours = re.search(r'(\d+)\s*h', duration_str)
        minutes = re.search(r'(\d+)\s*m(?:in)?', duration_str)
        seconds = re.search(r'(\d+)\s*s(?:ec)?', duration_str)
        
        if hours:
            total_seconds += int(hours.group(1)) * 3600
        if minutes:
            total_seconds += int(minutes.group(1)) * 60
        if seconds:
            total_seconds += int(seconds.group(1))
        
        if total_seconds > 0:
            return total_seconds
        
        # Try parsing "X hours Y minutes" format
        hours = re.search(r'(\d+)\s*hours?', duration_str)
        minutes = re.search(r'(\d+)\s*minutes?', duration_str)
        seconds = re.search(r'(\d+)\s*seconds?', duration_str)
        
        if hours:
            total_seconds += int(hours.group(1)) * 3600
        if minutes:
            total_seconds += int(minutes.group(1)) * 60
        if seconds:
            total_seconds += int(seconds.group(1))
        
        if total_seconds > 0:
            return total_seconds
        
        # Try parsing just a number (assume minutes)
        just_number = re.match(r'^(\d+)$', duration_str)
        if just_number:
            return int(just_number.group(1)) * 60
        
        raise ValueError(f"Could not parse duration: {duration_str}")


# Global timer manager instance
_timer_manager: Optional[TimerManager] = None


def _get_timer_manager() -> TimerManager:
    """Get or create the global timer manager."""
    global _timer_manager
    if _timer_manager is None:
        # Try to get event bus
        event_bus = None
        try:
            import sys
            if 'wakeword' in sys.modules and hasattr(sys.modules['wakeword'], "event_bus"):
                event_bus = sys.modules['wakeword'].event_bus
            elif '__main__' in sys.modules and hasattr(sys.modules['__main__'], "event_bus"):
                event_bus = sys.modules['__main__'].event_bus
        except:
            pass
        _timer_manager = TimerManager(event_bus)
    return _timer_manager


# === Tool Functions for Gemini ===

def set_timer(duration_minutes: int, name: str = None) -> str:
    """
    Set a timer for a specified duration.
    
    Args:
        duration_minutes: Duration in minutes
        name: Optional name for the timer (e.g., "pasta", "eggs")
    
    Returns:
        Confirmation message
    """
    manager = _get_timer_manager()
    duration_seconds = duration_minutes * 60
    timer = manager.set_timer(duration_seconds, name)
    
    formatted = manager._format_duration(duration_seconds)
    if name:
        return f"Timer '{name}' set for {formatted}"
    return f"Timer set for {formatted}"


def cancel_timer(name: str = None) -> str:
    """
    Cancel an active timer.
    
    Args:
        name: Name of timer to cancel. If not provided, cancels the most recent timer.
    
    Returns:
        Confirmation message
    """
    manager = _get_timer_manager()
    
    if manager.cancel_timer(name=name):
        if name:
            return f"Cancelled the {name} timer"
        return "Timer cancelled"
    
    return "No active timer found to cancel"


def list_timers() -> str:
    """
    List all active timers.
    
    Returns:
        List of active timers with remaining time
    """
    manager = _get_timer_manager()
    timers = manager.list_timers()
    
    if not timers:
        return "No active timers"
    
    lines = []
    for t in timers:
        remaining = manager._format_duration(t.remaining_seconds)
        lines.append(f"- {t.name}: {remaining} remaining")
    
    return "Active timers:\n" + "\n".join(lines)


def check_timer(name: str = None) -> str:
    """
    Check how much time is left on a timer.
    
    Args:
        name: Name of timer to check. If not provided, checks the most recent timer.
    
    Returns:
        Time remaining message
    """
    manager = _get_timer_manager()
    timers = manager.list_timers()
    
    if not timers:
        return "No active timers"
    
    timer = None
    if name:
        for t in timers:
            if t.name.lower() == name.lower():
                timer = t
                break
        if not timer:
            return f"No timer named '{name}' found"
    else:
        timer = timers[0]
    
    remaining = manager._format_duration(timer.remaining_seconds)
    return f"{timer.name} has {remaining} remaining"
