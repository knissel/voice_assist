"""
Audio output routing for macOS/Linux.
Routes audio to specific sinks via system commands.
"""
import subprocess
import platform

def get_audio_sinks():
    """Get available audio output sinks."""
    system = platform.system()
    
    if system == "Darwin":  # macOS
        try:
            result = subprocess.run(
                ["system_profiler", "SPAudioDataType"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout
        except Exception as e:
            return f"Failed to get audio sinks: {str(e)}"
    
    elif system == "Linux":
        # Try wpctl first (PipeWire)
        try:
            result = subprocess.run(
                ["wpctl", "status"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return result.stdout
        except FileNotFoundError:
            pass
        
        # Fall back to pactl (PulseAudio)
        try:
            result = subprocess.run(
                ["pactl", "list", "sinks", "short"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout
        except Exception as e:
            return f"Failed to get audio sinks: {str(e)}"
    
    return "Unsupported platform for audio routing"

def set_audio_sink(sink_id: str):
    """
    Set the default audio output sink.
    
    Args:
        sink_id: Sink identifier (device name or ID)
    
    Returns:
        Success or error message
    """
    system = platform.system()
    
    if system == "Darwin":  # macOS
        try:
            # Use SwitchAudioSource if available
            result = subprocess.run(
                ["SwitchAudioSource", "-s", sink_id],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return f"Audio output switched to {sink_id}"
            return f"Failed to switch audio: {result.stderr}"
        except FileNotFoundError:
            return "SwitchAudioSource not installed. Install via: brew install switchaudio-osx"
        except Exception as e:
            return f"Error switching audio: {str(e)}"
    
    elif system == "Linux":
        # Try wpctl first (PipeWire)
        try:
            result = subprocess.run(
                ["wpctl", "set-default", sink_id],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return f"Audio output switched to sink {sink_id}"
        except FileNotFoundError:
            pass
        
        # Fall back to pactl (PulseAudio)
        try:
            result = subprocess.run(
                ["pactl", "set-default-sink", sink_id],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return f"Audio output switched to sink {sink_id}"
            return f"Failed to switch audio: {result.stderr}"
        except Exception as e:
            return f"Error switching audio: {str(e)}"
    
    return "Unsupported platform for audio routing"

def route_to_bluetooth(device_name: str = None):
    """
    Route audio to Bluetooth device.
    Falls back to default sink if Bluetooth fails.
    
    Args:
        device_name: Optional specific Bluetooth device name
    
    Returns:
        Success or error message with fallback info
    """
    system = platform.system()
    
    if system == "Darwin":  # macOS
        try:
            # List audio devices and find Bluetooth
            result = subprocess.run(
                ["SwitchAudioSource", "-a", "-t", "output"],
                capture_output=True,
                text=True,
                timeout=5
            )
            devices = result.stdout.strip().split('\n')
            
            # Find Bluetooth device
            bt_device = None
            for device in devices:
                if device_name and device_name.lower() in device.lower():
                    bt_device = device
                    break
                elif "bluetooth" in device.lower() or "airpods" in device.lower():
                    bt_device = device
                    break
            
            if bt_device:
                switch_result = set_audio_sink(bt_device)
                return switch_result
            else:
                return "No Bluetooth audio device found. Using default output."
        except FileNotFoundError:
            return "SwitchAudioSource not installed. Using default output."
        except Exception as e:
            return f"Bluetooth routing failed: {str(e)}. Using default output."
    
    elif system == "Linux":
        # Try to find Bluetooth sink
        try:
            sinks = get_audio_sinks()
            # Look for bluez in sink names
            for line in sinks.split('\n'):
                if 'bluez' in line.lower() or (device_name and device_name.lower() in line.lower()):
                    # Extract sink ID (first column in pactl output)
                    sink_id = line.split()[0]
                    return set_audio_sink(sink_id)
            return "No Bluetooth audio sink found. Using default output."
        except Exception as e:
            return f"Bluetooth routing failed: {str(e)}. Using default output."
    
    return "Bluetooth audio routing not supported on this platform. Using default output."

def get_current_volume():
    """
    Get the current system volume level.
    
    Returns:
        Volume level as integer (0-100) or error message
    """
    system = platform.system()
    
    if system == "Darwin":  # macOS
        try:
            result = subprocess.run(
                ["osascript", "-e", "output volume of (get volume settings)"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return int(result.stdout.strip())
            return f"Failed to get volume: {result.stderr}"
        except Exception as e:
            return f"Error getting volume: {str(e)}"
    
    elif system == "Linux":
        # Try wpctl first (PipeWire)
        try:
            result = subprocess.run(
                ["wpctl", "get-volume", "@DEFAULT_AUDIO_SINK@"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Output format: "Volume: 0.50" -> convert to percentage
                volume_str = result.stdout.strip().split()[-1]
                volume = int(float(volume_str) * 100)
                return volume
        except FileNotFoundError:
            pass
        
        # Fall back to pactl (PulseAudio)
        try:
            result = subprocess.run(
                ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse output like: "Volume: front-left: 32768 /  50% / -18.06 dB"
                for part in result.stdout.split():
                    if '%' in part:
                        return int(part.replace('%', ''))
            return "Failed to parse volume"
        except Exception as e:
            return f"Error getting volume: {str(e)}"
    
    return "Unsupported platform for volume control"

def set_volume(level: int):
    """
    Set the system volume to a specific level.
    
    Args:
        level: Volume level (0-100)
    
    Returns:
        Success or error message
    """
    # Clamp level to valid range
    level = max(0, min(100, level))
    
    system = platform.system()
    
    if system == "Darwin":  # macOS
        try:
            result = subprocess.run(
                ["osascript", "-e", f"set volume output volume {level}"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return f"Volume set to {level}%"
            return f"Failed to set volume: {result.stderr}"
        except Exception as e:
            return f"Error setting volume: {str(e)}"
    
    elif system == "Linux":
        # Try wpctl first (PipeWire)
        try:
            # wpctl expects decimal (0.0-1.0)
            volume_decimal = level / 100.0
            result = subprocess.run(
                ["wpctl", "set-volume", "@DEFAULT_AUDIO_SINK@", str(volume_decimal)],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return f"Volume set to {level}%"
        except FileNotFoundError:
            pass
        
        # Fall back to pactl (PulseAudio)
        try:
            result = subprocess.run(
                ["pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{level}%"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                return f"Volume set to {level}%"
            return f"Failed to set volume: {result.stderr}"
        except Exception as e:
            return f"Error setting volume: {str(e)}"
    
    return "Unsupported platform for volume control"

def adjust_volume(direction: str, amount: int = 10):
    """
    Adjust the system volume up or down by a specific amount.
    
    Args:
        direction: "up" or "down"
        amount: Amount to adjust by (default 10%)
    
    Returns:
        Success or error message with new volume level
    """
    current = get_current_volume()
    
    if isinstance(current, str):
        return current
    
    if direction.lower() == "up":
        new_level = min(100, current + amount)
    elif direction.lower() == "down":
        new_level = max(0, current - amount)
    else:
        return f"Invalid direction: {direction}. Use 'up' or 'down'."
    
    result = set_volume(new_level)
    return f"{result} (was {current}%)"

def control_volume(action: str, level: int = None):
    """
    Control system volume with various actions.
    
    Args:
        action: "up", "down", or "set"
        level: Target level for "set" action (0-100), or adjustment amount for up/down (default 10)
    
    Returns:
        Success or error message
    """
    action = action.lower()
    
    if action == "set":
        if level is None:
            return "Error: level required for 'set' action"
        return set_volume(level)
    
    elif action in ["up", "down"]:
        amount = level if level is not None else 10
        return adjust_volume(action, amount)
    
    else:
        return f"Invalid action: {action}. Use 'up', 'down', or 'set'."
