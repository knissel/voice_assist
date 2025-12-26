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
