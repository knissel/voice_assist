# Volume Control Tool

## Overview
Control system volume through voice commands. Supports macOS and Linux (PipeWire/PulseAudio).

## Usage

### Set Volume to Specific Level
```python
from tools.audio import control_volume

# Set volume to 50%
control_volume("set", 50)

# Set volume to 0% (mute)
control_volume("set", 0)

# Set volume to 100% (max)
control_volume("set", 100)
```

### Adjust Volume Up/Down
```python
# Turn volume up by 10% (default)
control_volume("up")

# Turn volume up by 20%
control_volume("up", 20)

# Turn volume down by 10% (default)
control_volume("down")

# Turn volume down by 15%
control_volume("down", 15)
```

## Voice Assistant Integration

The tool is registered in `tools/registry.py` and can be called by the AI assistant:

**Example voice commands:**
- "Turn the volume up"
- "Turn the volume down by 20"
- "Set the volume to 50 percent"
- "Increase volume by 5"
- "Mute" (set to 0)
- "Max volume" (set to 100)

## Function Signature

```python
def control_volume(action: str, level: int = None) -> str:
    """
    Control system volume with various actions.
    
    Args:
        action: "up", "down", or "set"
        level: Target level for "set" action (0-100), 
               or adjustment amount for up/down (default 10)
    
    Returns:
        Success or error message
    """
```

## Features

- **Platform Support**: macOS (osascript) and Linux (wpctl/pactl)
- **Boundary Protection**: Automatically clamps values to 0-100 range
- **Case Insensitive**: Actions work with any case (UP, down, Set)
- **Default Adjustments**: Up/down default to 10% if no amount specified
- **Error Handling**: Clear error messages for invalid inputs

## Testing

Three test files are provided:

1. **test_volume_control.py** - Comprehensive functionality tests
2. **test_volume_interactive.py** - Interactive test with audio feedback
3. **test_volume_edge_cases.py** - Edge cases and error handling

Run tests:
```bash
python test_volume_control.py
python test_volume_edge_cases.py
python test_volume_interactive.py  # Listen for volume changes
```

## Implementation Details

### Helper Functions
- `get_current_volume()` - Returns current volume level (0-100)
- `set_volume(level)` - Sets volume to specific level
- `adjust_volume(direction, amount)` - Adjusts volume up/down
- `control_volume(action, level)` - Main control function

### Platform Commands
- **macOS**: Uses `osascript` AppleScript commands
- **Linux**: Tries `wpctl` (PipeWire) first, falls back to `pactl` (PulseAudio)
