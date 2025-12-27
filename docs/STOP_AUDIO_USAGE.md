# Stop Audio Tool

## Overview
Stop all audio playback including music and text-to-speech. This is useful when you want to interrupt long TTS responses or stop music playback.

## Expanded Functionality

The `stop_music()` function now stops **all audio**, not just music:

### What Gets Stopped
- **Music Players**: mpv, ffplay (YouTube Music playback)
- **Text-to-Speech**: 
  - macOS: `say` command
  - Linux: espeak, espeak-ng, festival, flite
- **Any other audio processes**

## Usage

### Direct Function Call
```python
from tools.youtube_music import stop_music

# Stop all audio playback
result = stop_music()
print(result)
# Output: "Stopped audio playback (mpv, say)" or "No audio is currently playing"
```

### Voice Assistant Integration

The tool is registered in `tools/registry.py` and can be called by the AI assistant.

**Example voice commands:**
- "Stop" / "Stop it" / "Stop that"
- "Stop the music"
- "Stop talking" / "Be quiet" / "Silence"
- "Pause" / "Turn it off"
- "Stop the speech"

## Use Cases

### 1. Interrupt Long TTS Responses
When the assistant is giving a long response and you want it to stop:
```
User: "Tell me about the history of computers"
Assistant: "Computers have a fascinating history that dates back to..."
User: "Stop" [interrupts the long response]
```

### 2. Stop Music Playback
```
User: "Play Bohemian Rhapsody"
Assistant: [Music starts playing]
User: "Stop the music"
```

### 3. Emergency Silence
Quickly silence all audio output:
```
User: "Stop everything"
```

## Function Signature

```python
def stop_music() -> str:
    """
    Stop all audio playback including music and text-to-speech.
    Kills mpv, ffplay, pyttsx3, espeak, and other audio processes.
    
    Returns:
        Result message indicating what was stopped
    """
```

## Platform Support

### macOS
- Stops `mpv`, `ffplay`, `say` (TTS)
- Uses `pkill -9` to forcefully terminate processes

### Linux
- Stops `mpv`, `ffplay`, `espeak`, `espeak-ng`, `festival`, `flite`
- Uses `pkill -9` to forcefully terminate processes

## Testing

Three test files are provided:

1. **test_stop_all_audio.py** - Comprehensive tests for all audio types
2. **test_tts_interrupt.py** - Interactive test for interrupting TTS
3. **test_stop_music.py** - Original music-only tests (still works)

Run tests:
```bash
python test_stop_all_audio.py
python test_tts_interrupt.py  # Listen for TTS being interrupted
```

## Implementation Details

The function:
1. Attempts to kill all known audio processes using `pkill -9`
2. Returns a list of what was stopped
3. Returns "No audio is currently playing" if nothing was running
4. Handles errors gracefully

### Process Termination
Uses `pkill -9` (SIGKILL) for immediate termination:
- `-9` ensures processes are killed immediately
- Works across different audio backends
- Safe to call even if processes aren't running

## Safety Notes

- Uses `SIGKILL` (-9) for immediate termination
- Will not kill the calling Python process itself
- Safe to call repeatedly
- No side effects if no audio is playing
