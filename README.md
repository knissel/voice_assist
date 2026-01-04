# Voice Assistant with Gemini Flash

A voice-controlled assistant powered by Google Gemini Flash that can control smart home devices, manage Bluetooth connections, and answer questions using natural language.

## Features

- 🎤 **Wake Word Detection**: Uses OpenWakeWord (open source) for built-in or custom wake words
- 🗣️ **Speech-to-Text**: Local transcription with Whisper.cpp
- 🤖 **AI Processing**: Google Gemini 2.5 Flash Lite for natural language understanding
- 🔧 **Function Calling**: Control smart home lights, Bluetooth devices, audio routing, and YouTube Music
- 🔊 **Text-to-Speech**: Ultra-low latency local TTS using Piper (~100-200ms)
- ⌨️ **Push-to-Talk Mode**: Alternative mode using spacebar to activate
- 🎵 **YouTube Music**: Play songs, albums, artists, and playlists
- 🔉 **Volume Control**: Adjust system volume with voice commands

## Prerequisites

- Python 3.9+
- macOS or Linux (Raspberry Pi supported)
- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) installed locally
- Google Gemini API key
- OpenWakeWord models (no API key required)
- Piper TTS voice model (for local text-to-speech)

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd voice_assist
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install system dependencies**
   ```bash
   # Linux/Raspberry Pi
   sudo apt-get update
   sudo apt-get install libportaudio2
   ```

4. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up Whisper.cpp**
   ```bash
   # Clone and build whisper.cpp
   git clone https://github.com/ggerganov/whisper.cpp.git
   cd whisper.cpp
   make
   
   # Download a model (tiny is recommended for speed)
   bash ./models/download-ggml-model.sh tiny
   cd ..
   ```

6. **Set up Piper TTS**
   
   Download a voice model (see [PIPER_TTS_SETUP.md](docs/PIPER_TTS_SETUP.md) for details):
   ```bash
   mkdir -p piper_models
   cd piper_models
   wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
   wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
   cd ..
   ```

7. **Configure environment variables**
   
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```bash
   # Gemini API Configuration
   GEMINI_API_KEY=your_gemini_api_key_here
   MODEL_NAME=gemini-2.5-flash-lite
   
   # OpenWakeWord Configuration (open source wake word)
   # WAKEWORD_MODELS=hey_jarvis
   # WAKEWORD_THRESHOLD=0.5
   # WAKEWORD_FRAME_LENGTH=1280
   # WAKEWORD_INPUT_SAMPLE_RATE=16000
   
   # Whisper Configuration (update paths if needed)
   WHISPER_PATH=/path/to/whisper.cpp/build/bin/whisper-cli
   MODEL_PATH=/path/to/whisper.cpp/models/ggml-tiny.bin
   
   # Piper TTS Configuration (optional, uses default if not set)
   # PIPER_MODEL=/path/to/your/piper/model.onnx
   
   # Control4 Configuration (if using Control4 smart home)
   CONTROL4_USERNAME=your_username
   CONTROL4_PASSWORD=your_password
   CONTROL4_CONTROLLER_IP=192.168.x.x
   ```

## Getting API Keys

### Google Gemini API Key
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key to your `.env` file

### OpenWakeWord Models
1. No API keys required
2. Default: `WAKEWORD_MODELS=hey_jarvis` (auto-downloads on first run)
3. Custom model: set `WAKEWORD_MODELS=/path/to/your_model.onnx`
4. Training guide: see `docs/WAKEWORD_OPENWAKEWORD.md`

### Piper TTS Voice Models
1. See [PIPER_TTS_SETUP.md](docs/PIPER_TTS_SETUP.md) for detailed setup
2. Download voice models from [Piper Voices](https://huggingface.co/rhasspy/piper-voices)
3. Recommended: `en_US-lessac-medium` for Raspberry Pi
4. Completely free and runs locally (no API costs)

## Usage

### Wake Word Mode (Continuous Listening)

Run the wake word listener (default wake word: `hey_jarvis`):

```bash
python wakeword.py
```

Say the wake word, then speak your command. The assistant will:
1. Detect the wake word
2. Record your voice command (4 seconds)
3. Transcribe it using Whisper.cpp
4. Process with Gemini Flash Lite
5. Execute actions or respond verbally with ultra-fast local TTS

**Recommended for Raspberry Pi** - hands-free operation without keyboard

### Push-to-Talk Mode

Run the push-to-talk assistant:

```bash
python run_assistant.py
```

- Press **SPACE** to record a command
- Press **ESC** to quit

## Supported Commands

### Lighting Control
- "Turn on the kitchen lights"
- "Set kitchen island to 50%"
- "Turn off the family room lights"
- "Dim the foyer lights"

**Device IDs:**
- Kitchen Cans: 85
- Foyer: 87
- Stairs: 89
- Upstairs Hall: 91
- Front Door: 93
- Kitchen Island: 95
- Downstairs Hallway: 97
- Upstairs Deck: 99
- Family Room: 204
- Breakfast: 206

### Bluetooth & Audio Control
- "Connect to [device name]"
- "Disconnect Bluetooth"
- "Route audio to Bluetooth"
- "Turn up the volume"
- "Set volume to 50%"
- "Turn down the volume"

### YouTube Music
- "Play [song name]"
- "Play [artist name]"
- "Play [album name]"
- "Stop the music"
- "Pause"

### General Questions
Ask any question and Computer will respond with concise answers.

## Project Structure

```
voice_assist/
├── .env                    # Environment variables (not in git)
├── .env.example           # Template for environment variables
├── .gitignore             # Git ignore rules
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── run_assistant.py      # Push-to-talk mode (main entry point)
├── wakeword.py           # Wake word detection mode (main entry point)
├── main.py               # Basic example script
├── docs/                 # Documentation
│   ├── RASPBERRY_PI_SETUP.md
│   ├── PIPER_TTS_SETUP.md
│   ├── YOUTUBE_MUSIC_SETUP.md
│   ├── STOP_AUDIO_USAGE.md
│   └── VOLUME_CONTROL_USAGE.md
├── tests/                # Test files
│   ├── test_volume_control.py
│   ├── test_stop_music.py
│   ├── test_youtube_music.py
│   └── ... (other test files)
├── scripts/              # Utility scripts
│   ├── setup_youtube_music.py
│   ├── find_devices.py
│   ├── find_scenes.py
│   └── ... (other utility scripts)
├── tools/                # Core tool modules
│   ├── registry.py       # Tool definitions and dispatch
│   ├── control4_tool.py  # Control4 smart home integration
│   ├── lights.py         # Lighting control utilities
│   ├── bluetooth.py      # Bluetooth device management
│   ├── audio.py          # Audio routing and volume control
│   └── youtube_music.py  # YouTube Music integration
└── whisper.cpp/          # Whisper.cpp installation (gitignored)
```

## Customization

### Change Wake Words

Set `WAKEWORD_MODELS` in your `.env`:
```bash
WAKEWORD_MODELS=hey_jarvis
```

Use a custom model:
```bash
WAKEWORD_MODELS=models/wakeword/my_wakeword.onnx
```

Multiple models can be comma-separated:
```bash
WAKEWORD_MODELS=hey_jarvis,models/wakeword/my_wakeword.onnx
```

To train your own wake word, record samples and follow `docs/WAKEWORD_OPENWAKEWORD.md`.

### Change Voice Model

Download a different Piper voice model and update your `.env`:
```bash
PIPER_MODEL=/path/to/different/model.onnx
```

Available voices: Browse [Piper Voices](https://huggingface.co/rhasspy/piper-voices/tree/main)

Popular options:
- `en_US-lessac-medium` - Clear, neutral (recommended)
- `en_US-amy-medium` - British English female
- `en_US-ryan-medium` - American English male

See [PIPER_TTS_SETUP.md](docs/PIPER_TTS_SETUP.md) for more voice options

### Add New Tools

1. Define your function in a new file or existing tool file
2. Add the function declaration to `tools/registry.py` in `GEMINI_TOOLS`
3. Add the function to `TOOL_FUNCTIONS` mapping
4. The dispatcher will automatically handle calls

## Profiling & Debugging

### CPU Profiling with py-spy
```bash
# Install
pip install py-spy

# Profile running assistant (requires sudo on Linux)
sudo py-spy top --pid $(pgrep -f wakeword.py)

# Generate flame graph
sudo py-spy record -o profile.svg --pid $(pgrep -f wakeword.py)
```

### Memory Profiling
```bash
pip install memory_profiler
python -m memory_profiler wakeword.py
```

### cProfile for Function-Level Analysis
```bash
python -m cProfile -s cumtime wakeword.py 2>&1 | head -50
```

### Audio Debugging
```bash
# List audio devices
python -c "import pyaudio; p = pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"

# Test recording (Linux)
arecord -d 5 -f S16_LE -r 16000 test.wav
aplay test.wav

# Test recording (macOS)
rec -r 16000 -c 1 test.wav trim 0 5
play test.wav
```

### Latency Measurement
The assistant logs timing for each stage. Look for:
- `🎧 Transcribing...` → ASR latency
- `🧠 Processing...` → LLM latency  
- `💬` → TTS start

## Remote Deployment

Deploy code changes to your Raspberry Pi without physical access.

### Prerequisites

1. **SSH key authentication** set up with your Pi:
   ```bash
   # Generate SSH key if you don't have one
   ssh-keygen -t ed25519
   
   # Copy to Pi
   ssh-copy-id pi@raspberrypi.local
   ```

2. **rsync** available on your dev machine:
   - **Windows**: Install via WSL, Git Bash, or [cwRsync](https://itefix.net/cwrsync)
   - **macOS/Linux**: Pre-installed

3. **Configure deployment**:
   ```bash
   cp deploy.config.example deploy.config
   # Edit deploy.config with your Pi's hostname/IP
   ```

### Deploy Commands

**PowerShell (Windows):**
```powershell
.\deploy.ps1 --all              # Deploy everything and restart
.\deploy.ps1 --ui               # Deploy UI only (no restart needed)
.\deploy.ps1 --wakeword         # Deploy wakeword code and restart
.\deploy.ps1 --wakeword --logs  # Deploy and tail logs
.\deploy.ps1 --restart          # Just restart the service
.\deploy.ps1 --dry-run --all    # Preview what would be deployed
```

**Bash (WSL/Git Bash/Linux/macOS):**
```bash
./deploy.sh --all              # Deploy everything and restart
./deploy.sh --ui               # Deploy UI only
./deploy.sh --wakeword --logs  # Deploy wakeword and show logs
```

### What Gets Deployed

| Flag | Files | Restart |
|------|-------|---------|
| `--ui` | `ui/` | No |
| `--wakeword` | `wakeword.py`, `core/`, `tools/`, `adapters/`, `schemas/` | Yes |
| `--all` | Everything above + config files | Yes |

## Deployment as systemd Service

### Install Service (Raspberry Pi)

1. Copy the service file:
```bash
sudo cp voice-assistant.service /etc/systemd/system/
```

2. Edit paths if needed:
```bash
sudo nano /etc/systemd/system/voice-assistant.service
```

3. Enable and start:
```bash
sudo systemctl daemon-reload
sudo systemctl enable voice-assistant
sudo systemctl start voice-assistant
```

4. Check status:
```bash
sudo systemctl status voice-assistant
journalctl -u voice-assistant -f  # Live logs
```

### Service Management
```bash
sudo systemctl stop voice-assistant     # Stop
sudo systemctl restart voice-assistant  # Restart
sudo systemctl disable voice-assistant  # Disable auto-start
```

## Troubleshooting

### "No speech detected"
- Check microphone permissions
- Speak louder or closer to the microphone
- Increase `RECORD_SECONDS` in the script

### "Gemini API failed"
- Verify your API key in `.env`
- Check your internet connection
- Ensure you haven't exceeded API rate limits

### Wake word not detecting
- Verify `WAKEWORD_MODELS` points to a valid model
- Try speaking the wake word more clearly
- Check microphone input levels
- Lower `WAKEWORD_THRESHOLD` if detection is too strict

### Whisper transcription errors
- Ensure Whisper.cpp is properly built
- Verify paths in `.env` are correct
- Try a different model (base or small for better accuracy)

## Contributing

Feel free to submit issues and pull requests!

## License

MIT License - feel free to use this project for personal or commercial purposes.

## Acknowledgments

- [Google Gemini](https://ai.google.dev/) for the LLM
- [Piper](https://github.com/rhasspy/piper) for ultra-fast local text-to-speech
- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) for speech recognition
- [OpenWakeWord](https://github.com/dscripka/openWakeWord) for wake word detection
- [pyControl4](https://github.com/lawtancool/pyControl4) for Control4 integration
