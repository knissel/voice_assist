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
- 🐳 **Docker Support**: Distributed architecture with Edge/Compute node split

## Docker Deployment (Distributed Architecture)

The assistant is now optimized for a distributed setup using Docker. This separates the high-performance "Brain" (Compute Node) from the "Ear/Mouth" (Endpoint Node).

### 1. Compute Node (RTX 5090 / High-End PC)
Handles STT, LLM orchestration, and high-quality TTS.

1.  **Prerequisites**: Install [Docker](https://docs.docker.com/get-docker/) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
2.  **Configuration**: Set your `GEMINI_API_KEY` in the root `.env` file.
3.  **Launch**:
    ```bash
    docker compose up compute --build
    ```

### 2. Edge Node (Mini PC / Raspberry Pi 5)
Handles microphone input, wake word detection, and audio playback.

1.  **Prerequisites**: Install Docker.
2.  **Configuration**: In `.env`, set `COMPUTE_SERVER_URL` to your Compute Node's IP (e.g., `http://192.168.1.50:8000`).
3.  **Launch**:
    ```bash
    docker compose up edge --build
    ```

> **Note for Pi 5 Users**: Ensure your Docker user is in the `audio` group to allow the container to access hardware sound devices (`/dev/snd`).

## Prerequisites (Non-Docker)

- Python 3.9+
- macOS or Linux (Raspberry Pi supported)
- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) installed locally
- Google Gemini API key
- OpenWakeWord models (no API key required)
- Piper TTS voice model (for local text-to-speech)

## Project Structure (New Architecture)

```
voice_assist/
├── docker-compose.yml      # Orchestrates Edge & Compute containers
├── services/
│   ├── compute/            # Brain Service (GPU/ML Heavy)
│   │   ├── Dockerfile
│   │   └── src/            # FastAPI server + Core Logic
│   └── edge/               # Endpoint Service (Audio I/O)
│       ├── Dockerfile
│       └── src/            # Wake Word Detection Loop
├── core/                   # Shared libraries
├── tools/                  # Specific tool integrations
├── docs/                   # Documentation & Setup Guides
├── tests/                  # Automated tests
└── scripts/                # Setup & Maintenance scripts
```

## Legacy Installation (Manual)

See [Installation Guide](docs/INSTALLATION_MANUAL.md) for non-docker setups.

## Usage (Edge Node)

### Wake Word Mode (Continuous Listening)

The Edge Node runs the wake word listener:

```bash
python services/edge/src/main.py
```

Say the wake word, then speak your command. The assistant will:
1. Detect the wake word locally on the Edge Node
2. Record your voice command and send it to the Compute Node
3. Transcribe, process with Gemini, and synthesize audio on the Compute Node
4. Play back the response audio on the Edge Node speakers

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

## Profiling & Debugging

See [Debugging Guide](docs/DEBUGGING.md) for more details.

## Troubleshooting

### "No speech detected"
- Check microphone permissions
- Speak louder or closer to the microphone
- Ensure your `MIC_DEVICE_INDEX` is correct in `.env`

### "Gemini API failed"
- Verify your API key in `.env`
- Check your internet connection

### Wake word not detecting
- Verify `WAKEWORD_MODELS` points to a valid model
- Check microphone input levels with `scripts/find_devices.py`

## Contributing

Feel free to submit issues and pull requests!

## License

MIT License

## Acknowledgments

- [Google Gemini](https://ai.google.dev/) for the LLM
- [Piper](https://github.com/rhasspy/piper) for fast local TTS
- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) for ASR
- [OpenWakeWord](https://github.com/dscripka/openWakeWord) for wake word detection
