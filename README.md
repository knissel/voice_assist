# Voice Assistant with Gemini Flash

A distributed voice assistant powered by Google Gemini Flash with real-time speech processing, smart home control, and WebSocket audio streaming.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    COMPUTE NODE (RTX 5090)                       │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │   Whisper   │    │   Gemini    │    │    XTTS     │          │
│  │     STT     │    │     LLM     │    │     TTS     │          │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘          │
│         │                  │                  │                  │
│         └──────────────────┴──────────────────┘                  │
│                            │                                     │
│                   FastAPI Server (:8000)                         │
│                   WebSocket + HTTP                               │
└────────────────────────────┬────────────────────────────────────┘
                             │ WebSocket Audio Stream
                             ▼
┌─────────────────────────────────────────────────────────────────┐
│                     EDGE NODE (Mini PC)                          │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐          │
│  │ Microphone  │    │  Wake Word  │    │   Speaker   │          │
│  │    Input    │───▶│  Detection  │───▶│   Output    │          │
│  └─────────────┘    └─────────────┘    └─────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## Features

- 🎤 **Wake Word Detection**: OpenWakeWord with custom wake words
- 🗣️ **Speech-to-Text**: GPU-accelerated Whisper (~100ms latency)
- 🤖 **AI Processing**: Gemini 2.5 Flash Lite with function calling
- 🔊 **Streaming TTS**: XTTS v2 with real-time audio streaming (~0.5s to first audio)
- 🏠 **Smart Home Control**: Control4 lighting integration
- 🎵 **YouTube Music**: Play songs, albums, and playlists
- ⏱️ **Timers**: Voice-controlled timers with announcements

## Quick Start

### 1. Start XTTS Server (GPU TTS)
```powershell
./scripts/refresh_xtts.ps1 -WaitForHealth
```

### 2. Start Compute Node (Docker)
```powershell
docker compose up compute -d
```

### 3. Deploy Edge Node
```powershell
./scripts/deploy_native.ps1
```

## Project Structure

```
voice_assist/
├── docker-compose.yml       # Compute container orchestration
├── xtts_server.py           # XTTS TTS server (runs natively)
├── speaker_reference.wav    # Voice cloning reference
├── services/
│   ├── compute/             # GPU/ML processing (Docker)
│   │   └── src/server.py    # FastAPI + WebSocket server
│   └── edge/                # Audio I/O (runs on Mini PC)
│       └── src/main.py      # Wake word + streaming client
├── scripts/
│   ├── deploy_native.ps1    # Deploy Edge to Mini PC
│   ├── refresh_xtts.ps1     # Restart XTTS server
│   └── refresh_whisper.ps1  # Restart Whisper server
├── docs/                    # Documentation
├── models/                  # Wake word models
└── legacy/                  # Old monolithic code (archived)
```

## Configuration

Create `.env` file in root:
```bash
# Gemini
GEMINI_API_KEY=your_key_here
MODEL_NAME=gemini-2.5-flash-lite

# Servers
XTTS_SERVER_URL=http://192.168.20.148:5001
WHISPER_SERVER_URL=http://192.168.20.148:5000

# Control4 (Smart Home)
CONTROL4_USERNAME=your_username
CONTROL4_PASSWORD=your_password
CONTROL4_CONTROLLER_IP=192.168.20.1
```

Create `.env` in `services/edge/`:
```bash
COMPUTE_SERVER_URL=http://192.168.20.148:8000
MIC_DEVICE_INDEX=0
WAKEWORD_MODELS=models/wakeword/oo_gway.onnx
WAKEWORD_THRESHOLD=0.5
```

## Updating Speaker Voice

To change the TTS voice, record a new `speaker_reference.wav` (6-30 seconds) and restart XTTS:
```powershell
./scripts/refresh_xtts.ps1 -Speaker "path/to/new_reference.wav" -WaitForHealth
```

## Performance

| Metric | Latency |
|--------|---------|
| Wake word → Transcript | ~125ms |
| Time to first audio | ~0.5s (streaming) |
| Full TTS synthesis | ~5-6s (241 chars) |

## Supported Commands

### Lighting
- "Turn on the kitchen lights"
- "Set kitchen island to 50%"
- "Turn off all the lights"

### YouTube Music
- "Play [song/artist/album]"
- "Pause" / "Stop the music"

### General
- Ask any question for AI-powered responses

## Troubleshooting

### No audio playback
- Check `MIC_DEVICE_INDEX` with `scripts/find_devices.py`
- Ensure XTTS server is healthy: `curl http://localhost:5001/health`

### Slow TTS
- XTTS requires GPU (~6s for long text is normal)
- For faster responses, ask shorter questions

### Multiple Python processes
The deploy script auto-kills stale processes, but manually:
```powershell
ssh kenny@192.168.20.48 "taskkill /F /IM python.exe"
```

## License

MIT License
