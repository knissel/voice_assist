# GPU TTS Setup Guide

This guide explains how to set up high-quality XTTS v2 text-to-speech on your RTX 5090 GPU with automatic fallback to local Piper TTS on the Raspberry Pi.

## Architecture

```
┌─────────────────┐     HTTP/JSON      ┌─────────────────┐
│  Raspberry Pi 5 │ ◄───────────────► │  RTX 5090 GPU   │
│                 │                    │                 │
│  - Wakeword     │   /synthesize      │  - XTTS v2      │
│  - STT          │   (text → wav)     │  - High quality │
│  - Piper (fallback)                  │  - ~200-300ms   │
└─────────────────┘                    └─────────────────┘
```

## GPU Server Setup (RTX 5090)

### 1. Install Dependencies

#### Windows with RTX 5090 (Blackwell Architecture)

The RTX 5090 uses the new Blackwell architecture (sm_120) which requires **PyTorch nightly** with CUDA 12.8:

```powershell
# Install PyTorch nightly with CUDA 12.8 (required for RTX 5090)
py -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install TTS and other dependencies
py -m pip install flask TTS transformers==4.35.2 soundfile
```

> **Note**: The `transformers==4.35.2` pinning is required for XTTS v2 compatibility.
> **Note**: `soundfile` is used for audio I/O to avoid FFmpeg dependency issues on Windows.

#### Linux with RTX 5090

```bash
# Install PyTorch nightly with CUDA 12.8
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

# Install TTS and dependencies
pip install flask TTS transformers==4.35.2 soundfile
```

#### Older GPUs (RTX 30xx/40xx)

For older NVIDIA GPUs, you can use the stable PyTorch release:

```bash
# On your GPU machine
cd voice_assist
pip install -r requirements_gpu_server.txt
```

This installs:
- `TTS>=0.22.0` - Coqui TTS library with XTTS v2
- `torch>=2.0.0` - PyTorch with CUDA support
- `torchaudio>=2.0.0` - Audio processing
- `flask>=3.0.0` - HTTP server

### 2. (Optional) Add a Speaker Reference

For best results, provide a 6-30 second WAV file of the voice you want to clone:

```bash
# Place your reference audio in the repo root
cp /path/to/your/voice.wav speaker_reference.wav
```

Or upload via API after starting the server:
```bash
curl -X POST -F "audio=@your_voice.wav" http://localhost:5001/set_speaker
```

### 3. Start the XTTS Server

```bash
# Basic usage (downloads model on first run, ~2GB)
python xtts_server.py --port 5001

# With custom speaker reference
python xtts_server.py --port 5001 --speaker /path/to/voice.wav

# Specify GPU device
python xtts_server.py --port 5001 --device cuda
```

The server exposes:
- `GET /health` - Health check
- `POST /synthesize` - Synthesize text to speech
- `POST /set_speaker` - Upload speaker reference audio

### 4. Test the Server

```bash
# Health check
curl http://localhost:5001/health

# Synthesize speech
curl -X POST -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test.", "language": "en"}' \
  http://localhost:5001/synthesize --output test.wav

# Play the result
ffplay test.wav
```

## Raspberry Pi Setup

### 1. Configure Environment Variables

Add to your `.env` file:

```bash
# Enable GPU TTS with Piper fallback
USE_GPU_TTS=true

# URL of your XTTS server (replace with your GPU machine's IP)
XTTS_SERVER_URL=http://192.168.1.100:5001
```

### 2. That's It!

The assistant will automatically:
1. Try the GPU XTTS server first
2. Fall back to local Piper if the server is unavailable or times out
3. Cache server health status to avoid repeated failed requests

## Configuration Options

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `USE_GPU_TTS` | `true` | Enable GPU TTS with fallback |
| `XTTS_SERVER_URL` | `http://localhost:5001` | XTTS server URL |

### GPU TTS Client Options

The `GPUTTSClient` class accepts:
- `server_url` - XTTS server URL
- `timeout_seconds` - Max wait time (default: 3.0s)
- `piper_voice` - Piper voice for fallback
- `language` - Language code (default: "en")

## Performance

### Expected Latency

| Component | Time |
|-----------|------|
| Network round-trip | 20-50ms |
| XTTS inference (5090) | 150-300ms |
| Audio transfer | 20-50ms |
| **Total GPU TTS** | **200-350ms** |
| **Piper (local)** | **100-150ms** |

### Quality Comparison

| TTS Engine | Quality | Naturalness | Voice Cloning |
|------------|---------|-------------|---------------|
| Piper (lessac-medium) | Good | Moderate | No |
| XTTS v2 | Excellent | High | Yes |

## Troubleshooting

### Server won't start
- Ensure CUDA is available: `python -c "import torch; print(torch.cuda.is_available())"`
- Check GPU memory: XTTS needs ~2GB VRAM
- **RTX 5090**: If you see "sm_120 is not compatible", install PyTorch nightly with CUDA 12.8:
  ```powershell
  py -m pip install --pre --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
  ```
- **Windows torchcodec errors**: Ensure `soundfile` is installed (`pip install soundfile`). The server uses soundfile for audio I/O to avoid FFmpeg dependencies.

### High latency
- Check network: `ping <gpu-server-ip>`
- Reduce timeout: Set `timeout_seconds=2.0` in client
- Ensure GPU isn't throttling

### Fallback always used
- Check server health: `curl http://<server>:5001/health`
- Check firewall allows port 5001
- Verify `XTTS_SERVER_URL` is correct

### No speaker voice
- Upload a reference: `curl -X POST -F "audio=@voice.wav" http://server:5001/set_speaker`
- Use 6-30 seconds of clear speech, single speaker

## Running Both Servers

For a complete GPU-accelerated setup, run both Whisper and XTTS:

```bash
# Terminal 1: Whisper STT server
python whisper_server.py --model large-v3 --port 5000

# Terminal 2: XTTS TTS server  
python xtts_server.py --port 5001 --speaker speaker_reference.wav
```

Or create a systemd service for production use.
