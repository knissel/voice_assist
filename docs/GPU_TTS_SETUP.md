# GPU TTS Setup Guide (Qwen3-TTS)

This guide explains how to set up high-quality **Qwen3-TTS** (Qwen/Qwen3-TTS-12Hz-1.7B-Base) on your RTX 5090 GPU with automatic fallback to local Piper TTS on the Raspberry Pi.

## Architecture

```
┌─────────────────┐     HTTP/JSON      ┌─────────────────┐
│  Raspberry Pi 5 │ ◄───────────────► │  RTX 5090 GPU   │
│                 │                    │                 │
│  - Wakeword     │   /synthesize      │  - Qwen3-TTS    │
│  - STT          │   (text → wav)     │  - Voice Cloning│
│  - Piper (fallback)                  │  - ~200-400ms   │
└─────────────────┘                    └─────────────────┘
```

## GPU Server Setup (RTX 5090)

### 1. Install Dependencies

#### Windows with RTX 5090
Ensure you are using the `.venv` environment used by the Voice Assistant.

```powershell
# Activate environment
.venv\Scripts\activate

# Install Qwen3-TTS dependencies
pip install -r requirements_qwen.txt

# (Optional) Install Flash Attention 2 for faster inference
# Note: Requires compilation environment or pre-built wheels
pip install flash-attn --no-build-isolation
```

**Requirements File (`requirements_qwen.txt`)**:
- `qwen-tts`
- `flask`
- `soundfile`
- `requests`

### 2. Configure Speaker Reference

Qwen3-TTS provides excellent zero-shot voice cloning.

1.  **Audio File**: Place a 6-10 second WAV file of the target voice at `service\edge\src\speaker_reference.wav` (or in the root `voice_assist` directory).
2.  **Transcript (Recommended)**: Create a text file `speaker_reference.txt` next to the WAV file containing the *exact content* of the audio.
    - Example: If `speaker_reference.wav` says "Hello, this is a test.", create `speaker_reference.txt` with "Hello, this is a test."
    - **Why?** Qwen3-TTS uses the text to extract better prosody and timbre matching. If simpler `x_vector` mode is used (no text), quality might be lower.

### 3. Start the Server

The server is automatically started by `start_5090_services.bat`.

To run manually:
```bash
.venv\Scripts\activate
python qwen_tts_server.py --port 5001
```

The server exposes:
- `GET /health` - Health check (shows loaded model and speaker status)
- `POST /synthesize` - Synthesize text to speech
- `POST /synthesize_stream` - Streaming synthesis (segment-based)
- `POST /set_speaker` - Upload new speaker reference audio

### 4. Test the Server

```bash
# Health check
curl http://localhost:5001/health

# Synthesize speech
curl -X POST -H "Content-Type: application/json" \
  -d '{"text": "Hello, this is a test of Qwen TTS.", "language": "English"}' \
  http://localhost:5001/synthesize --output test.wav
```

## Troubleshooting

### "Model not loaded"
- Check the console logs for download errors. The model `Qwen/Qwen3-TTS-12Hz-1.7B-Base` is large (~4GB) and downloads on first run.

### Flash Attention Errors
- If you see warnings about Flash Attention, the server will fall back to standard attention (`eager`). This is fine but slightly slower.

### Speaker Voice Doesn't Match
- Ensure `speaker_reference.txt` exists and matches the audio!
- Try a clearer audio sample (single speaker, no background noise).
