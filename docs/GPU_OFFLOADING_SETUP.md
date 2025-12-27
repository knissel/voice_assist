# GPU Offloading Setup Guide

This guide explains how to set up GPU-accelerated Whisper transcription on your RTX 5090 machine with automatic fallback to local CPU transcription.

## Architecture

```
[Local Device]                    [RTX 5090 Server]
├─ Wake word detection            ├─ Whisper Large-v3
├─ Audio recording                ├─ GPU acceleration
├─ Transcription client ──────────> Flask HTTP server
│  ├─ Try remote (GPU)            └─ faster-whisper
│  └─ Fallback to local (CPU)
├─ Gemini LLM
└─ Piper TTS
```

## Benefits

- **10-20x faster transcription** when GPU is available
- **Better accuracy** using large-v3 model vs tiny
- **Automatic fallback** to local CPU if GPU server is offline
- **Zero configuration changes** needed when switching between modes

## Setup Instructions

### 1. On Your RTX 5090 Machine

#### Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements_gpu_server.txt

# Or manually:
pip install faster-whisper flask torch
```

#### Start the Whisper Server

```bash
# Basic usage (large-v3 model on port 5000)
python whisper_server.py

# Custom configuration
python whisper_server.py --model large-v3 --port 5000 --host 0.0.0.0

# Available models: tiny, base, small, medium, large-v2, large-v3, turbo
```

**Server Options:**
- `--model`: Whisper model size (default: `large-v3`)
- `--port`: Server port (default: `5000`)
- `--host`: Server host (default: `0.0.0.0` - accessible from network)
- `--device`: Device to use (default: `cuda`, can use `cpu` for testing)
- `--compute-type`: Precision (default: `float16`, options: `float32`, `int8`)

#### Verify Server is Running

```bash
# From the 5090 machine
curl http://localhost:5000/health

# From your local device (replace with actual IP)
curl http://192.168.1.100:5000/health
```

Expected response:
```json
{"status": "healthy", "model": "large-v3"}
```

### 2. On Your Local Device (Raspberry Pi/Mac)

#### Update Environment Variables

Edit your `.env` file and add:

```bash
# Set this to your 5090 machine's IP and port
WHISPER_REMOTE_URL=http://192.168.1.100:5000

# Keep these for fallback (already configured)
WHISPER_PATH=/Users/kennynissel/voice_assist/whisper.cpp/build/bin/whisper-cli
MODEL_PATH=/Users/kennynissel/voice_assist/whisper.cpp/models/ggml-tiny.bin
```

#### Install Updated Dependencies

```bash
pip install -r requirements.txt
```

#### Run Your Assistant

```bash
# Wake word mode
python wakeword.py

# Push-to-talk mode
python run_assistant.py
```

## How It Works

### Automatic Fallback Logic

1. **On startup**: Client checks if remote server is available via `/health` endpoint
2. **On transcription request**:
   - If remote server is available → Use GPU (fast, accurate)
   - If remote server fails → Fallback to local whisper.cpp (slower, but works)
3. **Health check timeout**: 2 seconds (won't slow down startup)
4. **Transcription timeout**: 30 seconds

### Performance Comparison

| Method | Model | Device | Typical Speed | Accuracy |
|--------|-------|--------|---------------|----------|
| **Remote GPU** | large-v3 | RTX 5090 | 100-300ms | Excellent |
| **Local CPU** | tiny | CPU | 1-2s | Good |

### Logging

The system will log which transcription method is being used:

```
✅ Remote Whisper server available at http://192.168.1.100:5000
🚀 GPU transcription: 0.23s
```

Or if fallback occurs:

```
⚠️  Remote server unavailable
💻 Falling back to local CPU transcription...
```

## Troubleshooting

### Server Won't Start

**Error: CUDA not available**
```bash
# Check CUDA installation
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Error: Port already in use**
```bash
# Use a different port
python whisper_server.py --port 5001

# Update .env on client
WHISPER_REMOTE_URL=http://192.168.1.100:5001
```

### Client Can't Connect

**Check network connectivity:**
```bash
# From local device
ping 192.168.1.100
curl http://192.168.1.100:5000/health
```

**Firewall issues:**
```bash
# On 5090 machine, allow port 5000
sudo ufw allow 5000/tcp  # Linux
# Or configure Windows Firewall
```

### Slow Transcription

**If GPU transcription is slow:**
- Check GPU usage: `nvidia-smi`
- Try smaller model: `--model medium` or `--model turbo`
- Reduce precision: `--compute-type int8`

**If fallback is always used:**
- Check server logs for errors
- Verify `WHISPER_REMOTE_URL` is correct in `.env`
- Test health endpoint manually

## Advanced Configuration

### Running Server as Systemd Service (Linux)

Create `/etc/systemd/system/whisper-server.service`:

```ini
[Unit]
Description=Whisper GPU Inference Server
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/path/to/voice_assist
ExecStart=/usr/bin/python3 whisper_server.py --model large-v3 --port 5000
Restart=always

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable whisper-server
sudo systemctl start whisper-server
sudo systemctl status whisper-server
```

### Multiple Clients

The server supports multiple concurrent clients. Adjust `threaded=True` in `whisper_server.py` if needed.

### Model Selection

Choose based on your needs:

- **turbo**: Fastest, good accuracy (recommended for real-time)
- **large-v3**: Best accuracy, slightly slower
- **medium**: Balanced option
- **small/base/tiny**: Not recommended for GPU (use local instead)

## Disabling GPU Offloading

To disable GPU offloading and use only local transcription:

1. Comment out or remove `WHISPER_REMOTE_URL` from `.env`
2. Or stop the server on the 5090 machine

The system will automatically fall back to local transcription.
