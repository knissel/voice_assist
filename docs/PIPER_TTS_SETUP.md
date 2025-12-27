# Piper TTS Setup

This project uses [Piper](https://github.com/rhasspy/piper) for fast, local text-to-speech synthesis. Piper provides ultra-low latency (~100-200ms) compared to cloud TTS services (1-3 seconds).

## Installation

### 1. Install Dependencies

On your Raspberry Pi, install the required system libraries:

```bash
sudo apt-get update
sudo apt-get install libportaudio2
```

Then install Python packages:

```bash
pip install -r requirements.txt
```

### 2. Download a Voice Model

Piper requires a voice model file. Download one from the [official Piper voices repository](https://huggingface.co/rhasspy/piper-voices/tree/main).

**Recommended for Raspberry Pi:**
- `en_US-lessac-medium` - Good quality, reasonable size (~63MB)
- `en_US-lessac-low` - Faster, smaller (~18MB) but lower quality

**Download example (medium quality):**

```bash
# Create models directory
mkdir -p piper_models

# Download the model and config
cd piper_models
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
cd ..
```

**Important:** Both the `.onnx` and `.onnx.json` files must be in the same directory.

### 3. Configure Environment (Optional)

By default, the assistant looks for the model at:
```
./piper_models/en_US-lessac-medium.onnx
```

To use a different model, add to your `.env` file:

```bash
PIPER_MODEL=/path/to/your/model.onnx
```

## Voice Options

Browse all available voices at: https://huggingface.co/rhasspy/piper-voices/tree/main

Popular English voices:
- **lessac** - Clear, neutral American English (recommended)
- **amy** - British English female
- **ryan** - American English male
- **libritts** - Various speakers

Quality levels:
- **low** - Fastest, smallest (~18MB), good for Raspberry Pi Zero/3
- **medium** - Balanced quality/speed (~63MB), recommended for Pi 4/5
- **high** - Best quality (~100MB+), may be slow on older Pis

## Testing

Test TTS from command line:

```bash
echo "Hello, this is a test" | piper --model piper_models/en_US-lessac-medium.onnx --output_file test.wav
aplay test.wav
```

## Troubleshooting

### "Piper model not found" error
- Verify both `.onnx` and `.onnx.json` files exist in `piper_models/`
- Check the `PIPER_MODEL` path in your `.env` file

### "PortAudio library not found" error
```bash
sudo apt-get install libportaudio2
```

### Audio playback issues
- Ensure your audio output is configured: `sudo raspi-config` → System Options → Audio
- Test with: `speaker-test -t wav -c 2`

### Slow performance
- Try a "low" quality model instead of "medium"
- Ensure you're using a 64-bit OS on Raspberry Pi 4/5
- Close other applications to free up CPU/RAM

## Performance Comparison

| TTS Method | Latency | Quality | Internet Required |
|------------|---------|---------|-------------------|
| Piper (local) | 100-200ms | Good | No |
| Google Cloud TTS | 1-2s | Excellent | Yes |
| Gemini TTS | 1-3s | Excellent | Yes |

Piper is 5-15x faster than cloud TTS, making it ideal for responsive voice assistants on Raspberry Pi.
