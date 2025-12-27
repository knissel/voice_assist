# Raspberry Pi Headless Setup Guide

This guide explains how to run your voice assistant on a Raspberry Pi **without a screen**, with direct audio playback through speakers or Bluetooth.

## Why Headless Mode?

In your kitchen setup, the Pi may not have a screen. Headless mode:
- ✅ Streams music **directly** to the Pi's audio output
- ✅ Works with **speakers** or **Bluetooth** devices
- ✅ No browser needed
- ✅ Lower resource usage

## Installation on Raspberry Pi

### 1. Install System Dependencies

```bash
# Update system
sudo apt-get update
sudo apt-get upgrade -y

# Install audio player (choose one)
# Option A: mpv (recommended - lightweight and reliable)
sudo apt-get install mpv -y

# Option B: ffmpeg (alternative)
sudo apt-get install ffmpeg -y

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Configure Headless Mode

Edit your `.env` file on the Raspberry Pi:

```bash
nano .env
```

Add or update this line:
```
HEADLESS_PLAYBACK=true
```

Save and exit (Ctrl+X, Y, Enter).

### 3. Test Audio Output

Make sure your audio is working:

```bash
# Test speaker output
speaker-test -t wav -c 2

# Or test with mpv
mpv --no-video https://www.youtube.com/watch?v=dQw4w9WgXcQ
```

### 4. Configure Audio Output

**For Built-in Audio Jack:**
```bash
# Set audio to 3.5mm jack
sudo raspi-config
# Navigate to: System Options > Audio > Headphones
```

**For Bluetooth Speakers:**
```bash
# Pair Bluetooth device first
bluetoothctl
# Then in bluetoothctl:
# scan on
# pair XX:XX:XX:XX:XX:XX
# connect XX:XX:XX:XX:XX:XX
# trust XX:XX:XX:XX:XX:XX
# exit

# Your voice assistant can also connect Bluetooth via the bluetooth tool
```

## How It Works

### Desktop Mode (HEADLESS_PLAYBACK=false)
- Opens YouTube Music in your browser
- Good for machines with screens

### Headless Mode (HEADLESS_PLAYBACK=true)
1. Voice assistant searches YouTube Music
2. Finds the song/playlist
3. Streams audio directly using `mpv` or `ffplay`
4. Plays through Pi's audio output (speakers/Bluetooth)

## Usage Examples

Once running, say:
- "Play Bohemian Rhapsody"
- "Play Taylor Swift"
- "Play my workout playlist"
- "Play jazz music"

The music will play directly through your Pi's speakers!

## Troubleshooting

### No audio output
```bash
# Check audio devices
aplay -l

# Test audio
speaker-test -t wav -c 2

# Set default audio device
sudo raspi-config
```

### "mpv not found" error
```bash
# Install mpv
sudo apt-get install mpv -y
```

### Bluetooth audio issues
```bash
# Make sure Bluetooth device is connected
bluetoothctl devices
bluetoothctl info XX:XX:XX:XX:XX:XX

# Reconnect if needed
bluetoothctl connect XX:XX:XX:XX:XX:XX
```

### Music plays but no sound
```bash
# Check volume
alsamixer

# Unmute and increase volume with arrow keys
# Press M to unmute, arrow up to increase
```

## Performance Tips

**For Raspberry Pi Zero/3:**
- Use `mpv` (lighter than ffmpeg)
- Lower quality streams save bandwidth:
  ```bash
  # Edit the tool to use lower quality
  # In youtube_music.py, change mpv command to:
  # ["mpv", "--no-video", "--ytdl-format=bestaudio[height<=480]", url]
  ```

**For Raspberry Pi 4/5:**
- Default settings work great
- Can handle high-quality audio streams

## Switching Between Desktop and Pi

**On your Mac/Desktop:**
```bash
# In .env file
HEADLESS_PLAYBACK=false
```

**On your Raspberry Pi:**
```bash
# In .env file
HEADLESS_PLAYBACK=true
```

This way the same code works on both machines!

## Auto-start on Boot (Optional)

To start the assistant automatically when Pi boots:

```bash
# Create systemd service
sudo nano /etc/systemd/system/voice-assistant.service
```

Add:
```ini
[Unit]
Description=Voice Assistant
After=network.target sound.target

[Service]
Type=simple
User=pi
WorkingDirectory=/home/pi/voice_assist
ExecStart=/usr/bin/python3 /home/pi/voice_assist/run_assistant.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable voice-assistant
sudo systemctl start voice-assistant

# Check status
sudo systemctl status voice-assistant
```

## Summary

Your voice assistant now supports **two modes**:

| Mode | Use Case | Playback Method |
|------|----------|----------------|
| Desktop | Mac/PC with screen | Opens browser |
| Headless | Raspberry Pi in kitchen | Direct audio streaming |

Just set `HEADLESS_PLAYBACK=true` on your Pi and music will play directly through speakers without needing a screen!
