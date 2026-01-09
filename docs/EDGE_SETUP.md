# Edge Node Setup Notes (Linux)

This doc captures the commands used to run the edge assistant on Linux and to manage it with systemd.

## Run Manually

```bash
cd /home/kenny/voice_assist
.venv/bin/python services/edge/src/main.py
```

## MIC_DEVICE_INDEX (ReSpeaker selection)

Set in `.env`:

```bash
MIC_DEVICE_INDEX=3
# or name substring
MIC_DEVICE_INDEX=ReSpeaker
```

## ReSpeaker DSP Controls (AEC/NS/AGC)

The ReSpeaker mic array exposes DSP controls over USB. These settings are optional and only apply if the device is detected.

```bash
# Enable/disable DSP control
RESPEAKER_DSP_ENABLED=true

# Toggle features
RESPEAKER_AEC=true
RESPEAKER_NS=true
RESPEAKER_AGC=true

# Optional USB IDs (defaults shown)
RESPEAKER_USB_VID=0x2886
RESPEAKER_USB_PID=0x0018
```

## ReSpeaker LED Ring

```bash
# Enable/disable LED ring control
RESPEAKER_LED_ENABLED=true

# Optional: brightness 0-255
RESPEAKER_LED_BRIGHTNESS=64
```

## Systemd User Service

Install/enable (already done in this setup):

```bash
mkdir -p ~/.config/systemd/user
cp /home/kenny/voice_assist/voice-assistant-edge.service ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable voice-assistant-edge.service
```

Start/stop/status:

```bash
systemctl --user start voice-assistant-edge.service
systemctl --user stop voice-assistant-edge.service
systemctl --user status voice-assistant-edge.service --no-pager
```

Logs:

```bash
journalctl --user -u voice-assistant-edge.service -f
```

## Auto-start Without Login (Optional)

```bash
sudo loginctl enable-linger $USER
```
