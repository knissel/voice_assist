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
MIC_DEVICE_INDEX=pipewire
```

For the ReSpeaker 4 Mic Array on PipeWire, the most reliable wakeword behavior
was with a mono downmix:

```bash
MIC_CHANNELS=1
MIC_GAIN=1.0
```

If you explicitly capture multiple channels, you can pick one:

```bash
MIC_CHANNELS=3
MIC_CHANNEL_INDEX=0
```

## PipeWire Profile (ReSpeaker not showing as a source)

If the ReSpeaker device exists but no source node shows up, set the card profile
to a normal profile (not Pro Audio) and make the ReSpeaker the default source:

```bash
wpctl status
# Find the ReSpeaker device ID (e.g., 47) and source ID (e.g., 90)
# Use the non-pro profile index from `wpctl status` (usually 0 or 1)
wpctl set-profile <DEVICE_ID> <PROFILE_ID>
wpctl set-default <SOURCE_ID>
# Optional: set your speaker sink back to built-in audio
wpctl set-default <SINK_ID>
```

If `wpctl` hangs or the device still doesn't show, restart PipeWire/WirePlumber:

```bash
systemctl --user restart wireplumber pipewire pipewire-pulse
```

If you see "Dummy Output" or no devices in GNOME, restart PipeWire/WirePlumber
and replug the ReSpeaker.

## ALSA Direct Capture (Optional, bypass PipeWire)

Use this if PipeWire capture is unreliable. This takes exclusive access to the mic.

```bash
ALSA_INPUT_DEVICE=hw:3,0
ALSA_INPUT_CHANNELS=6
```

Note: If the device is busy, stop PipeWire from grabbing the ReSpeaker input or reboot before starting the service.

## Wakeword Debug (Quick sanity check)

Enable debug for one run to see scores:

```bash
WAKEWORD_DEBUG=true
```

Then restart the service and watch logs:

```bash
systemctl --user restart voice-assistant-edge.service
journalctl --user -u voice-assistant-edge.service -f
```

If scores stay near 0.0, confirm:
- `MIC_DEVICE_INDEX=pipewire`
- `MIC_CHANNELS=1`
- The ReSpeaker is the default source in `wpctl status`

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
