# Raspberry Pi Deployment Setup Guide

This guide walks you through setting up deployment from your Mac to your Raspberry Pi, including how to find all the necessary configuration values.

---

## Step 1: Find Your Raspberry Pi Username

### Option A: If you're already logged into the Pi (via keyboard/monitor or existing SSH)

```bash
# Run this on the Pi - it shows your current username
whoami
```

Common usernames:
- **`pi`** — Default on older Raspberry Pi OS (before April 2022)
- **`<your-name>`** — Newer Raspberry Pi OS requires you to create a username during setup
- **`root`** — Not recommended, but possible

### Option B: If you set up the Pi with Raspberry Pi Imager

When you used the Raspberry Pi Imager to flash the SD card, you may have configured a custom username. Check:

1. Open **Raspberry Pi Imager** on your Mac
2. Click the **gear icon ⚙️** (OS Customization)
3. Look at the **Username** field — this is what you set

### Option C: If you don't remember

Try connecting with the default username:
```bash
ssh pi@raspberrypi.local
```

If that fails with "Permission denied", try your Mac username or common alternatives:
```bash
ssh kenny@raspberrypi.local
ssh admin@raspberrypi.local
```

---

## Step 2: Find Your Raspberry Pi IP Address or Hostname

### Option A: Use mDNS hostname (easiest)

Most Raspberry Pis broadcast their hostname via mDNS. Try:

```bash
# From your Mac terminal
ping raspberrypi.local
```

If it responds, you can use `raspberrypi.local` as your hostname.

### Option B: Find IP address from your router

1. Log into your router's admin page (usually `192.168.1.1` or `192.168.0.1`)
2. Look for "Connected Devices" or "DHCP Clients"
3. Find the entry for "raspberrypi" or your Pi's hostname

### Option C: Scan your network from Mac

```bash
# Install nmap if you don't have it
brew install nmap

# Scan for devices (adjust the IP range to match your network)
nmap -sn 192.168.1.0/24 | grep -B2 "Raspberry"
```

### Option D: Find IP from the Pi itself

If you have a monitor/keyboard connected to the Pi:
```bash
hostname -I
```

---

## Step 3: Set Up SSH Key Authentication

The deploy script requires password-less SSH. Here's how to set it up:

### 3.1 Check if you already have an SSH key

```bash
ls -la ~/.ssh/id_*.pub
```

If you see files like `id_rsa.pub` or `id_ed25519.pub`, you have a key.

### 3.2 Generate a key if needed

```bash
ssh-keygen -t ed25519 -C "your-email@example.com"
```

Press Enter to accept defaults. You can skip the passphrase for convenience (just press Enter twice).

### 3.3 Copy your key to the Pi

```bash
# Replace with YOUR username and hostname/IP
ssh-copy-id pi@raspberrypi.local
```

You'll be prompted for your Pi's password one last time.

### 3.4 Test passwordless login

```bash
ssh pi@raspberrypi.local
```

If you connect without being asked for a password, you're set!

---

## Step 4: Create Your Deploy Configuration

### 4.1 Copy the example config

```bash
cd ~/voice_assist
cp deploy.config.example deploy.config
```

### 4.2 Edit the config with your values

```bash
nano deploy.config
```

Update these values:

```bash
# Your Pi connection (username@hostname or username@ip)
PI_HOST="pi@raspberrypi.local"

# Path where voice_assist will live on the Pi
# This should match your username's home directory
PI_PATH="/home/pi/voice_assist"

# Service name (usually leave as default)
SERVICE_NAME="voice-assistant"
```

**Important:** The `PI_PATH` should use your actual username:
- If username is `pi`: `/home/pi/voice_assist`
- If username is `kenny`: `/home/kenny/voice_assist`

Save with `Ctrl+O`, then `Ctrl+X` to exit.

---

## Step 5: Prepare the Pi

SSH into your Pi and create the destination directory:

```bash
ssh pi@raspberrypi.local

# On the Pi, create the voice_assist directory
mkdir -p ~/voice_assist

# Exit back to your Mac
exit
```

---

## Step 6: Test Your Deployment

### 6.1 Dry run first

```bash
./deploy.sh --all --dry-run
```

This shows what would be deployed without actually doing it.

### 6.2 Deploy for real

```bash
./deploy.sh --all
```

---

## Troubleshooting

### "SSH connection failed"

**Check network connectivity:**
```bash
ping raspberrypi.local
```

**Verify SSH is enabled on Pi:**
- Raspberry Pi OS: Enable via `raspi-config` → Interface Options → SSH
- Or create empty file named `ssh` in the boot partition of SD card

**Check hostname/IP is correct:**
```bash
# Try with IP instead of hostname
ssh pi@192.168.1.XXX
```

### "Permission denied (publickey)"

Your SSH key isn't set up correctly. Re-run:
```bash
ssh-copy-id pi@raspberrypi.local
```

### "Host key verification failed"

The Pi's identity changed (e.g., you reinstalled the OS). Fix with:
```bash
ssh-keygen -R raspberrypi.local
```

Then try connecting again.

### "No route to host"

The Pi isn't on the network or has a different IP. Check:
1. Pi is powered on with activity LED blinking
2. Ethernet cable is connected (or WiFi is configured)
3. Router shows the Pi as connected

---

## Quick Reference

| Setting | How to Find It |
|---------|----------------|
| Username | Run `whoami` on Pi |
| Hostname | Usually `raspberrypi.local` |
| IP Address | Run `hostname -I` on Pi, or check router |
| Home Path | `/home/<username>` |

### Example Configurations

**Default Pi user:**
```bash
PI_HOST="pi@raspberrypi.local"
PI_PATH="/home/pi/voice_assist"
```

**Custom username with static IP:**
```bash
PI_HOST="kenny@192.168.1.50"
PI_PATH="/home/kenny/voice_assist"
```

**Using .local hostname with custom user:**
```bash
PI_HOST="myuser@voicepi.local"
PI_PATH="/home/myuser/voice_assist"
```
