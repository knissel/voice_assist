#!/bin/bash
# scripts/setup_pi.sh
# Run this on the Raspberry Pi to set up the ops agent and legacy services.

set -e

REPO_DIR="/home/knissel/voice_assist"
SYSTEMD_DIR="/etc/systemd/system"

echo "🚀 Setting up Voice Assistant services on Pi..."

# 1. Create systemd symlinks
echo "🔗 Creating systemd symlinks..."
sudo ln -sf "$REPO_DIR/systemd/voice-assist-ops-agent.service" "$SYSTEMD_DIR/"
sudo ln -sf "$REPO_DIR/systemd/voice-assist-wakeword.service" "$SYSTEMD_DIR/"
sudo ln -sf "$REPO_DIR/systemd/voice-assist-ui-server.service" "$SYSTEMD_DIR/"

# 2. Reload systemd
echo "🔄 Reloading systemd..."
sudo systemctl daemon-reload

# 3. Enable services
echo "⚡ Enabling services..."
sudo systemctl enable voice-assist-ops-agent
sudo systemctl enable voice-assist-wakeword
sudo systemctl enable voice-assist-ui-server

# 4. Start Ops Agent (to manage the others)
echo "▶ Starting Ops Agent..."
# Set OPS_CONFIG in the service file update
sudo systemctl restart voice-assist-ops-agent

echo "✅ Setup complete! You should see the Raspberry Pi appear in the Ops Dashboard."
echo "You can now start/stop/deploy the wakeword and UI server from the dashboard."
