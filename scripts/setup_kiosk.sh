#!/usr/bin/env bash
set -euo pipefail

SERVICE_SRC="/home/kenny/voice_assist/systemd/kiosk-ui.service"
SERVICE_DST="${HOME}/.config/systemd/user/kiosk-ui.service"

mkdir -p "${HOME}/.config/systemd/user"
cp "${SERVICE_SRC}" "${SERVICE_DST}"

systemctl --user daemon-reload
systemctl --user enable kiosk-ui.service
systemctl --user start kiosk-ui.service

echo "Kiosk UI service installed and started."
