# Kiosk UI Setup (Ubuntu GNOME)

This guide pins the UI to the screen and keeps it running.

## 1) Enable Auto-login

Edit GDM:

```bash
sudo sed -n '1,120p' /etc/gdm3/custom.conf
```

Set:

```
[daemon]
AutomaticLoginEnable=true
AutomaticLogin=kenny
```

Reboot after changing this.

## 2) Install the Kiosk User Service

Copy the service template and enable it:

```bash
mkdir -p ~/.config/systemd/user
cp /home/kenny/voice_assist/systemd/kiosk-ui.service ~/.config/systemd/user/kiosk-ui.service
systemctl --user daemon-reload
systemctl --user enable kiosk-ui.service
systemctl --user start kiosk-ui.service
```

## 3) Disable Screen Blanking/Sleep

```bash
gsettings set org.gnome.desktop.session idle-delay 0
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-ac-type 'nothing'
gsettings set org.gnome.settings-daemon.plugins.power sleep-inactive-battery-type 'nothing'
gsettings set org.gnome.desktop.screensaver lock-enabled false
```

## 4) Optional: Run Services Without Login

```bash
sudo loginctl enable-linger $USER
```

## Kiosk Service Controls

```bash
systemctl --user status kiosk-ui.service --no-pager
systemctl --user restart kiosk-ui.service
journalctl --user -u kiosk-ui.service -f
```

