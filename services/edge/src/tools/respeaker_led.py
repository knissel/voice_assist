"""
ReSpeaker LED ring control via USB vendor requests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import usb.core
    import usb.util
except Exception:  # pragma: no cover - optional dependency
    usb = None  # type: ignore


@dataclass
class RespeakerLedConfig:
    brightness: Optional[int] = None  # 0-255


class PixelRing:
    TIMEOUT = 8000

    def __init__(self, dev):
        self.dev = dev
        self._claimed = False
        self._prepare_device()

    def _prepare_device(self) -> None:
        try:
            self.dev.set_configuration()
        except Exception:
            pass

        interface_number = 3
        try:
            if self.dev.is_kernel_driver_active(interface_number):
                self.dev.detach_kernel_driver(interface_number)
        except Exception:
            pass

        try:
            usb.util.claim_interface(self.dev, interface_number)
            self._claimed = True
        except Exception:
            self._claimed = False

    def trace(self) -> None:
        self.write(0)

    def mono(self, color: int) -> None:
        self.write(1, [(color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF, 0])

    def set_color(self, rgb: Optional[int] = None, r: int = 0, g: int = 0, b: int = 0) -> None:
        if rgb is not None:
            self.mono(rgb)
        else:
            self.write(1, [r, g, b, 0])

    def off(self) -> None:
        self.mono(0)

    def listen(self) -> None:
        self.write(2)

    wakeup = listen

    def speak(self) -> None:
        self.write(3)

    def think(self) -> None:
        self.write(4)

    wait = think

    def spin(self) -> None:
        self.write(5)

    def show(self, data) -> None:
        self.write(6, data)

    customize = show

    def set_brightness(self, brightness: int) -> None:
        self.write(0x20, [brightness])

    def set_vad_led(self, state: int) -> None:
        self.write(0x22, [state])

    def set_volume(self, volume: int) -> None:
        self.write(0x23, [volume])

    def write(self, cmd: int, data: list[int] | None = None) -> None:
        payload = data if data is not None else [0]
        self.dev.ctrl_transfer(
            usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0,
            cmd,
            0x1C,
            payload,
            self.TIMEOUT,
        )

    def close(self) -> None:
        if self._claimed:
            try:
                usb.util.release_interface(self.dev, 3)
            except Exception:
                pass
        usb.util.dispose_resources(self.dev)


def find(vid: int = 0x2886, pid: int = 0x0018) -> Optional[PixelRing]:
    if usb is None:
        return None
    dev = usb.core.find(idVendor=vid, idProduct=pid)
    if not dev:
        return None
    return PixelRing(dev)


class RespeakerLedController:
    def __init__(self, vid: int = 0x2886, pid: int = 0x0018, config: Optional[RespeakerLedConfig] = None):
        self._ring = find(vid=vid, pid=pid)
        self._config = config or RespeakerLedConfig()

        if self._ring and self._config.brightness is not None:
            try:
                self._ring.set_brightness(self._config.brightness)
            except Exception:
                pass

    @property
    def available(self) -> bool:
        return self._ring is not None

    def set_state(self, state: str) -> None:
        if not self._ring:
            return
        try:
            if state == "listening":
                self._ring.listen()
            elif state == "recording":
                self._ring.wakeup()
            elif state == "processing":
                self._ring.think()
            elif state == "speaking":
                self._ring.speak()
            elif state == "error":
                self._ring.set_color(r=255, g=0, b=0)
            elif state == "idle":
                self._ring.off()
        except Exception:
            self._ring = None

    def close(self) -> None:
        if self._ring:
            self._ring.close()
