"""
ReSpeaker USB DSP control (AEC/NS/AGC) via vendor USB requests.

Based on Seeed's usb_4_mic_array tuning interface.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Optional

try:
    import usb.core
    import usb.util
except Exception:  # pragma: no cover - optional dependency
    usb = None  # type: ignore


PARAMETERS = {
    "AGCONOFF": (19, 0, "int"),
    "STATNOISEONOFF": (19, 8, "int"),
    "NONSTATNOISEONOFF": (19, 11, "int"),
    "ECHOONOFF": (19, 14, "int"),
    "NLAEC_MODE": (19, 20, "int"),
    "STATNOISEONOFF_SR": (19, 33, "int"),
    "NONSTATNOISEONOFF_SR": (19, 34, "int"),
}


@dataclass
class RespeakerSettings:
    aec_enabled: bool = True
    ns_enabled: bool = True
    agc_enabled: bool = True


class RespeakerTuning:
    TIMEOUT = 100000

    def __init__(self, dev):
        self.dev = dev
        self._claimed = False
        self._prepare_device()

    def _prepare_device(self) -> None:
        try:
            self.dev.set_configuration()
        except Exception:
            pass

        # Vendor interface is typically 3 on the ReSpeaker 4-mic array.
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

    def write(self, name: str, value: int | float) -> None:
        data = PARAMETERS.get(name)
        if not data:
            raise KeyError(f"Unknown parameter: {name}")

        param_id, offset, dtype = data
        if dtype == "int":
            payload = struct.pack(b"iii", offset, int(value), 1)
        else:
            payload = struct.pack(b"ifi", offset, float(value), 0)

        self.dev.ctrl_transfer(
            usb.util.CTRL_OUT | usb.util.CTRL_TYPE_VENDOR | usb.util.CTRL_RECIPIENT_DEVICE,
            0,
            0,
            param_id,
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


def find(vid: int = 0x2886, pid: int = 0x0018) -> Optional[RespeakerTuning]:
    if usb is None:
        return None
    dev = usb.core.find(idVendor=vid, idProduct=pid)
    if not dev:
        return None
    return RespeakerTuning(dev)


def apply_settings(settings: RespeakerSettings, vid: int = 0x2886, pid: int = 0x0018) -> bool:
    tuning = find(vid=vid, pid=pid)
    if tuning is None:
        return False

    try:
        tuning.write("AGCONOFF", 1 if settings.agc_enabled else 0)
        tuning.write("STATNOISEONOFF", 1 if settings.ns_enabled else 0)
        tuning.write("NONSTATNOISEONOFF", 1 if settings.ns_enabled else 0)
        tuning.write("STATNOISEONOFF_SR", 1 if settings.ns_enabled else 0)
        tuning.write("NONSTATNOISEONOFF_SR", 1 if settings.ns_enabled else 0)
        tuning.write("ECHOONOFF", 1 if settings.aec_enabled else 0)
        tuning.write("NLAEC_MODE", 2 if settings.aec_enabled else 0)
        return True
    finally:
        tuning.close()
