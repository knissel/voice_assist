#!/usr/bin/env python3
import argparse
import os
import sys
from typing import List, Tuple


VID = 0x2886
PID = 0x0018


def _load_edge_tools() -> Tuple[object | None, object | None, str | None]:
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    edge_src = os.path.join(repo_root, "services", "edge", "src")
    sys.path.insert(0, edge_src)
    try:
        from tools import respeaker as respeaker_tools  # type: ignore
        from tools import respeaker_led as led_tools  # type: ignore
    except Exception as exc:
        return None, None, f"Failed to import edge tools from {edge_src}: {exc}"
    return respeaker_tools, led_tools, None


def _device_strings(usb_util, dev) -> tuple[str, str]:
    def _get_string(index: int) -> str:
        if not index:
            return "unknown"
        try:
            return usb_util.get_string(dev, index) or "unknown"
        except Exception:
            return "unknown"

    return _get_string(getattr(dev, "iManufacturer", 0)), _get_string(getattr(dev, "iProduct", 0))


def main() -> int:
    parser = argparse.ArgumentParser(description="ReSpeaker USB Mic Array diagnostic")
    parser.add_argument(
        "--timeout-ms",
        type=int,
        default=3000,
        help="USB control transfer timeout in ms (default: 3000)",
    )
    parser.add_argument(
        "--skip-nlaec",
        action="store_true",
        help="Skip NLAEC_MODE write (some firmware hangs on this)",
    )
    args = parser.parse_args()

    print("ReSpeaker USB Mic Array diagnostic")
    try:
        import usb.core  # type: ignore
        import usb.util  # type: ignore
    except Exception as exc:
        print(f"[FAIL] pyusb not available: {exc}")
        print("Install with: pip install pyusb")
        return 2

    dev = usb.core.find(idVendor=VID, idProduct=PID)
    if not dev:
        print(f"[FAIL] Device {VID:04x}:{PID:04x} not found")
        return 1

    bcd = getattr(dev, "bcdDevice", None)
    manufacturer, product = _device_strings(usb.util, dev)
    print(f"[INFO] Device: {VID:04x}:{PID:04x} bcdDevice={bcd} manufacturer={manufacturer} product={product}")

    respeaker_tools, led_tools, err = _load_edge_tools()
    if err:
        print(f"[FAIL] {err}")
        return 3

    tuning = respeaker_tools.find(vid=VID, pid=PID)
    if tuning is None:
        print("[FAIL] Tuning interface not available (pyusb missing or device not found)")
        return 4

    tests: List[Tuple[str, int]] = [
        ("AGCONOFF", 1),
        ("STATNOISEONOFF", 1),
        ("NONSTATNOISEONOFF", 1),
        ("STATNOISEONOFF_SR", 1),
        ("NONSTATNOISEONOFF_SR", 1),
        ("ECHOONOFF", 1),
        ("NLAEC_MODE", 2),
    ]
    if args.skip_nlaec:
        tests = [item for item in tests if item[0] != "NLAEC_MODE"]

    failures: List[Tuple[str, str]] = []
    try:
        tuning.TIMEOUT = max(100, int(args.timeout_ms))
        for name, value in tests:
            try:
                tuning.write(name, value)
                print(f"[OK] {name} set to {value}")
            except Exception as exc:
                failures.append((name, str(exc)))
                print(f"[FAIL] {name}: {exc}")
    finally:
        tuning.close()

    if failures:
        print("[WARN] DSP writes failed. Common causes:")
        print("  - Device in use (stop PipeWire/WirePlumber or the edge service)")
        print("  - Firmware variant that does not expose the tuning interface")
        return 5

    try:
        controller = led_tools.RespeakerLedController(vid=VID, pid=PID)
        print(f"[INFO] LED ring available: {controller.available}")
        controller.close()
    except Exception as exc:
        print(f"[WARN] LED ring check failed: {exc}")

    print("[PASS] ReSpeaker DSP control is working")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
