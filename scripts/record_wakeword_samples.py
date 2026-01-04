#!/usr/bin/env python3
"""
Record labeled wakeword audio samples for training.
Saves 16 kHz mono WAV files to output/<label>/.
"""
import argparse
import os
import time
import wave

import pyaudio


def _select_input_device_index(pa: pyaudio.PyAudio, preferred: str | None) -> int | None:
    """Resolve an input device index from an override (index or name substring)."""
    if not preferred:
        return None
    try:
        index = int(preferred)
        info = pa.get_device_info_by_index(index)
        if info.get("maxInputChannels", 0) > 0:
            print(f"Using input device index {index}: {info.get('name')}")
            return index
        print(f"Device index {index} has no input channels")
        return None
    except ValueError:
        needle = preferred.lower()
    except Exception as exc:
        print(f"Failed to read device index {preferred!r}: {exc}")
        return None

    for i in range(pa.get_device_count()):
        info = pa.get_device_info_by_index(i)
        name = info.get("name", "")
        if info.get("maxInputChannels", 0) > 0 and needle in name.lower():
            print(f"Using input device index {i}: {name}")
            return i

    print(f"No input device matched {preferred!r}")
    return None


def _record_clip(pa: pyaudio.PyAudio, rate: int, duration: float, chunk: int, device_index: int | None) -> bytes:
    frames = []
    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=rate,
        input=True,
        frames_per_buffer=chunk,
        input_device_index=device_index,
    )
    try:
        total_frames = int(rate * duration / chunk)
        for _ in range(total_frames):
            frames.append(stream.read(chunk, exception_on_overflow=False))
    finally:
        stream.stop_stream()
        stream.close()
    return b"".join(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description="Record labeled wakeword samples.")
    parser.add_argument("--output", default="data/wakeword_samples", help="Output directory")
    parser.add_argument("--label", default="positive", help="Label folder name (e.g., positive/negative)")
    parser.add_argument("--count", type=int, default=20, help="Number of samples to record")
    parser.add_argument("--duration", type=float, default=1.0, help="Seconds per sample")
    parser.add_argument("--rate", type=int, default=16000, help="Sample rate (Hz)")
    parser.add_argument("--chunk", type=int, default=1024, help="Frames per buffer")
    parser.add_argument("--device", default=None, help="Input device index or name substring")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    label_dir = os.path.join(args.output, args.label)
    os.makedirs(label_dir, exist_ok=True)

    pa = pyaudio.PyAudio()
    device_index = _select_input_device_index(pa, args.device)

    print(
        f"Recording {args.count} samples to {label_dir} "
        f"({args.rate} Hz, {args.duration}s each)."
    )
    print("Press Enter to record each sample. Speak the wakeword after the prompt.")

    try:
        for i in range(args.count):
            input(f"[{i + 1}/{args.count}] Ready. Press Enter to start...")
            data = _record_clip(pa, args.rate, args.duration, args.chunk, device_index)
            filename = f"{args.label}_{int(time.time() * 1000)}_{i:03d}.wav"
            path = os.path.join(label_dir, filename)
            with wave.open(path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(pa.get_sample_size(pyaudio.paInt16))
                wf.setframerate(args.rate)
                wf.writeframes(data)
            print(f"Saved {path}")
    finally:
        pa.terminate()


if __name__ == "__main__":
    main()
