"""
Test stopping all audio including music and TTS.
"""
import time
import subprocess
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.youtube_music import stop_music

print("=" * 60)
print("Stop All Audio Test")
print("=" * 60)

# Test 1: Stop when nothing is playing
print("\n🧪 Test 1: Stop when nothing is playing")
result = stop_music()
print(f"Result: {result}")
assert "No audio" in result or "Stopped" in result

# Test 2: Start music and stop it
print("\n🧪 Test 2: Start music playback and stop it")
print("Starting mpv in background...")
try:
    # Start a silent audio stream (1 second of silence)
    subprocess.Popen(
        ["mpv", "--no-video", "--really-quiet", "--length=10", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(2)  # Give it time to start
    
    print("Stopping music...")
    result = stop_music()
    print(f"Result: {result}")
    assert "mpv" in result.lower() or "Stopped" in result
except FileNotFoundError:
    print("⚠️  mpv not installed, skipping music test")

# Test 3: Start TTS and stop it (macOS)
print("\n🧪 Test 3: Start TTS and stop it")
import platform
if platform.system() == "Darwin":
    print("Starting macOS 'say' command...")
    subprocess.Popen(
        ["say", "This is a very long text to speech message that will keep going for a while so we can test stopping it"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    time.sleep(1)  # Give it time to start
    
    print("Stopping TTS...")
    result = stop_music()
    print(f"Result: {result}")
    assert "say" in result.lower() or "Stopped" in result
elif platform.system() == "Linux":
    print("Starting Linux espeak...")
    try:
        subprocess.Popen(
            ["espeak", "This is a very long text to speech message that will keep going for a while"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        time.sleep(1)
        
        print("Stopping TTS...")
        result = stop_music()
        print(f"Result: {result}")
        assert "espeak" in result.lower() or "Stopped" in result
    except FileNotFoundError:
        print("⚠️  espeak not installed, skipping TTS test")
else:
    print("⚠️  Platform not supported for TTS test")

# Test 4: Multiple audio sources
print("\n🧪 Test 4: Multiple audio sources at once")
started = []
try:
    # Start mpv
    subprocess.Popen(
        ["mpv", "--no-video", "--really-quiet", "--length=10", "https://www.youtube.com/watch?v=dQw4w9WgXcQ"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    started.append("mpv")
except FileNotFoundError:
    pass

if platform.system() == "Darwin":
    subprocess.Popen(
        ["say", "Testing multiple audio sources"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    started.append("say")

if started:
    time.sleep(2)
    print(f"Started: {', '.join(started)}")
    print("Stopping all audio...")
    result = stop_music()
    print(f"Result: {result}")
    assert "Stopped" in result
else:
    print("⚠️  No audio sources available to test")

print("\n✅ All stop audio tests complete!")
