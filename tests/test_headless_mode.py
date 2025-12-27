"""
Test YouTube Music in both browser and headless modes.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.youtube_music import play_youtube_music

print("=" * 60)
print("YouTube Music Playback Mode Test")
print("=" * 60)

# Test 1: Browser mode (default)
print("\n[Test 1] Browser Mode (HEADLESS_PLAYBACK=false)")
print("-" * 60)
os.environ['HEADLESS_PLAYBACK'] = 'false'
result = play_youtube_music("Bohemian Rhapsody", "song")
print(f"Result: {result}")

# Test 2: Headless mode (for Raspberry Pi)
print("\n[Test 2] Headless Mode (HEADLESS_PLAYBACK=true)")
print("-" * 60)
print("This will attempt to stream audio directly using mpv...")
os.environ['HEADLESS_PLAYBACK'] = 'true'
result = play_youtube_music("Bohemian Rhapsody", "song")
print(f"Result: {result}")

print("\n" + "=" * 60)
print("Test complete!")
print("=" * 60)
print("\nNote: On Raspberry Pi, headless mode will play audio")
print("through speakers without opening a browser.")
