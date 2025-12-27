"""
Test the stop_music functionality.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.youtube_music import play_youtube_music, stop_music

# Enable headless mode for testing
os.environ['HEADLESS_PLAYBACK'] = 'true'

print("=" * 60)
print("Testing Stop Music Functionality")
print("=" * 60)

# Start playing music
print("\n[Step 1] Starting music playback...")
result = play_youtube_music("Bohemian Rhapsody", "song")
print(f"Result: {result}")

print("\n[Step 2] Music should be playing now...")
print("Waiting 5 seconds...")
time.sleep(5)

print("\n[Step 3] Stopping music...")
result = stop_music()
print(f"Result: {result}")

print("\n[Step 4] Testing stop when nothing is playing...")
result = stop_music()
print(f"Result: {result}")

print("\n" + "=" * 60)
print("✅ Stop music test complete!")
print("=" * 60)
