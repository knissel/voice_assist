"""
Interactive test - you'll hear the music clearly.
"""
import os
import sys
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.youtube_music import play_youtube_music, stop_music

# Enable headless mode
os.environ['HEADLESS_PLAYBACK'] = 'true'

print("=" * 60)
print("Interactive Music Control Test")
print("=" * 60)

print("\n🎵 Starting music in 2 seconds...")
print("Listen for Bohemian Rhapsody to start playing...")
time.sleep(2)

print("\n[Playing music...]")
result = play_youtube_music("Bohemian Rhapsody", "song")
print(f"Status: {result}")

print("\n🔊 Music should be playing NOW through your speakers!")
print("Let it play for 10 seconds so you can hear it...")

for i in range(10, 0, -1):
    print(f"   Playing... {i} seconds remaining", end='\r')
    time.sleep(1)

print("\n\n🛑 Stopping music now...")
result = stop_music()
print(f"Status: {result}")

print("\n✅ Test complete!")
print("Did you hear the music playing and then stop?")
