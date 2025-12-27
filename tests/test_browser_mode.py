"""
Test browser mode - this will actually open YouTube Music in your browser.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.youtube_music import play_youtube_music

# Make sure we're in browser mode
os.environ['HEADLESS_PLAYBACK'] = 'false'

print("Testing YouTube Music - Browser Mode")
print("=" * 60)
print("\nThis will search for a song and open it in your browser.")
print("Press Ctrl+C within 3 seconds to cancel...\n")

import time
try:
    time.sleep(3)
except KeyboardInterrupt:
    print("\nCancelled!")
    exit(0)

print("Searching and playing 'Bohemian Rhapsody'...\n")
result = play_youtube_music("Bohemian Rhapsody", "song")
print(f"Result: {result}")
print("\n✅ Check your browser - YouTube Music should have opened!")
