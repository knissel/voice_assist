"""
Quick test of YouTube Music integration.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.youtube_music import play_youtube_music

print("Testing YouTube Music integration...")
print("-" * 60)

result = play_youtube_music("Bohemian Rhapsody", "song")
print(f"\nResult: {result}")
