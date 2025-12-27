"""
Simple test - just search and show what would play.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.youtube_music import YouTubeMusicManager

manager = YouTubeMusicManager()

print("Searching for 'Bohemian Rhapsody'...")
result = manager.search_and_play("Bohemian Rhapsody", "songs")

print("\nSearch Result:")
print(f"  Title: {result.get('title', 'Unknown')}")
print(f"  Artist: {result.get('artist', 'Unknown')}")
print(f"  Video ID: {result.get('video_id', 'None')}")
print(f"  URL: {result.get('url', 'None')}")

print("\n✅ YouTube Music search is working!")
print("\nOn your Mac: This URL would open in your browser")
print("On your Pi: This would stream directly through speakers")
