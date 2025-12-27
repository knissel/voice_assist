"""
Setup script for YouTube Music authentication.
Run this to authenticate with your YouTube Music Premium account.
"""
from ytmusicapi import setup

def setup_youtube_music():
    """Set up YouTube Music authentication using browser headers."""
    print("=" * 60)
    print("YouTube Music Authentication Setup")
    print("=" * 60)
    print("\nFollow these steps to authenticate:")
    print("\n1. Open YouTube Music in your browser: https://music.youtube.com")
    print("2. Make sure you're logged in with your Premium account")
    print("3. Open Developer Tools (Press F12 or Right-click > Inspect)")
    print("4. Go to the 'Network' tab")
    print("5. Click on any request to 'music.youtube.com'")
    print("6. Find the 'Request Headers' section")
    print("7. Copy ALL the request headers")
    print("\n" + "=" * 60)
    print("\nPaste the headers below and press Enter twice when done:")
    print("(The headers should start with something like 'accept: */*')")
    print("-" * 60)
    
    try:
        setup(filepath='oauth.json')
        print("\n✅ Success! YouTube Music authentication is set up.")
        print("You can now use voice commands to play music.")
        print("\nExample commands:")
        print("  - 'Play Bohemian Rhapsody'")
        print("  - 'Play Taylor Swift'")
        print("  - 'Play my Workout playlist'")
    except KeyboardInterrupt:
        print("\n\n❌ Setup cancelled by user.")
    except Exception as e:
        print(f"\n❌ Error during setup: {e}")
        print("\nMake sure you:")
        print("  - Copied the complete request headers")
        print("  - Are logged into YouTube Music Premium")
        print("  - Pressed Enter twice after pasting")

if __name__ == "__main__":
    setup_youtube_music()
