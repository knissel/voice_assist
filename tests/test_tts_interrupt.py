"""
Interactive test for interrupting TTS.
This simulates the real-world use case of stopping long TTS responses.
"""
import time
import subprocess
import platform
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.youtube_music import stop_music

print("=" * 60)
print("TTS Interrupt Test (Interactive)")
print("=" * 60)
print("\nThis test will start a long TTS message and then stop it.")
print("You should hear the speech start and then get cut off.\n")

if platform.system() == "Darwin":
    print("🔊 Starting long TTS message on macOS...")
    print("(You should hear this start speaking)")
    
    long_text = """
    This is a very long text to speech message that goes on and on.
    It talks about many different things and takes a long time to finish.
    The purpose of this message is to test whether we can interrupt it.
    If you're hearing this, the text to speech is working correctly.
    Now we're going to test stopping it in the middle of this sentence.
    """
    
    subprocess.Popen(
        ["say", long_text],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )
    
    print("\n⏳ Letting it speak for 3 seconds...")
    time.sleep(3)
    
    print("\n🛑 STOPPING TTS NOW!")
    result = stop_music()
    print(f"Result: {result}")
    
    print("\n✅ Did the speech stop abruptly? (It should have)")
    
elif platform.system() == "Linux":
    print("🔊 Starting long TTS message on Linux...")
    print("(You should hear this start speaking)")
    
    long_text = "This is a very long text to speech message that goes on and on. " * 10
    
    try:
        subprocess.Popen(
            ["espeak", long_text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        
        print("\n⏳ Letting it speak for 3 seconds...")
        time.sleep(3)
        
        print("\n🛑 STOPPING TTS NOW!")
        result = stop_music()
        print(f"Result: {result}")
        
        print("\n✅ Did the speech stop abruptly? (It should have)")
    except FileNotFoundError:
        print("⚠️  espeak not installed. Install with: sudo apt-get install espeak")
else:
    print("⚠️  Platform not supported for this test")

print("\n" + "=" * 60)
print("Test complete!")
