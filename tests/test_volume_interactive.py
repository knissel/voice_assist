"""
Interactive volume control test.
You'll hear the volume changes in real-time.
"""
import sys
import os
import time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.audio import control_volume, get_current_volume

print("=" * 60)
print("Interactive Volume Control Test")
print("=" * 60)
print("\nYou should HEAR the volume changing during this test!")
print("Make sure your speakers are on and playing something,")
print("or the system sound will demonstrate the changes.\n")

# Get initial volume
initial_volume = get_current_volume()
print(f"📊 Current volume: {initial_volume}%\n")

# Test sequence with audio feedback
print("🔊 Setting volume to 30% (quiet)...")
control_volume("set", 30)
print("   Listen - volume should be quiet now")
time.sleep(3)

print("\n🔊 Turning volume UP by 20%...")
result = control_volume("up", 20)
print(f"   {result}")
print("   Listen - volume should be louder now")
time.sleep(3)

print("\n🔊 Turning volume UP by another 20%...")
result = control_volume("up", 20)
print(f"   {result}")
print("   Listen - volume should be even louder")
time.sleep(3)

print("\n🔊 Turning volume DOWN by 15%...")
result = control_volume("down", 15)
print(f"   {result}")
print("   Listen - volume should decrease")
time.sleep(3)

print("\n🔊 Setting volume to 75% (loud)...")
control_volume("set", 75)
print("   Listen - volume should be loud now")
time.sleep(3)

print("\n🔊 Setting volume to 25% (quiet again)...")
control_volume("set", 25)
print("   Listen - volume should be quiet again")
time.sleep(3)

# Restore
print(f"\n🔄 Restoring volume to {initial_volume}%...")
if isinstance(initial_volume, int):
    control_volume("set", initial_volume)
    print(f"   Volume restored to {initial_volume}%")

print("\n✅ Interactive test complete!")
print("Did you hear the volume changing?")
