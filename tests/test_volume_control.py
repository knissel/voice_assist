"""
Test volume control functionality.
Tests setting volume to specific levels and adjusting up/down.
"""
import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.audio import control_volume, get_current_volume

print("=" * 60)
print("Volume Control Test")
print("=" * 60)

# Get initial volume
print("\n📊 Getting current volume...")
initial_volume = get_current_volume()
print(f"Current volume: {initial_volume}%")

# Test 1: Set volume to 50%
print("\n🔧 Test 1: Setting volume to 50%")
result = control_volume("set", 50)
print(f"Result: {result}")
time.sleep(1)
current = get_current_volume()
print(f"Verified: {current}%")

# Test 2: Turn volume up by 10%
print("\n🔧 Test 2: Turning volume up by 10%")
result = control_volume("up", 10)
print(f"Result: {result}")
time.sleep(1)
current = get_current_volume()
print(f"Verified: {current}%")

# Test 3: Turn volume down by 15%
print("\n🔧 Test 3: Turning volume down by 15%")
result = control_volume("down", 15)
print(f"Result: {result}")
time.sleep(1)
current = get_current_volume()
print(f"Verified: {current}%")

# Test 4: Turn volume up (default 10%)
print("\n🔧 Test 4: Turning volume up (default amount)")
result = control_volume("up")
print(f"Result: {result}")
time.sleep(1)
current = get_current_volume()
print(f"Verified: {current}%")

# Test 5: Set to minimum (0%)
print("\n🔧 Test 5: Setting volume to 0% (mute)")
result = control_volume("set", 0)
print(f"Result: {result}")
time.sleep(1)
current = get_current_volume()
print(f"Verified: {current}%")

# Test 6: Set to maximum (100%)
print("\n🔧 Test 6: Setting volume to 100%")
result = control_volume("set", 100)
print(f"Result: {result}")
time.sleep(1)
current = get_current_volume()
print(f"Verified: {current}%")

# Test 7: Test boundary - try to go above 100%
print("\n🔧 Test 7: Boundary test - setting to 150% (should clamp to 100%)")
result = control_volume("set", 150)
print(f"Result: {result}")
time.sleep(1)
current = get_current_volume()
print(f"Verified: {current}%")

# Test 8: Test boundary - try to go below 0%
print("\n🔧 Test 8: Boundary test - setting to -10% (should clamp to 0%)")
result = control_volume("set", -10)
print(f"Result: {result}")
time.sleep(1)
current = get_current_volume()
print(f"Verified: {current}%")

# Restore initial volume
print(f"\n🔄 Restoring initial volume to {initial_volume}%...")
if isinstance(initial_volume, int):
    control_volume("set", initial_volume)
    time.sleep(1)
    final = get_current_volume()
    print(f"Restored to: {final}%")

print("\n✅ All volume control tests complete!")
