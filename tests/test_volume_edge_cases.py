"""
Test edge cases and error handling for volume control.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.audio import control_volume, get_current_volume, set_volume, adjust_volume

print("=" * 60)
print("Volume Control Edge Cases Test")
print("=" * 60)

# Save initial volume
initial_volume = get_current_volume()
print(f"\n📊 Initial volume: {initial_volume}%")

# Test 1: Invalid action
print("\n🧪 Test 1: Invalid action")
result = control_volume("invalid_action", 50)
print(f"Result: {result}")
assert "Invalid action" in result, "Should reject invalid action"

# Test 2: Missing level for 'set' action
print("\n🧪 Test 2: Missing level for 'set' action")
result = control_volume("set")
print(f"Result: {result}")
assert "Error" in result or "required" in result.lower(), "Should require level for set"

# Test 3: Invalid direction for adjust_volume
print("\n🧪 Test 3: Invalid direction")
result = adjust_volume("sideways", 10)
print(f"Result: {result}")
assert "Invalid direction" in result, "Should reject invalid direction"

# Test 4: Volume at 100%, try to increase
print("\n🧪 Test 4: Volume at max, try to increase")
set_volume(100)
result = control_volume("up", 10)
print(f"Result: {result}")
current = get_current_volume()
print(f"Current volume: {current}%")
assert current == 100, "Should stay at 100%"

# Test 5: Volume at 0%, try to decrease
print("\n🧪 Test 5: Volume at min, try to decrease")
set_volume(0)
result = control_volume("down", 10)
print(f"Result: {result}")
current = get_current_volume()
print(f"Current volume: {current}%")
assert current == 0, "Should stay at 0%"

# Test 6: Large adjustment from middle
print("\n🧪 Test 6: Large adjustment (50 -> up 60)")
set_volume(50)
result = control_volume("up", 60)
print(f"Result: {result}")
current = get_current_volume()
print(f"Current volume: {current}%")
assert current == 100, "Should clamp at 100%"

# Test 7: Large adjustment down
print("\n🧪 Test 7: Large adjustment down (50 -> down 60)")
set_volume(50)
result = control_volume("down", 60)
print(f"Result: {result}")
current = get_current_volume()
print(f"Current volume: {current}%")
assert current == 0, "Should clamp at 0%"

# Test 8: Case insensitivity
print("\n🧪 Test 8: Case insensitivity")
result1 = control_volume("UP", 5)
print(f"Result (UP): {result1}")
result2 = control_volume("Down", 5)
print(f"Result (Down): {result2}")
result3 = control_volume("SET", 50)
print(f"Result (SET): {result3}")
assert "Invalid" not in result1 and "Invalid" not in result2 and "Invalid" not in result3

# Restore initial volume
print(f"\n🔄 Restoring initial volume to {initial_volume}%...")
if isinstance(initial_volume, int):
    set_volume(initial_volume)
    print(f"Restored to: {get_current_volume()}%")

print("\n✅ All edge case tests passed!")
