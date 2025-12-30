#!/usr/bin/env python3
"""
Simple test app for Control4 lighting tools.
No wake word or LLM required - just direct function calls.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tools.control4_tool import control_home_lighting

def print_menu():
    print("\n" + "="*50)
    print("CONTROL4 LIGHTING TEST TOOL")
    print("="*50)
    print("\nAvailable Commands:")
    print("  1. Kitchen Cans (ID: 85)")
    print("  2. Foyer (ID: 87)")
    print("  3. Stairs (ID: 89)")
    print("  4. Upstairs Hall (ID: 91)")
    print("  5. Front Door (ID: 93)")
    print("  6. Kitchen Island (ID: 95)")
    print("  7. Downstairs Hallway (ID: 97)")
    print("  8. Upstairs Deck (ID: 99)")
    print("  9. Family Room (ID: 204)")
    print(" 10. Breakfast (ID: 206)")
    print(" 11. ALL LIGHTS ON")
    print(" 12. ALL LIGHTS OFF")
    print(" 13. Custom (enter device ID and brightness)")
    print("  q. Quit")
    print("="*50)

def test_light(device_id, brightness, description=""):
    print(f"\n🔧 Testing: {description}")
    print(f"   Device ID: {device_id}, Brightness: {brightness}%")
    try:
        result = control_home_lighting(device_id, brightness)
        print(f"✅ {result}")
    except Exception as e:
        print(f"❌ Error: {e}")

def main():
    print("\n🏠 Control4 Lighting Test Tool")
    print("Test your lighting controls without wake word or LLM\n")
    
    while True:
        print_menu()
        choice = input("\nEnter your choice: ").strip().lower()
        
        if choice == 'q':
            print("\n👋 Goodbye!")
            break
        
        elif choice == '1':
            brightness = input("Enter brightness (0-100): ").strip()
            try:
                test_light(85, int(brightness), "Kitchen Cans")
            except ValueError:
                print("❌ Invalid brightness value")
        
        elif choice == '2':
            brightness = input("Enter brightness (0-100): ").strip()
            try:
                test_light(87, int(brightness), "Foyer")
            except ValueError:
                print("❌ Invalid brightness value")
        
        elif choice == '3':
            brightness = input("Enter brightness (0-100): ").strip()
            try:
                test_light(89, int(brightness), "Stairs")
            except ValueError:
                print("❌ Invalid brightness value")
        
        elif choice == '4':
            brightness = input("Enter brightness (0-100): ").strip()
            try:
                test_light(91, int(brightness), "Upstairs Hall")
            except ValueError:
                print("❌ Invalid brightness value")
        
        elif choice == '5':
            brightness = input("Enter brightness (0-100): ").strip()
            try:
                test_light(93, int(brightness), "Front Door")
            except ValueError:
                print("❌ Invalid brightness value")
        
        elif choice == '6':
            brightness = input("Enter brightness (0-100): ").strip()
            try:
                test_light(95, int(brightness), "Kitchen Island")
            except ValueError:
                print("❌ Invalid brightness value")
        
        elif choice == '7':
            brightness = input("Enter brightness (0-100): ").strip()
            try:
                test_light(97, int(brightness), "Downstairs Hallway")
            except ValueError:
                print("❌ Invalid brightness value")
        
        elif choice == '8':
            brightness = input("Enter brightness (0-100): ").strip()
            try:
                test_light(99, int(brightness), "Upstairs Deck")
            except ValueError:
                print("❌ Invalid brightness value")
        
        elif choice == '9':
            brightness = input("Enter brightness (0-100): ").strip()
            try:
                test_light(204, int(brightness), "Family Room")
            except ValueError:
                print("❌ Invalid brightness value")
        
        elif choice == '10':
            brightness = input("Enter brightness (0-100): ").strip()
            try:
                test_light(206, int(brightness), "Breakfast")
            except ValueError:
                print("❌ Invalid brightness value")
        
        elif choice == '11':
            test_light(999, 100, "ALL LIGHTS ON (Scene)")
        
        elif choice == '12':
            test_light(999, 0, "ALL LIGHTS OFF (Scene)")
        
        elif choice == '13':
            try:
                device_id = int(input("Enter device ID: ").strip())
                brightness = int(input("Enter brightness (0-100): ").strip())
                test_light(device_id, brightness, f"Custom Device {device_id}")
            except ValueError:
                print("❌ Invalid input")
        
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted. Goodbye!")
        sys.exit(0)
