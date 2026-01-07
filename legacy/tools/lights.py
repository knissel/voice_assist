import asyncio
import os
from pyControl4.account import C4Account
from pyControl4.director import C4Director
from pyControl4.light import C4Light
from dotenv import load_dotenv

load_dotenv()

USERNAME = os.getenv("CONTROL4_USERNAME")
PASSWORD = os.getenv("CONTROL4_PASSWORD")
CONTROLLER_IP = os.getenv("CONTROL4_CONTROLLER_IP", "192.168.20.12")

async def main():
    # 1. Login to Control4 Cloud to get tokens
    account = C4Account(USERNAME, PASSWORD)
    await account.getAccountBearerToken()
    
    # 2. Get your Controller details
    controllers = await account.getAccountControllers()
    print(f"Controllers: {controllers}")
    controller_common_name = controllers['controllerCommonName']
    print(f"Controller Common Name: {controller_common_name}")
    
    # 3. Connect to the Local Director (Controller)
    director_token_data = await account.getDirectorBearerToken(controller_common_name)
    director_token = director_token_data['token']
    print(f"Director Token received")
    
    director = C4Director(CONTROLLER_IP, director_token) # Local IP of controller

    # 4. Get all devices to find the correct device IDs
    import json
    devices_raw = await director.getAllItemInfo()
    devices = json.loads(devices_raw)
    
    # Find all lights
    print("\nAvailable lights:")
    lights = []
    for device in devices:
        device_type = device.get('type', 0)
        device_name = device.get('name', 'unknown')
        device_id = device.get('id', 0)
        control = device.get('control', '')
        
        # Type 4 is typically lights, but skip agents and only get actual light devices
        if (device_type == 4 or 'light' in control.lower() or 'dimmer' in control.lower()) and 'agent' not in control.lower():
            lights.append((device_id, device_name, control))
            print(f"  ID: {device_id}, Name: {device_name}, Control: {control}")
    
    # Control all lights found
    if lights:
        print(f"\nTurning off all {len(lights)} lights...")
        for light_id, light_name, light_control in lights:
            try:
                light = C4Light(director, light_id)
                await light.setLevel(0)
                print(f"✓ {light_name} (ID: {light_id})")
            except Exception as e:
                print(f"✗ {light_name} (ID: {light_id}) - Error: {e}")
        print(f"\nCompleted turning off all lights")
    else:
        print("\nNo lights found in the system")

asyncio.run(main())