import asyncio
import json
from pyControl4.account import C4Account
from pyControl4.director import C4Director

USERNAME = "knissel@gmail.com"
PASSWORD = "army1337"
IP = "192.168.20.12"

async def main():
    # Connect to Control4
    account = C4Account(USERNAME, PASSWORD)
    await account.getAccountBearerToken()
    controllers = await account.getAccountControllers()
    name = controllers['controllerCommonName']
    token_data = await account.getDirectorBearerToken(name)
    director = C4Director(IP, token_data['token'])
    
    print("Connected to Control4 Director.\n")
    
    # Get all devices
    devices_raw = await director.getAllItemInfo()
    devices = json.loads(devices_raw)
    
    # Find lights
    print("=== LIGHTS FOUND ===")
    for device in devices:
        device_type = device.get('type', 0)
        device_name = device.get('name', 'unknown')
        device_id = device.get('id', 0)
        control = device.get('control', '')
        
        # Type 4 is typically lights
        if device_type == 4 or 'light' in control.lower() or 'dimmer' in control.lower():
            if 'agent' not in control.lower():
                print(f"ID: {device_id:4d} | Name: {device_name}")

asyncio.run(main())
