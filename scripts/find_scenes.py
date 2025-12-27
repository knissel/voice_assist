import asyncio
import json
from pyControl4.account import C4Account
from pyControl4.director import C4Director
from dotenv import load_dotenv
import os

load_dotenv()

USERNAME = os.getenv("CONTROL4_USERNAME")
PASSWORD = os.getenv("CONTROL4_PASSWORD")
IP = os.getenv("CONTROL4_CONTROLLER_IP", "192.168.20.12")

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
    
    # Find Advanced Lighting agents and scenes
    print("=== SEARCHING FOR LIGHTING SCENES/AGENTS ===")
    for device in devices:
        device_type = device.get('type', 0)
        device_name = device.get('name', 'unknown')
        device_id = device.get('id', 0)
        control = device.get('control', '')
        proxy = device.get('proxy', '')
        
        # Look for anything related to scenes, advanced lighting, or "all"
        if any(keyword in device_name.lower() for keyword in ['scene', 'all', 'lighting', 'agent']) or \
           any(keyword in control.lower() for keyword in ['scene', 'advanced', 'lighting']) or \
           any(keyword in proxy.lower() for keyword in ['scene', 'lighting']):
            print(f"\nID: {device_id}")
            print(f"  Name: {device_name}")
            print(f"  Type: {device_type}")
            print(f"  Control: {control}")
            print(f"  Proxy: {proxy}")

asyncio.run(main())
