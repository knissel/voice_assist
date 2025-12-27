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
    
    # Get scenes from Advanced Lighting agent
    print("=== AVAILABLE LIGHTING SCENES ===\n")
    
    try:
        scenes_response = await director.sendGetRequest("/api/v1/agents/advanced_lighting")
        scenes = json.loads(scenes_response)
        print(json.dumps(scenes, indent=2))
    except Exception as e:
        print(f"Error getting scenes: {e}")

asyncio.run(main())
