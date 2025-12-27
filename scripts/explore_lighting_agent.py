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
    
    # Advanced Lighting Agent ID
    lighting_agent_id = 4
    
    # Try to get variables for the Advanced Lighting agent
    print(f"=== EXPLORING ADVANCED LIGHTING AGENT (ID: {lighting_agent_id}) ===\n")
    
    # Try to get item info
    try:
        item_info = await director.getItemInfo(lighting_agent_id)
        print("Item Info:")
        print(json.dumps(json.loads(item_info), indent=2))
    except Exception as e:
        print(f"Could not get item info: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Try to get item variables
    try:
        variables = await director.getItemVariables(lighting_agent_id)
        print("Item Variables:")
        print(json.dumps(json.loads(variables), indent=2))
    except Exception as e:
        print(f"Could not get item variables: {e}")
    
    print("\n" + "="*60 + "\n")
    
    # Try to get item commands
    try:
        commands = await director.getItemCommands(lighting_agent_id)
        print("Available Commands:")
        print(json.dumps(json.loads(commands), indent=2))
    except Exception as e:
        print(f"Could not get item commands: {e}")

asyncio.run(main())
