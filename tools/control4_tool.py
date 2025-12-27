import asyncio
import json
import os
from pyControl4.account import C4Account
from pyControl4.director import C4Director
from pyControl4.light import C4Light
from dotenv import load_dotenv

load_dotenv()

class Control4Manager:
    def __init__(self, username, password, ip):
        self.username = username
        self.password = password
        self.ip = ip
        self.director = None

    async def connect(self):
        account = C4Account(self.username, self.password)
        await account.getAccountBearerToken()
        controllers = await account.getAccountControllers()
        name = controllers['controllerCommonName']
        token_data = await account.getDirectorBearerToken(name)
        self.director = C4Director(self.ip, token_data['token'])
        print("Connected to Control4 Director.")

    async def set_light(self, device_id, level):
        if not self.director: await self.connect()
        light = C4Light(self.director, device_id)
        await light.setLevel(level)
        return f"Successfully set light {device_id} to {level}%"
    
    async def activate_scene(self, scene_id):
        if not self.director: await self.connect()
        await self.director.sendPostRequest(
            "/api/v1/items/4/commands",
            "ACTIVATE_SCENE",
            {"SCENE_ID": scene_id}
        )
        scene_names = {0: "Sunrise", 1: "Sunset", 2: "All ON", 3: "All OFF"}
        return f"Activated scene: {scene_names.get(scene_id, scene_id)}"

# The actual function Gemini will "see"
def control_home_lighting(device_id: int, brightness: int):
    """
    Adjusts home lights. 
    Kitchen Cans=85, Kitchen Island=95, Family Room=204, Foyer=87, Stairs=89.
    For ALL lights: use device_id=999 with brightness=100 for All ON or brightness=0 for All OFF.
    """
    username = os.getenv("CONTROL4_USERNAME")
    password = os.getenv("CONTROL4_PASSWORD")
    controller_ip = os.getenv("CONTROL4_CONTROLLER_IP", "192.168.20.12")
    manager = Control4Manager(username, password, controller_ip)
    
    # Special handling for "all lights" using scene activation
    if device_id == 999:
        if brightness == 0:
            return asyncio.run(manager.activate_scene(3))  # All OFF scene
        else:
            return asyncio.run(manager.activate_scene(2))  # All ON scene
    
    return asyncio.run(manager.set_light(device_id, brightness))