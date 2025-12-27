import asyncio
import threading
import time
import os
from pyControl4.account import C4Account
from pyControl4.director import C4Director
from pyControl4.light import C4Light
from dotenv import load_dotenv

load_dotenv()


class Control4Manager:
    """Manages connection to Control4 Director with automatic reconnection."""
    
    # Token refresh interval (tokens typically expire after ~24 hours)
    TOKEN_REFRESH_INTERVAL = 3600  # Refresh every hour to be safe
    
    def __init__(self, username, password, ip):
        self.username = username
        self.password = password
        self.ip = ip
        self.director = None
        self._last_connect_time = 0
        self._lock = threading.Lock()

    async def connect(self):
        """Connect to Control4 Director and get auth token."""
        account = C4Account(self.username, self.password)
        await account.getAccountBearerToken()
        controllers = await account.getAccountControllers()
        name = controllers['controllerCommonName']
        token_data = await account.getDirectorBearerToken(name)
        self.director = C4Director(self.ip, token_data['token'])
        self._last_connect_time = time.time()
        print("✅ Connected to Control4 Director.")
    
    async def ensure_connected(self):
        """Ensure we have a valid connection, reconnecting if needed."""
        needs_reconnect = (
            self.director is None or
            (time.time() - self._last_connect_time) > self.TOKEN_REFRESH_INTERVAL
        )
        if needs_reconnect:
            await self.connect()

    async def set_light(self, device_id, level):
        await self.ensure_connected()
        light = C4Light(self.director, device_id)
        await light.setLevel(level)
        return f"Successfully set light {device_id} to {level}%"
    
    async def activate_scene(self, scene_id):
        await self.ensure_connected()
        await self.director.sendPostRequest(
            "/api/v1/items/4/commands",
            "ACTIVATE_SCENE",
            {"SCENE_ID": scene_id}
        )
        scene_names = {0: "Sunrise", 1: "Sunset", 2: "All ON", 3: "All OFF"}
        return f"Activated scene: {scene_names.get(scene_id, scene_id)}"


# === Singleton Pattern for Connection Caching ===
# This avoids creating a new connection for every light command

_control4_manager = None
_control4_lock = threading.Lock()
_event_loop = None
_loop_lock = threading.Lock()


def _get_event_loop():
    """Get or create a dedicated event loop for Control4 operations."""
    global _event_loop
    with _loop_lock:
        if _event_loop is None or _event_loop.is_closed():
            _event_loop = asyncio.new_event_loop()
        return _event_loop


def _get_manager():
    """Get or create the singleton Control4Manager."""
    global _control4_manager
    with _control4_lock:
        if _control4_manager is None:
            username = os.getenv("CONTROL4_USERNAME")
            password = os.getenv("CONTROL4_PASSWORD")
            controller_ip = os.getenv("CONTROL4_CONTROLLER_IP", "192.168.20.12")
            _control4_manager = Control4Manager(username, password, controller_ip)
        return _control4_manager


def _run_async(coro):
    """Run an async coroutine in the dedicated event loop."""
    loop = _get_event_loop()
    return loop.run_until_complete(coro)


# The actual function Gemini will "see"
def control_home_lighting(device_id: int, brightness: int):
    """
    Adjusts home lights. 
    Kitchen Cans=85, Kitchen Island=95, Family Room=204, Foyer=87, Stairs=89.
    For ALL lights: use device_id=999 with brightness=100 for All ON or brightness=0 for All OFF.
    """
    manager = _get_manager()
    
    # Special handling for "all lights" using scene activation
    if device_id == 999:
        if brightness == 0:
            return _run_async(manager.activate_scene(3))  # All OFF scene
        else:
            return _run_async(manager.activate_scene(2))  # All ON scene
    
    return _run_async(manager.set_light(device_id, brightness))