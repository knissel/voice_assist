"""
Tool registry for voice assistant.
Centralizes tool schemas and dispatch logic.
"""
import json
from datetime import datetime
from google.genai import types
from tools.control4_tool import control_home_lighting
from tools.bluetooth import connect_bluetooth_device, disconnect_bluetooth_device, get_bluetooth_status
from tools.audio import route_to_bluetooth, set_audio_sink, get_audio_sinks, control_volume, pause_audio, resume_audio
from tools.youtube_music import play_youtube_music, stop_music
from tools.timer import set_timer, cancel_timer, list_timers, check_timer
from tools.recipes import pizza_dough_recipe

# Tool specifications in Gemini format
GEMINI_TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="control_home_lighting",
                description="Controls home lights. Kitchen Cans=85, Foyer=87, Stairs=89, Upstairs Hall=91, Front Door=93, Kitchen Island=95, Downstairs Hallway=97, Upstairs Deck=99, Family Room=204, Breakfast=206. For all lights use device_id=999 with brightness=100 (on) or 0 (off).",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "device_id": types.Schema(type="INTEGER", description="The device ID of the light to control (use 999 for all lights)"),
                        "brightness": types.Schema(type="INTEGER", description="Brightness level 0-100 (for all lights: 100 on, 0 off)")
                    },
                    required=["device_id", "brightness"]
                )
            ),
            types.FunctionDeclaration(
                name="connect_bluetooth_device",
                description="Connect to a Bluetooth device by MAC address",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "device_mac": types.Schema(type="STRING", description="MAC address of the Bluetooth device (e.g., AA:BB:CC:DD:EE:FF)")
                    },
                    required=["device_mac"]
                )
            ),
            types.FunctionDeclaration(
                name="disconnect_bluetooth_device",
                description="Disconnect from a Bluetooth device by MAC address",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "device_mac": types.Schema(type="STRING", description="MAC address of the Bluetooth device")
                    },
                    required=["device_mac"]
                )
            ),
            types.FunctionDeclaration(
                name="route_to_bluetooth",
                description="Route audio output to Bluetooth speaker. Falls back to default if Bluetooth unavailable.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "device_name": types.Schema(type="STRING", description="Optional: specific Bluetooth device name to route to")
                    },
                    required=[]
                )
            ),
            types.FunctionDeclaration(
                name="set_audio_sink",
                description="Set the default audio output sink/device",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "sink_id": types.Schema(type="STRING", description="Audio sink identifier or device name")
                    },
                    required=["sink_id"]
                )
            ),
            types.FunctionDeclaration(
                name="play_youtube_music",
                description="Play music on YouTube Music. Search and play songs, albums, artists, playlists, or videos.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "query": types.Schema(type="STRING", description="What to play (song name, artist name, playlist name, etc.)"),
                        "content_type": types.Schema(type="STRING", description="Type of content: 'song', 'video', 'album', 'artist', or 'playlist'. Defaults to 'song'.")
                    },
                    required=["query"]
                )
            ),
            types.FunctionDeclaration(
                name="stop_music",
                description="Stop all audio playback including music and text-to-speech. Use when user asks to stop or turn off music/audio.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={},
                    required=[]
                )
            ),
            types.FunctionDeclaration(
                name="pause_audio",
                description="Pause current audio or music playback. Use when user says pause, hold on, or wait.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={},
                    required=[]
                )
            ),
            types.FunctionDeclaration(
                name="resume_audio",
                description="Resume audio or music playback after a pause. Use when user says resume, continue, or play again.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={},
                    required=[]
                )
            ),
            types.FunctionDeclaration(
                name="control_volume",
                description="Control system volume. Can turn volume up, down, or set to a specific percentage (0-100).",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "action": types.Schema(type="STRING", description="Action to perform: 'up', 'down', or 'set'"),
                        "level": types.Schema(type="INTEGER", description="For 'set': target volume 0-100. For 'up'/'down': amount to adjust (default 10)")
                    },
                    required=["action"]
                )
            ),
            types.FunctionDeclaration(
                name="set_timer",
                description="Set a timer for a specified duration. Use for cooking, reminders, etc.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "duration_minutes": types.Schema(type="INTEGER", description="Duration in minutes"),
                        "name": types.Schema(type="STRING", description="Optional name for the timer (e.g., 'pasta', 'eggs', 'laundry')")
                    },
                    required=["duration_minutes"]
                )
            ),
            types.FunctionDeclaration(
                name="cancel_timer",
                description="Cancel an active timer. If no name provided, cancels the most recent timer.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "name": types.Schema(type="STRING", description="Name of timer to cancel (optional)")
                    },
                    required=[]
                )
            ),
            types.FunctionDeclaration(
                name="check_timer",
                description="Check how much time is left on a timer.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "name": types.Schema(type="STRING", description="Name of timer to check (optional, defaults to most recent)")
                    },
                    required=[]
                )
            ),
            types.FunctionDeclaration(
                name="list_timers",
                description="List all active timers with their remaining time.",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={},
                    required=[]
                )
            ),
            types.FunctionDeclaration(
                name="pizza_dough_recipe",
                description=(
                    "Create or adjust a pizza dough recipe using baker's percentages. "
                    "Use for Neapolitan dough requests or when user says to adjust hydration. "
                    "If only a change is provided (e.g., hydration_percent), reuse the last recipe."
                ),
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "style": types.Schema(type="STRING", description="Pizza style, e.g. 'neapolitan'"),
                        "ball_weight_g": types.Schema(type="NUMBER", description="Target dough ball weight in grams"),
                        "ball_count": types.Schema(type="INTEGER", description="Number of dough balls"),
                        "hydration_percent": types.Schema(type="NUMBER", description="Hydration percentage (e.g., 62)"),
                        "salt_percent": types.Schema(type="NUMBER", description="Salt percentage (e.g., 2.8)"),
                        "yeast_percent": types.Schema(type="NUMBER", description="Instant yeast percentage (e.g., 0.1)"),
                        "oil_percent": types.Schema(type="NUMBER", description="Oil percentage (0 for Neapolitan)"),
                        "bake_temp_f": types.Schema(type="INTEGER", description="Bake temperature in Fahrenheit"),
                        "cold_ferment_hours": types.Schema(type="NUMBER", description="Cold ferment duration in hours"),
                        "room_temp_hours": types.Schema(type="NUMBER", description="Room temperature proof duration in hours"),
                        "use_last": types.Schema(type="BOOLEAN", description="Use last recipe as base when fields are omitted")
                    },
                    required=[]
                )
            )
        ]
    )
]

# Tool specifications in OpenAI format (kept for reference)
TOOL_SPECS = [
    {
        "type": "function",
        "function": {
            "name": "control_home_lighting",
            "description": "Controls home lights. Kitchen Cans=85, Foyer=87, Stairs=89, Upstairs Hall=91, Front Door=93, Kitchen Island=95, Downstairs Hallway=97, Upstairs Deck=99, Family Room=204, Breakfast=206. For all lights use device_id=999 with brightness=100 (on) or 0 (off).",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {
                        "type": "integer",
                        "description": "The device ID of the light to control (use 999 for all lights)"
                    },
                    "brightness": {
                        "type": "integer",
                        "description": "Brightness level 0-100 (for all lights: 100 on, 0 off)"
                    }
                },
                "required": ["device_id", "brightness"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "connect_bluetooth_device",
            "description": "Connect to a Bluetooth device by MAC address",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_mac": {
                        "type": "string",
                        "description": "MAC address of the Bluetooth device (e.g., AA:BB:CC:DD:EE:FF)"
                    }
                },
                "required": ["device_mac"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "disconnect_bluetooth_device",
            "description": "Disconnect from a Bluetooth device by MAC address",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_mac": {
                        "type": "string",
                        "description": "MAC address of the Bluetooth device"
                    }
                },
                "required": ["device_mac"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "route_to_bluetooth",
            "description": "Route audio output to Bluetooth speaker. Falls back to default if Bluetooth unavailable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_name": {
                        "type": "string",
                        "description": "Optional: specific Bluetooth device name to route to"
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "set_audio_sink",
            "description": "Set the default audio output sink/device",
            "parameters": {
                "type": "object",
                "properties": {
                    "sink_id": {
                        "type": "string",
                        "description": "Audio sink identifier or device name"
                    }
                },
                "required": ["sink_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "play_youtube_music",
            "description": "Play music on YouTube Music. Search and play songs, albums, artists, playlists, or videos.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "What to play (song name, artist name, playlist name, etc.)"
                    },
                    "content_type": {
                        "type": "string",
                        "description": "Type of content: 'song', 'video', 'album', 'artist', or 'playlist'. Defaults to 'song'."
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "stop_music",
            "description": "Stop all audio playback including music and text-to-speech. Use when user asks to stop or turn off music/audio.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pause_audio",
            "description": "Pause current audio or music playback. Use when user says pause, hold on, or wait.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "resume_audio",
            "description": "Resume audio or music playback after a pause. Use when user says resume, continue, or play again.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "control_volume",
            "description": "Control system volume. Can turn volume up, down, or set to a specific percentage (0-100).",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action to perform: 'up', 'down', or 'set'"
                    },
                    "level": {
                        "type": "integer",
                        "description": "For 'set': target volume 0-100. For 'up'/'down': amount to adjust (default 10)"
                    }
                },
                "required": ["action"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pizza_dough_recipe",
            "description": "Create or adjust a pizza dough recipe using baker's percentages. Use for Neapolitan dough requests or hydration adjustments. If only a change is provided, reuse the last recipe.",
            "parameters": {
                "type": "object",
                "properties": {
                    "style": {
                        "type": "string",
                        "description": "Pizza style, e.g. 'neapolitan'"
                    },
                    "ball_weight_g": {
                        "type": "number",
                        "description": "Target dough ball weight in grams"
                    },
                    "ball_count": {
                        "type": "integer",
                        "description": "Number of dough balls"
                    },
                    "hydration_percent": {
                        "type": "number",
                        "description": "Hydration percentage (e.g., 62)"
                    },
                    "salt_percent": {
                        "type": "number",
                        "description": "Salt percentage (e.g., 2.8)"
                    },
                    "yeast_percent": {
                        "type": "number",
                        "description": "Instant yeast percentage (e.g., 0.1)"
                    },
                    "oil_percent": {
                        "type": "number",
                        "description": "Oil percentage (0 for Neapolitan)"
                    },
                    "bake_temp_f": {
                        "type": "integer",
                        "description": "Bake temperature in Fahrenheit"
                    },
                    "cold_ferment_hours": {
                        "type": "number",
                        "description": "Cold ferment duration in hours"
                    },
                    "room_temp_hours": {
                        "type": "number",
                        "description": "Room temperature proof duration in hours"
                    },
                    "use_last": {
                        "type": "boolean",
                        "description": "Use last recipe as base when fields are omitted"
                    }
                },
                "required": []
            }
        }
    }
]

# Tool function mapping
TOOL_FUNCTIONS = {
    "control_home_lighting": control_home_lighting,
    "connect_bluetooth_device": connect_bluetooth_device,
    "disconnect_bluetooth_device": disconnect_bluetooth_device,
    "route_to_bluetooth": route_to_bluetooth,
    "set_audio_sink": set_audio_sink,
    "play_youtube_music": play_youtube_music,
    "stop_music": stop_music,
    "pause_audio": pause_audio,
    "resume_audio": resume_audio,
    "control_volume": control_volume,
    "set_timer": set_timer,
    "cancel_timer": cancel_timer,
    "check_timer": check_timer,
    "list_timers": list_timers,
    "pizza_dough_recipe": pizza_dough_recipe
}

def dispatch_tool(name: str, args: dict) -> str:
    """
    Dispatch a tool call by name with arguments.
    Logs every invocation.
    
    Args:
        name: Tool function name
        args: Dictionary of arguments
        
    Returns:
        Result string from tool execution
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] 🔧 Tool Call: {name}")
    print(f"[{timestamp}] 📋 Args: {json.dumps(args, indent=2)}")
    
    if name not in TOOL_FUNCTIONS:
        error_msg = f"Unknown tool: {name}"
        print(f"[{timestamp}] ❌ Error: {error_msg}")
        return error_msg
    
    try:
        result = TOOL_FUNCTIONS[name](**args)
        print(f"[{timestamp}] ✅ Result: {result}")
        return result
    except Exception as e:
        error_msg = f"Tool execution failed: {str(e)}"
        print(f"[{timestamp}] ❌ Error: {error_msg}")
        return error_msg
