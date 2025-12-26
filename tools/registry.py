"""
Tool registry for voice assistant.
Centralizes tool schemas and dispatch logic.
"""
import json
from datetime import datetime
from google.genai import types
from control4_tool import control_home_lighting
from tools.bluetooth import connect_bluetooth_device, disconnect_bluetooth_device, get_bluetooth_status
from tools.audio import route_to_bluetooth, set_audio_sink, get_audio_sinks

# Tool specifications in Gemini format
GEMINI_TOOLS = [
    types.Tool(
        function_declarations=[
            types.FunctionDeclaration(
                name="control_home_lighting",
                description="Controls home lights. Kitchen Cans=85, Kitchen Island=95, Family Room=204, Foyer=87, Stairs=89",
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        "device_id": types.Schema(type="INTEGER", description="The device ID of the light to control"),
                        "brightness": types.Schema(type="INTEGER", description="Brightness level 0-100")
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
            "description": "Controls home lights. Kitchen Cans=85, Kitchen Island=95, Family Room=204, Foyer=87, Stairs=89",
            "parameters": {
                "type": "object",
                "properties": {
                    "device_id": {
                        "type": "integer",
                        "description": "The device ID of the light to control"
                    },
                    "brightness": {
                        "type": "integer",
                        "description": "Brightness level 0-100"
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
    }
]

# Tool function mapping
TOOL_FUNCTIONS = {
    "control_home_lighting": control_home_lighting,
    "connect_bluetooth_device": connect_bluetooth_device,
    "disconnect_bluetooth_device": disconnect_bluetooth_device,
    "route_to_bluetooth": route_to_bluetooth,
    "set_audio_sink": set_audio_sink
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
