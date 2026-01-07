"""
Bluetooth device management via bluetoothctl.
Provides connect/disconnect/status functionality.
"""
import subprocess
import re

def get_bluetooth_status():
    """Get current bluetooth controller status."""
    try:
        result = subprocess.run(
            ["bluetoothctl", "show"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if "Powered: yes" in result.stdout:
            return "Bluetooth is powered on"
        return "Bluetooth is powered off"
    except Exception as e:
        return f"Failed to get bluetooth status: {str(e)}"

def connect_bluetooth_device(device_mac: str):
    """
    Connect to a Bluetooth device by MAC address.
    
    Args:
        device_mac: MAC address like "AA:BB:CC:DD:EE:FF"
    
    Returns:
        Success or error message
    """
    try:
        result = subprocess.run(
            ["bluetoothctl", "connect", device_mac],
            capture_output=True,
            text=True,
            timeout=10
        )
        if "Connection successful" in result.stdout or "Connected: yes" in result.stdout:
            return f"Successfully connected to {device_mac}"
        return f"Failed to connect to {device_mac}: {result.stdout}"
    except subprocess.TimeoutExpired:
        return f"Connection to {device_mac} timed out"
    except Exception as e:
        return f"Error connecting to {device_mac}: {str(e)}"

def disconnect_bluetooth_device(device_mac: str):
    """
    Disconnect from a Bluetooth device by MAC address.
    
    Args:
        device_mac: MAC address like "AA:BB:CC:DD:EE:FF"
    
    Returns:
        Success or error message
    """
    try:
        result = subprocess.run(
            ["bluetoothctl", "disconnect", device_mac],
            capture_output=True,
            text=True,
            timeout=10
        )
        if "Successful disconnected" in result.stdout or "Disconnected" in result.stdout:
            return f"Successfully disconnected from {device_mac}"
        return f"Failed to disconnect from {device_mac}: {result.stdout}"
    except Exception as e:
        return f"Error disconnecting from {device_mac}: {str(e)}"

def list_paired_devices():
    """List all paired Bluetooth devices."""
    try:
        result = subprocess.run(
            ["bluetoothctl", "devices", "Paired"],
            capture_output=True,
            text=True,
            timeout=5
        )
        devices = []
        for line in result.stdout.split('\n'):
            match = re.match(r'Device\s+([0-9A-F:]+)\s+(.+)', line)
            if match:
                devices.append({"mac": match.group(1), "name": match.group(2)})
        return devices if devices else []
    except Exception as e:
        return []
