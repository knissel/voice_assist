"""
Calendar tool for voice assistant edge service.
Wraps Google Calendar functions for agenda and event management.
"""
import sys
import os
import importlib.util

# Add project root to path and import calendar_assistant using direct file import
# to avoid conflict with Python's built-in 'calendar' module
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))

# Load calendar_assistant module directly from file path
_calendar_module_path = os.path.join(_project_root, "calendar", "calendar_assistant.py")
_spec = importlib.util.spec_from_file_location("calendar_assistant", _calendar_module_path)
_calendar_assistant = importlib.util.module_from_spec(_spec)
sys.modules["calendar_assistant"] = _calendar_assistant
_spec.loader.exec_module(_calendar_assistant)

# Import the wrapped functions
tool_get_agenda = _calendar_assistant.tool_get_agenda
tool_add_event = _calendar_assistant.tool_add_event

# Re-export for use in registry
__all__ = ["tool_get_agenda", "tool_add_event"]
