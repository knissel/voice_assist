# pip3 install google-api-python-client google-auth-httplib2 google-auth-oauthlib python-dotenv

import os
import datetime as dt
from typing import List, Dict

from dotenv import load_dotenv

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build

# --------------------------------------------------
# ENV + CONSTANTS
# --------------------------------------------------

load_dotenv()

SCOPES = ["https://www.googleapis.com/auth/calendar"]  # read + write
TIMEZONE = os.getenv("TIMEZONE", "America/New_York")

# Determine credentials directory - defaults to calendar/ in project root
def _get_credentials_dir() -> str:
    """Get the directory containing credentials.json and token.json."""
    # First check for explicit env var
    creds_dir = os.getenv("CALENDAR_CREDENTIALS_DIR")
    if creds_dir and os.path.isdir(creds_dir):
        return creds_dir
    
    # Default to calendar/ directory relative to this file
    return os.path.dirname(os.path.abspath(__file__))

def _get_credentials_path() -> str:
    """Get full path to credentials.json."""
    return os.path.join(_get_credentials_dir(), "credentials.json")

def _get_token_path() -> str:
    """Get full path to token.json."""
    return os.path.join(_get_credentials_dir(), "token.json")


# --------------------------------------------------
# GOOGLE CALENDAR AUTH
# --------------------------------------------------

def get_calendar_service():
    """
    Returns an authenticated Google Calendar service.
    Uses OAuth2 with token.json to cache credentials.
    """
    creds = None
    token_path = _get_token_path()
    credentials_path = _get_credentials_path()
    
    if os.path.exists(token_path):
        creds = Credentials.from_authorized_user_file(token_path, SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                credentials_path, SCOPES
            )
            creds = flow.run_local_server(port=0)
        with open(token_path, "w") as token:
            token.write(creds.to_json())

    service = build("calendar", "v3", credentials=creds)
    return service


# --------------------------------------------------
# CORE FUNCTIONS FOR AGENDA
# --------------------------------------------------

def get_agenda(
    when: str = "today", calendar_id: str = "primary"
) -> List[Dict]:
    """
    Get events for 'today' or 'week'.
    Returns a list of events (dicts from Google API).
    """
    service = get_calendar_service()

    if when == "today":
        # Start of today in local timezone
        today_local = dt.datetime.now().astimezone()
        start = today_local.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + dt.timedelta(days=1)
    elif when == "week":
        # Monday-Sunday of current week in local timezone
        today_local = dt.datetime.now().astimezone()
        start = today_local - dt.timedelta(days=today_local.weekday())  # Monday
        start = start.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + dt.timedelta(days=7)
    else:
        raise ValueError("when must be 'today' or 'week'")

    time_min = start.isoformat()
    time_max = end.isoformat()

    events_result = (
        service.events()
        .list(
            calendarId=calendar_id,
            timeMin=time_min,
            timeMax=time_max,
            singleEvents=True,
            orderBy="startTime",
        )
        .execute()
    )
    events = events_result.get("items", [])
    return events


def format_agenda_text(events: List[Dict], when: str = "today") -> str:
    """
    Make a human-friendly text agenda suitable for TTS.
    """
    if not events:
        return f"No events on your {when} calendar."

    header = "Today's agenda." if when == "today" else "This week's agenda."
    lines = [header, ""]

    for event in events:
        start = event["start"].get("dateTime", event["start"].get("date"))
        end = event["end"].get("dateTime", event["end"].get("date"))
        summary = event.get("summary", "(No title)")

        # Try to parse times for nicer formatting
        try:
            start_dt = dt.datetime.fromisoformat(start.replace("Z", "+00:00"))
            end_dt = dt.datetime.fromisoformat(end.replace("Z", "+00:00"))
            start_str = start_dt.astimezone().strftime("%A %b %d, %I:%M %p")
            end_str = end_dt.astimezone().strftime("%I:%M %p")
            lines.append(f"- {start_str} to {end_str}: {summary}")
        except Exception:
            # All-day or unparseable
            lines.append(f"- On {start}: {summary}")

    return "\n".join(lines)


# --------------------------------------------------
# ADD EVENT TO CALENDAR
# --------------------------------------------------

def add_event(
    title: str,
    start_time: dt.datetime,
    end_time: dt.datetime,
    description: str = "",
    location: str = "",
    calendar_id: str = "primary",
) -> Dict:
    """
    Add a new event to Google Calendar.
    start_time and end_time should be timezone-aware datetimes.
    Returns created event object.
    """
    service = get_calendar_service()

    event_body = {
        "summary": title,
        "location": location,
        "description": description,
        "start": {
            "dateTime": start_time.isoformat(),
            "timeZone": TIMEZONE,
        },
        "end": {
            "dateTime": end_time.isoformat(),
            "timeZone": TIMEZONE,
        },
    }

    created_event = (
        service.events()
        .insert(calendarId=calendar_id, body=event_body)
        .execute()
    )
    return created_event


# --------------------------------------------------
# LLM / HOME-ASSISTANT FRIENDLY WRAPPERS
# --------------------------------------------------

def tool_get_agenda(when: str = "today") -> str:
    """
    Wrapper for LLM or home assistant.
    Returns a human-readable agenda string for 'today' or 'week'.
    """
    events = get_agenda(when=when)
    return format_agenda_text(events, when=when)


def tool_add_event(
    title: str,
    start_iso: str,
    end_iso: str,
    description: str = "",
    location: str = "",
) -> str:
    """
    Wrapper for LLM / assistant: expects ISO8601 strings for start/end.
    Example: 2026-01-10T09:00:00-05:00
    """
    start_dt = dt.datetime.fromisoformat(start_iso)
    end_dt = dt.datetime.fromisoformat(end_iso)
    created = add_event(
        title=title,
        start_time=start_dt,
        end_time=end_dt,
        description=description,
        location=location,
    )
    link = created.get("htmlLink", "")
    if link:
        return f"Event created: {title} at {start_iso}. Link: {link}"
    return f"Event created: {title} at {start_iso}."


# --------------------------------------------------
# CLI / DEBUG ENTRYPOINT
# --------------------------------------------------

if __name__ == "__main__":
    # When you run: python3 calendar_assistant.py
    # It will just print today's agenda to stdout.
    agenda_text = tool_get_agenda("today")
    print(agenda_text)
