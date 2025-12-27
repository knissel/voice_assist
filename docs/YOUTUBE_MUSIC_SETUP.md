# YouTube Music Setup Guide

This guide will help you set up YouTube Music integration for your voice assistant.

## Prerequisites

- YouTube Music Premium account
- Python 3.7+

## Installation

1. Install the required dependency:
```bash
pip install -r requirements.txt
```

## Authentication Setup

You need to authenticate with your YouTube Music account to use this tool.

### Method 1: OAuth Authentication (Recommended)

1. Run the setup script in Python:
```python
from ytmusicapi import YTMusic
YTMusic.setup(filepath='oauth.json')
```

2. This will:
   - Open your browser
   - Ask you to log in to your Google account
   - Request permission to access YouTube Music
   - Save the authentication token to `oauth.json`

3. The `oauth.json` file will be created in your project root directory

### Method 2: Browser Headers (Alternative)

If OAuth doesn't work, you can use browser headers:

1. Open YouTube Music in your browser (music.youtube.com)
2. Open Developer Tools (F12)
3. Go to the Network tab
4. Click on any request to `music.youtube.com`
5. Copy the request headers
6. Run:
```python
from ytmusicapi import YTMusic
YTMusic.setup(filepath='oauth.json', headers_raw='<paste headers here>')
```

## Usage

Once set up, you can use voice commands like:

- "Play [song name] on YouTube Music"
- "Play [artist name] on YouTube Music"
- "Play my [playlist name] playlist"
- "Play the album [album name]"

The tool will:
1. Search YouTube Music for your request
2. Find the best match
3. Open it in your default browser

## Troubleshooting

### "YouTube Music not initialized" error
- Make sure `oauth.json` exists in the project root
- Re-run the setup if the token expired

### Authentication expired
- Delete `oauth.json`
- Run the setup process again

### No results found
- Try being more specific with song/artist names
- Check that you're logged into YouTube Music Premium

## File Location

The authentication file should be located at:
```
/Users/kennynissel/voice_assist/oauth.json
```

Do NOT commit this file to version control (it's sensitive).
