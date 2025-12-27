# Voice Assistant with Gemini Flash

A voice-controlled assistant powered by Google Gemini Flash that can control smart home devices, manage Bluetooth connections, and answer questions using natural language.

## Features

- 🎤 **Wake Word Detection**: Uses Porcupine for "americano", "computer", or custom wake words
- 🗣️ **Speech-to-Text**: Local transcription with Whisper.cpp
- 🤖 **AI Processing**: Google Gemini 2.0 Flash Lite for natural language understanding
- 🔧 **Function Calling**: Control smart home lights, Bluetooth devices, audio routing, and YouTube Music
- 🔊 **Text-to-Speech**: High-quality neural voices using Google Cloud TTS
- ⌨️ **Push-to-Talk Mode**: Alternative mode using spacebar to activate
- 🎵 **YouTube Music**: Play songs, albums, artists, and playlists
- 🔉 **Volume Control**: Adjust system volume with voice commands

## Prerequisites

- Python 3.9+
- macOS or Linux (Raspberry Pi supported)
- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) installed locally
- Google Gemini API key
- Google Cloud TTS credentials (for high-quality voice)
- Porcupine access key (for wake word detection)
- mpg123 audio player (for TTS playback)

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd voice_assist
   ```

2. **Create a virtual environment**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. **Install system dependencies**
   ```bash
   # macOS
   brew install mpg123
   
   # Linux/Raspberry Pi
   sudo apt-get install mpg123
   ```

4. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

5. **Set up Whisper.cpp**
   ```bash
   # Clone and build whisper.cpp
   git clone https://github.com/ggerganov/whisper.cpp.git
   cd whisper.cpp
   make
   
   # Download a model (tiny is recommended for speed)
   bash ./models/download-ggml-model.sh tiny
   cd ..
   ```

6. **Set up Google Cloud TTS**
   
   1. Go to [Google Cloud Console](https://console.cloud.google.com)
   2. Create a new project or select existing
   3. Enable "Cloud Text-to-Speech API"
   4. Create a service account:
      - Go to IAM & Admin → Service Accounts
      - Click "Create Service Account"
      - Grant "Cloud Text-to-Speech User" role
      - Create and download JSON key
   5. Save the JSON key file to your project directory

7. **Configure environment variables**
   
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```bash
   # Gemini API Configuration
   GEMINI_API_KEY=your_gemini_api_key_here
   MODEL_NAME=gemini-flash-lite-latest
   
   # Porcupine Wake Word Configuration
   PORCUPINE_ACCESS_KEY=your_porcupine_access_key_here
   
   # Whisper Configuration (update paths if needed)
   WHISPER_PATH=/path/to/whisper.cpp/build/bin/whisper-cli
   MODEL_PATH=/path/to/whisper.cpp/models/ggml-tiny.bin
   
   # Google Cloud TTS Configuration
   GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account-key.json
   
   # Control4 Configuration (if using Control4 smart home)
   CONTROL4_USERNAME=your_username
   CONTROL4_PASSWORD=your_password
   CONTROL4_CONTROLLER_IP=192.168.x.x
   ```

## Getting API Keys

### Google Gemini API Key
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Sign in with your Google account
3. Create a new API key
4. Copy the key to your `.env` file

### Porcupine Access Key
1. Visit [Picovoice Console](https://console.picovoice.ai/)
2. Sign up for a free account
3. Create an access key
4. Copy the key to your `.env` file

### Google Cloud TTS Credentials
1. Follow the setup instructions in step 6 of Installation
2. Download the service account JSON key
3. Update `GOOGLE_APPLICATION_CREDENTIALS` in `.env` with the path to your JSON key file
4. Free tier includes 1 million characters per month

## Usage

### Wake Word Mode (Continuous Listening)

Run the wake word listener that responds to "americano" or "computer":

```bash
python wakeword.py
```

Say the wake word, then speak your command. The assistant will:
1. Detect the wake word
2. Record your voice command (4 seconds)
3. Transcribe it using Whisper.cpp
4. Process with Gemini Flash Lite
5. Execute actions or respond verbally with high-quality TTS

**Recommended for Raspberry Pi** - hands-free operation without keyboard

### Push-to-Talk Mode

Run the push-to-talk assistant:

```bash
python run_assistant.py
```

- Press **SPACE** to record a command
- Press **ESC** to quit

## Supported Commands

### Lighting Control
- "Turn on the kitchen lights"
- "Set kitchen island to 50%"
- "Turn off the family room lights"
- "Dim the foyer lights"

**Device IDs:**
- Kitchen Cans: 85
- Kitchen Island: 95
- Family Room: 204
- Foyer: 87
- Stairs: 89

### Bluetooth & Audio Control
- "Connect to [device name]"
- "Disconnect Bluetooth"
- "Route audio to Bluetooth"
- "Turn up the volume"
- "Set volume to 50%"
- "Turn down the volume"

### YouTube Music
- "Play [song name]"
- "Play [artist name]"
- "Play [album name]"
- "Stop the music"
- "Pause"

### General Questions
Ask any question and Jarvis will respond with concise answers.

## Project Structure

```
voice_assist/
├── .env                    # Environment variables (not in git)
├── .env.example           # Template for environment variables
├── .gitignore             # Git ignore rules
├── requirements.txt       # Python dependencies
├── README.md             # This file
├── run_assistant.py      # Push-to-talk mode (main entry point)
├── wakeword.py           # Wake word detection mode (main entry point)
├── main.py               # Basic example script
├── docs/                 # Documentation
│   ├── RASPBERRY_PI_SETUP.md
│   ├── YOUTUBE_MUSIC_SETUP.md
│   ├── STOP_AUDIO_USAGE.md
│   └── VOLUME_CONTROL_USAGE.md
├── tests/                # Test files
│   ├── test_volume_control.py
│   ├── test_stop_music.py
│   ├── test_youtube_music.py
│   └── ... (other test files)
├── scripts/              # Utility scripts
│   ├── setup_youtube_music.py
│   ├── find_devices.py
│   ├── find_scenes.py
│   └── ... (other utility scripts)
├── tools/                # Core tool modules
│   ├── registry.py       # Tool definitions and dispatch
│   ├── control4_tool.py  # Control4 smart home integration
│   ├── lights.py         # Lighting control utilities
│   ├── bluetooth.py      # Bluetooth device management
│   ├── audio.py          # Audio routing and volume control
│   └── youtube_music.py  # YouTube Music integration
└── whisper.cpp/          # Whisper.cpp installation (gitignored)
```

## Customization

### Change Wake Words

Edit `wakeword.py` line 161:
```python
keywords=['americano', 'computer']  # Change to your preferred wake words
```

Or use a custom wake word:
1. Train a custom wake word at [Picovoice Console](https://console.picovoice.ai/)
2. Download the `.ppn` file
3. Update the keywords parameter:
```python
keywords=['/path/to/custom_wakeword.ppn']
```

### Adjust Voice Settings

Edit the Google Cloud TTS settings in `run_assistant.py` or `wakeword.py`:
```python
tts_voice = texttospeech.VoiceSelectionParams(
    language_code="en-US",
    name="en-US-Neural2-J",  # Change voice (A-J available)
    ssml_gender=texttospeech.SsmlVoiceGender.MALE
)
tts_audio_config = texttospeech.AudioConfig(
    audio_encoding=texttospeech.AudioEncoding.MP3,
    speaking_rate=1.1,  # Adjust speed (0.25 to 4.0)
    pitch=0.0           # Adjust pitch (-20.0 to 20.0)
)
```

Available voices: See [Google Cloud TTS Voices](https://cloud.google.com/text-to-speech/docs/voices)

### Add New Tools

1. Define your function in a new file or existing tool file
2. Add the function declaration to `tools/registry.py` in `GEMINI_TOOLS`
3. Add the function to `TOOL_FUNCTIONS` mapping
4. The dispatcher will automatically handle calls

## Troubleshooting

### "No speech detected"
- Check microphone permissions
- Speak louder or closer to the microphone
- Increase `RECORD_SECONDS` in the script

### "Gemini API failed"
- Verify your API key in `.env`
- Check your internet connection
- Ensure you haven't exceeded API rate limits

### Wake word not detecting
- Verify Porcupine access key
- Try speaking the wake word more clearly
- Check microphone input levels

### Whisper transcription errors
- Ensure Whisper.cpp is properly built
- Verify paths in `.env` are correct
- Try a different model (base or small for better accuracy)

## Contributing

Feel free to submit issues and pull requests!

## License

MIT License - feel free to use this project for personal or commercial purposes.

## Acknowledgments

- [Google Gemini](https://ai.google.dev/) for the LLM
- [Google Cloud TTS](https://cloud.google.com/text-to-speech) for high-quality text-to-speech
- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) for speech recognition
- [Porcupine](https://picovoice.ai/platform/porcupine/) for wake word detection
- [pyControl4](https://github.com/lawtancool/pyControl4) for Control4 integration
