# Voice Assistant with Gemini Flash

A voice-controlled assistant powered by Google Gemini Flash that can control smart home devices, manage Bluetooth connections, and answer questions using natural language.

## Features

- 🎤 **Wake Word Detection**: Uses Porcupine for "Jarvis" or custom wake words
- 🗣️ **Speech-to-Text**: Local transcription with Whisper.cpp
- 🤖 **AI Processing**: Google Gemini 2.0 Flash for natural language understanding
- 🔧 **Function Calling**: Control smart home lights, Bluetooth devices, and audio routing
- 🔊 **Text-to-Speech**: Natural voice responses using pyttsx3
- ⌨️ **Push-to-Talk Mode**: Alternative mode using spacebar to activate

## Prerequisites

- Python 3.9+
- macOS (for audio features)
- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) installed locally
- Google Gemini API key
- Porcupine access key (for wake word detection)

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

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Whisper.cpp**
   ```bash
   # Clone and build whisper.cpp
   git clone https://github.com/ggerganov/whisper.cpp.git
   cd whisper.cpp
   make
   
   # Download a model (tiny is recommended for speed)
   bash ./models/download-ggml-model.sh tiny
   cd ..
   ```

5. **Configure environment variables**
   
   Copy the example environment file:
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```bash
   # Gemini API Configuration
   GEMINI_API_KEY=your_gemini_api_key_here
   
   # Porcupine Wake Word Configuration
   PORCUPINE_ACCESS_KEY=your_porcupine_access_key_here
   
   # Whisper Configuration (update paths if needed)
   WHISPER_PATH=/path/to/whisper.cpp/build/bin/whisper-cli
   MODEL_PATH=/path/to/whisper.cpp/models/ggml-tiny.bin
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

## Usage

### Wake Word Mode (Continuous Listening)

Run the wake word listener that responds to "Jarvis" or "Computer":

```bash
python wakeword.py
```

Say the wake word, then speak your command. The assistant will:
1. Detect the wake word
2. Record your voice command
3. Transcribe it using Whisper
4. Process with Gemini Flash
5. Execute actions or respond verbally

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

### Bluetooth Control
- "Connect to [device name]"
- "Disconnect Bluetooth"
- "Route audio to Bluetooth"

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
keywords=['jarvis', 'computer']  # Change to your preferred wake words
```

### Adjust Voice Settings

Edit the TTS settings in any script:
```python
tts_engine.setProperty('rate', 165)    # Speech rate
tts_engine.setProperty('volume', 0.95) # Volume level
```

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
- [Whisper.cpp](https://github.com/ggerganov/whisper.cpp) for speech recognition
- [Porcupine](https://picovoice.ai/platform/porcupine/) for wake word detection
- [pyttsx3](https://github.com/nateshmbhat/pyttsx3) for text-to-speech
