@echo off
TITLE Voice Assistant 5090 Services
ECHO Starting Voice Assistant Services...

:: 1. Start Whisper Server
start "Whisper Server (Port 5000)" cmd /k "call .venv\Scripts\activate && python whisper_server.py --port 5000"

:: 2. Start Qwen3-TTS Server
start "Qwen3-TTS Server (Port 5001)" cmd /k "call .venv\Scripts\activate && python qwen_tts_server.py --port 5001"

:: 3. Start Ops Agent
start "Ops Agent (Port 8010)" cmd /k "call .venv\Scripts\activate && cd services\ops_agent && python main.py"

:: 4. Start Ops Dashboard
start "Ops Dashboard" cmd /k "cd services\ops_dashboard && npm run dev"

ECHO All services started in background windows.
TIMEOUT /T 5
EXIT
