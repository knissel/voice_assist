param (
    [string]$EDGE_IP = "192.168.20.48",
    [string]$EDGE_USER = "kenny"
)

Write-Host "[SETUP] Setting up Edge Node ($EDGE_IP)..." -ForegroundColor Cyan

$sshTarget = "$EDGE_USER@$EDGE_IP"

# 1. Create directories
Write-Host "[1/5] Creating directories..." -ForegroundColor Yellow
ssh $sshTarget 'powershell -Command New-Item -ItemType Directory -Force -Path C:\Users\Kenny\voice_assist'
ssh $sshTarget 'powershell -Command New-Item -ItemType Directory -Force -Path C:\Users\Kenny\voice_assist_edge'

# 2. Create virtual environment (if not exists)
Write-Host "[2/5] Setting up Python virtual environment..." -ForegroundColor Yellow
ssh $sshTarget 'powershell -Command if (-not (Test-Path C:\Users\Kenny\voice_assist\venv)) { python -m venv C:\Users\Kenny\voice_assist\venv }'

# 3. Install dependencies
Write-Host "[3/5] Installing Python dependencies (this may take a few minutes)..." -ForegroundColor Yellow
ssh $sshTarget 'C:\Users\Kenny\voice_assist\venv\Scripts\pip install --upgrade pip'
ssh $sshTarget 'C:\Users\Kenny\voice_assist\venv\Scripts\pip install openwakeword pyaudio requests python-dotenv sounddevice flask piper-tts numpy'
ssh $sshTarget 'C:\Users\Kenny\voice_assist\venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu'

# 4. Install Visual C++ Redistributables reminder
Write-Host "[4/5] Reminder: Ensure Visual C++ Redistributables are installed on Edge" -ForegroundColor Magenta
Write-Host "       Download from: https://aka.ms/vs/17/release/vc_redist.x64.exe" -ForegroundColor Magenta

# 5. Open firewall for dashboard
Write-Host "[5/5] Note: You may need to open firewall port 5000 for the dashboard" -ForegroundColor Magenta
Write-Host "       Run this on the Edge (as Admin): New-NetFirewallRule -Name 'VoiceAssistDashboard' -DisplayName 'Voice Assistant Dashboard' -Enabled True -Direction Inbound -Protocol TCP -Action Allow -LocalPort 5000" -ForegroundColor Gray

Write-Host ""
Write-Host "[DONE] Edge setup complete!" -ForegroundColor Green
Write-Host "Now run ./scripts/deploy_native.ps1 to deploy and start the Edge Assistant" -ForegroundColor Cyan
