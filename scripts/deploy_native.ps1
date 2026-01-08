param (
    [string]$EDGE_IP = "192.168.20.48",
    [string]$EDGE_USER = "kenny",
    [switch]$RegisterStartup
)

Write-Host "[SYNC] Syncing code to Edge ($EDGE_IP)..." -ForegroundColor Cyan

# Kill any existing Python processes to prevent stale instances
$sshTarget = "$EDGE_USER@$EDGE_IP"
Write-Host "[CLEAN] Stopping any running Python processes..." -ForegroundColor Yellow
ssh $sshTarget 'taskkill /F /IM python.exe 2>$null' 2>$null

# Create the destination directory on the remote machine
ssh $sshTarget 'powershell -Command New-Item -ItemType Directory -Force -Path C:\Users\Kenny\voice_assist_edge'

# Copy the src files and the .env
scp -r services/edge/src/* "${sshTarget}:C:/Users/Kenny/voice_assist_edge/"
scp services/edge/.env "${sshTarget}:C:/Users/Kenny/voice_assist_edge/"

if ($RegisterStartup) {
    ./scripts/manage_startup.ps1 -Action "register" -EDGE_IP $EDGE_IP -EDGE_USER $EDGE_USER
}

Write-Host "[START] Starting Edge Assistant natively..." -ForegroundColor Green

# Run the python script using the venv with unbuffered output (-u)
ssh $sshTarget 'C:\Users\Kenny\voice_assist\venv\Scripts\python.exe -u C:\Users\Kenny\voice_assist_edge\main.py'
