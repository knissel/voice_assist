param (
    [string]$EDGE_IP = "192.168.20.48",
    [string]$EDGE_USER = "kenny"
)

Write-Host "[SYNC] Syncing code to Edge ($EDGE_IP)..." -ForegroundColor Cyan

# Create the destination directory on the remote machine
$sshTarget = "$EDGE_USER@$EDGE_IP"
ssh $sshTarget 'powershell -Command New-Item -ItemType Directory -Force -Path C:\Users\Kenny\voice_assist_edge'

# Copy the src files and the .env
scp -r services/edge/src/* "${sshTarget}:C:/Users/Kenny/voice_assist_edge/"
scp services/edge/.env "${sshTarget}:C:/Users/Kenny/voice_assist_edge/"

Write-Host "[START] Starting Edge Assistant natively..." -ForegroundColor Green

# Run the python script using the venv with unbuffered output (-u)
ssh $sshTarget 'C:\Users\Kenny\voice_assist\venv\Scripts\python.exe -u C:\Users\Kenny\voice_assist_edge\main.py'
