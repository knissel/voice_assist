param (
    [string]$EDGE_IP = "192.168.20.48",
    [string]$EDGE_USER = "kenny"
)

Write-Host "📦 Syncing code to Edge ($EDGE_IP)..." -ForegroundColor Cyan

# Use SCP to copy the 'src' and requirements to the Mini PC folder
# We'll put it in a folder called 'voice_assist_edge' in your home directory
ssh $EDGE_USER@$EDGE_IP "powershell -Command 'New-Item -ItemType Directory -Force -Path `\"`$env:USERPROFILE\voice_assist_edge`\" '"

# Copy the src files and the .env
scp -r services/edge/src/* "${EDGE_USER}@${EDGE_IP}:~/voice_assist_edge/"
scp services/edge/.env "${EDGE_USER}@${EDGE_IP}:~/voice_assist_edge/"

Write-Host "🚀 Starting Edge Assistant natively..." -ForegroundColor Green

# Start the assistant remotely
# We use --windowstyle normal so you can see the terminal if you're looking at the Mini PC monitor
ssh $EDGE_USER@$EDGE_IP "cd ~/voice_assist_edge; python main.py"
