param (
    [Parameter(Mandatory = $false)]
    [string]$Target = "edge",

    [Parameter(Mandatory = $false)]
    [switch]$Logs
)

# Configuration - EDIT THESE FOR YOUR NETWORK
$EDGE_IP = "192.168.20.48"
$EDGE_USER = "kenny"
$CONTEXT_NAME = "voice-edge"

# check if context exists
$existingContext = docker context ls --format '{{.Name}}' | Select-String -Pattern "^$CONTEXT_NAME$"

if (-not $existingContext) {
    Write-Host "Creating new Docker Context: $CONTEXT_NAME..." -ForegroundColor Cyan
    docker context create $CONTEXT_NAME --description "Remote Mini PC" --docker "host=ssh://$EDGE_USER@$EDGE_IP"
}

if ($Logs) {
    Write-Host "Showing logs from $CONTEXT_NAME..." -ForegroundColor Yellow
    docker --context $CONTEXT_NAME compose logs -f
    exit
}

Write-Host "🚀 Deploying to $CONTEXT_NAME ($EDGE_IP)..." -ForegroundColor Green

# Use docker compose with the remote context
# We use --build to ensure the latest local code is sent and built on the remote host
docker --context $CONTEXT_NAME compose up -d --build edge

Write-Host "✅ Deployment successful!" -ForegroundColor Green
