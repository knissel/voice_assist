# Deploy Voice Assistant to Raspberry Pi
# Usage: .\deploy.ps1 [-UI] [-Wakeword] [-All] [-Restart] [-Logs]
#
# Prerequisites:
#   1. SSH key-based authentication set up with your Pi
#   2. rsync available (via WSL, Git Bash, or native Windows)
#
# Examples:
#   .\deploy.ps1 -All              # Deploy everything and restart
#   .\deploy.ps1 -UI               # Deploy only UI changes
#   .\deploy.ps1 -Wakeword         # Deploy wakeword and related modules
#   .\deploy.ps1 -Restart          # Just restart the service
#   .\deploy.ps1 -All -Logs        # Deploy, restart, and tail logs

param(
    [switch]$UI,
    [switch]$Wakeword,
    [switch]$All,
    [switch]$Restart,
    [switch]$Logs,
    [switch]$DryRun
)

# === CONFIGURATION ===
# Edit these to match your Pi setup
$PI_HOST = "pi@raspberrypi.local"  # or use IP like "pi@192.168.1.100"
$PI_PATH = "/home/pi/voice_assist"
$SERVICE_NAME = "voice-assistant"

# Files/folders to deploy for each component
$UI_FILES = @(
    "ui/"
)

$WAKEWORD_FILES = @(
    "wakeword.py",
    "core/",
    "tools/",
    "adapters/",
    "schemas/"
)

$CONFIG_FILES = @(
    "requirements_pi.txt",
    "voice-assistant.service"
)

# === HELPER FUNCTIONS ===

function Write-Status($message, $color = "Cyan") {
    Write-Host "[$((Get-Date).ToString('HH:mm:ss'))] " -NoNewline -ForegroundColor DarkGray
    Write-Host $message -ForegroundColor $color
}

function Test-SshConnection {
    Write-Status "Testing SSH connection to $PI_HOST..."
    $result = ssh -o ConnectTimeout=5 -o BatchMode=yes $PI_HOST "echo ok" 2>$null
    if ($result -eq "ok") {
        Write-Status "SSH connection OK" "Green"
        return $true
    } else {
        Write-Status "SSH connection failed. Make sure:" "Red"
        Write-Host "  1. Pi is powered on and connected to network"
        Write-Host "  2. SSH key authentication is configured"
        Write-Host "  3. Hostname/IP in script is correct: $PI_HOST"
        return $false
    }
}

function Invoke-Rsync($source, $destination, $dryRun = $false) {
    $rsyncArgs = @("-avz", "--progress", "--delete")
    if ($dryRun) {
        $rsyncArgs += "--dry-run"
    }
    
    # Convert Windows path to rsync-compatible path
    $sourcePath = $source -replace '\\', '/'
    
    $cmd = "rsync $($rsyncArgs -join ' ') `"$sourcePath`" `"${PI_HOST}:${destination}`""
    Write-Status "Running: $cmd" "DarkGray"
    
    Invoke-Expression $cmd
}

function Invoke-Scp($source, $destination, $isDir = $false) {
    $scpArgs = if ($isDir) { "-r" } else { "" }
    $cmd = "scp $scpArgs `"$source`" `"${PI_HOST}:${destination}`""
    Write-Status "Running: $cmd" "DarkGray"
    Invoke-Expression $cmd
}

function Deploy-Files($files, $component) {
    Write-Status "Deploying $component files..." "Yellow"
    
    foreach ($file in $files) {
        $localPath = Join-Path $PSScriptRoot $file
        
        if (Test-Path $localPath) {
            $isDir = (Get-Item $localPath).PSIsContainer
            $destPath = "$PI_PATH/$file"
            
            if ($isDir) {
                # For directories, sync the contents
                $destDir = Split-Path $destPath -Parent
                Invoke-Rsync "$localPath/" "$PI_PATH/$file" $DryRun
            } else {
                # For files, copy directly
                if (-not $DryRun) {
                    Invoke-Scp $localPath $destPath
                } else {
                    Write-Status "[DRY-RUN] Would copy $file" "Magenta"
                }
            }
            Write-Status "  ✓ $file" "Green"
        } else {
            Write-Status "  ✗ $file (not found)" "Red"
        }
    }
}

function Restart-PiService {
    Write-Status "Restarting $SERVICE_NAME service on Pi..." "Yellow"
    
    if ($DryRun) {
        Write-Status "[DRY-RUN] Would restart service" "Magenta"
        return
    }
    
    ssh $PI_HOST "sudo systemctl restart $SERVICE_NAME"
    Start-Sleep -Seconds 2
    
    $status = ssh $PI_HOST "sudo systemctl is-active $SERVICE_NAME"
    if ($status -eq "active") {
        Write-Status "Service restarted successfully ✓" "Green"
    } else {
        Write-Status "Service may have failed to start. Check logs with: .\deploy.ps1 -Logs" "Red"
    }
}

function Show-ServiceLogs {
    Write-Status "Showing live logs (Ctrl+C to exit)..." "Yellow"
    ssh $PI_HOST "sudo journalctl -u $SERVICE_NAME -f --no-hostname -n 50"
}

# === MAIN EXECUTION ===

Write-Host ""
Write-Host "╔══════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║     Voice Assistant Pi Deployment        ║" -ForegroundColor Cyan
Write-Host "╚══════════════════════════════════════════╝" -ForegroundColor Cyan
Write-Host ""

# Default to -All if no specific component selected
if (-not $UI -and -not $Wakeword -and -not $All -and -not $Restart -and -not $Logs) {
    Write-Status "No options specified. Use -Help for usage or -All to deploy everything." "Yellow"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -UI        Deploy UI files only"
    Write-Host "  -Wakeword  Deploy wakeword and related modules"
    Write-Host "  -All       Deploy everything"
    Write-Host "  -Restart   Restart the service (auto with -Wakeword/-All)"
    Write-Host "  -Logs      Show live service logs"
    Write-Host "  -DryRun    Show what would be deployed without doing it"
    Write-Host ""
    exit 0
}

# Test SSH connection first
if (-not (Test-SshConnection)) {
    exit 1
}

# Deploy components
if ($All) {
    Deploy-Files $UI_FILES "UI"
    Deploy-Files $WAKEWORD_FILES "Wakeword"
    Deploy-Files $CONFIG_FILES "Config"
    $Restart = $true
}
elseif ($UI) {
    Deploy-Files $UI_FILES "UI"
}
elseif ($Wakeword) {
    Deploy-Files $WAKEWORD_FILES "Wakeword"
    $Restart = $true
}

# Restart service if needed
if ($Restart -and -not $Logs) {
    Restart-PiService
}

# Show logs if requested
if ($Logs) {
    if ($Restart) {
        Restart-PiService
    }
    Show-ServiceLogs
}

Write-Host ""
Write-Status "Deployment complete!" "Green"
