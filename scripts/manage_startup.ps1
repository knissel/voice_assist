# VERSION: 3.0.0 - Using Startup Folder (Batch File)
# This method is more reliable over SSH than Task Scheduler as it avoids CIM/RPC permission issues.
param (
    [string]$Action = "register",
    [string]$EDGE_IP = "192.168.20.48",
    [string]$EDGE_USER = "kenny"
)

$StartupFolder = "C:\Users\$EDGE_USER\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup"
$BatchFileName = "voice_assist_startup.bat"
$RemotePath = "C:\Users\$EDGE_USER\voice_assist_edge"
$PythonPath = "C:\Users\$EDGE_USER\voice_assist\venv\Scripts\python.exe"

$sshTarget = "$EDGE_USER@$EDGE_IP"

if ($Action -eq "register") {
    Write-Host "[STARTUP] Creating Startup Batch File..." -ForegroundColor Cyan
    
    # Create a simple batch file content
    # We use 'start /min' to run it minimized, or just run it directly.
    # We cd to the directory first to ensure relative paths work if needed.
    $batchContent = "@echo off`r`n" +
    "cd /d $RemotePath`r`n" +
    "start `"VoiceAssistant`" /min `"$PythonPath`" main.py"
    
    # We use a temp file locally then scp it, or echo it over ssh. 
    # Echoing over SSH is cleaner for a single file.
    
    # PowerShell escaping for the remote echo command is tricky.
    # Let's create a local temp file and SCP it. It's safer.
    
    $tempFile = [System.IO.Path]::GetTempFileName() + ".bat"
    Set-Content -Path $tempFile -Value $batchContent
    
    # Ensure destination directory exists (it should, but good to be safe)
    ssh $sshTarget "if not exist `"$StartupFolder`" mkdir `"$StartupFolder`""
    
    # SCP the file
    $remoteDest = "$StartupFolder\$BatchFileName"
    # Note: scp requires specific path format for windows usually, but let's try standard
    # We might need to use the local file path for scp
    
    Write-Host "   - Copying startup script to: $remoteDest" -ForegroundColor Gray
    scp $tempFile "${sshTarget}:`"$remoteDest`""
    
    Remove-Item $tempFile
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "[SUCCESS] Startup script created!" -ForegroundColor Green
        Write-Host "NOTE: You must configure Windows Auto-Login (netplwiz) for this to work on boot." -ForegroundColor Yellow
    }
    else {
        Write-Host "[ERROR] Failed to copy startup script." -ForegroundColor Red
    }
}
elseif ($Action -eq "unregister") {
    Write-Host "[STARTUP] Removing Startup Script..." -ForegroundColor Yellow
    $remoteFile = "$StartupFolder\$BatchFileName"
    ssh $sshTarget "del `"$remoteFile`""
}
elseif ($Action -eq "status") {
    Write-Host "[STARTUP] Checking for Startup Script..." -ForegroundColor Cyan
    $remoteFile = "$StartupFolder\$BatchFileName"
    ssh $sshTarget "if exist `"$remoteFile`" (echo [FOUND] Startup script exists.) else (echo [MISSING] Startup script not found.)"
}
else {
    Write-Host "[ERROR] Invalid action: $Action" -ForegroundColor Red
    Write-Host "Usage: ./scripts/manage_startup.ps1 -Action [register|unregister|status]"
}
