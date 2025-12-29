param(
    [switch]$Pull,
    [switch]$InstallDeps,
    [switch]$WaitForHealth,
    [int]$HealthTimeoutSeconds = 30,
    [int]$Port = 5001,
    [string]$ServerHost = "0.0.0.0",
    [string]$Device = "cuda",
    [string]$Speaker = "",
    [string]$Python = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $repoRoot

function Resolve-PythonExe {
    param([string]$Override)
    if ($Override) {
        return $Override
    }
    $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
    if (Test-Path $venvPython) {
        return $venvPython
    }
    $pythonCmd = Get-Command python -ErrorAction SilentlyContinue
    if ($pythonCmd) {
        return $pythonCmd.Path
    }
    $pyCmd = Get-Command py -ErrorAction SilentlyContinue
    if ($pyCmd) {
        return $pyCmd.Path
    }
    throw "Python not found. Provide -Python or install python."
}

function Stop-XttsServer {
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and $_.CommandLine -match "xtts_server\.py"
    }
    foreach ($proc in $procs) {
        Write-Host ("Stopping XTTS server PID {0}" -f $proc.ProcessId)
        Stop-Process -Id $proc.ProcessId -Force
    }
}

function Start-XttsServer {
    param(
        [string]$PythonExe,
        [string]$ServerHost,
        [int]$Port,
        [string]$Device,
        [string]$SpeakerPath
    )

    $logPath = Join-Path $repoRoot "xtts_server.log"
    $errPath = Join-Path $repoRoot "xtts_server.err"

    $argsList = @(
        "xtts_server.py",
        "--host", $ServerHost,
        "--port", $Port,
        "--device", $Device
    )
    if ($SpeakerPath) {
        $argsList += @("--speaker", $SpeakerPath)
    }

    Write-Host "Starting XTTS server..."
    Start-Process `
        -FilePath $PythonExe `
        -ArgumentList $argsList `
        -WorkingDirectory $repoRoot `
        -RedirectStandardOutput $logPath `
        -RedirectStandardError $errPath `
        -WindowStyle Hidden `
        | Out-Null
}

function Wait-ForHealth {
    param([int]$TimeoutSeconds, [int]$Port)

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    $url = "http://localhost:$Port/health"
    do {
        try {
            $resp = Invoke-RestMethod -Uri $url -Method Get -TimeoutSec 3
            if ($resp.status -eq "healthy") {
                Write-Host "XTTS healthy: model=$($resp.model) speaker_loaded=$($resp.speaker_loaded)"
                return $true
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    } while ((Get-Date) -lt $deadline)

    Write-Warning "XTTS health check timed out after $TimeoutSeconds seconds."
    return $false
}

if ($Pull) {
    if (Get-Command git -ErrorAction SilentlyContinue) {
        Write-Host "Pulling latest code..."
        git -C $repoRoot pull
    } else {
        Write-Warning "git not found. Skipping pull."
    }
}

$pythonExe = Resolve-PythonExe -Override $Python

if ($InstallDeps) {
    $reqFile = Join-Path $repoRoot "requirements_gpu_server.txt"
    if (-not (Test-Path $reqFile)) {
        $reqFile = Join-Path $repoRoot "requirements.txt"
    }
    Write-Host "Installing dependencies from $reqFile..."
    & $pythonExe -m pip install -r $reqFile
}

Stop-XttsServer
Start-XttsServer -PythonExe $pythonExe -ServerHost $ServerHost -Port $Port -Device $Device -SpeakerPath $Speaker

if ($WaitForHealth) {
    Wait-ForHealth -TimeoutSeconds $HealthTimeoutSeconds -Port $Port | Out-Null
}
