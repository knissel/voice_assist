param(
    [switch]$Pull,
    [switch]$InstallDeps,
    [switch]$WaitForHealth,
    [int]$HealthTimeoutSeconds = 30,
    [int]$Port = 5000,
    [string]$ServerHost = "0.0.0.0",
    [string]$Device = "cuda",
    [string]$Model = "large-v3",
    [string]$ComputeType = "float16",
    [string]$Language = "",
    [int]$BeamSize = -1,
    [int]$BestOf = -1,
    [double]$Temperature = [double]::NaN,
    [switch]$VadFilter,
    [switch]$WordTimestamps,
    [switch]$WithoutTimestamps,
    [switch]$ConditionOnPreviousText,
    [switch]$InMemory,
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

function Stop-WhisperServer {
    $procs = Get-CimInstance Win32_Process | Where-Object {
        $_.CommandLine -and $_.CommandLine -match "whisper_server\.py"
    }
    foreach ($proc in $procs) {
        Write-Host ("Stopping Whisper server PID {0}" -f $proc.ProcessId)
        Stop-Process -Id $proc.ProcessId -Force
    }
}

function Start-WhisperServer {
    param(
        [string]$PythonExe,
        [string]$ServerHost,
        [int]$Port,
        [string]$Device,
        [string]$Model,
        [string]$ComputeType,
        [string]$Language,
        [int]$BeamSize,
        [int]$BestOf,
        [double]$Temperature,
        [switch]$VadFilter,
        [switch]$WordTimestamps,
        [switch]$WithoutTimestamps,
        [switch]$ConditionOnPreviousText,
        [switch]$InMemory
    )

    $logPath = Join-Path $repoRoot "whisper_server.log"
    $errPath = Join-Path $repoRoot "whisper_server.err"

    $argsList = @(
        "whisper_server.py",
        "--host", $ServerHost,
        "--port", $Port,
        "--device", $Device,
        "--model", $Model,
        "--compute-type", $ComputeType
    )
    if ($Language) {
        $argsList += @("--language", $Language)
    }
    if ($BeamSize -ge 0) {
        $argsList += @("--beam-size", $BeamSize)
    }
    if ($BestOf -ge 0) {
        $argsList += @("--best-of", $BestOf)
    }
    if (-not [double]::IsNaN($Temperature)) {
        $argsList += @("--temperature", $Temperature)
    }
    if ($VadFilter) { $argsList += "--vad-filter" }
    if ($WordTimestamps) { $argsList += "--word-timestamps" }
    if ($WithoutTimestamps) { $argsList += "--without-timestamps" }
    if ($ConditionOnPreviousText) { $argsList += "--condition-on-previous-text" }
    if ($InMemory) { $argsList += "--in-memory" }

    Write-Host "Starting Whisper server..."
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
                Write-Host "Whisper healthy: model=$($resp.model)"
                return $true
            }
        } catch {
            Start-Sleep -Seconds 1
        }
    } while ((Get-Date) -lt $deadline)

    Write-Warning "Whisper health check timed out after $TimeoutSeconds seconds."
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

Stop-WhisperServer
Start-WhisperServer `
    -PythonExe $pythonExe `
    -ServerHost $ServerHost `
    -Port $Port `
    -Device $Device `
    -Model $Model `
    -ComputeType $ComputeType `
    -Language $Language `
    -BeamSize $BeamSize `
    -BestOf $BestOf `
    -Temperature $Temperature `
    -VadFilter:$VadFilter `
    -WordTimestamps:$WordTimestamps `
    -WithoutTimestamps:$WithoutTimestamps `
    -ConditionOnPreviousText:$ConditionOnPreviousText `
    -InMemory:$InMemory

if ($WaitForHealth) {
    Wait-ForHealth -TimeoutSeconds $HealthTimeoutSeconds -Port $Port | Out-Null
}
