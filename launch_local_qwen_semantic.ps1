[CmdletBinding()]
param(
    [string]$ModelDir = "J:\Models\Qwen\Qwen3-VL-8B-Instruct",
    [string]$ModelHost = "127.0.0.1",
    [int]$ModelPort = 8008,
    [string]$AppHost = "127.0.0.1",
    [int]$AppPort = 5000,
    [int]$ServiceStartupTimeoutSeconds = 300,
    [int]$WebStartupTimeoutSeconds = 60
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$localQwenUrl = "http://${ModelHost}:$ModelPort"
$webUrl = "http://${AppHost}:$AppPort"

function Write-Info([string]$Message) {
    Write-Host "[INFO] $Message" -ForegroundColor Cyan
}

function Write-Success([string]$Message) {
    Write-Host "[OK] $Message" -ForegroundColor Green
}

function Write-WarnMessage([string]$Message) {
    Write-Host "[WARN] $Message" -ForegroundColor Yellow
}

function Escape-SingleQuoted([string]$Value) {
    return $Value.Replace("'", "''")
}

function Convert-ToEncodedCommand([string]$Value) {
    $bytes = [System.Text.Encoding]::Unicode.GetBytes($Value)
    return [Convert]::ToBase64String($bytes)
}

function Test-PythonAvailable {
    return [bool](Get-Command python -ErrorAction SilentlyContinue)
}

function Test-PortListening([int]$Port) {
    $listeners = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
    return [bool]($listeners | Where-Object { $_.Port -eq $Port } | Select-Object -First 1)
}

function Test-LocalQwenHealthy {
    try {
        $response = Invoke-RestMethod -Uri "$localQwenUrl/health" -TimeoutSec 5
        return ($response.loaded -eq $true)
    } catch {
        return $false
    }
}

function Test-WebUiHealthy {
    try {
        $response = Invoke-WebRequest -Uri $webUrl -UseBasicParsing -TimeoutSec 5
        if ($response.StatusCode -ne 200) {
            return $false
        }
        $content = [string]$response.Content
        return $content.Contains("panel-classify") -and $content.Contains("panel-bundle")
    } catch {
        return $false
    }
}

function Wait-UntilHealthy(
    [string]$Name,
    [scriptblock]$Probe,
    [int]$TimeoutSeconds,
    $StartedProcess = $null
) {
    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        if (& $Probe) {
            return
        }
        if ($StartedProcess -and $StartedProcess.HasExited) {
            throw "$Name failed to start: the new window exited early."
        }
        Start-Sleep -Seconds 2
    }
    throw "$Name startup timed out after $TimeoutSeconds seconds."
}

function Start-LocalQwenWindow {
    $runScript = Join-Path $repoRoot "run_local_qwen3_vl_server.ps1"
    if (-not (Test-Path $runScript)) {
        throw "Missing model service script: $runScript"
    }

    $escapedRepoRoot = Escape-SingleQuoted $repoRoot
    $escapedRunScript = Escape-SingleQuoted $runScript
    $escapedModelDir = Escape-SingleQuoted $ModelDir
    $command = @"
`$Host.UI.RawUI.WindowTitle = 'Local Qwen3-VL Service'
Set-Location -LiteralPath '$escapedRepoRoot'
& '$escapedRunScript' -ModelDir '$escapedModelDir' -ListenHost '$ModelHost' -Port $ModelPort
"@
    $encodedCommand = Convert-ToEncodedCommand $command

    return Start-Process powershell.exe `
        -WorkingDirectory $repoRoot `
        -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encodedCommand `
        -PassThru
}

function Start-WebWindow {
    $escapedRepoRoot = Escape-SingleQuoted $repoRoot
    $escapedLocalQwenUrl = Escape-SingleQuoted $localQwenUrl
    $command = @"
`$Host.UI.RawUI.WindowTitle = 'Furniture AI Toolbox (Local Qwen Semantic)'
Set-Location -LiteralPath '$escapedRepoRoot'
`$env:SEMANTIC_BACKEND = 'local_qwen3_vl'
`$env:LOCAL_QWEN3_VL_URL = '$escapedLocalQwenUrl'
python app.py
"@
    $encodedCommand = Convert-ToEncodedCommand $command

    return Start-Process powershell.exe `
        -WorkingDirectory $repoRoot `
        -ArgumentList "-NoExit", "-ExecutionPolicy", "Bypass", "-EncodedCommand", $encodedCommand `
        -PassThru
}

try {
    Set-Location -LiteralPath $repoRoot

    if (-not (Test-PythonAvailable)) {
        throw "python was not found in PATH."
    }
    if (-not (Test-Path $ModelDir)) {
        throw "Model directory does not exist: $ModelDir"
    }

    Write-Info "Repo root: $repoRoot"
    Write-Info "Model dir: $ModelDir"

    if (Test-LocalQwenHealthy) {
        Write-WarnMessage "Reusing healthy local Qwen service: $localQwenUrl"
    } elseif (Test-PortListening $ModelPort) {
        throw "Port $ModelPort is occupied, but the existing service is not a healthy local Qwen service."
    } else {
        Write-Info "Starting local Qwen3-VL service window..."
        $modelProcess = Start-LocalQwenWindow
        Wait-UntilHealthy -Name "Local Qwen3-VL service" -Probe { Test-LocalQwenHealthy } -TimeoutSeconds $ServiceStartupTimeoutSeconds -StartedProcess $modelProcess
        Write-Success "Local Qwen3-VL service is ready: $localQwenUrl"
    }

    if (Test-WebUiHealthy) {
        Write-WarnMessage "Reusing existing toolbox page: $webUrl"
    } elseif (Test-PortListening $AppPort) {
        throw "Port $AppPort is occupied, but the existing service is not the toolbox page."
    } else {
        Write-Info "Starting toolbox web window..."
        $webProcess = Start-WebWindow
        Wait-UntilHealthy -Name "Toolbox web app" -Probe { Test-WebUiHealthy } -TimeoutSeconds $WebStartupTimeoutSeconds -StartedProcess $webProcess
        Write-Success "Toolbox web app is ready: $webUrl"
    }

    Write-Success "Launch completed. Semantic tagging will use the local Qwen backend."
    exit 0
} catch {
    Write-Host "[ERROR] $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}
