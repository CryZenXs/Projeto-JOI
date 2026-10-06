<#
.SYNOPSIS
    Projeto JOI - Server Start Script
.DESCRIPTION
    Verifies that the .venv is healthy and starts the uvicorn server on port 8001.
    Pure ASCII version - works on all PowerShell versions.
.NOTES
    File name: run.ps1
    Run as:    Regular user (do NOT run as Administrator)
#>

[CmdletBinding()]
param(
    [int]$Port = 8001,
    [string]$Host = "127.0.0.1",
    [switch]$Reload,
    [switch]$Help
)

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = $ScriptPath
Set-Location $ProjectRoot

$VenvPath = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$VenvUvicorn = Join-Path $VenvPath "Scripts\uvicorn.exe"

# ===========================================================================
# HELP
# ===========================================================================

if ($Help) {
    Write-Host ""
    Write-Host "Projeto JOI - Server Start Script" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Usage: .\run.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -Port <int>     Server port (default: 8001)"
    Write-Host "  -Host <string>  Bind address (default: 127.0.0.1)"
    Write-Host "  -Reload         Enable auto-reload on file changes"
    Write-Host "  -Help           Show this help message"
    Write-Host ""
    Write-Host "Examples:"
    Write-Host "  .\run.ps1                       # Start on port 8001"
    Write-Host "  .\run.ps1 -Port 8080            # Custom port"
    Write-Host "  .\run.ps1 -Reload               # With auto-reload (dev mode)"
    Write-Host "  .\run.ps1 -Host 0.0.0.0         # Accessible on network"
    Write-Host ""
    exit 0
}

# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

function Write-Step {
    param([string]$Message)
    Write-Host "  > $Message" -ForegroundColor Cyan
}

function Write-OK {
    param([string]$Message = "")
    Write-Host "    [OK] $Message" -ForegroundColor Green
}

function Write-Warn {
    param([string]$Message)
    Write-Host "    [!] $Message" -ForegroundColor Yellow
}

function Write-Err {
    param([string]$Message)
    Write-Host "    [X] $Message" -ForegroundColor Red
}

function Write-Info {
    param([string]$Message)
    Write-Host "    [i] $Message" -ForegroundColor DarkGray
}

# ===========================================================================
# CHECK VENV
# ===========================================================================

Write-Host ""
Write-Host "======================================================================" -ForegroundColor DarkYellow
Write-Host "  PROJETO JOI - Server Start" -ForegroundColor Yellow
Write-Host "======================================================================" -ForegroundColor DarkYellow
Write-Host ""

Write-Step "Checking virtual environment..."

# Check if .venv directory exists
if (-not (Test-Path $VenvPath)) {
    Write-Err "Virtual environment not found at .venv"
    Write-Host ""
    Write-Host "  The .venv directory does not exist. You need to run setup first:" -ForegroundColor Yellow
    Write-Host "    .\setup.bat" -ForegroundColor White
    Write-Host ""
    exit 1
}

# Check if python.exe exists inside venv
if (-not (Test-Path $VenvPython)) {
    Write-Err "Python not found in venv: $VenvPython"
    Write-Host ""
    Write-Host "  The venv exists but python.exe is missing. It may be corrupted." -ForegroundColor Yellow
    Write-Host "  Re-run setup with: .\setup.bat -CleanVenv" -ForegroundColor White
    Write-Host ""
    exit 1
}

# Verify venv python works
try {
    $pyVersion = & $VenvPython --version 2>&1
    if ($LASTEXITCODE -ne 0) {
        throw "Python exited with code $LASTEXITCODE"
    }
    Write-OK "Python: $pyVersion"
} catch {
    Write-Err "Failed to run venv Python: $_"
    Write-Host "  The venv may be corrupted. Re-run: .\setup.bat -CleanVenv" -ForegroundColor Yellow
    exit 1
}

# Check if uvicorn is installed in venv
if (-not (Test-Path $VenvUvicorn)) {
    Write-Warn "uvicorn not found in venv. Installing..."
    try {
        & $VenvPython -m pip install "uvicorn[standard]" --quiet 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "pip install failed"
        }
        Write-OK "uvicorn installed"
    } catch {
        Write-Err "Failed to install uvicorn: $_"
        Write-Host "  Try: .\.venv\Scripts\pip.exe install uvicorn[standard]" -ForegroundColor Yellow
        exit 1
    }
}

# Verify project is installed (importable)
Write-Step "Verifying project installation..."
try {
    & $VenvPython -c "import app.main; print('OK')" 2>&1 | Out-Null
    if ($LASTEXITCODE -ne 0) {
        throw "import failed"
    }
    Write-OK "Project is importable"
} catch {
    Write-Warn "Project not installed in venv. Installing..."
    try {
        & $VenvPython -m pip install -e ".[dev]" --quiet 2>&1 | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "pip install failed"
        }
        Write-OK "Project installed"
    } catch {
        Write-Err "Failed to install project: $_"
        Write-Host "  Try: .\.venv\Scripts\pip.exe install -e .[dev]" -ForegroundColor Yellow
        exit 1
    }
}

# ===========================================================================
# CHECK .env
# ===========================================================================

Write-Step "Checking .env file..."
$EnvFile = Join-Path $ProjectRoot ".env"
if (Test-Path $EnvFile) {
    Write-OK ".env file found"
    # Quick check for GROQ_API_KEY
    $envContent = Get-Content $EnvFile -Raw
    if ($envContent -match "GROQ_API_KEY=\s*(gsk_\S+)") {
        Write-OK "Groq API key configured"
    } else {
        Write-Warn "GROQ_API_KEY not set (will use MockLLMClient)"
    }
} else {
    Write-Warn ".env file not found"
    Write-Info "Server will use default settings (MockLLMClient)"
    Write-Info "Run .\setup.bat to configure properly"
}

# ===========================================================================
# CHECK PORT AVAILABILITY
# ===========================================================================

Write-Step "Checking if port $Port is available..."
try {
    $connection = Get-NetTCPConnection -LocalPort $Port -ErrorAction SilentlyContinue
    if ($connection) {
        $processId = $connection.OwningProcess | Select-Object -First 1
        $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
        $processName = if ($process) { $process.ProcessName } else { "unknown" }
        Write-Warn "Port $Port is already in use by: $processName (PID: $processId)"
        Write-Host ""
        $kill = Read-Host "Kill this process and continue? [y/N]"
        if ($kill.Trim().ToLower() -in @("y", "yes", "s", "sim")) {
            Stop-Process -Id $processId -Force
            Write-OK "Process killed"
            Start-Sleep -Seconds 1
        } else {
            Write-Host "  Aborting. Choose a different port: .\run.ps1 -Port 8002" -ForegroundColor Yellow
            exit 1
        }
    } else {
        Write-OK "Port $Port is available"
    }
} catch {
    # Get-NetTCPConnection may not work on older systems
    Write-Info "Could not check port (continuing anyway)"
}

# ===========================================================================
# START SERVER
# ===========================================================================

Write-Host ""
Write-Host "======================================================================" -ForegroundColor Green
Write-Host "  Starting server on ${Host}:$Port" -ForegroundColor Green
Write-Host "======================================================================" -ForegroundColor Green
Write-Host ""
Write-Host "  Endpoints:" -ForegroundColor White
Write-Host "    API:      http://${Host}:$Port/api/v1/health" -ForegroundColor DarkGray
Write-Host "    Docs:     http://${Host}:$Port/docs" -ForegroundColor DarkGray
Write-Host "    LLM:      http://${Host}:$Port/api/v1/llm/status" -ForegroundColor DarkGray
Write-Host "    Chat:     http://${Host}:$Port/api/v1/chat" -ForegroundColor DarkGray
Write-Host "    Stream:   http://${Host}:$Port/api/v1/chat/stream" -ForegroundColor DarkGray
Write-Host ""
Write-Host "  Press Ctrl+C to stop the server." -ForegroundColor DarkGray
Write-Host ""

# Build uvicorn arguments
$uvicornArgs = @("app.main:app", "--host", $Host, "--port", $Port)

# Auto-enable reload if no flag specified and we're in dev (heuristic)
$envContent = if (Test-Path $EnvFile) { Get-Content $EnvFile -Raw } else { "" }
$isDev = $envContent -match "ENVIRONMENT=development"
if ($Reload -or $isDev) {
    $uvicornArgs += "--reload"
    Write-Info "Auto-reload enabled (development mode)"
}

# Start uvicorn
try {
    & $VenvUvicorn @uvicornArgs
} catch {
    Write-Err "Server failed to start: $_"
    exit 1
}
