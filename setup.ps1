<#
.SYNOPSIS
    Projeto JOI - Automated Windows Setup Script
.DESCRIPTION
    This script automates the entire setup process for the Projeto JOI on Windows.
    Pure ASCII version - works on all PowerShell versions regardless of encoding.
.NOTES
    File name: setup.ps1
    Author:    Projeto JOI
    Requires:  PowerShell 5.1+ (Windows 10/11 ships with this)
    Run as:    Regular user (do NOT run as Administrator unless asked)
#>

[CmdletBinding()]
param(
    [switch]$SkipDocker,
    [switch]$SkipOllama,
    [switch]$SkipTests,
    [switch]$CleanVenv,
    [switch]$Force,
    [switch]$Help
)

# ===========================================================================
# CONFIGURATION
# ===========================================================================

$ErrorActionPreference = "Stop"
$ProgressPreference = "SilentlyContinue"

$ScriptPath = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = $ScriptPath
Set-Location $ProjectRoot

$VenvName = ".venv"
$VenvPath = Join-Path $ProjectRoot $VenvName
$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
$VenvPip = Join-Path $VenvPath "Scripts\pip.exe"
$EnvFile = Join-Path $ProjectRoot ".env"
$EnvExample = Join-Path $ProjectRoot ".env.example"

$MinPythonVersion = [version]"3.12.0"

# ===========================================================================
# HELPER FUNCTIONS (ASCII-only output for max compatibility)
# ===========================================================================

function Write-Header {
    param([string]$Title)
    $line = "=" * 70
    Write-Host ""
    Write-Host $line -ForegroundColor DarkYellow
    Write-Host "  $Title" -ForegroundColor Yellow
    Write-Host $line -ForegroundColor DarkYellow
}

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

function Test-Command {
    param([string]$Name)
    $null = Get-Command $Name -ErrorAction SilentlyContinue
    return $?
}

function Get-SecureInput {
    param(
        [string]$Prompt,
        [switch]$Required,
        [switch]$Secret,
        [string]$Default = "",
        [string]$ValidatePattern = "",
        [string]$ErrorMessage = ""
    )
    do {
        if ($Secret) {
            $secure = Read-Host -Prompt $Prompt -AsSecureString
            $value = [System.Runtime.InteropServices.Marshal]::PtrToStringAuto(
                [System.Runtime.InteropServices.Marshal]::SecureStringToBSTR($secure)
            )
        } else {
            $value = Read-Host -Prompt $Prompt
        }

        if ([string]::IsNullOrWhiteSpace($value)) {
            if ($Required) {
                Write-Warn "This field is required. Please enter a value."
                continue
            } else {
                return $Default
            }
        }

        if ($ValidatePattern -and ($value -notmatch $ValidatePattern)) {
            Write-Err $ErrorMessage
            continue
        }

        return $value
    } while ($true)
}

function Get-YesNo {
    param([string]$Prompt, [bool]$Default = $true)
    $hint = if ($Default) { "[Y/n]" } else { "[y/N]" }
    $response = Read-Host -Prompt "$Prompt $hint"
    if ([string]::IsNullOrWhiteSpace($response)) {
        return $Default
    }
    return $response.Trim().ToLower() -in @("y", "yes", "s", "sim")
}

function Invoke-SafeCommand {
    param(
        [scriptblock]$ScriptBlock,
        [string]$ErrorMessage = "Command failed"
    )
    try {
        & $ScriptBlock
        if ($LASTEXITCODE -ne 0 -and $null -ne $LASTEXITCODE) {
            throw "$ErrorMessage (exit code: $LASTEXITCODE)"
        }
    } catch {
        throw "$ErrorMessage`: $_"
    }
}

# ===========================================================================
# HELP
# ===========================================================================

if ($Help) {
    Write-Host ""
    Write-Host "Projeto JOI - Windows Setup Script" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Usage: .\setup.ps1 [options]"
    Write-Host ""
    Write-Host "Options:"
    Write-Host "  -SkipDocker    Skip Docker installation prompt"
    Write-Host "  -SkipOllama    Skip Ollama installation prompt"
    Write-Host "  -SkipTests     Skip running the test suite"
    Write-Host "  -CleanVenv     Remove existing venv before creating a new one"
    Write-Host "  -Force         Skip confirmation prompts (use with caution)"
    Write-Host "  -Help          Show this help message"
    Write-Host ""
    exit 0
}

# ===========================================================================
# EXECUTION POLICY
# ===========================================================================

try {
    $currentPolicy = Get-ExecutionPolicy -Scope CurrentUser
    if ($currentPolicy -eq "Restricted") {
        Write-Warn "PowerShell execution policy is Restricted."
        Write-Info "Attempting to set policy to RemoteSigned for CurrentUser (no admin needed)..."
        try {
            Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser -Force -ErrorAction Stop
            Write-OK "Execution policy updated to RemoteSigned (CurrentUser)."
        } catch {
            Write-Warn "Could not update execution policy automatically."
            Write-Info "Run this manually in an Admin PowerShell:"
            Write-Host "    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser" -ForegroundColor White
            Write-Info "Then re-run this script."
            exit 1
        }
    }
} catch {
    # Non-fatal - continue
}

# ===========================================================================
# WELCOME BANNER
# ===========================================================================

Write-Header "PROJETO JOI - WINDOWS SETUP"
Write-Host ""
Write-Host "  This script will set up the complete development environment." -ForegroundColor White
Write-Host "  You will be asked for:" -ForegroundColor White
Write-Host "    * Groq API key (required for LLM -- get one free at https://console.groq.com)" -ForegroundColor White
Write-Host "    * Optional: Docker, Ollama, custom ports" -ForegroundColor White
Write-Host ""
Write-Host "  Press Ctrl+C at any time to abort." -ForegroundColor DarkGray
Write-Host ""
$proceed = Get-YesNo "Continue with setup?" -Default $true
if (-not $proceed) {
    Write-Host "  Setup aborted by user." -ForegroundColor Yellow
    exit 0
}

# ===========================================================================
# STEP 1: CHECK PREREQUISITES
# ===========================================================================

Write-Header "Step 1/7: Checking Prerequisites"

# --- Python ---
Write-Step "Checking Python..."
$pythonExe = $null
foreach ($cmd in @("python", "python3", "py")) {
    if (Test-Command $cmd) {
        try {
            $versionOutput = & $cmd --version 2>&1
            if ($versionOutput -match "Python (\d+\.\d+\.\d+)") {
                $version = [version]$Matches[1]
                if ($version -ge $MinPythonVersion) {
                    $pythonExe = $cmd
                    Write-OK "Found $cmd $version"
                    break
                } else {
                    Write-Warn "$cmd $version is too old (need $MinPythonVersion or newer)"
                }
            }
        } catch {
            # Continue to next candidate
        }
    }
}

if (-not $pythonExe) {
    Write-Warn "Python $MinPythonVersion+ not found."
    $installPython = Get-YesNo "Install Python 3.12 via winget?" -Default $true
    if ($installPython) {
        Write-Step "Installing Python 3.12..."
        try {
            Invoke-SafeCommand {
                winget install Python.Python.3.12 --silent --accept-package-agreements --accept-source-agreements
            } -ErrorMessage "Failed to install Python"
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
            $pythonExe = "python"
            Write-OK "Python installed"
        } catch {
            Write-Err $_
            Write-Host ""
            Write-Host "  Please install Python 3.12+ manually from:" -ForegroundColor Yellow
            Write-Host "    https://www.python.org/downloads/" -ForegroundColor White
            Write-Host "  Then re-run this script." -ForegroundColor DarkGray
            exit 1
        }
    } else {
        Write-Host "  Setup aborted. Python is required." -ForegroundColor Yellow
        exit 1
    }
}

# --- Git ---
Write-Step "Checking Git..."
if (Test-Command "git") {
    $gitVersion = (& git --version) -replace "git version ", ""
    Write-OK "Found git $gitVersion"
} else {
    Write-Warn "Git not found."
    $installGit = Get-YesNo "Install Git via winget?" -Default $true
    if ($installGit) {
        Write-Step "Installing Git..."
        try {
            Invoke-SafeCommand {
                winget install Git.Git --silent --accept-package-agreements --accept-source-agreements
            } -ErrorMessage "Failed to install Git"
            $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
            Write-OK "Git installed"
        } catch {
            Write-Err $_
            Write-Host "  Please install Git manually from: https://git-scm.com/download/win" -ForegroundColor Yellow
            exit 1
        }
    } else {
        Write-Warn "Git is recommended. Continuing anyway."
    }
}

# --- pip ---
Write-Step "Checking pip..."
$hasPip = & $pythonExe -m pip --version 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Warn "pip not available. Installing..."
    Invoke-SafeCommand {
        & $pythonExe -m ensurepip --upgrade
    } -ErrorMessage "Failed to install pip"
    Write-OK "pip installed"
} else {
    Write-OK "pip available"
}

# --- Docker (optional) ---
$hasDocker = $false
if (-not $SkipDocker) {
    Write-Step "Checking Docker (optional)..."
    if (Test-Command "docker") {
        $dockerVersion = (& docker --version) 2>&1
        Write-OK "Found $dockerVersion"
        $hasDocker = $true
    } else {
        Write-Warn "Docker not found (optional, recommended for Postgres/Redis/Ollama)"
        $installDocker = Get-YesNo "Install Docker Desktop via winget?" -Default $false
        if ($installDocker) {
            Write-Step "Installing Docker Desktop..."
            Write-Info "This will require a system restart after installation."
            try {
                Invoke-SafeCommand {
                    winget install Docker.DockerDesktop --silent --accept-package-agreements --accept-source-agreements
                } -ErrorMessage "Failed to install Docker"
                Write-OK "Docker Desktop installed"
                Write-Warn "Please restart your computer before using Docker."
            } catch {
                Write-Err $_
                Write-Host "  You can install Docker Desktop manually later from:" -ForegroundColor Yellow
                Write-Host "    https://docs.docker.com/desktop/install/windows-install/" -ForegroundColor White
            }
        }
    }
}

# ===========================================================================
# STEP 2: CREATE VIRTUAL ENVIRONMENT
# ===========================================================================

Write-Header "Step 2/7: Creating Virtual Environment"

if ((Test-Path $VenvPath) -and $CleanVenv) {
    Write-Step "Removing existing venv (clean mode)..."
    Remove-Item -Recurse -Force $VenvPath
    Write-OK "Old venv removed"
}

if (Test-Path $VenvPath) {
    Write-Info "Virtual environment already exists at $VenvName"
    $recreate = Get-YesNo "Recreate it (will lose installed packages)?" -Default $false
    if ($recreate) {
        Remove-Item -Recurse -Force $VenvPath
    }
}

if (-not (Test-Path $VenvPath)) {
    Write-Step "Creating venv at $VenvName..."
    Invoke-SafeCommand {
        & $pythonExe -m venv $VenvName
    } -ErrorMessage "Failed to create virtual environment"
    Write-OK "Virtual environment created"
}

if (-not (Test-Path $VenvPython)) {
    Write-Err "Virtual environment Python not found at $VenvPython"
    exit 1
}

Write-Step "Upgrading pip..."
Invoke-SafeCommand {
    & $VenvPython -m pip install --upgrade pip setuptools wheel --quiet
} -ErrorMessage "Failed to upgrade pip"
Write-OK "pip upgraded"

# ===========================================================================
# STEP 3: INSTALL DEPENDENCIES
# ===========================================================================

Write-Header "Step 3/7: Installing Dependencies"

Write-Step "Installing project (this may take 2-5 minutes)..."
Write-Info "Installing in editable mode with dev dependencies..."
try {
    & $VenvPip install -e ".[dev]" --quiet 2>&1 | ForEach-Object {
        if ($_ -match "ERROR|WARN|Successfully installed") {
            Write-Host "    $_" -ForegroundColor DarkGray
        }
    }
    if ($LASTEXITCODE -ne 0) {
        throw "pip install failed with exit code $LASTEXITCODE"
    }
    Write-OK "Dependencies installed"
} catch {
    Write-Err "Failed to install dependencies: $_"
    Write-Host ""
    Write-Host "  This is often caused by:" -ForegroundColor Yellow
    Write-Host "    1. Network issues -- check your internet connection" -ForegroundColor White
    Write-Host "    2. Build tools missing -- install Visual Studio Build Tools:" -ForegroundColor White
    Write-Host "       https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor DarkGray
    Write-Host "    3. Conflicting packages -- try: .\setup.ps1 -CleanVenv" -ForegroundColor White
    exit 1
}

Write-Step "Installing pre-commit hooks..."
if (Test-Path ".git") {
    try {
        $preCommitExe = Join-Path $VenvPath "Scripts\pre-commit.exe"
        & $preCommitExe install 2>&1 | Out-Null
        Write-OK "pre-commit hooks installed"
    } catch {
        Write-Warn "Could not install pre-commit hooks: $_"
    }
} else {
    Write-Info "Not a git repo yet -- skipping pre-commit install"
}

# ===========================================================================
# STEP 4: COLLECT API KEYS AND CONFIGURATION
# ===========================================================================

Write-Header "Step 4/7: Configuration"

$existingEnv = $null
$skipEnvConfig = $false
if (Test-Path $EnvFile) {
    Write-Info "Found existing .env file"
    $existingEnv = Get-Content $EnvFile -Raw
    $overwrite = Get-YesNo "Reconfigure settings (will back up existing .env)?" -Default $false
    if (-not $overwrite) {
        Write-OK "Keeping existing .env configuration"
        $skipEnvConfig = $true
    } else {
        $backupFile = "$EnvFile.backup.$(Get-Date -Format 'yyyyMMdd-HHmmss')"
        Copy-Item $EnvFile $backupFile
        Write-Info "Backed up to $(Split-Path -Leaf $backupFile)"
    }
}

$groqKey = ""
$ollamaHost = "http://localhost:11434"
$ollamaInstalled = $false
$port = 8000
$envValue = "development"
$env = $true

if (-not $skipEnvConfig) {
    Write-Host ""
    Write-Host "  +-----------------------------------------------------------------+" -ForegroundColor DarkYellow
    Write-Host "  |  GROQ API KEY                                                   |" -ForegroundColor Yellow
    Write-Host "  |                                                                 |" -ForegroundColor Yellow
    Write-Host "  |  Required for the LLM. Get one FREE at:                         |" -ForegroundColor Yellow
    Write-Host "  |     https://console.groq.com                                    |" -ForegroundColor White
    Write-Host "  |                                                                 |" -ForegroundColor Yellow
    Write-Host "  |  Sign in -> API Keys -> Create API Key                          |" -ForegroundColor Yellow
    Write-Host "  |  Free tier: ~14,000 requests/day -- plenty for development      |" -ForegroundColor Yellow
    Write-Host "  +-----------------------------------------------------------------+" -ForegroundColor DarkYellow
    Write-Host ""

    $groqKey = Get-SecureInput -Prompt "Paste your Groq API key (starts with gsk_)" -Secret `
        -ValidatePattern "^(gsk_[A-Za-z0-9_]{30,}|)$" `
        -ErrorMessage "Invalid format. Groq keys start with 'gsk_' and are 40+ chars. Try again or press Enter to skip."

    if ([string]::IsNullOrWhiteSpace($groqKey)) {
        Write-Warn "No Groq key provided. Server will use MockLLMClient (echo responses)."
        Write-Info "You can add the key later by editing .env"
        $groqKey = ""
    } else {
        Write-OK "Groq API key captured"
    }

    # --- Ollama (optional) ---
    if (-not $SkipOllama) {
        Write-Host ""
        Write-Host "  +-----------------------------------------------------------------+" -ForegroundColor DarkYellow
        Write-Host "  |  OLLAMA (Local LLM Fallback)                                    |" -ForegroundColor Yellow
        Write-Host "  |                                                                 |" -ForegroundColor Yellow
        Write-Host "  |  Optional but recommended. Runs a local model when:             |" -ForegroundColor Yellow
        Write-Host "  |    * Groq API is unreachable                                    |" -ForegroundColor Yellow
        Write-Host "  |    * Rate limits are hit                                        |" -ForegroundColor Yellow
        Write-Host "  |    * You want privacy (no data leaves your machine)             |" -ForegroundColor Yellow
        Write-Host "  |                                                                 |" -ForegroundColor Yellow
        Write-Host "  |  Install: https://ollama.com/download/windows                   |" -ForegroundColor White
        Write-Host "  +-----------------------------------------------------------------+" -ForegroundColor DarkYellow
        Write-Host ""

        if (Test-Command "ollama") {
            Write-OK "Ollama already installed"
            $ollamaInstalled = $true
        } else {
            $installOllama = Get-YesNo "Install Ollama via winget?" -Default $true
            if ($installOllama) {
                Write-Step "Installing Ollama..."
                try {
                    Invoke-SafeCommand {
                        winget install Ollama.Ollama --silent --accept-package-agreements --accept-source-agreements
                    } -ErrorMessage "Failed to install Ollama"
                    $env:Path = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
                    Write-OK "Ollama installed"
                    $ollamaInstalled = $true
                } catch {
                    Write-Warn "Could not install Ollama automatically: $_"
                    Write-Host "  Install manually from: https://ollama.com/download/windows" -ForegroundColor Yellow
                }
            }
        }

        if ($ollamaInstalled) {
            Write-Step "Checking if Ollama service is running..."
            $ollamaRunning = $false
            try {
                $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
                if ($response.StatusCode -eq 200) {
                    Write-OK "Ollama service is running"
                    $ollamaRunning = $true
                }
            } catch {
                Write-Warn "Ollama service not running. Attempting to start..."
                try {
                    Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
                    Start-Sleep -Seconds 3
                    $response = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 3 -ErrorAction Stop
                    if ($response.StatusCode -eq 200) {
                        Write-OK "Ollama service started"
                        $ollamaRunning = $true
                    }
                } catch {
                    Write-Warn "Could not start Ollama. You may need to run 'ollama serve' manually."
                }
            }

            if ($ollamaRunning) {
                Write-Step "Checking for llama3.1:8b model..."
                $hasModel = $false
                try {
                    $models = (Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing).Content | ConvertFrom-Json
                    $hasModel = $models.models.name -contains "llama3.1:8b"
                } catch {}

                if ($hasModel) {
                    Write-OK "Model llama3.1:8b already present"
                } else {
                    $pullModel = Get-YesNo "Pull llama3.1:8b model (4.7 GB download, runs in background)?" -Default $true
                    if ($pullModel) {
                        Write-Step "Pulling model (this runs in a separate window)..."
                        Start-Process "ollama" -ArgumentList "pull", "llama3.1:8b"
                        Write-OK "Download started in separate window. You can continue while it downloads."
                        Write-Info "The model will be available once the download completes."
                    }
                }
            }
        }
    }

    # --- Port configuration ---
    Write-Host ""
    $customPort = Get-SecureInput -Prompt "Server port (default 8000)" -Default "8000" `
        -ValidatePattern "^\d{1,5}$" -ErrorMessage "Port must be a number"
    $port = [int]$customPort

    $env = Get-YesNo "Use 'development' environment (verbose logs, auto-reload)?" -Default $true
    $envValue = if ($env) { "development" } else { "production" }
}

# ===========================================================================
# STEP 5: GENERATE .env FILE
# ===========================================================================

Write-Header "Step 5/7: Generating .env Configuration"

if (-not $skipEnvConfig) {
    # Generate a strong SECRET_KEY
    $secretKeyBytes = New-Object byte[] 32
    $rng = [System.Security.Cryptography.RandomNumberGenerator]::Create()
    $rng.GetBytes($secretKeyBytes)
    $secretKey = [Convert]::ToBase64String($secretKeyBytes)

    $debugValue = if ($env) { "true" } else { "false" }
    $ollamaEnabledValue = if ($ollamaInstalled) { "true" } else { "false" }

    $envContent = @"
# ===========================================================================
# Projeto JOI - Environment Configuration
# Generated by setup.ps1 on $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')
# ===========================================================================

# --- Application ---
APP_NAME="Projeto JOI"
APP_VERSION="0.1.0"
ENVIRONMENT=$envValue
DEBUG=$debugValue
LOG_LEVEL=INFO
HOST=127.0.0.1
PORT=$port

# --- Security ---
SECRET_KEY=$secretKey
ALLOWED_ORIGINS_RAW=http://localhost:3000,http://localhost:$port,http://127.0.0.1:$port

# --- Groq API (PRIMARY LLM PROVIDER) ---
GROQ_API_KEY=$groqKey
GROQ_MODEL_PRIMARY=llama-3.1-70b-versatile
GROQ_MODEL_FALLBACK=llama-3.1-8b-instant
GROQ_MAX_TOKENS=2048
GROQ_TEMPERATURE=0.7
GROQ_TIMEOUT_SECONDS=30.0
GROQ_MAX_RETRIES=3

# --- Ollama (LOCAL FALLBACK LLM) ---
OLLAMA_HOST=$ollamaHost
OLLAMA_MODEL=llama3.1:8b
OLLAMA_TIMEOUT_SECONDS=60.0
OLLAMA_ENABLED=$ollamaEnabledValue

# --- PostgreSQL (optional, uses Docker) ---
DATABASE_URL=postgresql+asyncpg://joi:joi_dev_password@localhost:5432/joi
DB_POOL_SIZE=10
DB_MAX_OVERFLOW=20
DB_ECHO=false

# --- Redis (optional, uses Docker) ---
REDIS_URL=redis://localhost:6379/0
REDIS_PASSWORD=
REDIS_NAMESPACE=joi

# --- ChromaDB (vector store) ---
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_EPISODES=episodic_memory
CHROMA_COLLECTION_FACTS=semantic_facts
CHROMA_EMBEDDING_MODEL=bge-m3

# --- Memory System ---
MEMORY_WORKING_TTL_SECONDS=1800
MEMORY_MAX_CONTEXT_TURNS=12
MEMORY_RETRIEVAL_TOP_K=5
MEMORY_RERANK_ENABLED=true

# --- Persona ---
PERSONA_CONFIG_DIR=./config/persona
PERSONA_REINJECTION_INTERVAL=8
PERSONA_CONSISTENCY_CHECK_ENABLED=true

# --- Rate Limiting & Budget ---
RATE_LIMIT_PER_MINUTE=60
MONTHLY_BUDGET_USD=100.0
BUDGET_ALERT_THRESHOLD=0.8

# --- Feature Flags ---
FEATURE_STREAMING=true
FEATURE_MEMORY_PERSISTENCE=true
FEATURE_PERSONA_CONSISTENCY_CHECK=true
FEATURE_FALLBACK_LOCAL=true
"@

    $envContent | Out-File -FilePath $EnvFile -Encoding ascii -NoNewline
    Write-OK ".env file generated at $(Split-Path -Leaf $EnvFile)"
    if ($groqKey) {
        Write-OK "Groq API key configured"
    } else {
        Write-Warn "No Groq key -- server will use MockLLMClient"
    }
}

# ===========================================================================
# STEP 6: RUN TESTS
# ===========================================================================

if (-not $SkipTests) {
    Write-Header "Step 6/7: Validating Installation (running tests)"

    Write-Step "Running pytest (this takes ~10 seconds)..."
    try {
        $pytestExe = Join-Path $VenvPath "Scripts\pytest.exe"
        $testOutput = & $pytestExe --tb=short -q 2>&1
        $testResult = $testOutput | Select-Object -Last 10
        Write-Host $testResult -ForegroundColor DarkGray

        if ($LASTEXITCODE -eq 0) {
            Write-OK "All tests passed!"
        } else {
            Write-Warn "Some tests failed. The server may still work, but please review."
            Write-Info "Run tests manually: .\.venv\Scripts\pytest.exe -v"
        }
    } catch {
        Write-Warn "Could not run tests: $_"
    }
}

# ===========================================================================
# STEP 7: FINAL SUMMARY
# ===========================================================================

Write-Header "Step 7/7: Setup Complete!"

Write-Host ""
Write-Host "  +-----------------------------------------------------------------+" -ForegroundColor Green
Write-Host "  |                                                                 |" -ForegroundColor Green
Write-Host "  |   [OK]  PROJETO JOI is ready to run!                            |" -ForegroundColor Green
Write-Host "  |                                                                 |" -ForegroundColor Green
Write-Host "  +-----------------------------------------------------------------+" -ForegroundColor Green
Write-Host ""

Write-Host "  Quick commands (run from $ProjectRoot):" -ForegroundColor White
Write-Host ""
Write-Host "    # Activate the virtual environment" -ForegroundColor DarkGray
Write-Host "    .\.venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host ""
Write-Host "    # Start the development server" -ForegroundColor DarkGray
Write-Host "    .\.venv\Scripts\uvicorn.exe app.main:app --reload" -ForegroundColor White
Write-Host ""
Write-Host "    # Open the API docs in your browser" -ForegroundColor DarkGray
Write-Host "    start http://localhost:$port/docs" -ForegroundColor White
Write-Host ""
Write-Host "    # Chat with JOI via CLI" -ForegroundColor DarkGray
Write-Host "    .\.venv\Scripts\python.exe scripts\chat_cli.py" -ForegroundColor White
Write-Host ""
Write-Host "    # Run the latency benchmark (requires Groq key)" -ForegroundColor DarkGray
Write-Host "    .\.venv\Scripts\python.exe scripts\benchmark_groq.py" -ForegroundColor White
Write-Host ""

if ($hasDocker) {
    Write-Host "    # Start Postgres + Redis + Ollama via Docker" -ForegroundColor DarkGray
    Write-Host "    docker compose -f docker\docker-compose.yml up -d" -ForegroundColor White
    Write-Host ""
}

if ($groqKey) {
    Write-Host "  Groq API key: [OK] configured" -ForegroundColor Green
} else {
    Write-Host "  Groq API key: [X] not set (using mock client)" -ForegroundColor Yellow
    Write-Host "    Get one at: https://console.groq.com" -ForegroundColor DarkGray
}

if ($ollamaInstalled) {
    Write-Host "  Ollama:       [OK] installed" -ForegroundColor Green
} else {
    Write-Host "  Ollama:       [X] not installed (optional)" -ForegroundColor Yellow
}

if ($hasDocker) {
    Write-Host "  Docker:       [OK] available" -ForegroundColor Green
} else {
    Write-Host "  Docker:       [X] not available (optional)" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  Documentation: https://github.com/CryZenXs/Projeto-JOI" -ForegroundColor DarkGray
Write-Host ""

# --- Optional: start the server now ---
$startNow = Get-YesNo "Start the development server now?" -Default $true
if ($startNow) {
    Write-Host ""
    Write-Step "Starting server..."
    Write-Info "Press Ctrl+C to stop. The server will auto-reload on file changes."
    Write-Info "Open http://localhost:$port/docs in your browser."
    Write-Host ""
    & $VenvPython -m uvicorn app.main:app --reload --host 127.0.0.1 --port $port
}

Write-Host ""
Write-Host "  Setup finished. Press any key to exit..." -ForegroundColor DarkGray
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
