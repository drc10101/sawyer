<#
.SYNOPSIS
    Install Sawyer — Distributed MoE Inference Network
.DESCRIPTION
    Installs Python 3.12 (if needed), sawyer-core from PyPI,
    downloads Sawyer Fast Llama binary, creates shortcuts, and validates.
    ONE-LINE INSTALL:
      irm https://sawyer.infill.systems/install.ps1 | iex
    OR:
      ./install_sawyer.ps1
.EXAMPLE
    irm https://sawyer.infill.systems/install.ps1 | iex
#>

param(
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$AppName = "Sawyer"
$AppPkg = "sawyer-core"
$Version = "0.6.0"
$FastLlamaTag = "sawyer-fast-llama-v0.6.0"
$FastLlamaRepo = "drc10101/llama.cpp"
$BinDir = Join-Path $env:USERPROFILE ".sawyer\bin"
$VenvDir = Join-Path $env:USERPROFILE ".sawyer\venv"

# ── Uninstall ──
if ($Uninstall) {
    Write-Host "Uninstalling Sawyer..." -ForegroundColor Yellow
    $DesktopShortcut = "$env:PUBLIC\Desktop\Sawyer.lnk"
    $StartShortcut = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Sawyer.lnk"
    if (Test-Path $DesktopShortcut) { Remove-Item $DesktopShortcut -Force; Write-Host "  Removed desktop shortcut" }
    if (Test-Path $StartShortcut) { Remove-Item $StartShortcut -Force; Write-Host "  Removed Start Menu shortcut" }
    if (Test-Path $BinDir) {
        Write-Host "  Removing Fast Llama binaries from $BinDir" -ForegroundColor DarkGray
        Remove-Item $BinDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    if (Test-Path $VenvDir) {
        Write-Host "  Removing virtual environment from $VenvDir" -ForegroundColor DarkGray
        Remove-Item $VenvDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    # Remove from PATH
    $userPath = [Environment]::GetEnvironmentVariable("Path", "User")
    if ($userPath -like "*$BinDir*") {
        $newPath = ($userPath -split ";" | Where-Object { $_ -ne $BinDir }) -join ";"
        [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
        Write-Host "  Removed from PATH" -ForegroundColor DarkGray
    }
    Write-Host "Sawyer uninstalled." -ForegroundColor Green
    return
}

# ── Banner ──
Write-Host ""
Write-Host "  Sawyer - Distributed MoE Inference Network" -ForegroundColor Cyan
Write-Host "  The load is split. Friends help." -ForegroundColor DarkGray
Write-Host ""

# ── Step 1: Python ──
Write-Host "  Step 1/4: Checking Python 3.11+..." -ForegroundColor Cyan

$python = $null
foreach ($cmd in @("python", "python3", "py")) {
    try {
        $ver = & $cmd --version 2>&1
        if ($ver -match "3\.(\d+)") {
            $minor = [int]$Matches[1]
            if ($minor -ge 11) { $python = $cmd; break }
        }
    } catch {}
}

if (-not $python) {
    Write-Host "  Python 3.11+ not found. Installing Python 3.12..." -ForegroundColor Yellow

    # Download Python 3.12 installer
    $pythonUrl = "https://www.python.org/ftp/python/3.12.7/python-3.12.7-amd64.exe"
    $pythonInstaller = Join-Path $env:TEMP "python-3.12.7-installer.exe"

    Write-Host "  Downloading Python 3.12..." -ForegroundColor Cyan
    try {
        Invoke-WebRequest -Uri $pythonUrl -OutFile $pythonInstaller -UseBasicParsing
    } catch {
        Write-Host "  ERROR: Could not download Python installer." -ForegroundColor Red
        Write-Host "  Download manually from: https://www.python.org/downloads/" -ForegroundColor White
        Write-Host "  Make sure to check 'Add Python to PATH' during install." -ForegroundColor White
        exit 1
    }

    Write-Host "  Installing Python 3.12 (this may take a minute)..." -ForegroundColor Cyan
    Write-Host "  IMPORTANT: If a UAC prompt appears, click Yes." -ForegroundColor Yellow

    # Install Python with PATH and pip
    $installArgs = "/quiet InstallAllUsers=0 PrependPath=1 Include_pip=1 Include_launcher=0"
    $proc = Start-Process -FilePath $pythonInstaller -ArgumentList $installArgs -Wait -PassThru -NoNewWindow

    if ($proc.ExitCode -ne 0) {
        Write-Host "  WARNING: Python installer exited with code $($proc.ExitCode)." -ForegroundColor Yellow
        Write-Host "  Python may have been installed anyway. Continuing..." -ForegroundColor Yellow
    }

    # Clean up installer
    Remove-Item $pythonInstaller -Force -ErrorAction SilentlyContinue

    # Re-check for Python
    foreach ($cmd in @("python", "python3", "py")) {
        try {
            $ver = & $cmd --version 2>&1
            if ($ver -match "3\.(\d+)") {
                $minor = [int]$Matches[1]
                if ($minor -ge 11) { $python = $cmd; break }
            }
        } catch {}
    }

    if (-not $python) {
        # Try refreshing PATH — the installer may have added Python but our shell doesn't see it
        $newPath = [Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [Environment]::GetEnvironmentVariable("Path", "User")
        $env:Path = $newPath

        foreach ($cmd in @("python", "python3", "py")) {
            try {
                $ver = & $cmd --version 2>&1
                if ($ver -match "3\.(\d+)") {
                    $minor = [int]$Matches[1]
                    if ($minor -ge 11) { $python = $cmd; break }
                }
            } catch {}
        }
    }

    if (-not $python) {
        Write-Host "  ERROR: Python installation failed or not on PATH." -ForegroundColor Red
        Write-Host "  Install Python 3.11+ manually from: https://www.python.org/downloads/" -ForegroundColor White
        Write-Host "  Check 'Add Python to PATH' during install, then re-run this script." -ForegroundColor White
        exit 1
    }
}

$pyVer = & $python --version 2>&1
Write-Host "  Using $pyVer" -ForegroundColor Green

# ── Step 2: sawyer-core ──
Write-Host "  Step 2/4: Installing sawyer-core..." -ForegroundColor Cyan

# Create venv for isolation
if (-not (Test-Path (Join-Path $VenvDir "Scripts\Activate.ps1"))) {
    Write-Host "  Creating virtual environment..." -ForegroundColor DarkGray
    & $python -m venv $VenvDir 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  WARNING: venv creation failed. Installing to user site." -ForegroundColor Yellow
        & $python -m pip install --user --upgrade $AppPkg
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ERROR: pip install failed." -ForegroundColor Red
            exit 1
        }
    }
}

if (Test-Path (Join-Path $VenvDir "Scripts\Activate.ps1")) {
    # Activate venv
    & (Join-Path $VenvDir "Scripts\Activate.ps1")
    $python = "python"
    $pip = "pip"

    # Upgrade pip
    & $python -m pip install --upgrade pip --quiet 2>$null

    # Install sawyer-core
    & $pip install --upgrade $AppPkg
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: pip install failed." -ForegroundColor Red
        exit 1
    }

    # Create wrapper script for PATH access
    $sawyerWrapper = Join-Path $BinDir "sawyer-agent.cmd"
    $sawyerWrapperContent = @"
@echo off
call "$VenvDir\Scripts\Activate.bat"
python -m sawyer.cli %*
"@
    New-Item -ItemType Directory -Path $BinDir -Force | Out-Null
    Set-Content -Path $sawyerWrapper -Value $sawyerWrapperContent -Force

    # Also create a PowerShell wrapper
    $sawyerPs1 = Join-Path $BinDir "sawyer-agent.ps1"
    $sawyerPs1Content = @"
& "$VenvDir\Scripts\Activate.ps1"
python -m sawyer.cli `$args
"@
    Set-Content -Path $sawyerPs1 -Value $sawyerPs1Content -Force
}

Write-Host "  $AppPkg installed" -ForegroundColor Green

# ── Step 3: Sawyer Fast Llama binary ──
# Available assets on GitHub:
#   sawyer-fast-llama-cli-windows-x64.zip (CPU CLI, on llama.cpp repo)
#   sawyer-fast-llama-windows-cuda.zip (CUDA server + runtime, on llama.cpp repo)
# Both are .zip archives, not bare .exe files.

Write-Host "  Step 3/4: Downloading Sawyer Fast Llama..." -ForegroundColor Cyan

New-Item -ItemType Directory -Path $BinDir -Force | Out-Null

# Check for NVIDIA GPU
$HasNvidia = $false
try {
    $nvidia = Get-WmiObject Win32_VideoController -ErrorAction SilentlyContinue | Where-Object { $_.Name -like "*NVIDIA*" }
    if ($nvidia) { $HasNvidia = $true }
} catch {}

# Download CUDA variant if NVIDIA GPU present
if ($HasNvidia) {
    $CudaZipName = "sawyer-fast-llama-windows-cuda.zip"
    $CudaZipUrl = "https://github.com/$FastLlamaRepo/releases/download/$FastLlamaTag/$CudaZipName"
    $CudaDest = Join-Path $BinDir "sawyer-fast-llama-windows-cuda"

    if (Test-Path (Join-Path $CudaDest "llama-server.exe")) {
        Write-Host "  CUDA binary already cached" -ForegroundColor Green
    } else {
        Write-Host "  NVIDIA GPU detected — downloading CUDA binary..." -ForegroundColor Cyan
        $CudaZipTemp = Join-Path $env:TEMP "sawyer-cuda.zip"
        try {
            Invoke-WebRequest -Uri $CudaZipUrl -OutFile $CudaZipTemp -UseBasicParsing
            Expand-Archive -Path $CudaZipTemp -DestinationPath $CudaDest -Force
            Remove-Item $CudaZipTemp -Force -ErrorAction SilentlyContinue
            Write-Host "  Downloaded CUDA binary" -ForegroundColor Green
        } catch {
            Write-Host "  WARNING: CUDA download failed. CPU-only binary will be used." -ForegroundColor Yellow
            Remove-Item $CudaZipTemp -Force -ErrorAction SilentlyContinue
        }
    }
}

# Download CPU CLI binary (always needed)
$CliZipName = "sawyer-fast-llama-cli-windows-x64.zip"
$CliZipUrl = "https://github.com/$FastLlamaRepo/releases/download/$FastLlamaTag/$CliZipName"
$CliExeDest = Join-Path $BinDir "sawyer-fast-llama-windows-x64.exe"

if (Test-Path $CliExeDest) {
    Write-Host "  Fast Llama CLI already cached" -ForegroundColor Green
} else {
    Write-Host "  Downloading Fast Llama CLI for Windows..." -ForegroundColor Cyan
    $CliZipTemp = Join-Path $env:TEMP "sawyer-cli.zip"
    try {
        Invoke-WebRequest -Uri $CliZipUrl -OutFile $CliZipTemp -UseBasicParsing
        $TempExtract = Join-Path $env:TEMP "sawyer-cli-extract"
        if (Test-Path $TempExtract) { Remove-Item $TempExtract -Recurse -Force }
        Expand-Archive -Path $CliZipTemp -DestinationPath $TempExtract -Force
        # Find the exe in extracted contents
        $FoundExe = Get-ChildItem -Path $TempExtract -Filter "*.exe" -Recurse | Select-Object -First 1
        if ($FoundExe) {
            Copy-Item $FoundExe.FullName -Destination $CliExeDest -Force
            Write-Host "  Downloaded CLI binary" -ForegroundColor Green
        } else {
            Write-Host "  WARNING: No exe found in zip archive." -ForegroundColor Yellow
        }
        Remove-Item $TempExtract -Recurse -Force -ErrorAction SilentlyContinue
        Remove-Item $CliZipTemp -Force -ErrorAction SilentlyContinue
    } catch {
        Write-Host "  WARNING: Fast Llama download failed." -ForegroundColor Yellow
        Write-Host "  Download manually from: $CliZipUrl" -ForegroundColor Yellow
        Write-Host "  Extract and place in: $BinDir" -ForegroundColor DarkGray
        Remove-Item $CliZipTemp -Force -ErrorAction SilentlyContinue
    }
}

Write-Host "  Fast Llama ready" -ForegroundColor Green

# ── Create desktop shortcuts ──
$PkgDir = & $python -c "import sawyer, os; print(os.path.dirname(sawyer.__file__))" 2>&1
$BatPath = Join-Path $PkgDir "sawyer.bat"
$IconPath = Join-Path $PkgDir "sawyer.ico"

if (-not (Test-Path $BatPath)) {
    Write-Host "  WARNING: Launcher not found at $BatPath" -ForegroundColor Yellow
}
if (-not (Test-Path $IconPath)) {
    Write-Host "  WARNING: Icon not found at $IconPath" -ForegroundColor Yellow
}

$WshShell = New-Object -ComObject WScript.Shell

function New-Shortcut {
    param([string]$Path, [string]$Target, [string]$Icon, [string]$Desc)
    $Shortcut = $WshShell.CreateShortcut($Path)
    $Shortcut.TargetPath = $Target
    $Shortcut.WorkingDirectory = $env:USERPROFILE
    $Shortcut.Description = $Desc
    if ($Icon -and (Test-Path $Icon)) { $Shortcut.IconLocation = "$Icon,0" }
    $Shortcut.Save()
    Write-Host "  Shortcut: $Path" -ForegroundColor DarkGray
}

$DesktopShortcut = "$env:USERPROFILE\Desktop\Sawyer.lnk"
$StartShortcut = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Sawyer.lnk"

New-Shortcut $DesktopShortcut $BatPath $IconPath "Sawyer - Distributed MoE Inference"
New-Shortcut $StartShortcut $BatPath $IconPath "Sawyer - Distributed MoE Inference"

# ── Add to PATH ──
$userPath = [Environment]::GetEnvironmentVariable("Path", "User")
if ($userPath -notlike "*$BinDir*") {
    $newPath = "$BinDir;$userPath"
    [Environment]::SetEnvironmentVariable("Path", $newPath, "User")
    $env:Path = "$BinDir;$($env:Path)"
    Write-Host "  Added $BinDir to PATH" -ForegroundColor DarkGray
}

# ── Step 4: Validate ──
Write-Host "  Step 4/4: Validating installation..." -ForegroundColor Cyan

$validationErrors = 0

# Test sawyer-agent command
$sawyerBin = Join-Path $BinDir "sawyer-agent.cmd"
if (Test-Path $sawyerBin) {
    $helpOutput = & $sawyerBin --help 2>&1
    if ($helpOutput -match "sawyer") {
        Write-Host "  sawyer-agent command works" -ForegroundColor Green
    } else {
        Write-Host "  WARNING: sawyer-agent command returned unexpected output" -ForegroundColor Yellow
        $validationErrors++
    }
} else {
    # Try python -m sawyer as fallback
    try {
        & $python -m sawyer.cli --help | Out-Null 2>&1
        Write-Host "  sawyer-agent works via python -m sawyer" -ForegroundColor Green
    } catch {
        Write-Host "  ERROR: sawyer-agent command not found" -ForegroundColor Red
        $validationErrors++
    }
}

# Test Python imports
try {
    & $python -c "from sawyer.config import SawyerConfig; c = SawyerConfig(); print(f'  Config OK: router={c.router_url}')" 2>&1
    Write-Host "  Python imports work" -ForegroundColor Green
} catch {
    Write-Host "  ERROR: Python import validation failed" -ForegroundColor Red
    $validationErrors++
}

# ── Results ──
Write-Host ""
if ($validationErrors -eq 0) {
    Write-Host "  Sawyer installed successfully!" -ForegroundColor Green
} else {
    Write-Host "  Sawyer installed with $validationErrors warning(s). See above." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "  Quick start:" -ForegroundColor White
Write-Host "    Desktop shortcut: double-click Sawyer" -ForegroundColor Cyan
Write-Host "    Command line:     sawyer-agent chat" -ForegroundColor Cyan
Write-Host "    Serve a node:     sawyer-agent serve" -ForegroundColor Cyan
Write-Host "    All-in-one:       sawyer-agent run" -ForegroundColor Cyan
Write-Host ""

if ($validationErrors -gt 0) {
    Write-Host "  If 'sawyer-agent' command not found, restart your terminal or run:" -ForegroundColor Yellow
    Write-Host "    `$env:Path = `"$BinDir;`$env:Path`"" -ForegroundColor Cyan
    Write-Host ""
}