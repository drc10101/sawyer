<#
.SYNOPSIS
    Install Sawyer — Distributed MoE Inference Network
.DESCRIPTION
    Installs sawyer-core, creates a desktop shortcut and Start Menu entry.
    Requires Python 3.11+ and pip.
.USAGE
    irm https://infill.systems/install/sawyer.ps1 | iex
    OR
    ./install_sawyer.ps1
#>

param(
    [switch]$Uninstall
)

$ErrorActionPreference = "Stop"
$AppName = "Sawyer"
$AppCmd = "python -m sawyer serve"
$AppPkg = "sawyer-core"
$BatName = "sawyer.bat"
$IconUrl = "https://infill.systems/assets/sawyer-icon.ico"

# --- Uninstall ---
if ($Uninstall) {
    Write-Host "Uninstalling Sawyer..." -ForegroundColor Yellow
    $DesktopShortcut = "$env:PUBLIC\Desktop\Sawyer.lnk"
    $StartShortcut = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Sawyer.lnk"
    if (Test-Path $DesktopShortcut) { Remove-Item $DesktopShortcut -Force; Write-Host "  Removed desktop shortcut" }
    if (Test-Path $StartShortcut) { Remove-Item $StartShortcut -Force; Write-Host "  Removed Start Menu shortcut" }
    $ScriptsDir = pip show $AppPkg 2>$null | Select-String "Location:" | ForEach-Object { ($_ -split ": ")[1] }
    if ($ScriptsDir) {
        $BatPath = Join-Path $ScriptsDir "..\Scripts\$BatName"
        if (Test-Path $BatPath) { Remove-Item $BatPath -Force; Write-Host "  Removed launcher" }
    }
    Write-Host "Sawyer uninstalled." -ForegroundColor Green
    return
}

# --- Check Python ---
Write-Host ""
Write-Host "  Sawyer — Distributed MoE Inference Network" -ForegroundColor Cyan
Write-Host "  The load is split. Friends help." -ForegroundColor DarkGray
Write-Host ""

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
    Write-Host "  ERROR: Python 3.11+ not found." -ForegroundColor Red
    Write-Host "  Install Python: https://www.python.org/downloads/" -ForegroundColor White
    Write-Host "  Make sure to check 'Add Python to PATH' during install." -ForegroundColor White
    exit 1
}

$pyVer = & $python --version 2>&1
Write-Host "  Using $pyVer" -ForegroundColor Green

# --- Install package ---
Write-Host "  Installing $AppPkg..." -ForegroundColor Cyan
& $python -m pip install --upgrade $AppPkg
if ($LASTEXITCODE -ne 0) {
    Write-Host "  ERROR: pip install failed." -ForegroundColor Red
    exit 1
}

# --- Find Scripts directory ---
$SitePackages = & $python -c "import site; print(site.getsitepackages()[0])" 2>&1
$ScriptsDir = Join-Path (Split-Path $SitePackages -Parent) "Scripts"

# --- Write launcher .bat to Scripts dir ---
$BatContent = @"
@echo off
title Sawyer — Distributed MoE Inference Network
echo.
echo  Starting Sawyer node...
echo.
python -m sawyer serve %*
if errorlevel 1 (
    echo.
    echo  Sawyer exited with an error.
    echo  Make sure sawyer-core is installed: pip install sawyer-core
    echo.
    pause
)
"@
$BatPath = Join-Path $ScriptsDir $BatName
Set-Content -Path $BatPath -Value $BatContent -Encoding ASCII
Write-Host "  Launcher: $BatPath" -ForegroundColor DarkGray

# --- Download icon (optional, non-blocking) ---
$IconPath = Join-Path $ScriptsDir "sawyer-icon.ico"
try {
    Invoke-WebRequest -Uri $IconUrl -OutFile $IconPath -UseBasicParsing -ErrorAction Stop
} catch {
    Write-Host "  (Icon download skipped — no internet or icon not hosted yet)" -ForegroundColor DarkGray
    $IconPath = $null
}

# --- Create shortcuts ---
$WshShell = New-Object -ComObject WScript.Shell

function New-Shortcut {
    param([string]$Path, [string]$Target, [string]$Icon)
    $Shortcut = $WshShell.CreateShortcut($Path)
    $Shortcut.TargetPath = $Target
    $Shortcut.WorkingDirectory = $env:USERPROFILE
    if ($Icon) { $Shortcut.IconLocation = $Icon }
    $Shortcut.Save()
    Write-Host "  Shortcut: $Path" -ForegroundColor DarkGray
}

$DesktopShortcut = "$env:USERPROFILE\Desktop\Sawyer.lnk"
$StartShortcut = "$env:APPDATA\Microsoft\Windows\Start Menu\Programs\Sawyer.lnk"

New-Shortcut $DesktopShortcut $BatPath $IconPath
New-Shortcut $StartShortcut $BatPath $IconPath

# --- Done ---
Write-Host ""
Write-Host "  Sawyer installed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "  Quick start:" -ForegroundColor White
Write-Host "    Desktop shortcut: double-click Sawyer" -ForegroundColor Cyan
Write-Host "    Command line:     python -m sawyer serve" -ForegroundColor Cyan
Write-Host "    Register a node:  python -m sawyer register --name my-node --gpu" -ForegroundColor Cyan
Write-Host ""