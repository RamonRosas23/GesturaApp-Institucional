# GesturaApp - FUNCTIONAL Installation Script
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "    GesturaApp - Complete Installation Script" -ForegroundColor Cyan  
Write-Host "=====================================================" -ForegroundColor Cyan

# Check Python
$pythonCmd = Get-Command python -ErrorAction SilentlyContinue
if (-not $pythonCmd) {
    Write-Host "ERROR: Python not found!" -ForegroundColor Red
    Write-Host "Install Python 3.8+ from https://python.org" -ForegroundColor Yellow
    pause
    exit 1
}

$pythonVersion = python --version
Write-Host "Python found: $pythonVersion" -ForegroundColor Green

# Remove old venv if exists
if (Test-Path "venv") {
    Write-Host "Removing old virtual environment..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force "venv"
}

# Create venv
Write-Host "[1/7] Creating virtual environment..." -ForegroundColor Yellow
python -m venv venv
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to create venv" -ForegroundColor Red
    pause
    exit 1
}

# Activate venv and run commands in it
Write-Host "[2/7] Activating virtual environment..." -ForegroundColor Yellow
$venvPython = ".\venv\Scripts\python.exe"
$venvPip = ".\venv\Scripts\pip.exe"

Write-Host "[3/7] Upgrading pip..." -ForegroundColor Yellow
& $venvPython -m pip install --upgrade pip setuptools wheel
if ($LASTEXITCODE -ne 0) {
    Write-Host "Warning: pip upgrade failed, continuing..." -ForegroundColor Yellow
}

Write-Host "[4/7] Installing requirements..." -ForegroundColor Yellow
& $venvPip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install requirements" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "[5/7] Building leapc-cffi..." -ForegroundColor Yellow
Push-Location "leapc-cffi"
& $venvPython -m build
$buildResult = $LASTEXITCODE
Pop-Location

if ($buildResult -ne 0) {
    Write-Host "ERROR: Failed to build leapc-cffi" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "[6/7] Installing leapc-cffi..." -ForegroundColor Yellow
$tarFile = Get-ChildItem "leapc-cffi\dist\*.tar.gz" | Select-Object -First 1
if ($tarFile) {
    & $venvPip install $tarFile.FullName
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Failed to install leapc-cffi" -ForegroundColor Red
        pause
        exit 1
    }
} else {
    Write-Host "ERROR: leapc-cffi build file not found" -ForegroundColor Red
    pause
    exit 1
}

Write-Host "[7/7] Installing leapc-python-api..." -ForegroundColor Yellow
& $venvPip install -e leapc-python-api
if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to install leapc-python-api" -ForegroundColor Red
    pause
    exit 1
}

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Green
Write-Host "    Installation COMPLETED successfully!" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Testing opencv installation..." -ForegroundColor Yellow
& $venvPython -c "import cv2; print('OpenCV version:', cv2.__version__)"

Write-Host ""
Write-Host "To run the app:" -ForegroundColor Cyan
Write-Host "  .\run_app.ps1" -ForegroundColor White
Write-Host ""
pause