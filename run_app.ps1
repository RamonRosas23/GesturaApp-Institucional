# GesturaApp - FUNCTIONAL Application Launcher
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "    Starting GesturaApp" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan

# Check venv exists
if (-not (Test-Path "venv\Scripts\python.exe")) {
    Write-Host "ERROR: Virtual environment not found!" -ForegroundColor Red
    Write-Host "Run .\install.ps1 first" -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "Using virtual environment python..." -ForegroundColor Green
$venvPython = ".\venv\Scripts\python.exe"

Write-Host "Starting GesturaApp..." -ForegroundColor Yellow
Write-Host ""

& $venvPython "Aplicacion\GesturaV4.py"

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "Application exited with error code: $LASTEXITCODE" -ForegroundColor Red
    pause
}