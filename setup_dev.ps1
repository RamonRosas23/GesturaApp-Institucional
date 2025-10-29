# GesturaApp - Development Setup Script
# This script sets up the development environment with additional tools

Write-Host "=====================================================" -ForegroundColor Magenta
Write-Host "    GesturaApp - Development Environment Setup" -ForegroundColor Magenta
Write-Host "=====================================================" -ForegroundColor Magenta
Write-Host ""

# First run the standard installation
Write-Host "Running standard installation..." -ForegroundColor Yellow
& ".\install.ps1"

if ($LASTEXITCODE -ne 0) {
    Write-Host "Standard installation failed. Please check the logs." -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Installing development dependencies..." -ForegroundColor Yellow

# Activate virtual environment
& "venv\Scripts\Activate.ps1"

# Development packages
$devPackages = @(
    "jupyter",
    "notebook", 
    "pytest",
    "black",
    "flake8",
    "isort",
    "mypy",
    "pre-commit"
)

foreach ($package in $devPackages) {
    Write-Host "Installing $package..." -ForegroundColor Cyan
    pip install $package
}

Write-Host ""
Write-Host "Setting up pre-commit hooks..." -ForegroundColor Yellow
pre-commit install

Write-Host ""
Write-Host "=====================================================" -ForegroundColor Green
Write-Host "    Development environment ready!" -ForegroundColor Green
Write-Host "=====================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Development tools installed:" -ForegroundColor Cyan
Write-Host "  • Jupyter Notebook - jupyter notebook" -ForegroundColor White
Write-Host "  • Code formatting - black ." -ForegroundColor White
Write-Host "  • Linting - flake8 ." -ForegroundColor White
Write-Host "  • Import sorting - isort ." -ForegroundColor White
Write-Host "  • Type checking - mypy ." -ForegroundColor White
Write-Host "  • Testing - pytest" -ForegroundColor White
Write-Host ""
Read-Host "Press Enter to exit"