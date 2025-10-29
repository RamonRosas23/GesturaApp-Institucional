# GesturaApp - Installation Verification Script
# Run this after installation to verify everything is working

Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host "    GesturaApp - Installation Verification" -ForegroundColor Cyan
Write-Host "=====================================================" -ForegroundColor Cyan
Write-Host ""

# Check if virtual environment exists
if (Test-Path "venv\Scripts\Activate.ps1") {
    Write-Host "✓ Virtual environment found" -ForegroundColor Green
    & "venv\Scripts\Activate.ps1"
} else {
    Write-Host "✗ Virtual environment not found" -ForegroundColor Red
    Write-Host "Please run install.ps1 first" -ForegroundColor Yellow
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host ""
Write-Host "Checking required packages..." -ForegroundColor Yellow
Write-Host ""

# List of critical packages to check
$packages = @(
    "tensorflow",
    "keras", 
    "opencv-python",
    "mediapipe",
    "PyQt6",
    "numpy",
    "pandas",
    "scikit-learn",
    "flask",
    "mysql-connector-python",
    "python-dotenv",
    "google-generativeai",
    "leap"
)

$allInstalled = $true

foreach ($package in $packages) {
    try {
        $version = pip show $package 2>$null | Select-String "Version:" 
        if ($version) {
            $versionNum = ($version -split ":")[1].Trim()
            Write-Host "✓ $package ($versionNum)" -ForegroundColor Green
        } else {
            Write-Host "✗ $package - Not found" -ForegroundColor Red
            $allInstalled = $false
        }
    } catch {
        Write-Host "✗ $package - Error checking" -ForegroundColor Red
        $allInstalled = $false
    }
}

Write-Host ""

if ($allInstalled) {
    Write-Host "=====================================================" -ForegroundColor Green
    Write-Host "    All packages installed successfully!" -ForegroundColor Green
    Write-Host "=====================================================" -ForegroundColor Green
    Write-Host ""
    Write-Host "Testing basic imports..." -ForegroundColor Yellow
    
    # Test critical imports
    $testScript = @"
import sys
print("Python version:", sys.version)
print("Testing imports...")

try:
    import tensorflow as tf
    print("✓ TensorFlow:", tf.__version__)
except Exception as e:
    print("✗ TensorFlow error:", str(e))

try:
    import cv2
    print("✓ OpenCV:", cv2.__version__)
except Exception as e:
    print("✗ OpenCV error:", str(e))

try:
    import mediapipe as mp
    print("✓ MediaPipe:", mp.__version__)
except Exception as e:
    print("✗ MediaPipe error:", str(e))

try:
    import PyQt6
    print("✓ PyQt6: Available")
except Exception as e:
    print("✗ PyQt6 error:", str(e))

try:
    import leap
    print("✓ Leap Motion: Available")
except Exception as e:
    print("✗ Leap Motion error:", str(e))

print("Import test completed!")
"@

    $testScript | python

} else {
    Write-Host "=====================================================" -ForegroundColor Red
    Write-Host "    Some packages are missing!" -ForegroundColor Red  
    Write-Host "=====================================================" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please run install.ps1 again or install missing packages manually" -ForegroundColor Yellow
}

Write-Host ""
Read-Host "Press Enter to exit"