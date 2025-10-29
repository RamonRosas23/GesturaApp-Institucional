@echo off
echo =====================================================
echo    GesturaApp - Complete Installation Script
echo =====================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://python.org
    pause
    exit /b 1
)

echo [1/7] Creating virtual environment...
python -m venv venv
if errorlevel 1 (
    echo ERROR: Failed to create virtual environment
    pause
    exit /b 1
)

echo [2/7] Activating virtual environment...
call venv\Scripts\activate.bat

echo [3/7] Upgrading pip, setuptools and wheel...
python -m pip install --upgrade pip setuptools wheel

echo [4/7] Installing main requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install requirements
    pause
    exit /b 1
)

echo [5/7] Building leapc-cffi...
cd leapc-cffi
python -m build
if errorlevel 1 (
    echo ERROR: Failed to build leapc-cffi
    cd ..
    pause
    exit /b 1
)
cd ..

echo [6/7] Installing leapc-cffi package...
pip install leapc-cffi/dist/leapc_cffi-0.0.1.tar.gz
if errorlevel 1 (
    echo ERROR: Failed to install leapc-cffi
    pause
    exit /b 1
)

echo [7/7] Installing leapc-python-api in development mode...
pip install -e leapc-python-api
if errorlevel 1 (
    echo ERROR: Failed to install leapc-python-api
    pause
    exit /b 1
)

echo.
echo =====================================================
echo    Installation completed successfully!
echo =====================================================
echo.
echo To activate the virtual environment in the future, run:
echo    venv\Scripts\activate.bat
echo.
echo To run the application:
echo    python Aplicacion\GesturaV4.py
echo.
echo To test Leap Motion tracking:
echo    python examples\tracking_event_example.py
echo.
pause