@echo off
echo =====================================================
echo    Starting GesturaApp
echo =====================================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found
    echo Please run install.bat first
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo Starting GesturaApp...
python Aplicacion\GesturaV4.py

pause