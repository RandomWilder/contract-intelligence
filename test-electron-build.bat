@echo off
echo ========================================
echo Testing Electron Build Process
echo ========================================

cd electron-app

echo.
echo [1/4] Installing Node.js dependencies...
call npm ci
if errorlevel 1 (
    echo ERROR: Failed to install Node.js dependencies
    pause
    exit /b 1
)

echo.
echo [2/4] Installing Python dependencies...
cd ../backend
call python -m pip install --upgrade pip
call pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: Failed to install Python dependencies
    pause
    exit /b 1
)

cd ../electron-app

echo.
echo [3/4] Preparing Python backend for bundling...
mkdir python-backend 2>nul
xcopy /E /Y ..\backend\* python-backend\
copy ..\*.py python-backend\ 2>nul

echo.
echo [4/4] Building Electron app for Windows...
call npm run build-win
if errorlevel 1 (
    echo ERROR: Failed to build Electron app
    pause
    exit /b 1
)

echo.
echo ========================================
echo Build completed successfully!
echo ========================================
echo Check the electron-app/dist/ directory for the installer
dir dist\*.exe
echo.
pause
