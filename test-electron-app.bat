@echo off
echo ========================================
echo  Contract Intelligence - Electron Test
echo ========================================
echo.

echo 1. Checking Node.js installation...
node --version
if %errorlevel% neq 0 (
    echo ERROR: Node.js not found! Please install Node.js from https://nodejs.org/
    pause
    exit /b 1
)

echo 2. Checking Python installation...
python --version
if %errorlevel% neq 0 (
    echo ERROR: Python not found! Please install Python 3.10+
    pause
    exit /b 1
)

echo 3. Installing Node.js dependencies...
cd electron-app
call npm install
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Node.js dependencies
    pause
    exit /b 1
)

echo 4. Installing Python dependencies...
cd ..\backend
pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo ERROR: Failed to install Python dependencies
    echo Try: pip install fastapi uvicorn python-multipart
    pause
    exit /b 1
)

echo 5. Starting Electron app...
cd ..\electron-app
echo.
echo ========================================
echo  🚀 Starting Contract Intelligence App
echo ========================================
echo.
echo The app window should open automatically.
echo Backend will start on http://localhost:8502
echo.
echo To stop the app, close the window or press Ctrl+C here.
echo.

npm start

