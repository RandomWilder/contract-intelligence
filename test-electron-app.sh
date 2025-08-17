#!/bin/bash

echo "========================================"
echo " Contract Intelligence - Electron Test"
echo "========================================"
echo

echo "1. Checking Node.js installation..."
if ! command -v node &> /dev/null; then
    echo "ERROR: Node.js not found! Please install Node.js from https://nodejs.org/"
    exit 1
fi
node --version

echo "2. Checking Python installation..."
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo "ERROR: Python not found! Please install Python 3.10+"
    exit 1
fi
python3 --version 2>/dev/null || python --version

echo "3. Installing Node.js dependencies..."
cd electron-app
npm install
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install Node.js dependencies"
    exit 1
fi

echo "4. Installing Python dependencies..."
cd ../backend
pip3 install -r requirements.txt 2>/dev/null || pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install Python dependencies"
    echo "Try: pip install fastapi uvicorn python-multipart"
    exit 1
fi

echo "5. Starting Electron app..."
cd ../electron-app
echo
echo "========================================"
echo " 🚀 Starting Contract Intelligence App"
echo "========================================"
echo
echo "The app window should open automatically."
echo "Backend will start on http://localhost:8502"
echo
echo "To stop the app, close the window or press Ctrl+C here."
echo

npm start

