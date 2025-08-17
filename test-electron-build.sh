#!/bin/bash
echo "========================================"
echo "Testing Electron Build Process"
echo "========================================"

cd electron-app

echo
echo "[1/4] Installing Node.js dependencies..."
npm ci
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install Node.js dependencies"
    exit 1
fi

echo
echo "[2/4] Installing Python dependencies..."
cd ../backend
python -m pip install --upgrade pip
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to install Python dependencies"
    exit 1
fi

cd ../electron-app

echo
echo "[3/4] Preparing Python backend for bundling..."
mkdir -p python-backend
cp -r ../backend/* python-backend/
cp -r ../*.py python-backend/ 2>/dev/null || true

echo
echo "[4/4] Building Electron app for macOS..."
npm run build-mac
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to build Electron app"
    exit 1
fi

echo
echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo "Check the electron-app/dist/ directory for the installer"
ls -la dist/*.dmg 2>/dev/null || echo "No .dmg files found"
echo
