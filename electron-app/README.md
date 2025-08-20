# Contract Intelligence Desktop App

Simple Electron-based desktop application that replaces the Streamlit interface with a native desktop experience.

## Architecture

```
Electron Frontend (HTML/CSS/JS)
    ↕️ REST API calls
Python Backend (FastAPI)
    ↕️ Direct imports
Your existing business logic (LocalRAGFlow, etc.)
```

## Quick Start (Local Testing)

### 1. Install Node.js Dependencies
```bash
cd electron-app
npm install
```

### 2. Install Python Dependencies
```bash
cd ../backend
pip install -r requirements.txt
```

### 3. Set up Environment Variables
Make sure you have your OpenAI API key and Google credentials configured:
- OpenAI API key in environment or config
- Google credentials JSON file in the expected location

### 4. Start the Application
```bash
# From electron-app directory
npm start
```

This will:
1. Launch Electron window
2. Start Python backend automatically on port 8502
3. Connect frontend to backend
4. Show the application UI

## Development

- **Frontend code**: `src/` directory (HTML, CSS, JavaScript)
- **Backend code**: `python-backend/api_server_minimal.py` (FastAPI backend)
- **Main process**: `main.js` (Electron main process)

## Building for Distribution

```bash
# Build for current platform
npm run build

# Build for Windows
npm run build-win

# Build for macOS
npm run build-mac
```

## Features

✅ **Native desktop app** - No browser required
✅ **File drag & drop** - Easy document upload
✅ **RTL support** - Hebrew/Arabic text rendering
✅ **Google OAuth** - OCR capabilities
✅ **Chat interface** - Same functionality as Streamlit version
✅ **Document management** - Upload, organize, delete
✅ **Professional UI** - Modern, responsive design

## Troubleshooting

### Backend won't start
- Check Python dependencies are installed
- Verify OpenAI API key is configured
- Check port 8502 is not in use

### Frontend shows "Backend not ready"
- Wait a few seconds for Python backend to start
- Check console for error messages
- Verify backend is running on localhost:8502

### File upload fails
- Check file format is supported (PDF, DOCX, TXT, JPG, PNG)
- Verify you have write permissions
- Check backend logs for errors

## File Structure

```
electron-app/
├── package.json          # Electron configuration
├── main.js              # Main Electron process
├── src/
│   ├── index.html       # Main UI
│   ├── styles.css       # Styling
│   ├── app.js           # Frontend logic
│   └── preload.js       # Security bridge
└── build/               # Icons and build assets

python-backend/
├── api_server_minimal.py    # FastAPI backend (PRODUCTION)
└── requirements.txt         # Python dependencies
```
