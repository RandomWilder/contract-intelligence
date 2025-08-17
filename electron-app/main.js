const { app, BrowserWindow, ipcMain, dialog, shell } = require('electron');
const path = require('path');
const { spawn } = require('child_process');
const axios = require('axios');

// Keep a global reference of the window object
let mainWindow;
let pythonProcess;

// Python backend configuration
const PYTHON_BACKEND_PORT = 8503;
const PYTHON_BACKEND_URL = `http://127.0.0.1:${PYTHON_BACKEND_PORT}`;

function createWindow() {
    // Create the browser window
    mainWindow = new BrowserWindow({
        width: 1400,
        height: 900,
        webPreferences: {
            nodeIntegration: false,
            contextIsolation: true,
            preload: path.join(__dirname, 'src/preload.js')
        },
        icon: path.join(__dirname, 'build/icon.png'),
        show: false, // Don't show until ready
        titleBarStyle: 'default'
    });

    // Load the app
    mainWindow.loadFile('src/index.html');

    // Show window when ready to prevent visual flash
    mainWindow.once('ready-to-show', () => {
        mainWindow.show();
        
        // Start Python backend
        startPythonBackend();
    });

    // Handle window closed
    mainWindow.on('closed', () => {
        mainWindow = null;
        stopPythonBackend();
    });

    // Open external links in browser
    mainWindow.webContents.setWindowOpenHandler(({ url }) => {
        shell.openExternal(url);
        return { action: 'deny' };
    });
}

function startPythonBackend() {
    try {
        // Path to Python backend (we'll create this)
        const backendPath = path.join(__dirname, '..', 'backend', 'api_server.py');
        
        // Start Python process
        pythonProcess = spawn('python', [backendPath], {
            cwd: path.join(__dirname, '..'),
            stdio: ['pipe', 'pipe', 'pipe']
        });

        pythonProcess.stdout.on('data', (data) => {
            console.log(`Python Backend: ${data}`);
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error(`Python Backend Error: ${data}`);
        });

        pythonProcess.on('close', (code) => {
            console.log(`Python backend exited with code ${code}`);
        });

        // Wait for backend to be ready
        setTimeout(() => {
            checkBackendHealth();
        }, 3000);

    } catch (error) {
        console.error('Failed to start Python backend:', error);
        showErrorDialog('Failed to start Python backend. Please ensure Python is installed.');
    }
}

function stopPythonBackend() {
    if (pythonProcess) {
        pythonProcess.kill();
        pythonProcess = null;
    }
}

async function checkBackendHealth() {
    try {
        const response = await axios.get(`${PYTHON_BACKEND_URL}/health`);
        console.log('Backend is ready:', response.data);
        
        // Notify renderer that backend is ready
        if (mainWindow) {
            mainWindow.webContents.send('backend-ready', true);
        }
    } catch (error) {
        console.error('Backend not ready:', error.message);
        
        // Retry after 2 seconds
        setTimeout(checkBackendHealth, 2000);
    }
}

function showErrorDialog(message) {
    dialog.showErrorBox('Error', message);
}

// App event listeners
app.whenReady().then(createWindow);

app.on('window-all-closed', () => {
    stopPythonBackend();
    if (process.platform !== 'darwin') {
        app.quit();
    }
});

app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
        createWindow();
    }
});

// IPC handlers for renderer process
ipcMain.handle('get-backend-url', () => {
    return PYTHON_BACKEND_URL;
});

ipcMain.handle('show-open-dialog', async (event, options) => {
    const result = await dialog.showOpenDialog(mainWindow, options);
    return result;
});

ipcMain.handle('show-save-dialog', async (event, options) => {
    const result = await dialog.showSaveDialog(mainWindow, options);
    return result;
});

ipcMain.handle('open-external', async (event, url) => {
    shell.openExternal(url);
});

// Handle app closing
app.on('before-quit', () => {
    stopPythonBackend();
});
