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
        // **FIX #4: Backend path resolution for distribution**
        const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;
        
        let backendPath;
        let workingDir;
        
                    if (isDev) {
                // Development: use minimal backend for testing
                backendPath = path.join(__dirname, 'python-backend', 'api_server_minimal.py');
                workingDir = path.join(__dirname, 'python-backend');
            } else {
                // Distribution: use minimal backend from resources
                backendPath = path.join(process.resourcesPath, 'python-backend', 'api_server_minimal.py');
                workingDir = path.join(process.resourcesPath, 'python-backend');
            }
        
        console.log(`Starting Python backend from: ${backendPath}`);
        console.log(`Working directory: ${workingDir}`);
        
        // Determine Python executable
        let pythonCmd;
        if (isDev) {
            // Development: use system Python
            pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
        } else {
            // Production: use system Python (installed during build)
            pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
        }
        
        // Start Python process
        let pythonEnv = { ...process.env };
        
        if (!isDev) {
            // Production: ensure Python can find dependencies
            pythonEnv.PYTHONPATH = workingDir;
        } else {
            // Development: ensure Python can find dependencies
            pythonEnv.PYTHONPATH = workingDir;
        }
        
        pythonProcess = spawn(pythonCmd, [backendPath], {
            cwd: workingDir,
            stdio: ['pipe', 'pipe', 'pipe'],
            env: pythonEnv
        });

        pythonProcess.stdout.on('data', (data) => {
            console.log(`Python Backend: ${data}`);
        });

        pythonProcess.stderr.on('data', (data) => {
            console.error(`Python Backend Error: ${data}`);
        });

        pythonProcess.on('close', (code) => {
            console.log(`Python backend exited with code ${code}`);
            
            // If backend crashes, show error to user
            if (code !== 0 && mainWindow) {
                showErrorDialog(`Python backend stopped unexpectedly (code: ${code}). Please check your API configuration.`);
            }
        });

        pythonProcess.on('error', (error) => {
            console.error('Failed to start Python process:', error);
            showErrorDialog('Failed to start Python backend. Please ensure Python is available on your system.');
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
        
        // Retry after 2 seconds, but give up after reasonable time
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