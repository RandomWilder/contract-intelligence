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
            // Development: use Python script directly
            backendPath = path.join(__dirname, 'python-backend', 'api_server_minimal.py');
            workingDir = path.join(__dirname, 'python-backend');
        } else {
            // Distribution: use PyInstaller executable
            const executableName = process.platform === 'win32' ? 'api_server.exe' : 'api_server';
            backendPath = path.join(process.resourcesPath, executableName);
            workingDir = process.resourcesPath;
        }
        
        console.log(`Starting Python backend from: ${backendPath}`);
        console.log(`Working directory: ${workingDir}`);
        console.log(`Platform: ${process.platform}, isDev: ${isDev}, isPackaged: ${app.isPackaged}`);
        
        // **FIX #5: macOS executable permissions check**
        if (!isDev && process.platform !== 'win32') {
            const fs = require('fs');
            try {
                // Check if executable exists
                if (!fs.existsSync(backendPath)) {
                    throw new Error(`Backend executable not found at: ${backendPath}`);
                }
                
                // Ensure executable permissions on macOS/Linux
                const stats = fs.statSync(backendPath);
                if (!(stats.mode & parseInt('111', 8))) {
                    console.log('Setting executable permissions on backend...');
                    fs.chmodSync(backendPath, stats.mode | parseInt('755', 8));
                }
            } catch (error) {
                console.error('Backend executable check failed:', error);
                throw error;
            }
        }
        
        // Start backend process
        let spawnOptions = {
            cwd: workingDir,
            stdio: ['pipe', 'pipe', 'pipe']
        };
        
        if (isDev) {
            // Development: use Python interpreter
            const pythonCmd = process.platform === 'win32' ? 'python' : 'python3';
            let pythonEnv = { ...process.env };
            pythonEnv.PYTHONPATH = workingDir;
            spawnOptions.env = pythonEnv;
            
            pythonProcess = spawn(pythonCmd, [backendPath], spawnOptions);
        } else {
            // Production: run PyInstaller executable directly
            // **FIX #6: Enhanced environment setup for production**
            let productionEnv = { ...process.env };
            
            // Set library paths for macOS
            if (process.platform === 'darwin') {
                productionEnv.DYLD_LIBRARY_PATH = workingDir + ':' + (productionEnv.DYLD_LIBRARY_PATH || '');
                productionEnv.DYLD_FALLBACK_LIBRARY_PATH = workingDir + ':' + (productionEnv.DYLD_FALLBACK_LIBRARY_PATH || '');
            }
            
            spawnOptions.env = productionEnv;
            pythonProcess = spawn(backendPath, [], spawnOptions);
        }

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
                const errorMessage = process.platform === 'darwin' 
                    ? `Python backend stopped unexpectedly (code: ${code}). This might be due to missing dependencies or permissions. Please check the console for details.`
                    : `Python backend stopped unexpectedly (code: ${code}). Please check your API configuration.`;
                showErrorDialog(errorMessage);
            }
        });

        pythonProcess.on('error', (error) => {
            console.error('Failed to start Python process:', error);
            const errorMessage = process.platform === 'darwin'
                ? `Failed to start Python backend: ${error.message}. This might be due to missing executable permissions or dependencies.`
                : 'Failed to start Python backend. Please ensure Python is available on your system.';
            showErrorDialog(errorMessage);
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