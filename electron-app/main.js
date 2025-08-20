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
    console.log('=== CREATING MAIN WINDOW ===');
    
    // Send immediate diagnostic to any existing renderer
    if (mainWindow && mainWindow.webContents) {
        mainWindow.webContents.executeJavaScript(`console.log("Main process: Creating window...")`);
    }
    
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
        console.log('=== WINDOW READY TO SHOW ===');
        mainWindow.show();
        
        // Send diagnostic to renderer
        mainWindow.webContents.executeJavaScript(`console.log("Main process: Window ready, starting backend...")`);
        
        // Start Python backend
        console.log('=== ABOUT TO START PYTHON BACKEND ===');
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
    console.log(`=== STARTING PYTHON BACKEND FUNCTION ===`);
    
    // Send to renderer immediately
    if (mainWindow && mainWindow.webContents) {
        mainWindow.webContents.executeJavaScript(`console.log("Main process: Starting Python backend function...")`);
    }
    
    try {
        // **FIX #4: Backend path resolution for distribution**
        const isDev = process.env.NODE_ENV === 'development' || !app.isPackaged;
        console.log(`Environment check: NODE_ENV=${process.env.NODE_ENV}, isPackaged=${app.isPackaged}, isDev=${isDev}`);
        
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
            
            // **CRITICAL FIX: Ensure executable exists before proceeding**
            const fs = require('fs');
            if (!fs.existsSync(backendPath)) {
                console.error(`CRITICAL ERROR: Backend executable not found at: ${backendPath}`);
                console.error(`Available files in ${process.resourcesPath}:`);
                try {
                    const files = fs.readdirSync(process.resourcesPath);
                    files.forEach(file => console.error(`  - ${file}`));
                } catch (e) {
                    console.error(`  Error reading directory: ${e.message}`);
                }
                throw new Error(`Backend executable not found: ${backendPath}`);
            }
        }
        
        console.log(`Starting Python backend from: ${backendPath}`);
        console.log(`Working directory: ${workingDir}`);
        console.log(`Platform: ${process.platform}, isDev: ${isDev}, isPackaged: ${app.isPackaged}`);
        
        // **CRITICAL DIAGNOSTIC: Check if backend executable exists and is accessible**
        if (!isDev) {
            const fs = require('fs');
            const diagnosticMessages = [];
            
            function logDiagnostic(message) {
                console.log(message);
                diagnosticMessages.push(message);
                // Also send to renderer immediately
                if (mainWindow && mainWindow.webContents) {
                    mainWindow.webContents.executeJavaScript(`console.log("${message.replace(/"/g, '\\"')}")`);
                }
            }
            
            logDiagnostic(`=== PRODUCTION BACKEND DIAGNOSTIC ===`);
            logDiagnostic(`process.resourcesPath: ${process.resourcesPath}`);
            logDiagnostic(`Expected backend path: ${backendPath}`);
            
            try {
                if (fs.existsSync(backendPath)) {
                    const stats = fs.statSync(backendPath);
                    logDiagnostic(`Backend executable found!`);
                    logDiagnostic(`Size: ${stats.size} bytes`);
                    logDiagnostic(`Permissions: ${stats.mode.toString(8)}`);
                    
                    // **FIX: Platform-specific executable check**
                    let isExecutable = false;
                    if (process.platform === 'win32') {
                        // On Windows, check if file has .exe extension and can be accessed
                        isExecutable = backendPath.toLowerCase().endsWith('.exe');
                        try {
                            // Additional check: try to access the file for execution
                            fs.accessSync(backendPath, fs.constants.F_OK | fs.constants.R_OK);
                            logDiagnostic(`Windows executable access check: PASSED`);
                        } catch (accessError) {
                            logDiagnostic(`Windows executable access check: FAILED - ${accessError.message}`);
                            isExecutable = false;
                        }
                    } else {
                        // Unix-like systems: check permission bits
                        isExecutable = !!(stats.mode & parseInt('111', 8));
                    }
                    
                    logDiagnostic(`Is executable: ${isExecutable}`);
                    
                    // **FIX: Windows-specific permission fix**
                    if (process.platform === 'win32' && !isExecutable) {
                        logDiagnostic(`ATTEMPTING WINDOWS PERMISSION FIX...`);
                        try {
                            // On Windows, try to set file attributes to ensure it's executable
                            const { execSync } = require('child_process');
                            execSync(`attrib -R "${backendPath}"`, { stdio: 'ignore' });
                            logDiagnostic(`Windows permission fix applied successfully`);
                        } catch (winFixError) {
                            logDiagnostic(`Windows permission fix failed: ${winFixError.message}`);
                        }
                    }
                } else {
                    logDiagnostic(`Backend executable NOT FOUND at: ${backendPath}`);
                    logDiagnostic(`THIS IS THE PROBLEM - BACKEND MISSING!`);
                    
                    // List what's actually in the resources directory
                    logDiagnostic(`Contents of ${process.resourcesPath}:`);
                    try {
                        const files = fs.readdirSync(process.resourcesPath);
                        files.forEach(file => {
                            const filePath = path.join(process.resourcesPath, file);
                            const stat = fs.statSync(filePath);
                            logDiagnostic(`  ${stat.isDirectory() ? 'DIR' : 'FILE'}: ${file} (${stat.size || 'N/A'} bytes)`);
                        });
                    } catch (dirError) {
                        logDiagnostic(`Error reading directory: ${dirError.message}`);
                    }
                }
            } catch (error) {
                logDiagnostic(`Error checking backend: ${error.message}`);
            }
            logDiagnostic(`=== END DIAGNOSTIC ===`);
        }
        
        // **FIX #5: Cross-platform executable permissions check**
        if (!isDev) {
            const fs = require('fs');
            try {
                // Check if executable exists
                if (!fs.existsSync(backendPath)) {
                    throw new Error(`Backend executable not found at: ${backendPath}`);
                }
                
                if (process.platform === 'win32') {
                    // **WINDOWS FIX: Ensure file is not read-only and has proper attributes**
                    console.log('Checking Windows executable attributes...');
                    try {
                        const { execSync } = require('child_process');
                        // Remove read-only attribute if present
                        execSync(`attrib -R "${backendPath}"`, { stdio: 'pipe' });
                        console.log('Windows executable attributes verified');
                    } catch (attrError) {
                        console.warn('Windows attribute fix failed:', attrError.message);
                        // Don't fail the whole process for this
                    }
                } else {
                    // Ensure executable permissions on macOS/Linux
                    const stats = fs.statSync(backendPath);
                    if (!(stats.mode & parseInt('111', 8))) {
                        console.log('Setting executable permissions on backend...');
                        fs.chmodSync(backendPath, stats.mode | parseInt('755', 8));
                    }
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

        // Note: stdout handling moved to startup timing section above

        pythonProcess.stderr.on('data', (data) => {
            const output = data.toString();
            console.error(`Python Backend Error: ${output}`);
            
            // **ENHANCED: Capture specific import errors**
            if (output.includes('ModuleNotFoundError') || output.includes('ImportError')) {
                console.error(`IMPORT ERROR DETECTED: ${output}`);
            }
            if (output.includes('No module named')) {
                console.error(`MISSING MODULE: ${output}`);
            }
            if (output.includes('Traceback')) {
                console.error(`PYTHON TRACEBACK: ${output}`);
            }
            
            handleBackendOutput(data); // Also check stderr for readiness signals
        });

        pythonProcess.on('close', (code) => {
            console.log(`Python backend exited with code ${code}`);
            
            // Enhanced error reporting for production debugging
            if (code !== 0 && mainWindow) {
                console.log(`=== BACKEND CRASH DIAGNOSTIC ===`);
                console.log(`Exit code: ${code}`);
                console.log(`Platform: ${process.platform}`);
                console.log(`isDev: ${isDev}`);
                console.log(`Backend path: ${backendPath}`);
                console.log(`Working directory: ${workingDir}`);
                console.log(`=== END CRASH DIAGNOSTIC ===`);
                
                const errorMessage = process.platform === 'darwin' 
                    ? `Python backend stopped unexpectedly (code: ${code}). This might be due to missing dependencies, permissions, or macOS security restrictions. Please check the console for details and try right-clicking the app and selecting "Open" to bypass security warnings.`
                    : `Python backend stopped unexpectedly (code: ${code}). Please check your API configuration and ensure all dependencies are available.`;
                showErrorDialog(errorMessage);
            }
        });

        pythonProcess.on('error', (error) => {
            console.error('Failed to start Python process:', error);
            
            // Send error to renderer console
            if (mainWindow && mainWindow.webContents) {
                mainWindow.webContents.executeJavaScript(`console.error("PYTHON PROCESS SPAWN ERROR: ${error.message.replace(/"/g, '\\"')}")`);
                mainWindow.webContents.executeJavaScript(`console.error("Error code: ${error.code || 'unknown'}");`);
                mainWindow.webContents.executeJavaScript(`console.error("Error path: ${error.path || 'unknown'}");`);
            }
            
            // **ENHANCED: Platform-specific error messages**
            let errorMessage;
            if (process.platform === 'win32') {
                if (error.code === 'EACCES' || error.code === 'EPERM') {
                    errorMessage = `Failed to start Python backend: Permission denied. The executable may be blocked by Windows security or antivirus software. Try running the application as administrator or adding it to your antivirus exceptions.`;
                } else if (error.code === 'ENOENT') {
                    errorMessage = `Failed to start Python backend: Executable not found. The backend file may be missing or corrupted. Please reinstall the application.`;
                } else {
                    errorMessage = `Failed to start Python backend: ${error.message}. This might be due to Windows security restrictions or missing dependencies.`;
                }
            } else if (process.platform === 'darwin') {
                errorMessage = `Failed to start Python backend: ${error.message}. This might be due to missing executable permissions or macOS security restrictions. Try right-clicking the app and selecting "Open" to bypass security warnings.`;
            } else {
                errorMessage = `Failed to start Python backend: ${error.message}. Please ensure the backend executable exists and has proper permissions.`;
            }
            
            showErrorDialog(errorMessage);
        });

        // **IMPROVED: Reliable backend startup detection**
        let backendReadyDetected = false;
        let healthCheckStarted = false;
        
        // Listen for backend readiness signals from both stdout AND stderr
        const handleBackendOutput = (data) => {
            const output = data.toString();
            
            // Detect when Uvicorn is actually running and ready
            if ((output.includes('Uvicorn running on') || output.includes('Application startup complete')) && !backendReadyDetected) {
                backendReadyDetected = true;
                console.log('✅ Backend process signals ready - starting health checks...');
                
                // Start health checks immediately when backend signals ready
                if (!healthCheckStarted) {
                    healthCheckStarted = true;
                    setTimeout(() => checkBackendHealth(), 2000); // Give it a moment to fully initialize
                }
            }
        };
        
        pythonProcess.stdout.on('data', (data) => {
            console.log(`Python Backend: ${data}`);
            handleBackendOutput(data);
        });
        
        // **IMPROVED: Start health checks proactively after reasonable delay**
        // Don't wait only for stdout signals - they might be missed
        setTimeout(() => {
            if (!healthCheckStarted) {
                console.log('⏰ Starting proactive health checks (no startup signal detected yet)...');
                healthCheckStarted = true;
                checkBackendHealth();
            }
        }, 8000); // Start checking after 8 seconds regardless
        
        // **SAFETY NET: Emergency health checks if nothing worked**
        setTimeout(() => {
            if (!backendReadyDetected && healthCheckStarted) {
                console.log('🚨 Emergency health check - backend should be ready by now...');
                checkBackendHealth();
            }
        }, 20000); // Emergency check after 20 seconds

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

async function checkBackendHealth(retryCount = 0, maxRetries = 20) {
    try {
        console.log(`🔍 Health check attempt ${retryCount + 1}/${maxRetries}...`);
        
        const response = await axios.get(`${PYTHON_BACKEND_URL}/health`, {
            timeout: 10000 // 10 second timeout
        });
        
        console.log('✅ Backend is ready:', response.data);
        
        // Notify renderer that backend is ready
        if (mainWindow) {
            mainWindow.webContents.send('backend-ready', true);
            // Also send to console for debugging
            mainWindow.webContents.executeJavaScript(`console.log("✅ Main process confirmed: Backend is ready!")`);
        }
        
    } catch (error) {
        const errorType = error.code || error.message;
        console.log(`❌ Health check ${retryCount + 1} failed: ${errorType}`);
        
        // Progressive backoff with maximum retry limit
        if (retryCount < maxRetries) {
            // More aggressive initial retries, then slower
            const delays = [1000, 2000, 3000, 4000, 5000, 8000, 12000]; 
            const delayIndex = Math.min(retryCount, delays.length - 1);
            const delay = delays[delayIndex];
            
            console.log(`⏰ Retrying health check in ${delay}ms (attempt ${retryCount + 1}/${maxRetries})...`);
            
            // Send progress update to renderer
            if (mainWindow) {
                const progress = Math.round((retryCount / maxRetries) * 100);
                mainWindow.webContents.executeJavaScript(`console.log("⏳ Backend startup progress: ${progress}% (attempt ${retryCount + 1}/${maxRetries})")`);
            }
            
            setTimeout(() => checkBackendHealth(retryCount + 1, maxRetries), delay);
        } else {
            console.error('❌ Backend failed to start after maximum retries. Showing error to user.');
            
            // Send final error to renderer
            if (mainWindow) {
                mainWindow.webContents.executeJavaScript(`console.error("❌ Backend startup failed after ${maxRetries} attempts")`);
            }
            
            showErrorDialog('Backend services failed to start after multiple attempts. Please check your system permissions and try restarting the application.');
        }
    }
}

function showErrorDialog(message) {
    dialog.showErrorBox('Error', message);
}

// App event listeners
console.log('=== ELECTRON APP STARTING ===');
app.whenReady().then(() => {
    console.log('=== ELECTRON APP READY ===');
    createWindow();
});

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