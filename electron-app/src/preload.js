const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
    // Backend communication
    getBackendUrl: () => ipcRenderer.invoke('get-backend-url'),
    
    // File dialogs
    showOpenDialog: (options) => ipcRenderer.invoke('show-open-dialog', options),
    showSaveDialog: (options) => ipcRenderer.invoke('show-save-dialog', options),
    
    // External links
    openExternal: (url) => ipcRenderer.invoke('open-external', url),
    
    // Backend ready listener
    onBackendReady: (callback) => ipcRenderer.on('backend-ready', callback),
    
    // Remove listeners
    removeAllListeners: (channel) => ipcRenderer.removeAllListeners(channel)
});
