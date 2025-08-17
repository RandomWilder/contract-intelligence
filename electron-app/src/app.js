// Main application JavaScript for Contract Intelligence Desktop App

class ContractIntelligenceApp {
    constructor() {
        this.backendUrl = '';
        this.isBackendReady = false;
        this.currentFile = null;
        this.documents = [];
        
        this.init();
    }

    async init() {
        console.log('🚀 Initializing Contract Intelligence App...');
        
        // Get backend URL from Electron
        this.backendUrl = await window.electronAPI.getBackendUrl();
        console.log('Backend URL:', this.backendUrl);
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Listen for backend ready
        window.electronAPI.onBackendReady((event, isReady) => {
            console.log('Backend ready:', isReady);
            this.isBackendReady = isReady;
            this.hideLoadingOverlay();
            this.loadInitialData();
        });
        
        // Show loading overlay initially
        this.showLoadingOverlay('Starting backend services...');
    }

    setupEventListeners() {
        // File input
        const fileInput = document.getElementById('file-input');
        fileInput.addEventListener('change', (e) => this.handleFileSelect(e));
        
        // Upload area drag & drop
        const uploadArea = document.getElementById('upload-area');
        uploadArea.addEventListener('dragover', (e) => this.handleDragOver(e));
        uploadArea.addEventListener('dragleave', (e) => this.handleDragLeave(e));
        uploadArea.addEventListener('drop', (e) => this.handleFileDrop(e));
        uploadArea.addEventListener('click', () => fileInput.click());
        
        // Chat input
        const chatInput = document.getElementById('chat-input');
        chatInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !e.shiftKey) {
                e.preventDefault();
                this.sendMessage();
            }
        });
        
        // Auto-resize textarea
        chatInput.addEventListener('input', (e) => {
            e.target.style.height = 'auto';
            e.target.style.height = Math.min(e.target.scrollHeight, 120) + 'px';
        });
    }

    async loadInitialData() {
        try {
            // Load status
            await this.updateStatus();
            
            // Load documents
            await this.loadDocuments();
            
            // Update UI
            this.updateScopeSelect();
            
        } catch (error) {
            console.error('Failed to load initial data:', error);
            this.showError('Failed to initialize application');
        }
    }

    async updateStatus() {
        try {
            const response = await fetch(`${this.backendUrl}/api/status`);
            const status = await response.json();
            
            // Get Google auth status
            const googleResponse = await fetch(`${this.backendUrl}/api/google/auth-status`);
            const googleStatus = await googleResponse.json();
            
            // Update status indicators
            this.updateStatusIndicator('backend-status', '✅', 'Backend Ready');
            this.updateStatusIndicator('openai-status', 
                status.openai_configured ? '✅' : '❌', 
                'OpenAI'
            );
            this.updateStatusIndicator('google-status', 
                googleStatus.authenticated ? '✅' : (googleStatus.needs_setup ? '❌' : '⚠️'), 
                'Google OCR'
            );
            this.updateStatusIndicator('docs-status', 
                '📊', 
                `${status.documents_count} Documents`
            );
            
            // Update Google auth section
            this.updateGoogleAuthSection(googleStatus);
            
        } catch (error) {
            console.error('Failed to update status:', error);
            this.updateStatusIndicator('backend-status', '❌', 'Backend Error');
        }
    }

    updateStatusIndicator(id, icon, text) {
        const element = document.getElementById(id);
        if (element) {
            element.innerHTML = `<span class="status-icon">${icon}</span><span>${text}</span>`;
        }
    }

    updateGoogleAuthSection(googleStatus) {
        const section = document.getElementById('google-auth-section');
        
        if (googleStatus.authenticated) {
            section.innerHTML = `
                <p>✅ Google services connected</p>
                <p class="small">OCR and Drive access available</p>
                <button class="btn btn-secondary" onclick="app.clearGoogleCredentials()">🗑️ Clear</button>
            `;
        } else if (googleStatus.needs_setup) {
            section.innerHTML = `
                <p>❌ Google credentials file missing</p>
                <p class="small">Path: ${googleStatus.credentials_path}</p>
                <div style="margin: 1rem 0;">
                    <strong>Setup Steps:</strong>
                    <ol style="font-size: 0.8rem; margin-top: 0.5rem;">
                        <li>Go to <a href="#" onclick="app.openGoogleConsole()">Google Cloud Console</a></li>
                        <li>Create OAuth 2.0 Client ID (Desktop App)</li>
                        <li>Download JSON file</li>
                        <li>Save as: <code>google_credentials.json</code></li>
                    </ol>
                </div>
                <button class="btn btn-primary" onclick="app.selectCredentialsFile()">📁 Select Credentials File</button>
            `;
        } else {
            section.innerHTML = `
                <p>⚠️ Google services not authenticated</p>
                <p class="small">Credentials file found, but not authenticated</p>
                <button class="btn btn-primary" onclick="app.authenticateGoogle()">🔗 Connect</button>
            `;
        }
    }

    async loadDocuments() {
        try {
            const response = await fetch(`${this.backendUrl}/api/documents`);
            const data = await response.json();
            
            this.documents = data.documents || [];
            this.renderDocumentsList(data.documents_by_folder || {});
            this.updateFolderSelect(Object.keys(data.documents_by_folder || {}));
            
        } catch (error) {
            console.error('Failed to load documents:', error);
        }
    }

    renderDocumentsList(documentsByFolder) {
        const container = document.getElementById('documents-list');
        
        if (Object.keys(documentsByFolder).length === 0) {
            container.innerHTML = '<p class="empty-state">No documents yet</p>';
            return;
        }
        
        let html = '';
        for (const [folder, docs] of Object.entries(documentsByFolder)) {
            html += `<div class="folder-section">
                <h4>📁 ${folder} (${docs.length})</h4>`;
            
            for (const doc of docs) {
                html += `
                    <div class="document-item">
                        <span class="document-name">📄 ${doc}</span>
                        <div class="document-actions">
                            <button class="delete-btn" onclick="app.deleteDocument('${doc}')">🗑️</button>
                        </div>
                    </div>
                `;
            }
            html += '</div>';
        }
        
        container.innerHTML = html;
    }

    updateFolderSelect(folders) {
        const select = document.getElementById('folder-select');
        select.innerHTML = '<option value="General">General</option>';
        
        folders.forEach(folder => {
            if (folder !== 'General') {
                select.innerHTML += `<option value="${folder}">${folder}</option>`;
            }
        });
    }

    updateScopeSelect() {
        const select = document.getElementById('scope-select');
        select.innerHTML = '<option value="all">All Documents</option>';
        
        // Add individual documents
        this.documents.forEach(doc => {
            select.innerHTML += `<option value="${doc}">${doc}</option>`;
        });
    }

    // File handling
    handleFileSelect(e) {
        const file = e.target.files[0];
        if (file) {
            this.currentFile = file;
            this.showUploadOptions();
        }
    }

    handleDragOver(e) {
        e.preventDefault();
        e.currentTarget.classList.add('dragover');
    }

    handleDragLeave(e) {
        e.currentTarget.classList.remove('dragover');
    }

    handleFileDrop(e) {
        e.preventDefault();
        e.currentTarget.classList.remove('dragover');
        
        const files = e.dataTransfer.files;
        if (files.length > 0) {
            this.currentFile = files[0];
            this.showUploadOptions();
        }
    }

    showUploadOptions() {
        document.querySelector('.upload-area').style.display = 'none';
        document.querySelector('.upload-options').style.display = 'block';
        
        // Auto-select OCR for images
        if (this.currentFile && this.currentFile.type.startsWith('image/')) {
            document.getElementById('use-ocr').checked = true;
        }
    }

    cancelUpload() {
        this.currentFile = null;
        document.querySelector('.upload-area').style.display = 'block';
        document.querySelector('.upload-options').style.display = 'none';
        document.getElementById('file-input').value = '';
    }

    async uploadFile() {
        if (!this.currentFile) return;
        
        const formData = new FormData();
        formData.append('file', this.currentFile);
        formData.append('folder', document.getElementById('folder-select').value);
        formData.append('use_ocr', document.getElementById('use-ocr').checked);
        
        this.showProgress('Processing document...');
        
        try {
            const response = await fetch(`${this.backendUrl}/api/documents/upload`, {
                method: 'POST',
                body: formData
            });
            
            const result = await response.json();
            
            if (response.ok && result.success) {
                this.showSuccess(result.message);
                this.cancelUpload();
                await this.loadDocuments();
                await this.updateStatus();
            } else {
                // Enhanced error handling
                const errorMsg = result.detail || result.message || 'Upload failed';
                console.error('Upload failed:', result);
                this.showError(`Upload failed: ${errorMsg}`);
            }
            
        } catch (error) {
            console.error('Upload failed:', error);
            this.showError('Upload failed: ' + error.message);
        } finally {
            this.hideProgress();
        }
    }

    async deleteDocument(docName) {
        if (!confirm(`Delete document "${docName}"?`)) return;
        
        try {
            const response = await fetch(`${this.backendUrl}/api/documents/${encodeURIComponent(docName)}`, {
                method: 'DELETE'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showSuccess(result.message);
                await this.loadDocuments();
                await this.updateStatus();
            } else {
                this.showError(result.message || 'Delete failed');
            }
            
        } catch (error) {
            console.error('Delete failed:', error);
            this.showError('Delete failed: ' + error.message);
        }
    }

    // Chat functionality
    async sendMessage() {
        const input = document.getElementById('chat-input');
        const query = input.value.trim();
        
        if (!query) return;
        
        // Clear input
        input.value = '';
        input.style.height = 'auto';
        
        // Add user message
        this.addMessage('user', query);
        
        // Disable send button
        const sendButton = document.getElementById('send-button');
        sendButton.disabled = true;
        sendButton.textContent = '⏳ Thinking...';
        
        try {
            // Get scope
            const scopeSelect = document.getElementById('scope-select');
            const scope = scopeSelect.value;
            
            const requestData = {
                query: query,
                target_documents: scope === 'all' ? null : [scope],
                target_folder: null
            };
            
            const response = await fetch(`${this.backendUrl}/api/chat`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(requestData)
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.addMessage('assistant', result.answer, result.source_info);
            } else {
                this.addMessage('assistant', '❌ Error: ' + (result.detail || 'Unknown error'));
            }
            
        } catch (error) {
            console.error('Chat failed:', error);
            this.addMessage('assistant', '❌ Error: Failed to process your question');
        } finally {
            // Re-enable send button
            sendButton.disabled = false;
            sendButton.textContent = '📤 Send';
        }
    }

    askQuickQuestion(question) {
        document.getElementById('chat-input').value = question;
        this.sendMessage();
    }

    addMessage(role, content, sourceInfo = null) {
        const container = document.getElementById('chat-messages');
        
        // Remove welcome message if it exists
        const welcome = container.querySelector('.welcome-message');
        if (welcome) {
            welcome.remove();
        }
        
        // Detect RTL text
        const isRTL = this.detectRTL(content);
        
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        const messageContent = document.createElement('div');
        messageContent.className = `message-content ${isRTL ? 'rtl' : ''}`;
        messageContent.textContent = content;
        
        const messageTime = document.createElement('div');
        messageTime.className = 'message-time';
        messageTime.textContent = new Date().toLocaleTimeString();
        
        messageDiv.appendChild(messageContent);
        messageDiv.appendChild(messageTime);
        
        container.appendChild(messageDiv);
        
        // Scroll to bottom
        container.scrollTop = container.scrollHeight;
    }

    detectRTL(text) {
        // Simple RTL detection for Hebrew/Arabic
        const rtlChars = /[\u0590-\u05FF\u0600-\u06FF]/;
        return rtlChars.test(text);
    }

    // Google Authentication
    async authenticateGoogle() {
        try {
            this.showProgress('Authenticating with Google...');
            
            const response = await fetch(`${this.backendUrl}/api/google/authenticate`, {
                method: 'POST'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showSuccess('Google authentication successful!');
                await this.updateStatus();
            } else {
                this.showError('Google authentication failed');
            }
            
        } catch (error) {
            console.error('Google auth failed:', error);
            this.showError('Authentication failed: ' + error.message);
        } finally {
            this.hideProgress();
        }
    }

    async openGoogleConsole() {
        // Open Google Cloud Console in external browser
        window.electronAPI.openExternal && window.electronAPI.openExternal('https://console.cloud.google.com/apis/credentials');
    }

    async selectCredentialsFile() {
        try {
            const result = await window.electronAPI.showOpenDialog({
                title: 'Select Google Credentials File',
                filters: [
                    { name: 'JSON Files', extensions: ['json'] },
                    { name: 'All Files', extensions: ['*'] }
                ],
                properties: ['openFile']
            });

            if (!result.canceled && result.filePaths.length > 0) {
                const filePath = result.filePaths[0];
                
                // Copy file to the expected location
                this.showProgress('Setting up Google credentials...');
                
                const copyResponse = await fetch(`${this.backendUrl}/api/google/setup-credentials`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({
                        source_path: filePath
                    })
                });
                
                const result = await copyResponse.json();
                
                if (result.success) {
                    this.showSuccess('Credentials file set up successfully! Now you can authenticate.');
                    await this.updateStatus();
                } else {
                    this.showError('Failed to setup credentials: ' + result.message);
                }
            }
        } catch (error) {
            console.error('File selection failed:', error);
            this.showError('Failed to select credentials file: ' + error.message);
        } finally {
            this.hideProgress();
        }
    }

    async clearGoogleCredentials() {
        if (!confirm('Clear Google credentials?')) return;
        
        try {
            const response = await fetch(`${this.backendUrl}/api/google/clear-credentials`, {
                method: 'DELETE'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showSuccess('Credentials cleared');
                await this.updateStatus();
            } else {
                this.showError('Failed to clear credentials');
            }
            
        } catch (error) {
            console.error('Clear credentials failed:', error);
            this.showError('Failed to clear credentials: ' + error.message);
        }
    }

    // UI helpers
    showLoadingOverlay(message = 'Loading...') {
        const overlay = document.getElementById('loading-overlay');
        overlay.style.display = 'flex';
        overlay.querySelector('p').textContent = message;
    }

    hideLoadingOverlay() {
        document.getElementById('loading-overlay').style.display = 'none';
    }

    showProgress(message) {
        const container = document.querySelector('.progress-container');
        const text = container.querySelector('.progress-text');
        
        text.textContent = message;
        container.style.display = 'block';
        
        // Animate progress bar
        const fill = container.querySelector('.progress-fill');
        fill.style.width = '70%';
    }

    hideProgress() {
        const container = document.querySelector('.progress-container');
        container.style.display = 'none';
        
        const fill = container.querySelector('.progress-fill');
        fill.style.width = '0%';
    }

    showSuccess(message) {
        this.showNotification(message, 'success');
    }

    showError(message) {
        this.showNotification(message, 'error');
    }

    showNotification(message, type = 'info') {
        // Create a custom notification that supports RTL text
        const isRTL = this.detectRTL(message);
        
        // Remove any existing notifications
        const existing = document.querySelector('.notification-toast');
        if (existing) existing.remove();
        
        const toast = document.createElement('div');
        toast.className = `notification-toast ${type} ${isRTL ? 'rtl' : ''}`;
        
        const icon = type === 'success' ? '✅' : type === 'error' ? '❌' : 'ℹ️';
        toast.innerHTML = `
            <div class="notification-content">
                <span class="notification-icon">${icon}</span>
                <span class="notification-text">${message}</span>
                <button class="notification-close" onclick="this.parentElement.parentElement.remove()">×</button>
            </div>
        `;
        
        document.body.appendChild(toast);
        
        // Auto-remove after 5 seconds
        setTimeout(() => {
            if (toast && toast.parentElement) {
                toast.remove();
            }
        }, 5000);
    }
}

// Global functions for onclick handlers
function createNewFolder() {
    const name = prompt('Enter folder name:');
    if (name && name.trim()) {
        const select = document.getElementById('folder-select');
        const option = document.createElement('option');
        option.value = name.trim();
        option.textContent = name.trim();
        select.appendChild(option);
        select.value = name.trim();
    }
}

function uploadFile() {
    app.uploadFile();
}

function cancelUpload() {
    app.cancelUpload();
}

function sendMessage() {
    app.sendMessage();
}

function askQuickQuestion(question) {
    app.askQuickQuestion(question);
}

function authenticateGoogle() {
    app.authenticateGoogle();
}

// Initialize app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new ContractIntelligenceApp();
});
