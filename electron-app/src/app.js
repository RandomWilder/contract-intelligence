// Main application JavaScript for Contract Intelligence Desktop App

class ContractIntelligenceApp {
    constructor() {
        this.backendUrl = '';
        this.isBackendReady = false;
        this.currentFile = null;
        this.documents = [];
        this.documentsByFolder = {};
        this.contextMenuTarget = null;
        this.setupNeeded = false;
        
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
            this.checkAndHandleSetup();
        });
        
        // Show loading overlay initially
        this.showLoadingOverlay('Starting backend services...');
    }

    async checkAndHandleSetup() {
        try {
            // Check if setup is needed
            const response = await fetch(`${this.backendUrl}/api/config/check-setup`);
            const setupStatus = await response.json();
            
            if (setupStatus.setup_needed) {
                this.setupNeeded = true;
                this.showSetupModal();
            } else {
                this.setupNeeded = false;
                this.loadInitialData();
            }
            
        } catch (error) {
            console.error('Failed to check setup status:', error);
            // If we can't check setup, try to load normally
            this.loadInitialData();
        }
    }

    showSetupModal() {
        // Create setup modal HTML
        const modalHTML = `
            <div id="setup-modal" class="setup-modal">
                <div class="setup-modal-content">
                    <div class="setup-header">
                        <h2>📄 Welcome to Contract Intelligence</h2>
                        <p>Let's get you set up with the essentials</p>
                    </div>
                    
                    <div class="setup-form">
                        <div class="setup-step">
                            <h3>🔑 OpenAI API Key</h3>
                            <p class="step-description">Required for AI-powered contract analysis</p>
                            <div class="input-group">
                                <input type="password" id="setup-openai-key" placeholder="sk-..." />
                                <button type="button" id="show-key-btn" onclick="this.previousElementSibling.type = this.previousElementSibling.type === 'password' ? 'text' : 'password'">👁️</button>
                            </div>
                            <div class="setup-help">
                                <small>Get your API key from <a href="#" onclick="window.electronAPI.openExternal && window.electronAPI.openExternal('https://platform.openai.com/api-keys')">OpenAI Platform</a></small>
                            </div>
                        </div>
                        
                        <div class="setup-step">
                            <h3>🔍 Google Cloud Credentials (Optional)</h3>
                            <p class="step-description">For OCR capabilities with scanned documents</p>
                            <div class="input-group">
                                <input type="text" id="setup-google-path" placeholder="Select credentials JSON file..." readonly />
                                <button type="button" onclick="app.selectCredentialsForSetup()">Browse</button>
                            </div>
                            <div class="setup-help">
                                <small>
                                    <a href="#" onclick="window.electronAPI.openExternal && window.electronAPI.openExternal('https://console.cloud.google.com/apis/credentials')">Create credentials</a> 
                                    in Google Cloud Console (OAuth 2.0 Client ID, Desktop Application)
                                </small>
                            </div>
                        </div>
                        
                        <div class="setup-actions">
                            <button id="setup-save-btn" class="btn btn-primary" onclick="app.saveSetupConfig()">
                                💾 Save & Continue
                            </button>
                            <button class="btn btn-secondary" onclick="app.skipSetup()">
                                ⏭️ Skip Google Setup
                            </button>
                        </div>
                        
                        <div id="setup-status" class="setup-status"></div>
                    </div>
                </div>
            </div>
        `;
        
        // Add modal to page
        document.body.insertAdjacentHTML('beforeend', modalHTML);
        
        // Add modal styles
        this.addSetupModalStyles();
    }

    addSetupModalStyles() {
        if (document.getElementById('setup-modal-styles')) return;
        
        const styles = `
            <style id="setup-modal-styles">
                .setup-modal {
                    position: fixed;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    background: rgba(0, 0, 0, 0.8);
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    z-index: 10000;
                }
                
                .setup-modal-content {
                    background: white;
                    border-radius: 12px;
                    padding: 2rem;
                    max-width: 500px;
                    width: 90%;
                    max-height: 80vh;
                    overflow-y: auto;
                    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.3);
                }
                
                .setup-header {
                    text-align: center;
                    margin-bottom: 2rem;
                }
                
                .setup-header h2 {
                    margin: 0 0 0.5rem 0;
                    color: #333;
                }
                
                .setup-header p {
                    margin: 0;
                    color: #666;
                }
                
                .setup-step {
                    margin-bottom: 2rem;
                }
                
                .setup-step h3 {
                    margin: 0 0 0.5rem 0;
                    color: #333;
                    font-size: 1.1rem;
                }
                
                .step-description {
                    margin: 0 0 1rem 0;
                    color: #666;
                    font-size: 0.9rem;
                }
                
                .input-group {
                    display: flex;
                    gap: 0.5rem;
                }
                
                .input-group input {
                    flex: 1;
                    padding: 0.75rem;
                    border: 1px solid #ddd;
                    border-radius: 6px;
                    font-size: 0.9rem;
                }
                
                .input-group button {
                    padding: 0.75rem 1rem;
                    border: 1px solid #ddd;
                    border-radius: 6px;
                    background: #f8f9fa;
                    cursor: pointer;
                }
                
                .input-group button:hover {
                    background: #e9ecef;
                }
                
                .setup-help {
                    margin-top: 0.5rem;
                }
                
                .setup-help small {
                    color: #888;
                }
                
                .setup-help a {
                    color: #007bff;
                    text-decoration: none;
                }
                
                .setup-help a:hover {
                    text-decoration: underline;
                }
                
                .setup-actions {
                    display: flex;
                    gap: 1rem;
                    justify-content: center;
                    margin-top: 2rem;
                }
                
                .setup-status {
                    margin-top: 1rem;
                    padding: 0.75rem;
                    border-radius: 6px;
                    display: none;
                }
                
                .setup-status.success {
                    background: #d4edda;
                    color: #155724;
                    border: 1px solid #c3e6cb;
                }
                
                .setup-status.error {
                    background: #f8d7da;
                    color: #721c24;
                    border: 1px solid #f5c6cb;
                }
                
                .setup-status.loading {
                    background: #d1ecf1;
                    color: #0c5460;
                    border: 1px solid #bee5eb;
                }
            </style>
        `;
        
        document.head.insertAdjacentHTML('beforeend', styles);
    }

    async selectCredentialsForSetup() {
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
                document.getElementById('setup-google-path').value = result.filePaths[0];
            }
        } catch (error) {
            console.error('File selection failed:', error);
            this.showSetupStatus('error', 'Failed to select file: ' + error.message);
        }
    }

    async saveSetupConfig() {
        const openaiKey = document.getElementById('setup-openai-key').value.trim();
        const googlePath = document.getElementById('setup-google-path').value.trim();
        
        if (!openaiKey) {
            this.showSetupStatus('error', 'OpenAI API key is required');
            return;
        }
        
        this.showSetupStatus('loading', 'Validating configuration...');
        
        try {
            const setupData = {
                openai_key: openaiKey,
                google_creds_path: googlePath || undefined
            };
            
            const response = await fetch(`${this.backendUrl}/api/config/setup`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(setupData)
            });
            
            const result = await response.json();
            
            if (response.ok && result.success) {
                this.showSetupStatus('success', 'Configuration saved successfully!');
                
                // Wait a moment then close modal and continue
                setTimeout(() => {
                    this.closeSetupModal();
                    this.setupNeeded = false;
                    this.loadInitialData();
                }, 1500);
                
            } else {
                this.showSetupStatus('error', result.message || 'Setup failed');
            }
            
        } catch (error) {
            console.error('Setup failed:', error);
            this.showSetupStatus('error', 'Setup failed: ' + error.message);
        }
    }

    skipSetup() {
        const openaiKey = document.getElementById('setup-openai-key').value.trim();
        
        if (!openaiKey) {
            this.showSetupStatus('error', 'OpenAI API key is required to continue');
            return;
        }
        
        // Save just the OpenAI key
        this.saveSetupConfig();
    }

    showSetupStatus(type, message) {
        const statusDiv = document.getElementById('setup-status');
        statusDiv.className = `setup-status ${type}`;
        statusDiv.textContent = message;
        statusDiv.style.display = 'block';
    }

    closeSetupModal() {
        const modal = document.getElementById('setup-modal');
        if (modal) {
            modal.remove();
        }
        
        const styles = document.getElementById('setup-modal-styles');
        if (styles) {
            styles.remove();
        }
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

        // Context menu - hide on click outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('#context-menu')) {
                this.hideContextMenu();
            }
        });

        // Context menu actions
        document.addEventListener('click', (e) => {
            if (e.target.closest('.context-menu-item')) {
                const action = e.target.closest('.context-menu-item').dataset.action;
                this.handleContextMenuAction(action);
            }
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
                <button class="btn btn-primary" onclick="app.selectCredentialsFile()">🔍 Select Credentials File</button>
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
            this.documentsByFolder = data.documents_by_folder || {};
            this.renderDocumentsTree(this.documentsByFolder);
            this.updateFolderSelect(Object.keys(this.documentsByFolder));
            this.updateFolderScopeSelect(Object.keys(this.documentsByFolder));
            
        } catch (error) {
            console.error('Failed to load documents:', error);
        }
    }

    renderDocumentsTree(documentsByFolder) {
        const container = document.getElementById('documents-tree');
        
        if (Object.keys(documentsByFolder).length === 0) {
            container.innerHTML = '<p class="empty-state">No documents yet</p>';
            return;
        }
        
        let html = '';
        for (const [folder, docs] of Object.entries(documentsByFolder)) {
            const folderId = `folder-${folder.replace(/\s+/g, '-').toLowerCase()}`;
            html += `
                <div class="tree-folder">
                    <div class="folder-header" data-folder="${folder}" data-folder-id="${folderId}">
                        <span class="folder-toggle" id="${folderId}-toggle">▶</span>
                        <span class="folder-icon">📁</span>
                        <span class="folder-name">${folder}</span>
                        <span class="folder-count">${docs.length}</span>
                    </div>
                    <div class="folder-documents" id="${folderId}-docs">`;
            
            for (const doc of docs) {
                const docIcon = this.getDocumentIcon(doc);
                html += `
                    <div class="document-item" data-document="${doc}" data-folder="${folder}">
                        <span class="document-icon">${docIcon}</span>
                        <span class="document-name" title="${doc}">${doc}</span>
                    </div>
                `;
            }
            html += `
                    </div>
                </div>
            `;
        }
        
        container.innerHTML = html;
        
        // Add event listeners after rendering
        this.attachTreeEventListeners();
    }

    getDocumentIcon(filename) {
        const extension = filename.split('.').pop().toLowerCase();
        switch (extension) {
            case 'pdf': return '📄';
            case 'docx':
            case 'doc': return '📝';
            case 'txt': return '📃';
            case 'jpg':
            case 'jpeg':
            case 'png':
            case 'gif': return '🖼️';
            default: return '📄';
        }
    }

    attachTreeEventListeners() {
        // Folder click handlers
        const folderHeaders = document.querySelectorAll('.folder-header');
        folderHeaders.forEach(header => {
            // Click to toggle folder
            header.addEventListener('click', (e) => {
                e.stopPropagation();
                const folderId = header.dataset.folderId;
                this.toggleFolder(folderId);
            });
            
            // Right-click context menu for folders
            header.addEventListener('contextmenu', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const folderName = header.dataset.folder;
                this.showContextMenu(e, 'folder', folderName);
            });
        });
        
        // Document right-click handlers
        const documentItems = document.querySelectorAll('.document-item');
        documentItems.forEach(item => {
            item.addEventListener('contextmenu', (e) => {
                e.preventDefault();
                e.stopPropagation();
                const documentName = item.dataset.document;
                const folderName = item.dataset.folder;
                this.showContextMenu(e, 'document', documentName, folderName);
            });
            
            // Optional: Double-click to open document
            item.addEventListener('dblclick', (e) => {
                e.stopPropagation();
                const documentName = item.dataset.document;
                const folderName = item.dataset.folder;
                this.openItem('document', documentName, folderName);
            });
        });
    }

    toggleFolder(folderId) {
        const docsContainer = document.getElementById(`${folderId}-docs`);
        const toggle = document.getElementById(`${folderId}-toggle`);
        
        if (docsContainer && toggle) {
            if (docsContainer.classList.contains('expanded')) {
                docsContainer.classList.remove('expanded');
                toggle.classList.remove('expanded');
                toggle.textContent = '▶';
            } else {
                docsContainer.classList.add('expanded');
                toggle.classList.add('expanded');
                toggle.textContent = '▼';
            }
        }
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

    updateFolderScopeSelect(folders) {
        const select = document.getElementById('folder-scope-select');
        select.innerHTML = '<option value="all">All Folders</option>';
        
        folders.forEach(folder => {
            select.innerHTML += `<option value="${folder}">${folder}</option>`;
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
            // Get scope and folder
            const scopeSelect = document.getElementById('scope-select');
            const folderScopeSelect = document.getElementById('folder-scope-select');
            const scope = scopeSelect.value;
            const folderScope = folderScopeSelect.value;
            
            const requestData = {
                query: query,
                target_documents: scope === 'all' ? null : [scope],
                target_folder: folderScope === 'all' ? null : folderScope
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
 
    // Context Menu functionality
    showContextMenu(event, type, name, folder = null) {
        event.preventDefault();
        event.stopPropagation();
        
        const contextMenu = document.getElementById('context-menu');
        this.contextMenuTarget = { type, name, folder };
        
        // Show menu first to get dimensions
        contextMenu.style.display = 'block';
        contextMenu.style.left = '0px';
        contextMenu.style.top = '0px';
        
        // Get menu dimensions
        const rect = contextMenu.getBoundingClientRect();
        
        // Calculate position
        let x = event.clientX;
        let y = event.clientY;
        
        // Adjust if menu would go off screen
        if (x + rect.width > window.innerWidth) {
            x = window.innerWidth - rect.width - 5;
        }
        if (y + rect.height > window.innerHeight) {
            y = window.innerHeight - rect.height - 5;
        }
        
        // Position the menu
        contextMenu.style.left = x + 'px';
        contextMenu.style.top = y + 'px';
        
        console.log(`Context menu shown for ${type}: ${name}${folder ? ` in ${folder}` : ''}`);
    }
 
    hideContextMenu() {
        const contextMenu = document.getElementById('context-menu');
        contextMenu.style.display = 'none';
        this.contextMenuTarget = null;
    }
 
    handleContextMenuAction(action) {
        if (!this.contextMenuTarget) return;
        
        const { type, name, folder } = this.contextMenuTarget;
        this.hideContextMenu();
        
        switch (action) {
            case 'open':
                this.openItem(type, name, folder);
                break;
            case 'rename':
                this.renameItem(type, name, folder);
                break;
            case 'delete':
                this.deleteItemWithConfirmation(type, name, folder);
                break;
            case 'properties':
                this.showItemProperties(type, name, folder);
                break;
        }
    }
 
    openItem(type, name, folder) {
        if (type === 'document') {
            this.showNotification(`Opening ${name}...`, 'info');
            // TODO: Implement document opening functionality
        } else if (type === 'folder') {
            const folderId = `folder-${name.replace(/\s+/g, '-').toLowerCase()}`;
            this.toggleFolder(folderId);
        }
    }
 
    renameItem(type, name, folder) {
        const newName = prompt(`Enter new name for ${type}:`, name);
        if (newName && newName.trim() && newName.trim() !== name) {
            this.showNotification(`Renaming ${type} functionality coming soon...`, 'info');
            // TODO: Implement rename functionality
        }
    }
 
    async deleteItemWithConfirmation(type, name, folder) {
        let confirmMessage;
        if (type === 'folder') {
            const folderDocs = this.documents.filter(doc => 
                Object.entries(this.documentsByFolder || {}).some(([f, docs]) => 
                    f === name && docs.includes(doc)
                )
            );
            confirmMessage = `Delete folder "${name}" and all ${folderDocs.length} documents inside it?\n\nThis action cannot be undone.`;
        } else {
            confirmMessage = `Delete document "${name}"?\n\nThis action cannot be undone.`;
        }
        
        if (confirm(confirmMessage)) {
            if (type === 'document') {
                await this.deleteDocument(name);
            } else if (type === 'folder') {
                await this.deleteFolder(name);
            }
        }
    }
 
    async deleteFolder(folderName) {
        try {
            this.showProgress(`Deleting folder "${folderName}"...`);
            
            const response = await fetch(`${this.backendUrl}/api/folders/${encodeURIComponent(folderName)}`, {
                method: 'DELETE'
            });
            
            const result = await response.json();
            
            if (result.success) {
                this.showSuccess(`Folder "${folderName}" deleted successfully`);
                await this.loadDocuments();
                await this.updateStatus();
            } else {
                this.showError(result.message || 'Failed to delete folder');
            }
            
        } catch (error) {
            console.error('Delete folder failed:', error);
            this.showError('Failed to delete folder: ' + error.message);
        } finally {
            this.hideProgress();
        }
    }
 
    showItemProperties(type, name, folder) {
        let info = `${type.charAt(0).toUpperCase() + type.slice(1)}: ${name}`;
        if (folder && type === 'document') {
            info += `\nFolder: ${folder}`;
        }
        if (type === 'folder') {
            const folderDocs = Object.entries(this.documentsByFolder || {})
                .find(([f, docs]) => f === name)?.[1] || [];
            info += `\nDocuments: ${folderDocs.length}`;
        }
        
        alert(info);
        // TODO: Implement proper properties dialog
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
 
 // Open settings page
function openSettings() {
    window.location.href = 'settings.html';
}

// Initialize app when DOM is loaded
let app;
document.addEventListener('DOMContentLoaded', () => {
    app = new ContractIntelligenceApp();
});