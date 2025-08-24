// Settings page functionality
const API_BASE_URL = 'http://127.0.0.1:8503';

// Initialize page
document.addEventListener('DOMContentLoaded', function() {
    refreshStatus();
});

// Go back to main page
function goBack() {
    // Add parameter to indicate we're returning from settings
    // This allows the main app to skip setup checks and load faster
    window.location.replace('index.html?from=settings');
}

// Show/hide messages
function showMessage(elementId, message, type = 'success') {
    const messageEl = document.getElementById(elementId);
    messageEl.textContent = message;
    messageEl.className = `message ${type}`;
    messageEl.style.display = 'block';
    
    // Auto-hide after 5 seconds
    setTimeout(() => {
        messageEl.style.display = 'none';
    }, 5000);
}

// Update status indicator
function updateStatusIndicator(elementId, connected, text) {
    const statusEl = document.getElementById(elementId);
    statusEl.className = `status-indicator ${connected ? 'status-connected' : 'status-disconnected'}`;
    statusEl.textContent = text;
}

// Refresh overall status
async function refreshStatus() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/settings`);
        const data = await response.json();
        
        // Update status indicators
        updateStatusIndicator('openai-status', data.openai.configured, 
            data.openai.configured ? 'Connected' : 'Not Connected');
        
        updateStatusIndicator('google-status', data.google.configured, 
            data.google.configured ? 'Connected' : 'Not Connected');
        
        // Update status overview
        document.getElementById('status-openai').textContent = 
            data.openai.configured ? '✅ Connected' : '❌ Not Connected';
        
        document.getElementById('status-google').textContent = 
            data.google.configured ? '✅ Connected' : '❌ Not Connected';
        
        // Update available features
        const features = [];
        if (data.openai.configured) features.push('AI Chat', 'Document Analysis');
        if (data.google.configured) features.push('OCR', 'Drive Access');
        
        document.getElementById('status-features').textContent = 
            features.length > 0 ? features.join(', ') : 'None (configure services above)';
            
    } catch (error) {
        console.error('Failed to refresh status:', error);
        showMessage('openai-message', 'Failed to connect to backend. Please ensure the server is running.', 'error');
        
        // Update status indicators to show disconnected state
        updateStatusIndicator('openai-status', false, 'Not Connected');
        updateStatusIndicator('google-status', false, 'Not Connected');
        document.getElementById('status-openai').textContent = '❌ Not Connected';
        document.getElementById('status-google').textContent = '❌ Not Connected';
        document.getElementById('status-features').textContent = 'None (backend not available)';
    }
}

// OpenAI Functions
async function testAndSaveOpenAI() {
    const apiKey = document.getElementById('openai-key').value.trim();
    const saveBtn = document.getElementById('openai-save-btn');
    
    if (!apiKey) {
        showMessage('openai-message', 'Please enter an API key', 'error');
        return;
    }
    
    // Show loading state
    saveBtn.disabled = true;
    saveBtn.innerHTML = 'Testing Key... <span class="loading">⏳</span>';
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/settings/openai`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ api_key: apiKey })
        });
        
        const data = await response.json();
        
        if (response.ok) {
            // More descriptive success message
            showMessage('openai-message', 
                `✅ ${data.message}. Your OpenAI API key is now active.`, 'success');
            document.getElementById('openai-key').value = ''; // Clear for security
            // Make sure to refresh status to update UI
            setTimeout(refreshStatus, 500);
        } else {
            showMessage('openai-message', `❌ ${data.detail}`, 'error');
        }
        
    } catch (error) {
        console.error('OpenAI test failed:', error);
        showMessage('openai-message', '❌ Failed to test API key. Check your connection.', 'error');
    } finally {
        // Reset button
        saveBtn.disabled = false;
        saveBtn.innerHTML = 'Test & Save Key';
    }
}

// Google Functions
async function uploadGoogleCredentials() {
    const fileInput = document.getElementById('google-credentials');
    const uploadBtn = document.getElementById('google-upload-btn');
    
    if (!fileInput.files[0]) {
        showMessage('google-message', 'Please select a service account credentials file', 'error');
        return;
    }
    
    const file = fileInput.files[0];
    
    // Validate file type
    if (!file.name.endsWith('.json')) {
        showMessage('google-message', 'Please select a JSON file', 'error');
        return;
    }
    
    // Show loading state
    uploadBtn.disabled = true;
    uploadBtn.innerHTML = 'Uploading... <span class="loading">⏳</span>';
    
    try {
        const formData = new FormData();
        formData.append('file', file);
        
        const response = await fetch(`${API_BASE_URL}/api/settings/google/upload-credentials`, {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (response.ok) {
            showMessage('google-message', 
                `✅ ${data.message}. Services available: ${data.services.join(', ')}`, 'success');
            fileInput.value = ''; // Clear file input
            // Make sure to refresh status to update UI
            setTimeout(refreshStatus, 500);
        } else {
            showMessage('google-message', `❌ ${data.detail}`, 'error');
        }
        
    } catch (error) {
        console.error('Upload failed:', error);
        showMessage('google-message', '❌ Failed to upload credentials file', 'error');
    } finally {
        // Reset button
        uploadBtn.disabled = false;
        uploadBtn.innerHTML = 'Upload Credentials';
    }
}

// Service account credentials don't need browser authentication

function showGoogleHelp() {
    const helpDiv = document.getElementById('google-help');
    helpDiv.style.display = helpDiv.style.display === 'none' ? 'block' : 'none';
}

// Auto-refresh status every 30 seconds
setInterval(refreshStatus, 30000);
