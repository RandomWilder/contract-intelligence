# Contract Intelligence Platform - Desktop Distribution

Transform your Streamlit app into a distributable desktop application for Windows and macOS.

## 🚀 Quick Start (For Developers)

### 1. Build the Desktop App

```bash
# Install build dependencies
pip install -r requirements_desktop.txt

# Run the build script
python build_desktop.py
```

### 2. Test Locally

```bash
# Run the desktop launcher
python desktop_launcher.py
```

### 3. Distribute

The built executable will be in the `dist/` directory:
- **Windows**: `dist/ContractIntelligence/ContractIntelligence.exe`
- **macOS**: `dist/Contract Intelligence.app`

## 📦 What Gets Built

### Desktop Launcher
- **GUI Setup**: User-friendly configuration for API keys and credentials
- **Auto-Launch**: Automatically starts Streamlit server and opens browser
- **Telemetry**: Optional anonymous usage analytics (user consent required)

### Bundled Components
- Complete Python runtime (no user installation needed)
- All dependencies (Streamlit, OpenAI, ChromaDB, etc.)
- Your application code
- Configuration management

## 🔧 User Installation Process

### For End Users (What they see):

1. **Download & Install**
   - Download installer (Windows: `.exe`, macOS: `.pkg`)
   - Run installer (standard OS installation process)

2. **First Launch Setup**
   - Application opens with setup wizard
   - Enter OpenAI API key
   - Select Google Cloud credentials JSON file
   - Choose telemetry preference
   - Click "Launch App"

3. **Normal Usage**
   - Click desktop shortcut
   - App launches automatically in browser
   - All data stored locally on their machine

## 🔒 Privacy & Security

### Local-First Architecture
- All processing happens on user's machine
- No data sent to your servers (except optional telemetry)
- User owns their data completely

### Telemetry (Optional)
- **Anonymous**: No personal data collected
- **User Consent**: Explicitly opt-in during setup
- **Transparent**: Users can disable anytime
- **Minimal**: Only usage patterns and error reports

### Data Collected (if enabled):
- App usage statistics (features used, session duration)
- Performance metrics (processing times, error rates)
- System info (OS version, Python version - for compatibility)
- **NOT collected**: API keys, documents, chat content, personal data

## 🛠️ Build Process Details

### PyInstaller Configuration
- **Single Directory**: All files bundled together
- **Hidden Imports**: Automatically includes Streamlit and dependencies
- **Cross-Platform**: Separate builds for Windows/macOS
- **Optimized**: Excludes unnecessary packages (matplotlib, scipy, etc.)

### Platform-Specific Features

#### Windows
- **Installer**: Inno Setup script for professional installation
- **Registry**: Proper Windows integration
- **Shortcuts**: Desktop and Start Menu shortcuts
- **Uninstaller**: Clean removal process

#### macOS
- **App Bundle**: Native `.app` format
- **Code Signing**: Ready for notarization (requires developer certificate)
- **Installer**: `.pkg` installer for App Store-style installation
- **Retina Support**: High-resolution display optimization

## 📊 Telemetry Server (Optional)

If you want usage analytics, set up a simple server:

### Simple Express.js Server
```javascript
const express = require('express');
const app = express();

app.use(express.json());

app.post('/api/usage', (req, res) => {
    console.log('Telemetry:', req.body);
    // Store in database, send to analytics service, etc.
    res.json({ status: 'ok' });
});

app.listen(3000);
```

### Or use existing services:
- **Mixpanel**: Full analytics platform
- **PostHog**: Open-source analytics
- **Google Analytics**: Free but less privacy-focused
- **Custom Database**: Store in your own database

## 🚨 Important Considerations

### Distribution Size
- **Large**: ~500MB-1GB (includes Python runtime)
- **Optimization**: Can be reduced with careful dependency management
- **Trade-off**: Size vs. user convenience (no Python installation needed)

### Updates
- **Manual**: Users download new versions
- **Auto-Update**: Can implement with additional complexity
- **Versioning**: Include version checking in telemetry

### Support
- **Error Tracking**: Telemetry helps identify common issues
- **Logs**: Application creates local log files for troubleshooting
- **User Guidance**: Include troubleshooting guide

## 🔄 Development Workflow

### Testing
1. Test locally with `python desktop_launcher.py`
2. Build with `python build_desktop.py`
3. Test built executable on clean machine
4. Create installer and test full installation process

### Release Process
1. Update version numbers in build scripts
2. Build for all target platforms
3. Create installers
4. Test on multiple machines
5. Distribute (website, email, etc.)

### Maintenance
- Monitor telemetry for errors and usage patterns
- Update dependencies regularly
- Provide user support documentation

## 💡 Advanced Features

### Custom Branding
- Replace icons (`icon.ico`, `icon.icns`)
- Update app names in build scripts
- Customize installer appearance

### Additional Security
- Code signing certificates (Windows/macOS)
- Virus scanning integration
- Secure update mechanisms

### Enterprise Features
- Silent installation options
- Group policy integration (Windows)
- Enterprise license management

## 🎯 Success Metrics

With telemetry enabled, you can track:
- **Adoption**: How many users install and use the app
- **Usage Patterns**: Which features are most popular
- **Performance**: Average processing times, error rates
- **Platform Distribution**: Windows vs macOS usage
- **Retention**: How often users return to the app

This approach gives you a professional, distributable desktop application while maintaining user privacy and providing valuable usage insights.
