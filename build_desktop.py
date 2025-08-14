#!/usr/bin/env python3
"""
Build script for creating desktop distribution of Contract Intelligence Platform
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import platform

def create_pyinstaller_spec():
    """Create PyInstaller spec file for the application"""
    
    spec_content = '''# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from pathlib import Path

# Get the directory containing this spec file
spec_dir = Path(SPECPATH)

block_cipher = None

# Define the main script
main_script = str(spec_dir / 'desktop_launcher.py')

# Collect all necessary files
added_files = [
    (str(spec_dir / 'streamlit_app.py'), '.'),
    (str(spec_dir / 'local_rag_app.py'), '.'),
    (str(spec_dir / 'contract_intelligence.py'), '.'),
    (str(spec_dir / 'simple_ocr.py'), '.'),
    (str(spec_dir / 'desktop_launcher.py'), '.'),
    (str(spec_dir / 'telemetry_client.py'), '.'),
    # Add any other Python files your app needs
]

# Add data directories if they exist
data_dirs = ['data', 'conf']
for data_dir in data_dirs:
    data_path = spec_dir / data_dir
    if data_path.exists():
        added_files.append((str(data_path), data_dir))

a = Analysis(
    [main_script],
    pathex=[str(spec_dir)],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'streamlit',
        'streamlit.web.cli',
        'streamlit.runtime',
        'streamlit.runtime.scriptrunner',
        'streamlit.runtime.state',
        'chromadb',
        'chromadb.utils.embedding_functions',
        'chromadb.utils.embedding_functions.openai_embedding_function',
        'openai',
        'google.auth',
        'google_auth_oauthlib',
        'PyPDF2',
        'docx',
        'cv2',
        'PIL',
        'numpy',
        'requests',
        'tiktoken',
        'tiktoken_ext',
        'tiktoken_ext.openai_public',
        'altair',
        'plotly',
        'pandas',
        'pytz',
        'tzdata',
        'keyring',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        'matplotlib',
        'scipy',
        'sklearn',
        'jupyter',
        'notebook',
        'IPython',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='ContractIntelligence',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to False for Windows GUI app
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='icon.ico' if os.path.exists('icon.ico') else None,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ContractIntelligence',
)

# For macOS, create an app bundle
if sys.platform == 'darwin':
    app = BUNDLE(
        coll,
        name='Contract Intelligence.app',
        icon='icon.icns' if os.path.exists('icon.icns') else None,
        bundle_identifier='com.yourcompany.contractintelligence',
        version='1.4.7',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'CFBundleDisplayName': 'Contract Intelligence Platform',
            'CFBundleExecutable': 'ContractIntelligence',
            'CFBundleName': 'Contract Intelligence',
            'CFBundleVersion': '1.4.7',
            'CFBundleShortVersionString': '1.4.7',
        },
    )
'''
    
    with open('contract_intelligence.spec', 'w') as f:
        f.write(spec_content)
    
    print("[SUCCESS] PyInstaller spec file created")

def install_dependencies():
    """Install required dependencies for building"""
    # In GitHub Actions, dependencies are already installed via requirements_desktop.txt
    # This function now just verifies they're available
    required_modules = [
        'pyinstaller',
        'streamlit',
        'openai',
        'chromadb',
        'google.auth',
        'google_auth_oauthlib',
        'google.cloud.vision',
        'PyPDF2',
        'docx',
        'cv2',
        'PIL',
        'numpy',
        'requests',
        'dotenv',
        'keyring',
    ]
    
    print("[INFO] Verifying dependencies...")
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
            print(f"[OK] {module}")
        except ImportError:
            print(f"[ERROR] Missing: {module}")
            missing.append(module)
    
    if missing:
        print(f"[WARNING] Missing modules: {missing}")
        print("Installing missing dependencies...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install'] + missing)
            return True
        except subprocess.CalledProcessError:
            return False
    
    return True

def build_executable():
    """Build the executable using PyInstaller"""
    print("[INFO] Building executable...")
    
    try:
        # Clean previous builds
        for dir_name in ['build', 'dist']:
            if os.path.exists(dir_name):
                shutil.rmtree(dir_name)
                print(f"🧹 Cleaned {dir_name}/")
        
        # Build with PyInstaller
        cmd = [sys.executable, '-m', 'PyInstaller', 'contract_intelligence.spec']
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("[SUCCESS] Build completed successfully!")
            
            # Show output location
            if platform.system() == 'Darwin':
                app_path = Path('dist') / 'Contract Intelligence.app'
                if app_path.exists():
                    print(f"📱 macOS app created: {app_path}")
            else:
                exe_path = Path('dist') / 'ContractIntelligence'
                if exe_path.exists():
                    print(f"[SUCCESS] Executable created: {exe_path}")
            
            return True
        else:
            print("[ERROR] Build failed!")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False
            
    except Exception as e:
        print(f"[ERROR] Build error: {e}")
        return False

def create_installer():
    """Create installer packages for different platforms"""
    system = platform.system()
    
    if system == 'Windows':
        create_windows_installer()
    elif system == 'Darwin':
        create_macos_installer()
    else:
        print("[INFO] Linux installer not implemented yet")

def create_windows_installer():
    """Create Windows installer using NSIS or Inno Setup"""
    print("Creating Windows installer...")
    
    # Create Inno Setup script
    iss_content = f'''[Setup]
AppName=Contract Intelligence Platform
AppVersion=1.0.0
AppPublisher=Your Company
AppPublisherURL=https://yourwebsite.com
DefaultDirName={{autopf}}\\ContractIntelligence
DefaultGroupName=Contract Intelligence Platform
OutputDir=installers
OutputBaseFilename=ContractIntelligence-Setup-Windows
Compression=lzma
SolidCompression=yes
WizardStyle=modern
SetupIconFile=icon.ico
UninstallDisplayIcon={{app}}\\ContractIntelligence.exe

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{{cm:CreateDesktopIcon}}"; GroupDescription: "{{cm:AdditionalIcons}}"; Flags: unchecked

[Files]
Source: "dist\\ContractIntelligence\\*"; DestDir: "{{app}}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{{group}}\\Contract Intelligence Platform"; Filename: "{{app}}\\ContractIntelligence.exe"
Name: "{{autodesktop}}\\Contract Intelligence Platform"; Filename: "{{app}}\\ContractIntelligence.exe"; Tasks: desktopicon

[Run]
Filename: "{{app}}\\ContractIntelligence.exe"; Description: "{{cm:LaunchProgram,Contract Intelligence Platform}}"; Flags: nowait postinstall skipifsilent
'''
    
    os.makedirs('installers', exist_ok=True)
    with open('contract_intelligence.iss', 'w') as f:
        f.write(iss_content)
    
    print("[SUCCESS] Windows installer script created (contract_intelligence.iss)")
    print("[INFO] To build installer, run: iscc contract_intelligence.iss")

def create_macos_installer():
    """Create macOS installer"""
    print("Creating macOS installer...")
    
    # Create a simple script to package the .app
    script_content = '''#!/bin/bash
# Create macOS installer package

APP_NAME="Contract Intelligence.app"
PKG_NAME="ContractIntelligence-Setup-macOS.pkg"
IDENTIFIER="com.yourcompany.contractintelligence"

if [ -d "dist/$APP_NAME" ]; then
    echo "Creating macOS installer..."
    pkgbuild --root dist --identifier $IDENTIFIER --version 1.0.0 --install-location /Applications installers/$PKG_NAME
    echo "[SUCCESS] macOS installer created: installers/$PKG_NAME"
else
    echo "[ERROR] App bundle not found at dist/$APP_NAME"
    exit 1
fi
'''
    
    os.makedirs('installers', exist_ok=True)
    with open('create_macos_installer.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('create_macos_installer.sh', 0o755)
    print("[SUCCESS] macOS installer script created (create_macos_installer.sh)")

def main():
    print("Contract Intelligence Platform - Desktop Build Script")
    print("=" * 60)
    
    # Check if we're in the right directory
    if not os.path.exists('streamlit_app.py'):
        print("[ERROR] streamlit_app.py not found. Run this script from the ragflow directory.")
        return
    
    # Install dependencies
    if not install_dependencies():
        print("[ERROR] Failed to install dependencies")
        return
    
    # Create PyInstaller spec
    create_pyinstaller_spec()
    
    # Build executable
    if not build_executable():
        print("[ERROR] Build failed")
        return
    
    # Create installer
    create_installer()
    
    print("\n[SUCCESS] Build process completed!")
    print("\n[INFO] Next steps:")
    print("1. Test the executable in dist/ directory")
    print("2. Run installer creation scripts if needed")
    print("3. Distribute to users")

if __name__ == "__main__":
    main()
