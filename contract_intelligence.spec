# -*- mode: python ; coding: utf-8 -*-

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
        version='1.0.0',
        info_plist={
            'NSHighResolutionCapable': 'True',
            'CFBundleDisplayName': 'Contract Intelligence Platform',
            'CFBundleExecutable': 'ContractIntelligence',
            'CFBundleName': 'Contract Intelligence',
            'CFBundleVersion': '1.0.0',
            'CFBundleShortVersionString': '1.0.0',
        },
    )
