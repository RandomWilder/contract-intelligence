# -*- mode: python ; coding: utf-8 -*-

import os
import sys
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

block_cipher = None

# Collect all submodules from key packages
openai_imports = collect_submodules('openai')
chromadb_imports = collect_submodules('chromadb')
fastapi_imports = collect_submodules('fastapi')
starlette_imports = collect_submodules('starlette')
pydantic_imports = collect_submodules('pydantic')  # Required by FastAPI
tokenizers_imports = collect_submodules('tokenizers')  # Required by ChromaDB's embedding function
sentence_transformers_imports = collect_submodules('sentence_transformers')  # Often used with tokenizers

# Critical ChromaDB dependencies
onnxruntime_imports = []
try:
    onnxruntime_imports = collect_submodules('onnxruntime')  # Required by ChromaDB even if not used directly
except ImportError:
    print("WARNING: onnxruntime not found, but will be installed during build")
protobuf_imports = collect_submodules('protobuf')  # Required for ChromaDB's data serialization
tiktoken_imports = collect_submodules('tiktoken')  # Required for OpenAI embeddings

# Collect binary dependencies
from PyInstaller.utils.hooks import collect_dynamic_libs
tokenizers_binaries = collect_dynamic_libs('tokenizers')
onnxruntime_binaries = []
try:
    onnxruntime_binaries = collect_dynamic_libs('onnxruntime')
except ImportError:
    print("WARNING: onnxruntime binaries not found, but will be installed during build")

# Collect data files
openai_datas = collect_data_files('openai')
chromadb_datas = collect_data_files('chromadb')
tiktoken_datas = collect_data_files('tiktoken')
tokenizers_datas = collect_data_files('tokenizers')
sentence_transformers_datas = collect_data_files('sentence_transformers')

# Additional required data files
onnxruntime_datas = []
try:
    onnxruntime_datas = collect_data_files('onnxruntime')
except ImportError:
    print("WARNING: onnxruntime not found for data files collection")
protobuf_datas = collect_data_files('protobuf')

# Add contract_intelligence.py as a data file
contract_intelligence_file = [
    ('python-backend/contract_intelligence.py', '.'),
]

a = Analysis(
    ['api_server_minimal.py'],
    pathex=['.'],
    binaries=tokenizers_binaries + onnxruntime_binaries,
    datas=[
        ('requirements.txt', '.'),
        ('contract_intelligence.py', '.'),  # Explicitly include contract_intelligence.py
    ] + openai_datas + chromadb_datas + tiktoken_datas + tokenizers_datas + sentence_transformers_datas + onnxruntime_datas + protobuf_datas,
    hiddenimports=[
        # Standard library modules
        'tempfile', 'shutil', 'logging', 're', 'codecs', 'contextlib', 'pathlib', 'typing',
        'sqlite3', 'pickle', 'json', 'dataclasses', 'datetime', 'enum', 'functools',
        'io', 'os', 'sys', 'time', 'uuid', 'warnings', 'zlib', 'base64',
        
        # Contract Intelligence specific imports
        'contract_intelligence',
        'dataclasses',
        'datetime',
        're.sub',
        're.match',
        're.search',
        're.findall',
        
        # FastAPI and Uvicorn
        'uvicorn',
        'uvicorn.lifespan',
        'uvicorn.lifespan.on',
        'uvicorn.protocols',
        'uvicorn.protocols.websockets',
        'uvicorn.protocols.websockets.auto',
        'uvicorn.protocols.http',
        'uvicorn.protocols.http.auto',
        'uvicorn.protocols.http.h11_impl',
        'uvicorn.protocols.http.httptools_impl',
        'uvicorn.loops',
        'uvicorn.loops.auto',
        'fastapi',
        'fastapi.applications',
        'fastapi.routing',
        'fastapi.middleware',
        'fastapi.middleware.cors',
        'fastapi.responses',
        'fastapi.encoders',
        'fastapi.exceptions',
        'fastapi.params',
        'fastapi.dependencies',
        'fastapi.security',
        'starlette',
        'starlette.applications',
        'starlette.routing',
        'starlette.middleware',
        'starlette.middleware.cors',
        'starlette.responses',
        'starlette.requests',
        'starlette.datastructures',
        'pydantic',
        'pydantic.fields',
        'pydantic.main',
        'pydantic.error_wrappers',
        'pydantic.validators',
        'pydantic.typing',
        'pydantic.utils',
        'pydantic.errors',
        'pydantic.json',
        
        # AI and ML
        'openai',
        'openai.api_resources',
        'openai.api_resources.abstract',
        'openai.api_resources.chat_completion',
        'openai.api_resources.completion',
        'openai.api_resources.embedding',
        'openai._models',
        'openai._types',
        'openai._client',
        'openai._streaming',
        'openai._base_client',
        'openai._utils',
        'openai._legacy_response',
        'tiktoken',
        
        # Tokenizers and embeddings
        'tokenizers',
        'tokenizers.processors',
        'tokenizers.decoders',
        'tokenizers.models',
        'tokenizers.normalizers',
        'tokenizers.pre_tokenizers',
        'tokenizers.trainers',
        'sentence_transformers',
        'sentence_transformers.models',
        'sentence_transformers.util',
        
        # ChromaDB
        'chromadb',
        'chromadb.config',
        'chromadb.api',
        'chromadb.api.models',
        'chromadb.api.types',
        'chromadb.db',
        'chromadb.utils',
        'chromadb.utils.embedding_functions',
        'chromadb.utils.embedding_functions.openai_embedding_function',
        'chromadb.telemetry',
        'chromadb.segment',
        'chromadb.segment.impl',
        'onnxruntime',
        
        # Google Cloud and Auth
        'google.auth',
        'google.auth.transport',
        'google.auth.transport.requests',
        'google_auth_oauthlib',
        'google_auth_oauthlib.flow',
        'google.oauth2.credentials',
        'google.cloud',
        'google.cloud.vision',
        'requests',
        'requests.auth',
        'requests.adapters',
        'requests.models',
        'requests.sessions',
        'requests.utils',
        
        # Document processing
        'PIL',
        'PIL.Image',
        'PyPDF2',
        'PyPDF2.pdf',
        'PyPDF2.generic',
        'PyPDF2._page',
        'PyPDF2._reader',
        'docx',
        'docx.document',
        'docx.oxml',
        'docx.parts',
        'docx.text',
        'fitz',
        
        # Data processing
        'pandas',
        'numpy',
        
        # Environment and config
        'dotenv',
        
        # Windows runtime dependencies
        'win32api', 'win32con', 'win32gui', 'win32process', 'win32security', 'win32service', 
        'win32serviceutil', 'win32event', 'win32file',
        
        # Additional system modules
        'multiprocessing', 'multiprocessing.spawn', 'multiprocessing.util',
        'concurrent', 'concurrent.futures', 'concurrent.futures.thread',
        'asyncio', 'asyncio.events', 'asyncio.selector_events', 'asyncio.windows_events',
        
        # HTTP and networking
        'http', 'http.client', 'urllib', 'urllib.parse', 'urllib.request', 'urllib.error',
        'ssl', 'socket', 'email', 'email.message', 'email.parser', 'email.policy',
        
        # JSON and serialization
        'json.decoder', 'json.encoder', 'marshal', 'pickle', 'copyreg',
        
        # Critical ChromaDB dependencies
        'onnxruntime',
        'onnxruntime.capi',
        'onnxruntime.capi.onnxruntime_pybind11_state',
        'protobuf',
        'protobuf.internal',
        'protobuf.descriptor',
        'protobuf.message',
        'protobuf.reflection',
        'protobuf.json_format'
    ] + openai_imports + chromadb_imports + fastapi_imports + starlette_imports + pydantic_imports + tokenizers_imports + sentence_transformers_imports + onnxruntime_imports + protobuf_imports + tiktoken_imports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[
        # Add a runtime hook to handle PyInstaller environment
        'pyinstaller_runtime_hook.py'
    ],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='api_server',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=True,  # Enable argv emulation for macOS compatibility
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
