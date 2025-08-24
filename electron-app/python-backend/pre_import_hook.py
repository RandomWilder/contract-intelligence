#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-import hook to handle dependency issues for PyInstaller builds
This file is imported before any other modules to set up necessary monkey patches
"""

import sys
import types

print("[INFO] Running pre-import hook...")

# === SENTENCE TRANSFORMERS PATCHING ===
# Create stub module to replace sentence_transformers
stub_module = types.ModuleType('sentence_transformers')
stub_module.__path__ = []
sys.modules['sentence_transformers'] = stub_module

# Create necessary classes and functions inside the stub module
class DummyEmbeddingFunction:
    def __init__(self, *args, **kwargs):
        pass
    
    def __call__(self, texts):
        raise NotImplementedError("Dummy embedding function - not available in this build")

# Create the specific class that ChromaDB is looking for
class ONNXMiniLM_L6_V2:
    def __init__(self, *args, **kwargs):
        print("[INFO] Dummy ONNXMiniLM_L6_V2 initialized (stub)")
    
    def encode(self, texts, **kwargs):
        # Return a dummy embedding - must be compatible with what ChromaDB expects
        raise NotImplementedError("Dummy encoding function - not available in this build")

# Create proper submodules
models_module = types.ModuleType('sentence_transformers.models')
sys.modules['sentence_transformers.models'] = models_module

# Add the class to both the main module and the models module for maximum compatibility
stub_module.ONNXMiniLM_L6_V2 = ONNXMiniLM_L6_V2
models_module.ONNXMiniLM_L6_V2 = ONNXMiniLM_L6_V2
stub_module.models = models_module

# === GOOGLE API PATCHING ===
# Ensure Google API modules are properly loaded and initialized
try:
    print("[INFO] Setting up Google API modules...")
    
    # Create base modules
    google_module = types.ModuleType('google') if 'google' not in sys.modules else sys.modules['google']
    google_module.__path__ = []
    sys.modules['google'] = google_module
    
    # Create submodules
    oauth2_module = types.ModuleType('google.oauth2')
    oauth2_module.__path__ = []
    sys.modules['google.oauth2'] = oauth2_module
    
    auth_module = types.ModuleType('google.auth')
    auth_module.__path__ = []
    sys.modules['google.auth'] = auth_module
    
    transport_module = types.ModuleType('google.auth.transport')
    transport_module.__path__ = []
    sys.modules['google.auth.transport'] = transport_module
    
    requests_module = types.ModuleType('google.auth.transport.requests')
    requests_module.__path__ = []
    sys.modules['google.auth.transport.requests'] = requests_module
    
    # Set up the module hierarchy
    google_module.oauth2 = oauth2_module
    google_module.auth = auth_module
    auth_module.transport = transport_module
    transport_module.requests = requests_module
    
    # Try to import the real modules now that the hierarchy is established
    try:
        import google.oauth2.service_account
        import google.auth.transport.requests
        import googleapiclient.discovery
        import googleapiclient.errors
        import googleapiclient.http
        print("[INFO] Successfully loaded Google API modules")
    except ImportError as e:
        print(f"[WARNING] Some Google API modules could not be loaded: {e}")
        
        # Create stub classes if imports failed
        class DummyCredentials:
            @staticmethod
            def from_service_account_file(*args, **kwargs):
                raise ImportError("Google API credentials are not available in this build.")
        
        # Add stub to the modules
        if not hasattr(oauth2_module, 'service_account'):
            service_account_module = types.ModuleType('google.oauth2.service_account')
            service_account_module.__path__ = []
            sys.modules['google.oauth2.service_account'] = service_account_module
            service_account_module.Credentials = DummyCredentials
            oauth2_module.service_account = service_account_module
            
except Exception as e:
    print(f"[WARNING] Failed to set up Google API modules: {e}")

# === CHROMADB PATCHING ===
# Also directly patch the chromadb embedding function to bypass sentence_transformers
try:
    # This will pre-load the module where the error occurs to patch it directly
    import importlib.util
    import os
    
    # Define a patching function that will run when chromadb imports
    def patch_chromadb():
        try:
            # Try to directly modify chromadb's embedding functions
            import chromadb.utils.embedding_functions
            
            # Replace DefaultEmbeddingFunction to avoid ONNXMiniLM_L6_V2
            original_default = getattr(chromadb.utils.embedding_functions, 'DefaultEmbeddingFunction', None)
            if original_default:
                # Replace it with a function that raises an informative error
                def patched_default_embedding(*args, **kwargs):
                    raise ImportError(
                        "DefaultEmbeddingFunction is not available in this build. "
                        "Please use OpenAIEmbeddingFunction instead."
                    )
                chromadb.utils.embedding_functions.DefaultEmbeddingFunction = patched_default_embedding
                print("[INFO] Successfully patched chromadb.utils.embedding_functions.DefaultEmbeddingFunction")
        except Exception as e:
            print(f"[WARNING] Failed to patch chromadb: {e}")
    
    # Register our patching function to run when chromadb is imported
    import builtins
    original_import = builtins.__import__
    
    def patched_import(name, *args, **kwargs):
        module = original_import(name, *args, **kwargs)
        if name == 'chromadb' or name.startswith('chromadb.'):
            patch_chromadb()
        return module
    
    builtins.__import__ = patched_import
    print("[INFO] Installed import hook for chromadb")
    
except Exception as e:
    print(f"[WARNING] Could not set up chromadb patching: {e}")

print("[INFO] All pre-import hooks completed successfully")