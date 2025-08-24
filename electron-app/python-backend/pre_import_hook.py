#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pre-import hook to handle dependency issues for PyInstaller builds
This file is imported before any other modules to set up necessary monkey patches
"""

import sys
import types

print("[INFO] Running pre-import hook...")

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

print("[INFO] Successfully stubbed sentence_transformers dependencies")