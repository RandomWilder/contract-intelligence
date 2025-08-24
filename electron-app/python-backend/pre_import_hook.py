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

# Add the class to the stub module
stub_module.ONNXMiniLM_L6_V2 = ONNXMiniLM_L6_V2
stub_module.models = types.ModuleType('sentence_transformers.models')

print("[INFO] Successfully stubbed sentence_transformers dependencies")
