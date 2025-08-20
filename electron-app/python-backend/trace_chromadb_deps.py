#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChromaDB Dependency Tracer

This script traces all dependencies loaded by ChromaDB during initialization
and prints them in a structured format.
"""

import sys
import os
import importlib
import traceback
import time
from types import ModuleType

# Dictionary to store loaded modules
loaded_modules = {}
original_import = __import__

# Track import order
import_order = []

# Set up a custom import hook to track imports
def custom_import(name, *args, **kwargs):
    """Custom import function to track module imports"""
    module = original_import(name, *args, **kwargs)
    
    if name not in loaded_modules:
        loaded_modules[name] = {
            'order': len(loaded_modules),
            'file': getattr(module, '__file__', 'built-in'),
            'version': getattr(module, '__version__', 'unknown'),
            'submodules': []
        }
        import_order.append(name)
        
        # Track when this was imported
        loaded_modules[name]['timestamp'] = time.time()
        
    return module

# Replace the built-in import function with our custom one
sys.meta_path.insert(0, type('CustomImporter', (), {
    'find_spec': lambda self, fullname, path, target=None: None,
    'exec_module': lambda self, module: None,
    'create_module': lambda self, spec: None,
    '__import__': custom_import
}))

# Redirect stdout to capture ChromaDB's initialization output
import io
from contextlib import redirect_stdout

def try_import(module_name):
    """Try to import a module and return success status"""
    try:
        importlib.import_module(module_name)
        return True
    except ImportError:
        return False
    except Exception:
        return False

print("\n=== CHROMADB DEPENDENCY TRACER ===\n")

# First, import ChromaDB and initialize it
print("Importing and initializing ChromaDB...")
start_time = time.time()

# Capture ChromaDB's initialization output
output_buffer = io.StringIO()
with redirect_stdout(output_buffer):
    try:
        import chromadb
        print(f"ChromaDB version: {getattr(chromadb, '__version__', 'unknown')}")
        
        # Try to initialize with OpenAI embeddings
        try:
            from chromadb.utils import embedding_functions
            import openai
            
            # Create an OpenAI embedding function (with dummy key)
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key="dummy-key-for-testing",
                model_name="text-embedding-ada-002"
            )
            
            # Initialize ChromaDB with OpenAI embeddings
            client = chromadb.Client(embedding_function=openai_ef)
            print("ChromaDB initialized with OpenAI embeddings")
            
        except ImportError:
            # Fall back to default initialization
            client = chromadb.Client()
            print("ChromaDB initialized with default settings")
            
    except ImportError as e:
        print(f"Failed to import ChromaDB: {e}")
    except Exception as e:
        print(f"Error initializing ChromaDB: {e}")
        print(traceback.format_exc())

end_time = time.time()
initialization_time = end_time - start_time

print(f"\nChromaDB initialization completed in {initialization_time:.2f} seconds\n")

# Analyze and display the loaded modules
print("\n=== MODULES LOADED DURING CHROMADB INITIALIZATION ===\n")

# Sort modules by import order
sorted_modules = sorted([(name, info) for name, info in loaded_modules.items()], 
                        key=lambda x: x[1]['order'])

# Group modules by categories
categories = {
    'chromadb': [],
    'onnx': [],
    'numpy': [],
    'sqlite': [],
    'embedding': [],
    'standard_lib': [],
    'other': []
}

for name, info in sorted_modules:
    if name.startswith('chromadb'):
        categories['chromadb'].append((name, info))
    elif name.startswith('onnx') or 'onnx' in name:
        categories['onnx'].append((name, info))
    elif name.startswith('numpy') or name == 'np':
        categories['numpy'].append((name, info))
    elif 'sqlite' in name or name.startswith('sqlite'):
        categories['sqlite'].append((name, info))
    elif any(term in name for term in ['openai', 'embedding', 'tokenizer', 'sentence_transformers']):
        categories['embedding'].append((name, info))
    elif not info['file'] or info['file'] == 'built-in' or '<frozen' in str(info['file']):
        categories['standard_lib'].append((name, info))
    else:
        categories['other'].append((name, info))

# Print modules by category
for category, modules in categories.items():
    if modules:
        print(f"\n--- {category.upper()} MODULES ({len(modules)}) ---")
        for name, info in modules:
            file_info = info['file'] if info['file'] else 'built-in'
            version_info = f"v{info['version']}" if info['version'] != 'unknown' else ''
            print(f"  {name:<30} {version_info:<10} {file_info}")

# Print import timeline
print("\n\n=== IMPORT TIMELINE ===\n")
for i, name in enumerate(import_order):
    info = loaded_modules[name]
    time_offset = info['timestamp'] - start_time
    print(f"{i+1:3d}. [{time_offset:.4f}s] {name}")

# Print statistics
print("\n\n=== STATISTICS ===\n")
print(f"Total modules loaded: {len(loaded_modules)}")
print(f"ChromaDB modules: {len(categories['chromadb'])}")
print(f"ONNX related modules: {len(categories['onnx'])}")
print(f"NumPy related modules: {len(categories['numpy'])}")
print(f"SQLite related modules: {len(categories['sqlite'])}")
print(f"Embedding related modules: {len(categories['embedding'])}")
print(f"Standard library modules: {len(categories['standard_lib'])}")
print(f"Other modules: {len(categories['other'])}")
print(f"\nTotal initialization time: {initialization_time:.2f} seconds")

# Print system information
print("\n\n=== SYSTEM INFORMATION ===\n")
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")
print(f"Executable: {sys.executable}")

print("\n=== END OF CHROMADB DEPENDENCY TRACE ===\n")
