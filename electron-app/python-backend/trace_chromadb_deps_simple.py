#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ChromaDB Dependency Tracer (Simple Version)

This script captures all modules loaded during ChromaDB initialization
by comparing sys.modules before and after import.
"""

import sys
import os
import time
import traceback
import importlib.util

def get_module_info(module):
    """Get information about a module"""
    info = {
        'file': getattr(module, '__file__', 'built-in'),
        'version': getattr(module, '__version__', 'unknown'),
        'package': getattr(module, '__package__', 'unknown')
    }
    return info

def check_module_exists(name):
    """Check if a module can be imported without actually importing it"""
    return importlib.util.find_spec(name) is not None

def print_section(title):
    """Print a section header"""
    print(f"\n{'-' * 80}")
    print(f"{title}")
    print(f"{'-' * 80}")

print("\n=== CHROMADB DEPENDENCY TRACER (SIMPLE VERSION) ===\n")

# Check for required packages before starting
print("Checking for required packages...")
required_packages = [
    'chromadb', 'openai', 'numpy', 'onnxruntime', 'protobuf', 
    'tokenizers', 'sentence_transformers', 'sqlite3'
]

for package in required_packages:
    exists = check_module_exists(package)
    print(f"  {package:<25}: {'Available' if exists else 'Not found'}")

# Capture initial modules
print("\nCapturing initial module state...")
initial_modules = set(sys.modules.keys())
print(f"  Initial modules count: {len(initial_modules)}")

# Import and initialize ChromaDB
print_section("INITIALIZING CHROMADB")
start_time = time.time()

try:
    # First, just import ChromaDB
    print("Importing chromadb module...")
    import chromadb
    print(f"ChromaDB version: {getattr(chromadb, '__version__', 'unknown')}")
    print(f"ChromaDB location: {getattr(chromadb, '__file__', 'unknown')}")
    
    # Capture modules after import
    after_import_modules = set(sys.modules.keys())
    import_only_modules = after_import_modules - initial_modules
    print(f"Modules loaded during import: {len(import_only_modules)}")
    
    # Try to initialize with OpenAI embeddings
    print("\nInitializing ChromaDB with OpenAI embeddings...")
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
        
    except ImportError as e:
        print(f"Failed to import required modules for OpenAI embeddings: {e}")
        
        # Fall back to default initialization
        print("\nFalling back to default initialization...")
        client = chromadb.Client()
        print("ChromaDB initialized with default settings")
    
    # Try to create a collection
    print("\nAttempting to create a test collection...")
    try:
        collection = client.create_collection("test_collection")
        print("Test collection created successfully")
    except Exception as e:
        print(f"Failed to create collection: {e}")
    
except ImportError as e:
    print(f"Failed to import ChromaDB: {e}")
    print(traceback.format_exc())
except Exception as e:
    print(f"Error initializing ChromaDB: {e}")
    print(traceback.format_exc())

end_time = time.time()
initialization_time = end_time - start_time
print(f"\nChromaDB initialization completed in {initialization_time:.2f} seconds")

# Capture final modules
final_modules = set(sys.modules.keys())
new_modules = final_modules - initial_modules
print(f"Total new modules loaded: {len(new_modules)}")

# Analyze modules
print_section("MODULE ANALYSIS")

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

for name in sorted(new_modules):
    module = sys.modules.get(name)
    if not module:
        continue
        
    info = get_module_info(module)
    
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
            file_info = info['file'] if info['file'] != 'built-in' else 'built-in'
            version_info = f"v{info['version']}" if info['version'] != 'unknown' else ''
            print(f"  {name:<40} {version_info:<10} {file_info}")

# Check for specific important modules
print_section("CRITICAL DEPENDENCY CHECK")
critical_modules = [
    'chromadb', 'onnxruntime', 'numpy', 'protobuf', 'tokenizers', 
    'sentence_transformers', 'sqlite3', 'openai', 'tiktoken',
    'chromadb.api', 'chromadb.config', 'chromadb.utils.embedding_functions'
]

for module_name in critical_modules:
    is_loaded = module_name in sys.modules
    print(f"{module_name:<40}: {'✓ Loaded' if is_loaded else '✗ Not loaded'}")
    
    if is_loaded:
        module = sys.modules[module_name]
        file_path = getattr(module, '__file__', 'built-in')
        version = getattr(module, '__version__', 'unknown')
        print(f"  - Location: {file_path}")
        if version != 'unknown':
            print(f"  - Version: {version}")

# Print statistics
print_section("STATISTICS")
print(f"Total modules loaded: {len(new_modules)}")
print(f"ChromaDB modules: {len(categories['chromadb'])}")
print(f"ONNX related modules: {len(categories['onnx'])}")
print(f"NumPy related modules: {len(categories['numpy'])}")
print(f"SQLite related modules: {len(categories['sqlite'])}")
print(f"Embedding related modules: {len(categories['embedding'])}")
print(f"Standard library modules: {len(categories['standard_lib'])}")
print(f"Other modules: {len(categories['other'])}")
print(f"\nTotal initialization time: {initialization_time:.2f} seconds")

# Print system information
print_section("SYSTEM INFORMATION")
print(f"Python version: {sys.version}")
print(f"Platform: {sys.platform}")
print(f"Executable: {sys.executable}")
print(f"Working directory: {os.getcwd()}")

try:
    # Try to get more detailed OS info
    import platform
    print(f"OS: {platform.platform()}")
    print(f"Architecture: {platform.architecture()}")
    if sys.platform == 'darwin':  # macOS
        print(f"macOS version: {platform.mac_ver()}")
except ImportError:
    pass

print("\n=== END OF CHROMADB DEPENDENCY TRACE ===\n")
