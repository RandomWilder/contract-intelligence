# python-backend/pyinstaller_runtime_hook.py

import os
import sys
import importlib.util
import tempfile

# Handle PyInstaller runtime environment
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Running in PyInstaller bundle
    print("[INFO] Running in PyInstaller bundle")
    bundle_dir = sys._MEIPASS
    
    # Add bundle directory to path
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)
    
    # Set up ChromaDB data directory in a writable location
    app_data_dir = os.path.join(os.path.dirname(sys.executable), 'app_data') if getattr(sys, 'frozen', False) else './app_data'
    os.environ['CHROMADB_DIR'] = os.path.join(app_data_dir, 'chroma_db')
    print(f"[INFO] Setting ChromaDB directory to: {os.environ['CHROMADB_DIR']}")
    os.makedirs(os.environ['CHROMADB_DIR'], exist_ok=True)
    
    # Ensure contract_intelligence.py is accessible
    contract_intelligence_path = os.path.join(bundle_dir, 'contract_intelligence.py')
    if os.path.exists(contract_intelligence_path):
        print(f"[INFO] Found contract_intelligence.py at: {contract_intelligence_path}")
        
        # Create a module spec and import the module
        try:
            spec = importlib.util.spec_from_file_location('contract_intelligence', contract_intelligence_path)
            contract_intelligence_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(contract_intelligence_module)
            sys.modules['contract_intelligence'] = contract_intelligence_module
            print("[SUCCESS] Successfully loaded contract_intelligence module")
        except Exception as e:
            print(f"[ERROR] Failed to import contract_intelligence module: {e}")
    else:
        print(f"[ERROR] contract_intelligence.py not found at: {contract_intelligence_path}")
        print(f"[DEBUG] Bundle directory contents: {os.listdir(bundle_dir)}")
        
    # Print out key environment information for debugging
    print(f"[DEBUG] Python executable: {sys.executable}")
    print(f"[DEBUG] Working directory: {os.getcwd()}")
    print(f"[DEBUG] Sys path: {sys.path}")
    print(f"[DEBUG] Bundle directory: {bundle_dir}")
    print(f"[DEBUG] App data directory: {app_data_dir}")
    print(f"[DEBUG] ChromaDB directory: {os.environ['CHROMADB_DIR']}")
    
    # Ensure ChromaDB can find its dependencies
    try:
        import chromadb
        print("[SUCCESS] ChromaDB module imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import ChromaDB: {e}")
else:
    # Running in normal Python environment
    print("[INFO] Running in normal Python environment")
