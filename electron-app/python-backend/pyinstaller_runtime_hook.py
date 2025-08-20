# python-backend/pyinstaller_runtime_hook.py

import os
import sys
import importlib.util

# Handle PyInstaller runtime environment
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Running in PyInstaller bundle
    print("[INFO] Running in PyInstaller bundle")
    bundle_dir = sys._MEIPASS
    
    # Add bundle directory to path
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)
    
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
else:
    # Running in normal Python environment
    print("[INFO] Running in normal Python environment")
