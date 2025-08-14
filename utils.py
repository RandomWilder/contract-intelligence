#!/usr/bin/env python3
"""
Utility functions for the Contract Intelligence Platform
"""
import os

def get_google_credentials_path():
    """Get the path to Google credentials file in user's home directory"""
    home_dir = os.path.expanduser("~")
    credentials_dir = os.path.join(home_dir, ".contract_intelligence")
    os.makedirs(credentials_dir, exist_ok=True)
    new_path = os.path.join(credentials_dir, "google_credentials.json")
    
    # Migration: Check if old file exists and move it
    old_path = "google_credentials.json"
    if os.path.exists(old_path) and not os.path.exists(new_path):
        import shutil
        try:
            shutil.move(old_path, new_path)
            print(f"✅ Moved Google credentials to: {new_path}")
        except Exception as e:
            print(f"⚠️ Could not move credentials file: {e}")
    
    return new_path
