#!/usr/bin/env python3
"""
Setup script to configure GitHub Actions for cross-platform builds
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"🔧 {description}...")
    try:
        result = subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed")
        if result.stdout:
            print(f"   Output: {result.stdout.strip()}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed")
        print(f"   Error: {e.stderr}")
        return False

def main():
    print("🚀 Setting up GitHub Actions for Cross-Platform Desktop Builds")
    print("=" * 70)
    
    # Check if we're in a git repository
    if not Path('.git').exists():
        print("❌ This doesn't appear to be a git repository")
        print("   Make sure you're in the correct directory")
        return False
    
    # Check if the workflow file exists
    workflow_file = Path('.github/workflows/build-desktop.yml')
    if not workflow_file.exists():
        print("❌ GitHub Actions workflow file not found")
        print(f"   Expected: {workflow_file}")
        return False
    
    print(f"✅ Found workflow file: {workflow_file}")
    
    # Add and commit the workflow
    commands = [
        ("git add .github/", "Adding GitHub Actions workflow"),
        ("git add build_desktop.py", "Adding updated build script"),
        ("git add setup_github_actions.py", "Adding setup script"),
        ('git commit -m "Add GitHub Actions workflow for cross-platform desktop builds\n\n- Automated Windows .exe and macOS .app builds\n- Artifacts uploaded for easy distribution\n- Supports both push and release triggers"', "Committing changes"),
        ("git push origin main", "Pushing to GitHub")
    ]
    
    for cmd, desc in commands:
        if not run_command(cmd, desc):
            return False
    
    print("\n🎉 GitHub Actions setup completed!")
    print("\n📋 What happens next:")
    print("1. 🔄 GitHub Actions will automatically start building")
    print("2. ⏱️  Build takes ~10-15 minutes (both Windows & macOS)")
    print("3. 📦 Artifacts will be available for download")
    print("4. 🚀 You can distribute both .exe and .app files!")
    
    print("\n🔗 Check your build status at:")
    print("   https://github.com/RandomWilder/contract-intelligence/actions")
    
    print("\n💡 How to download built applications:")
    print("1. Go to GitHub Actions tab in your repository")
    print("2. Click on the latest successful workflow run")
    print("3. Scroll down to 'Artifacts' section")
    print("4. Download 'ContractIntelligence-Windows.zip' and 'ContractIntelligence-macOS.zip'")
    print("5. Extract and distribute to users!")
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
    
    print("\n✨ Ready for cross-platform distribution! ✨")
