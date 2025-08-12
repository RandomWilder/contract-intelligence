#!/usr/bin/env python3
"""
Development Workflow Script for Contract Intelligence Platform
Streamlines the development → build → distribute process
"""

import os
import subprocess
import sys
import argparse
from pathlib import Path
from datetime import datetime

class DevWorkflow:
    def __init__(self):
        self.project_root = Path(__file__).parent
        self.git_repo = "https://github.com/RandomWilder/contract-intelligence.git"
        
    def run_command(self, cmd, description, check=True):
        """Run a command with proper error handling"""
        print(f"🔧 {description}...")
        try:
            if isinstance(cmd, str):
                result = subprocess.run(cmd, shell=True, check=check, capture_output=True, text=True)
            else:
                result = subprocess.run(cmd, check=check, capture_output=True, text=True)
            
            if result.stdout:
                print(f"   ✅ {result.stdout.strip()}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"   ❌ Error: {e.stderr}")
            return False
    
    def test_local(self):
        """Test the application locally"""
        print("🧪 Testing Local Application")
        print("=" * 50)
        
        # Test imports
        test_imports = [
            "import streamlit",
            "import openai", 
            "import chromadb",
            "from desktop_launcher import ContractIntelligenceSetup",
            "from telemetry_client import TelemetryClient",
            "print('✅ All imports successful')"
        ]
        
        for test in test_imports:
            if not self.run_command(f'python -c "{test}"', f"Testing: {test.split()[1] if len(test.split()) > 1 else test}"):
                return False
        
        # Test desktop launcher
        print("\n💻 Testing Desktop Launcher...")
        if not self.run_command("python desktop_launcher.py --help", "Desktop launcher help", check=False):
            print("   ⚠️ Desktop launcher test skipped (GUI required)")
        
        return True
    
    def build_local(self):
        """Build application locally for testing"""
        print("🔨 Building Application Locally")
        print("=" * 50)
        
        # Clean previous builds
        build_dirs = ["build", "dist", "*.spec"]
        for dir_pattern in build_dirs:
            self.run_command(f"rm -rf {dir_pattern}", f"Cleaning {dir_pattern}", check=False)
        
        # Build with PyInstaller
        build_cmd = [
            "python", "-m", "PyInstaller",
            "--name=ContractIntelligence-Local",
            "--onedir",
            "--windowed" if sys.platform == "win32" else "--windowed",
            "--distpath=dist",
            "--workpath=build",
            "--add-data=streamlit_app.py:." if sys.platform != "win32" else "--add-data=streamlit_app.py;.",
            "--add-data=local_rag_app.py:." if sys.platform != "win32" else "--add-data=local_rag_app.py;.",
            "--hidden-import=streamlit",
            "--hidden-import=openai",
            "--hidden-import=chromadb",
            "--collect-all=streamlit",
            "desktop_launcher.py"
        ]
        
        return self.run_command(build_cmd, "Building with PyInstaller")
    
    def commit_and_push(self, message=None):
        """Commit changes and push to GitHub"""
        print("📤 Committing and Pushing Changes")
        print("=" * 50)
        
        # Check git status
        if not self.run_command("git status --porcelain", "Checking git status"):
            return False
        
        # Add changes
        if not self.run_command("git add .", "Adding changes"):
            return False
        
        # Commit with message
        if not message:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            message = f"Update Contract Intelligence Platform - {timestamp}"
        
        commit_cmd = f'git commit -m "{message}"'
        if not self.run_command(commit_cmd, "Committing changes"):
            print("   ℹ️ No changes to commit")
            return True
        
        # Push to GitHub
        return self.run_command("git push origin main", "Pushing to GitHub")
    
    def trigger_build(self):
        """Trigger GitHub Actions build"""
        print("🚀 Triggering GitHub Actions Build")
        print("=" * 50)
        
        # GitHub CLI method (if available)
        if self.run_command("gh --version", "Checking GitHub CLI", check=False):
            return self.run_command("gh workflow run build-release.yml", "Triggering build via GitHub CLI")
        else:
            print("   ℹ️ GitHub CLI not available")
            print("   🌐 Trigger build manually at:")
            print(f"   {self.git_repo.replace('.git', '')}/actions")
            return True
    
    def full_workflow(self, commit_message=None):
        """Run the complete development workflow"""
        print("🚀 Contract Intelligence Platform - Full Development Workflow")
        print("=" * 70)
        
        steps = [
            ("Testing locally", self.test_local),
            ("Building locally", self.build_local),
            ("Committing and pushing", lambda: self.commit_and_push(commit_message)),
            ("Triggering GitHub build", self.trigger_build)
        ]
        
        for step_name, step_func in steps:
            print(f"\n📋 Step: {step_name}")
            if not step_func():
                print(f"❌ Failed at: {step_name}")
                return False
            print(f"✅ Completed: {step_name}")
        
        print("\n🎉 Full Workflow Completed Successfully!")
        print("\n📋 Next Steps:")
        print("1. 🔄 Check GitHub Actions: https://github.com/RandomWilder/contract-intelligence/actions")
        print("2. ⏱️ Wait ~10-15 minutes for builds to complete")
        print("3. 📦 Download artifacts from successful workflow run")
        print("4. 🚀 Distribute to users!")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Contract Intelligence Platform Development Workflow")
    parser.add_argument("--test", action="store_true", help="Test application locally")
    parser.add_argument("--build", action="store_true", help="Build application locally")
    parser.add_argument("--push", action="store_true", help="Commit and push changes")
    parser.add_argument("--trigger", action="store_true", help="Trigger GitHub Actions build")
    parser.add_argument("--full", action="store_true", help="Run full workflow")
    parser.add_argument("--message", "-m", help="Commit message")
    
    args = parser.parse_args()
    
    workflow = DevWorkflow()
    
    if args.test:
        workflow.test_local()
    elif args.build:
        workflow.build_local()
    elif args.push:
        workflow.commit_and_push(args.message)
    elif args.trigger:
        workflow.trigger_build()
    elif args.full:
        workflow.full_workflow(args.message)
    else:
        print("🚀 Contract Intelligence Platform - Development Workflow")
        print("\nUsage:")
        print("  python dev-workflow.py --test     # Test locally")
        print("  python dev-workflow.py --build    # Build locally") 
        print("  python dev-workflow.py --push     # Commit & push")
        print("  python dev-workflow.py --trigger  # Trigger GitHub build")
        print("  python dev-workflow.py --full     # Run complete workflow")
        print("  python dev-workflow.py --full -m 'Custom commit message'")

if __name__ == "__main__":
    main()
