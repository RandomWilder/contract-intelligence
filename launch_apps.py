# launch_apps.py - Easy launcher for both Streamlit apps
import subprocess
import sys
import os
import time
from pathlib import Path

def launch_app(app_name, port):
    """Launch a Streamlit app on specific port"""
    print(f"🚀 Launching {app_name} on port {port}...")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_name,
        "--server.port", str(port),
        "--server.headless", "true",
        "--browser.gatherUsageStats", "false"
    ]
    
    return subprocess.Popen(cmd, cwd=os.getcwd())

def main():
    print("=" * 60)
    print("🎯 Contract Intelligence Platform - Dual App Launcher")
    print("=" * 60)
    
    # Check if files exist
    current_app = Path("streamlit_app.py")
    builtin_app = Path("streamlit_builtin.py")
    
    if not current_app.exists():
        print(f"❌ {current_app} not found!")
        return
    
    if not builtin_app.exists():
        print(f"❌ {builtin_app} not found!")
        return
    
    print("\n📋 Available Options:")
    print("1. 🔒 Current App (Google APIs) - Port 8501")
    print("2. 🚀 Built-in App (RAGFlow) - Port 8502")
    print("3. 🚀 Both Apps (Parallel) - Ports 8501 & 8502")
    print("4. ❌ Exit")
    
    while True:
        try:
            choice = input("\n👉 Enter your choice (1-4): ").strip()
            
            if choice == "1":
                print("\n🔒 Launching Current App (Google APIs)...")
                process = launch_app("streamlit_app.py", 8501)
                print(f"✅ Current App running on: http://localhost:8501")
                print("Press Ctrl+C to stop")
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print("\n🛑 Stopping Current App...")
                    process.terminate()
                break
                
            elif choice == "2":
                print("\n🚀 Launching Built-in App (RAGFlow)...")
                process = launch_app("streamlit_builtin.py", 8502)
                print(f"✅ Built-in App running on: http://localhost:8502")
                print("Press Ctrl+C to stop")
                try:
                    process.wait()
                except KeyboardInterrupt:
                    print("\n🛑 Stopping Built-in App...")
                    process.terminate()
                break
                
            elif choice == "3":
                print("\n🚀 Launching Both Apps in Parallel...")
                
                # Launch current app
                current_process = launch_app("streamlit_app.py", 8501)
                time.sleep(2)  # Give first app time to start
                
                # Launch built-in app
                builtin_process = launch_app("streamlit_builtin.py", 8502)
                time.sleep(2)  # Give second app time to start
                
                print("\n✅ Both apps are running:")
                print("   🔒 Current App (Google APIs): http://localhost:8501")
                print("   🚀 Built-in App (RAGFlow):   http://localhost:8502")
                print("\nPress Ctrl+C to stop both apps")
                
                try:
                    # Wait for user interruption
                    current_process.wait()
                except KeyboardInterrupt:
                    print("\n🛑 Stopping both apps...")
                    current_process.terminate()
                    builtin_process.terminate()
                    
                    # Wait for clean shutdown
                    current_process.wait()
                    builtin_process.wait()
                break
                
            elif choice == "4":
                print("👋 Goodbye!")
                break
                
            else:
                print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
