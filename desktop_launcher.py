#!/usr/bin/env python3
"""
Desktop Launcher for Contract Intelligence Platform
Handles user setup, configuration, and launches Streamlit app
"""

import os
import sys
import json
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading
import webbrowser
import time
import uuid
from pathlib import Path
import requests
from datetime import datetime

class ContractIntelligenceSetup:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Contract Intelligence Platform - Setup")
        self.root.geometry("600x500")
        self.root.resizable(False, False)
        
        # App data directory
        self.app_data_dir = Path.home() / ".contract_intelligence"
        self.app_data_dir.mkdir(exist_ok=True)
        
        self.config_file = self.app_data_dir / "config.json"
        self.credentials_dir = self.app_data_dir / "credentials"
        self.credentials_dir.mkdir(exist_ok=True)
        
        # Telemetry settings
        self.telemetry_url = "https://your-telemetry-server.com/api/usage"  # Replace with your server
        self.user_id = self.get_or_create_user_id()
        
        self.setup_ui()
        self.load_existing_config()
        
    def get_or_create_user_id(self):
        """Generate unique user ID for telemetry"""
        user_id_file = self.app_data_dir / "user_id.txt"
        if user_id_file.exists():
            return user_id_file.read_text().strip()
        else:
            user_id = str(uuid.uuid4())
            user_id_file.write_text(user_id)
            return user_id
    
    def setup_ui(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Title
        title_label = ttk.Label(main_frame, text="📄 Contract Intelligence Platform", 
                               font=("Arial", 16, "bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # OpenAI API Key
        ttk.Label(main_frame, text="OpenAI API Key:", font=("Arial", 10, "bold")).grid(
            row=1, column=0, sticky=tk.W, pady=(0, 5))
        self.openai_key_var = tk.StringVar()
        
        # Create frame for API key entry with show/hide toggle
        api_key_frame = ttk.Frame(main_frame)
        api_key_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        self.openai_entry = ttk.Entry(api_key_frame, textvariable=self.openai_key_var, 
                                     width=50, show="*")
        self.openai_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        # Add show/hide toggle button
        self.show_key_var = tk.BooleanVar()
        show_button = ttk.Checkbutton(api_key_frame, text="Show", 
                                     variable=self.show_key_var,
                                     command=self.toggle_api_key_visibility)
        show_button.grid(row=0, column=1, padx=(5, 0))
        
        # Enable right-click context menu for paste
        self.setup_context_menu(self.openai_entry)
        
        # Google Credentials
        ttk.Label(main_frame, text="Google Cloud Credentials:", font=("Arial", 10, "bold")).grid(
            row=3, column=0, sticky=tk.W, pady=(0, 5))
        
        cred_frame = ttk.Frame(main_frame)
        cred_frame.grid(row=4, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 15))
        
        self.google_cred_var = tk.StringVar()
        self.google_cred_entry = ttk.Entry(cred_frame, textvariable=self.google_cred_var, width=45)
        self.google_cred_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        
        ttk.Button(cred_frame, text="Browse...", 
                  command=self.browse_google_credentials).grid(row=0, column=1, padx=(10, 0))
        
        # Telemetry consent
        self.telemetry_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(main_frame, text="Send anonymous usage analytics (helps improve the app)", 
                       variable=self.telemetry_var).grid(row=5, column=0, columnspan=2, 
                                                        sticky=tk.W, pady=(0, 20))
        
        # Status display
        self.status_var = tk.StringVar(value="Ready to configure...")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, 
                                foreground="blue")
        status_label.grid(row=6, column=0, columnspan=2, pady=(0, 20))
        
        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=7, column=0, columnspan=2)
        
        ttk.Button(button_frame, text="Save Configuration", 
                  command=self.save_config).grid(row=0, column=0, padx=(0, 10))
        ttk.Button(button_frame, text="Launch App", 
                  command=self.launch_app).grid(row=0, column=1, padx=(10, 0))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        cred_frame.columnconfigure(0, weight=1)
        api_key_frame.columnconfigure(0, weight=1)
    
    def toggle_api_key_visibility(self):
        """Toggle API key visibility"""
        if self.show_key_var.get():
            self.openai_entry.config(show="")
        else:
            self.openai_entry.config(show="*")
    
    def setup_context_menu(self, entry_widget):
        """Setup right-click context menu for entry widget"""
        def show_context_menu(event):
            try:
                context_menu.tk_popup(event.x_root, event.y_root)
            finally:
                context_menu.grab_release()
        
        def paste_text():
            try:
                clipboard_text = self.root.clipboard_get()
                entry_widget.delete(0, tk.END)
                entry_widget.insert(0, clipboard_text)
            except tk.TclError:
                pass  # No clipboard content
        
        def copy_text():
            try:
                entry_widget.clipboard_clear()
                entry_widget.clipboard_append(entry_widget.get())
            except tk.TclError:
                pass
        
        def select_all():
            entry_widget.select_range(0, tk.END)
        
        # Create context menu
        context_menu = tk.Menu(self.root, tearoff=0)
        context_menu.add_command(label="Paste", command=paste_text)
        context_menu.add_command(label="Copy", command=copy_text)
        context_menu.add_separator()
        context_menu.add_command(label="Select All", command=select_all)
        
        # Bind right-click
        entry_widget.bind("<Button-3>", show_context_menu)
        
        # Also enable Ctrl+V for paste
        entry_widget.bind("<Control-v>", lambda e: paste_text())
        entry_widget.bind("<Control-c>", lambda e: copy_text())
        entry_widget.bind("<Control-a>", lambda e: select_all())

    def browse_google_credentials(self):
        """Browse for Google credentials JSON file"""
        file_path = filedialog.askopenfilename(
            title="Select Google Cloud Credentials JSON file",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if file_path:
            self.google_cred_var.set(file_path)
    
    def load_existing_config(self):
        """Load existing configuration if available"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    config = json.load(f)
                    self.openai_key_var.set(config.get('openai_api_key', ''))
                    self.google_cred_var.set(config.get('google_credentials_path', ''))
                    self.telemetry_var.set(config.get('telemetry_enabled', True))
                    self.status_var.set("✅ Configuration loaded")
            except Exception as e:
                self.status_var.set(f"❌ Error loading config: {e}")
    
    def validate_openai_key(self, api_key):
        """Validate OpenAI API key with specific error messages"""
        # Basic format validation first
        if not api_key or not api_key.strip():
            return False, "API key cannot be empty."
            
        api_key = api_key.strip()
        
        if not api_key.startswith("sk-"):
            return False, "Invalid API key format. OpenAI keys should start with 'sk-'."
            
        if len(api_key) < 20:  # Basic length check
            return False, "API key appears to be too short."
        
        try:
            from openai import OpenAI
            import openai
            
            # Create client with explicit timeout
            client = OpenAI(
                api_key=api_key,
                timeout=30.0  # 30 second timeout
            )
            
            # Test with a simple API call - models.list is the most reliable test
            response = client.models.list()
            
            # Verify we actually got a response
            if hasattr(response, 'data') and len(response.data) > 0:
                return True, "Valid API key"
            else:
                return False, "API key validation failed - no models returned."
            
        except openai.AuthenticationError as e:
            # Handle 401 authentication errors specifically
            return False, "Invalid OpenAI API key. Please check your key and try again."
        except openai.RateLimitError as e:
            # Handle 429 rate limit errors
            return False, "OpenAI rate limit exceeded. Please check your usage limits or try again later."
        except openai.APIConnectionError as e:
            # Handle network connection errors
            return False, "Cannot connect to OpenAI. Please check your internet connection."
        except openai.APITimeoutError as e:
            # Handle timeout errors
            return False, "OpenAI API request timed out. Please try again."
        except openai.APIStatusError as e:
            # Handle other API status errors (4xx, 5xx)
            if e.status_code == 402:
                return False, "OpenAI account has insufficient credits. Please add credits to your account."
            elif e.status_code == 403:
                return False, "OpenAI API access forbidden. Please check your account permissions."
            elif e.status_code >= 500:
                return False, "OpenAI server error. Please try again later."
            else:
                return False, f"OpenAI API error (status {e.status_code}): {str(e)}"
        except ImportError as e:
            return False, "OpenAI library not properly installed. Please reinstall the application."
        except Exception as e:
            # Fallback for any other errors with detailed logging
            error_str = str(e).lower()
            
            if "invalid api key" in error_str or "unauthorized" in error_str:
                return False, "Invalid OpenAI API key. Please check your key and try again."
            elif "rate limit" in error_str or "429" in error_str:
                return False, "OpenAI rate limit exceeded. Please check your usage limits or try again later."
            elif "insufficient" in error_str and "credit" in error_str:
                return False, "OpenAI account has insufficient credits. Please add credits to your account."
            elif "quota" in error_str or "billing" in error_str:
                return False, "OpenAI billing issue. Please check your account billing status."
            elif "network" in error_str or "connection" in error_str or "timeout" in error_str:
                return False, "Cannot connect to OpenAI. Please check your internet connection."
            else:
                # More detailed error for debugging
                return False, f"OpenAI API validation failed: {type(e).__name__}: {str(e)}"

    def save_config(self):
        """Save configuration to file"""
        try:
            # Validate inputs
            if not self.openai_key_var.get().strip():
                messagebox.showerror("Error", "Please enter your OpenAI API Key")
                return
            
            # Validate OpenAI API key
            self.status_var.set("🔍 Validating OpenAI API key...")
            self.root.update()
            
            is_valid, message = self.validate_openai_key(self.openai_key_var.get().strip())
            if not is_valid:
                messagebox.showerror("OpenAI API Key Error", message)
                self.status_var.set("❌ Invalid OpenAI API key")
                return
            
            if not self.google_cred_var.get().strip():
                messagebox.showerror("Error", "Please select your Google credentials file")
                return
            
            # Copy Google credentials to user home directory
            google_cred_source = Path(self.google_cred_var.get())
            if not google_cred_source.exists():
                messagebox.showerror("Error", "Google credentials file not found")
                return
            
            from utils import get_google_credentials_path
            google_cred_dest = get_google_credentials_path()
            import shutil
            shutil.copy2(google_cred_source, google_cred_dest)
            
            # Save configuration
            config = {
                'openai_api_key': self.openai_key_var.get().strip(),
                'google_credentials_path': str(google_cred_dest),
                'telemetry_enabled': self.telemetry_var.get(),
                'user_id': self.user_id,
                'setup_date': datetime.now().isoformat()
            }
            
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            self.status_var.set("✅ Configuration saved successfully!")
            
            # Send setup telemetry if enabled
            if self.telemetry_var.get():
                self.send_telemetry('app_setup', {'platform': sys.platform})
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save configuration: {e}")
            self.status_var.set(f"❌ Error: {e}")
    
    def send_telemetry(self, event_type, data=None):
        """Send telemetry data (non-blocking)"""
        if not self.telemetry_var.get():
            return
            
        def send_async():
            try:
                payload = {
                    'user_id': self.user_id,
                    'event_type': event_type,
                    'timestamp': datetime.now().isoformat(),
                    'platform': sys.platform,
                    'data': data or {}
                }
                
                requests.post(self.telemetry_url, json=payload, timeout=5)
            except:
                pass  # Silent fail for telemetry
        
        threading.Thread(target=send_async, daemon=True).start()
    
    def check_dependencies(self):
        """Check if all required dependencies are available without triggering model loading"""
        required_modules = [
            'streamlit',
            'openai', 
            'chromadb',
            'google.auth',
            'PyPDF2',
            'docx'
        ]
        
        missing_modules = []
        for module in required_modules:
            try:
                # Use importlib to avoid triggering module initialization
                import importlib.util
                spec = importlib.util.find_spec(module)
                if spec is None:
                    missing_modules.append(module)
            except ImportError:
                missing_modules.append(module)
        
        return missing_modules

    def launch_app(self):
        """Launch the Streamlit application"""
        if not self.config_file.exists():
            messagebox.showerror("Error", "Please save configuration first")
            return
        
        try:
            # Check dependencies first
            self.status_var.set("🔍 Checking dependencies...")
            missing_deps = self.check_dependencies()
            if missing_deps:
                error_msg = f"Missing required dependencies: {', '.join(missing_deps)}\nPlease reinstall the application."
                messagebox.showerror("Dependency Error", error_msg)
                self.status_var.set(f"❌ Missing dependencies: {', '.join(missing_deps)}")
                return
            
            # Set environment variables from config
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            os.environ['OPENAI_API_KEY'] = config['openai_api_key']
            from utils import get_google_credentials_path
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(get_google_credentials_path())
            
            self.status_var.set("🚀 Launching application...")
            
            # Send launch telemetry
            if config.get('telemetry_enabled', True):
                self.send_telemetry('app_launch')
            
            # Hide setup window
            self.root.withdraw()
            
            # Launch Streamlit in separate thread
            def launch_streamlit():
                try:
                    # Set environment variables for embedded Streamlit
                    os.environ['OPENAI_API_KEY'] = config['openai_api_key']
                    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(get_google_credentials_path())
                    
                    # Get the directory where this script is located
                    app_dir = Path(__file__).parent
                    streamlit_app = app_dir / "streamlit_app.py"
                    
                    print(f"✅ Environment variables set: OPENAI_API_KEY, GOOGLE_APPLICATION_CREDENTIALS")
                    print("🚀 Starting embedded Streamlit server...")
                    
                    # Import and run Streamlit directly (embedded approach)
                    import streamlit.web.bootstrap
                    
                    # Configure Streamlit server options
                    streamlit_args = [
                        "streamlit", "run", str(streamlit_app),
                        "--server.headless=true",
                        "--server.address=0.0.0.0", 
                        "--server.port=8501",
                        "--browser.gatherUsageStats=false",
                        "--server.enableCORS=false",
                        "--server.enableXsrfProtection=false"
                    ]
                    
                    # Run Streamlit directly without subprocess
                    streamlit.web.bootstrap.run(
                        str(streamlit_app),
                        command_line="streamlit run",
                        args=streamlit_args[2:],  # Skip 'streamlit run' part
                        flag_options={
                            "server.headless": True,
                            "server.address": "0.0.0.0",
                            "server.port": 8501,
                            "browser.gatherUsageStats": False,
                            "server.enableCORS": False,
                            "server.enableXsrfProtection": False
                        }
                    )
                    
                    print("✅ Embedded Streamlit server started successfully")
                    print("🚀 Streamlit should be available at: http://localhost:8501")
                    
                except Exception as e:
                    error_msg = f"Failed to start embedded Streamlit: {str(e)}"
                    print(f"❌ {error_msg}")
                    messagebox.showerror("Launch Error", error_msg)
                finally:
                    # Show setup window again when app closes
                    self.root.after(0, self.root.deiconify)
            
            threading.Thread(target=launch_streamlit, daemon=True).start()
            
            # Open browser after longer delay to ensure Streamlit is ready
            def open_browser():
                time.sleep(5)  # Increased delay for macOS
                try:
                    # Try to verify Streamlit is running before opening browser
                    import requests
                    for attempt in range(10):  # Try for up to 10 seconds
                        try:
                            response = requests.get("http://localhost:8501/healthz", timeout=1)
                            if response.status_code == 200:
                                break
                        except:
                            time.sleep(1)
                    
                    webbrowser.open("http://localhost:8501")
                except Exception as e:
                    print(f"Browser opening error: {e}")
                    # Fallback: just open the URL anyway
                    webbrowser.open("http://localhost:8501")
            
            threading.Thread(target=open_browser, daemon=True).start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch application: {e}")
            self.status_var.set(f"❌ Launch failed: {e}")
    
    def run(self):
        """Start the setup GUI"""
        self.root.mainloop()

if __name__ == "__main__":
    app = ContractIntelligenceSetup()
    app.run()
