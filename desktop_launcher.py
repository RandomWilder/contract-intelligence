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
from pathlib import Path
import requests
from datetime import datetime
import uuid

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
        try:
            from openai import OpenAI
            client = OpenAI(api_key=api_key)
            
            # Test with a simple API call
            response = client.models.list()
            return True, "Valid API key"
            
        except Exception as e:
            error_str = str(e).lower()
            
            if "invalid api key" in error_str or "unauthorized" in error_str:
                return False, "Invalid OpenAI API key. Please check your key and try again."
            elif "rate limit" in error_str or "429" in error_str:
                return False, "OpenAI rate limit exceeded. Please check your usage limits or try again later."
            elif "insufficient" in error_str and "credit" in error_str:
                return False, "OpenAI account has insufficient credits. Please add credits to your account."
            elif "quota" in error_str or "billing" in error_str:
                return False, "OpenAI billing issue. Please check your account billing status."
            elif "network" in error_str or "connection" in error_str:
                return False, "Cannot connect to OpenAI. Please check your internet connection."
            else:
                return False, f"OpenAI API error: {str(e)}"

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
    
    def launch_app(self):
        """Launch the Streamlit application"""
        if not self.config_file.exists():
            messagebox.showerror("Error", "Please save configuration first")
            return
        
        try:
            # Set environment variables from config
            with open(self.config_file, 'r') as f:
                config = json.load(f)
            
            os.environ['OPENAI_API_KEY'] = config['openai_api_key']
            from local_rag_app import get_google_credentials_path
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
                    # Get the directory where this script is located
                    app_dir = Path(__file__).parent
                    streamlit_app = app_dir / "streamlit_app.py"
                    
                    # Launch Streamlit
                    subprocess.run([
                        sys.executable, "-m", "streamlit", "run", 
                        str(streamlit_app),
                        "--server.headless=true",
                        "--server.port=8501",
                        "--browser.gatherUsageStats=false"
                    ])
                except Exception as e:
                    messagebox.showerror("Launch Error", f"Failed to launch app: {e}")
                finally:
                    # Show setup window again when app closes
                    self.root.after(0, self.root.deiconify)
            
            threading.Thread(target=launch_streamlit, daemon=True).start()
            
            # Open browser after short delay
            def open_browser():
                time.sleep(3)
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
