#!/usr/bin/env python3
"""
Telemetry Client for Contract Intelligence Platform
Collects anonymous usage statistics to help improve the application
"""

import os
import json
import requests
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
import uuid
import hashlib
import platform
import sys
from typing import Dict, Any, Optional
import logging

class TelemetryClient:
    """
    Privacy-focused telemetry client that collects anonymous usage data
    """
    
    def __init__(self, app_name: str = "ContractIntelligence", telemetry_url: str = None):
        self.app_name = app_name
        self.telemetry_url = telemetry_url or "https://your-telemetry-server.com/api/usage"
        
        # Setup paths
        self.app_data_dir = Path.home() / f".{app_name.lower()}"
        self.app_data_dir.mkdir(exist_ok=True)
        
        self.config_file = self.app_data_dir / "telemetry_config.json"
        self.session_file = self.app_data_dir / "session.json"
        
        # Load configuration
        self.config = self.load_config()
        self.user_id = self.config.get('user_id', self.generate_user_id())
        self.enabled = self.config.get('enabled', True)
        
        # Session tracking
        self.session_id = str(uuid.uuid4())
        self.session_start = datetime.now()
        self.events_queue = []
        self.last_heartbeat = datetime.now()
        
        # Setup logging
        logging.basicConfig(level=logging.WARNING)
        self.logger = logging.getLogger(__name__)
        
        # Start background worker
        if self.enabled:
            self.start_background_worker()
    
    def generate_user_id(self) -> str:
        """Generate anonymous user ID based on machine characteristics"""
        # Use machine-specific but non-personal identifiers
        machine_id = platform.node() + platform.machine() + platform.processor()
        user_id = hashlib.sha256(machine_id.encode()).hexdigest()[:16]
        return user_id
    
    def load_config(self) -> Dict[str, Any]:
        """Load telemetry configuration"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    return json.load(f)
            except Exception:
                pass
        return {}
    
    def save_config(self):
        """Save telemetry configuration"""
        try:
            config = {
                'user_id': self.user_id,
                'enabled': self.enabled,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
        except Exception as e:
            self.logger.warning(f"Failed to save telemetry config: {e}")
    
    def enable(self):
        """Enable telemetry collection"""
        self.enabled = True
        self.save_config()
        if not hasattr(self, '_worker_thread') or not self._worker_thread.is_alive():
            self.start_background_worker()
    
    def disable(self):
        """Disable telemetry collection"""
        self.enabled = False
        self.save_config()
        self.events_queue.clear()
    
    def track_event(self, event_type: str, properties: Optional[Dict[str, Any]] = None):
        """Track an application event"""
        if not self.enabled:
            return
        
        event = {
            'user_id': self.user_id,
            'session_id': self.session_id,
            'event_type': event_type,
            'timestamp': datetime.now().isoformat(),
            'properties': properties or {},
            'system_info': self.get_system_info()
        }
        
        self.events_queue.append(event)
        
        # Send immediately for important events
        if event_type in ['app_start', 'app_crash', 'error']:
            self.flush_events()
    
    def track_app_start(self):
        """Track application start"""
        self.track_event('app_start', {
            'app_version': '1.0.0',  # Update this with your version
            'python_version': sys.version,
        })
    
    def track_app_end(self):
        """Track application end"""
        session_duration = (datetime.now() - self.session_start).total_seconds()
        self.track_event('app_end', {
            'session_duration': session_duration,
            'events_count': len(self.events_queue)
        })
        self.flush_events()
    
    def track_document_processed(self, file_type: str, file_size: int, processing_time: float):
        """Track document processing"""
        self.track_event('document_processed', {
            'file_type': file_type,
            'file_size_mb': round(file_size / (1024 * 1024), 2),
            'processing_time_seconds': round(processing_time, 2)
        })
    
    def track_chat_interaction(self, model_used: str, response_time: float, tokens_used: Optional[int] = None):
        """Track chat interactions"""
        properties = {
            'model': model_used,
            'response_time_seconds': round(response_time, 2)
        }
        if tokens_used:
            properties['tokens_used'] = tokens_used
            
        self.track_event('chat_interaction', properties)
    
    def track_error(self, error_type: str, error_message: str, context: Optional[str] = None):
        """Track application errors"""
        # Hash error message to avoid sending sensitive data
        error_hash = hashlib.sha256(error_message.encode()).hexdigest()[:16]
        
        self.track_event('error', {
            'error_type': error_type,
            'error_hash': error_hash,
            'context': context
        })
    
    def track_feature_usage(self, feature_name: str, usage_data: Optional[Dict[str, Any]] = None):
        """Track feature usage"""
        self.track_event('feature_usage', {
            'feature': feature_name,
            'usage_data': usage_data or {}
        })
    
    def get_system_info(self) -> Dict[str, Any]:
        """Get anonymous system information"""
        return {
            'platform': platform.system(),
            'platform_version': platform.release(),
            'architecture': platform.machine(),
            'python_version': f"{sys.version_info.major}.{sys.version_info.minor}",
            'app_name': self.app_name
        }
    
    def send_heartbeat(self):
        """Send periodic heartbeat to track active usage"""
        if not self.enabled:
            return
            
        now = datetime.now()
        if (now - self.last_heartbeat) > timedelta(minutes=5):
            self.track_event('heartbeat', {
                'active_time_minutes': (now - self.last_heartbeat).total_seconds() / 60
            })
            self.last_heartbeat = now
    
    def flush_events(self):
        """Send all queued events to server"""
        if not self.enabled or not self.events_queue:
            return
        
        def send_async():
            try:
                # Prepare batch payload
                payload = {
                    'events': self.events_queue[:50],  # Send max 50 events at once
                    'client_version': '1.0.0',
                    'timestamp': datetime.now().isoformat()
                }
                
                # Send to server
                response = requests.post(
                    self.telemetry_url,
                    json=payload,
                    timeout=10,
                    headers={'Content-Type': 'application/json'}
                )
                
                if response.status_code == 200:
                    # Remove sent events
                    self.events_queue = self.events_queue[50:]
                    self.logger.debug(f"Sent {len(payload['events'])} telemetry events")
                else:
                    self.logger.warning(f"Telemetry server returned {response.status_code}")
                    
            except Exception as e:
                self.logger.warning(f"Failed to send telemetry: {e}")
        
        # Send asynchronously to avoid blocking the main application
        threading.Thread(target=send_async, daemon=True).start()
    
    def start_background_worker(self):
        """Start background worker for periodic tasks"""
        def worker():
            while self.enabled:
                try:
                    # Send heartbeat
                    self.send_heartbeat()
                    
                    # Flush events periodically
                    if len(self.events_queue) > 10:
                        self.flush_events()
                    
                    # Sleep for 1 minute
                    time.sleep(60)
                    
                except Exception as e:
                    self.logger.warning(f"Telemetry worker error: {e}")
                    time.sleep(60)
        
        self._worker_thread = threading.Thread(target=worker, daemon=True)
        self._worker_thread.start()

# Global telemetry instance
_telemetry_instance = None

def get_telemetry_client(telemetry_url: str = None) -> TelemetryClient:
    """Get global telemetry client instance"""
    global _telemetry_instance
    if _telemetry_instance is None:
        _telemetry_instance = TelemetryClient(telemetry_url=telemetry_url)
    return _telemetry_instance

def track_event(event_type: str, properties: Optional[Dict[str, Any]] = None):
    """Convenience function to track events"""
    client = get_telemetry_client()
    client.track_event(event_type, properties)

def track_app_start():
    """Convenience function to track app start"""
    client = get_telemetry_client()
    client.track_app_start()

def track_app_end():
    """Convenience function to track app end"""
    client = get_telemetry_client()
    client.track_app_end()

# Example usage and testing
if __name__ == "__main__":
    # Test the telemetry client
    client = TelemetryClient(telemetry_url="https://httpbin.org/post")  # Test endpoint
    
    client.track_app_start()
    client.track_document_processed("pdf", 1024*1024, 2.5)
    client.track_chat_interaction("gpt-4o-mini", 1.2, 150)
    client.track_feature_usage("ocr_processing", {"language": "hebrew"})
    client.track_error("FileNotFound", "Document not found", "upload_processing")
    
    # Flush events
    client.flush_events()
    
    print("✅ Telemetry test completed")
