#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal FastAPI backend for Contract Intelligence Platform
Uses only installed packages: FastAPI, OpenAI, ChromaDB, basic document processing
"""

import os
import sys
import json
import uuid
import platform
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
from io import BytesIO
import re
import math
import base64

# Essential imports - fail fast if these are missing
try:
    from fastapi import FastAPI, HTTPException, Depends, Query, UploadFile, File, Form, Request
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
    import uvicorn
    from contextlib import asynccontextmanager
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False

# Try to import OpenAI
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Try to import ChromaDB - with safeguards for dependencies
try:
    import chromadb
    from chromadb.config import Settings
    
    # Explicitly prevent sentence_transformers from being imported indirectly
    # This will prevent the ONNXMiniLM_L6_V2 error
    try:
        # Monkey-patch the embedding functions module if it tries to import sentence_transformers
        import sys
        import types
        
        # Create stub module to replace sentence_transformers if it's imported
        stub_module = types.ModuleType('sentence_transformers')
        stub_module.__path__ = []
        sys.modules['sentence_transformers'] = stub_module
        
        # Import chromadb utils directly to force early failure if there's an issue
        from chromadb.utils import embedding_functions
        print("[INFO] Successfully imported ChromaDB with embedding_functions")
    except Exception as ef_error:
        print(f"[WARNING] ChromaDB embedding_functions import issue: {ef_error}")
        # We'll continue anyway - we'll handle this in init_chromadb
        pass
        
    AI_CHROMADB_AVAILABLE = True
except ImportError as import_error:
    print(f"[WARNING] ChromaDB import failed: {import_error}")
    AI_CHROMADB_AVAILABLE = False

# Try to import Google API
try:
    from google.oauth2 import service_account
    # Rename the Request import to avoid conflict with FastAPI
    from google.auth.transport.requests import Request as GoogleRequest
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
    GOOGLE_API_AVAILABLE = True
    
    # Scope needed for Google APIs - Vision scope needs to be first for service account validation
    SCOPES = [
        'https://www.googleapis.com/auth/cloud-vision',  # Cloud Vision API scope - primary for OCR
        'https://www.googleapis.com/auth/cloud-platform',  # Full access - sometimes needed for Vision API
        'https://www.googleapis.com/auth/drive',
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.send',
        'https://www.googleapis.com/auth/gmail.compose',
        'https://www.googleapis.com/auth/gmail.modify'
    ]
except ImportError:
    GOOGLE_API_AVAILABLE = False

# Try to import Contract Intelligence Engine
try:
    from contract_intelligence import ContractIntelligenceEngine
    CONTRACT_INTELLIGENCE_AVAILABLE = True
except ImportError:
    CONTRACT_INTELLIGENCE_AVAILABLE = False

# Application settings
app_settings = {
    "openai_api_key": None,
    "google_credentials_path": None,
    "chromadb_dir": "./chroma_db",
}

# Global variables - initialized later
chroma_client = None
collection = None
openai_client = None
contract_intelligence_engine = None
google_credentials = None

# Simple HTTP handler for minimal mode (used if FastAPI is not available)
class MinimalHandler:
    def do_GET(self):
        return {"status": "ok", "message": "Contract Intelligence Platform API is running in minimal mode"}

# Settings management functions
def load_settings():
    """Load settings from file"""
    global app_settings
    try:
        for settings_file in ["app_settings.json", "ci_settings.json"]:
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    app_settings.update(json.load(f))
                print(f"[INFO] Settings loaded from {settings_file}")
                return
        print("[INFO] Settings file not found, using defaults")
    except Exception as e:
        print(f"[WARNING] Failed to load settings: {e}")

def save_settings():
    """Save settings to file"""
    try:
        with open("app_settings.json", 'w') as f:
            json.dump(app_settings, f, indent=2)
        print("[INFO] Settings saved successfully")
        return True
    except Exception as e:
        print(f"[WARNING] Failed to save settings: {e}")
        return False

# Removed save_google_credentials function as it's now handled directly in the upload endpoint

# Component initialization functions
def init_openai():
    """Initialize OpenAI client"""
    global openai_client
    
    api_key = app_settings.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("[INFO] No OpenAI API key found - configure in settings")
        return None
        
    if OPENAI_AVAILABLE:
        try:
            os.environ["OPENAI_API_KEY"] = api_key
            client = openai.OpenAI(api_key=api_key)
            print("[INFO] OpenAI client initialized")
            return client
        except Exception as e:
            print(f"[WARNING] Failed to initialize OpenAI client: {e}")
            return None
    return None

def init_google():
    """Initialize Google credentials from service account file"""
    global google_credentials
    
    if not GOOGLE_API_AVAILABLE:
        print("[WARNING] Google API module not available")
        return None
    
    creds_path = app_settings.get("google_credentials_path")
    if not creds_path or not os.path.exists(creds_path):
        print(f"[WARNING] Google credentials file not found at {creds_path}")
        return None
        
    try:
        # Load service account credentials with all required scopes
        creds = service_account.Credentials.from_service_account_file(
            creds_path, scopes=SCOPES)
            
        # Test Vision API access
        try:
            vision_service = build('vision', 'v1', credentials=creds)
            # Just build the service to verify credentials (no API call needed yet)
            print(f"[INFO] Google Vision API service successfully initialized")
        except Exception as vision_error:
            print(f"[WARNING] Vision API initialization failed: {vision_error}")
            # Continue anyway, we've loaded credentials
            
        print(f"[INFO] Google service account credentials loaded successfully")
        return creds
    except Exception as e:
        print(f"[WARNING] Failed to load Google credentials: {e}")
        traceback.print_exc()
        return None

def get_chromadb_dir():
    """Get ChromaDB directory path"""
    if getattr(sys, 'frozen', False) and 'CHROMADB_DIR' in os.environ:
        persist_dir = os.environ['CHROMADB_DIR']
    else:
        persist_dir = app_settings.get("chromadb_dir", "./chroma_db")
    
    try:
        os.makedirs(persist_dir, exist_ok=True)
        # Test if directory is writable
        test_file = os.path.join(persist_dir, 'test_write.txt')
        with open(test_file, 'w') as f:
            f.write('test')
        os.remove(test_file)
        return persist_dir
    except Exception:
        # Try fallback directory
        fallback_dir = os.path.join(os.path.expanduser('~'), '.contract_intelligence', 'chroma_db')
        os.makedirs(fallback_dir, exist_ok=True)
        return fallback_dir

def init_chromadb():
    """Initialize ChromaDB client and collection"""
    if not AI_CHROMADB_AVAILABLE or not OPENAI_AVAILABLE:
        return None, None
        
    api_key = app_settings.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None, None
    
    try:
        # Set up ChromaDB directory
        persist_dir = get_chromadb_dir()
        
        # MODIFIED: Create embedding function - ONLY use OpenAI, never fallback to sentence_transformers
        try:
            from chromadb.utils import embedding_functions
            
            # Ensure we're not trying to use any sentence_transformers based embedding
            if hasattr(embedding_functions, "DefaultEmbeddingFunction"):
                print("[INFO] Disabling ChromaDB's DefaultEmbeddingFunction to avoid sentence_transformers dependency")
                # Prevent DefaultEmbeddingFunction from being used
                embedding_functions.DefaultEmbeddingFunction = None
                
            # Set up OpenAI embedding explicitly
            os.environ["CHROMA_OPENAI_API_KEY"] = api_key
            openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=api_key,
                model_name="text-embedding-ada-002"  # Explicitly use ada-002 only
            )
        except Exception as ef_error:
            print(f"[ERROR] Failed to initialize OpenAI embedding function: {ef_error}")
            traceback.print_exc()
            return None, None
        
        # Initialize client
        client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Get or create collection
        collection_name = "contracts_electron"
        try:
            # Try to get existing collection
            collection = client.get_collection(
                name=collection_name, 
                embedding_function=openai_ef
            )
        except Exception:
            # Create new collection
            collection = client.create_collection(
                name=collection_name,
                embedding_function=openai_ef,
                metadata={"description": "Contract Intelligence document collection"}
            )
        
        return client, collection
    except Exception as e:
        print(f"[WARNING] ChromaDB initialization failed: {e}")
        return None, None

def init_contract_intelligence():
    """Initialize Contract Intelligence Engine"""
    global openai_client
    
    if not CONTRACT_INTELLIGENCE_AVAILABLE or not openai_client:
        return None
        
    try:
        engine = ContractIntelligenceEngine(openai_client)
        print("[INFO] Contract Intelligence Engine initialized")
        return engine
    except Exception:
        return None

def initialize():
    """Main initialization function"""
    global chroma_client, collection, openai_client, contract_intelligence_engine, google_credentials
    
    print("[INFO] Initializing backend components...")
    
    # Load settings first
    load_settings()
    
    # Initialize core components
    openai_client = init_openai()
    google_credentials = init_google()
    chroma_client, collection = init_chromadb()
    contract_intelligence_engine = init_contract_intelligence()
    
    print("[INFO] Backend initialization complete")

# FastAPI setup with modern lifespan pattern
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan event handler"""
    # Startup
    initialize()
    yield
    # Shutdown
    pass

# Create FastAPI app with lifespan handler
app = FastAPI(title="Contract Intelligence API", lifespan=lifespan) if FASTAPI_AVAILABLE else None

# Add CORS middleware
if FASTAPI_AVAILABLE:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# API Endpoints
if FASTAPI_AVAILABLE:
    @app.get("/health")
    async def get_health():
        """Health check endpoint - reports actual component status"""
        chromadb_ready = chroma_client is not None and collection is not None
        openai_ready = openai_client is not None
        
        return {
            "status": "healthy",
            "version": "1.5.60",
            "backend": "minimal",
            "chromadb_ready": chromadb_ready,
            "openai_ready": openai_ready,
            "features_available": {
                "document_processing": True,
                "vector_search": chromadb_ready,
                "contract_intelligence": contract_intelligence_engine is not None,
                "tokenizers": False,
                "sentence_transformers": False
            },
            "backend_mode": "minimal"
        }

    @app.get("/api/health")
    async def api_health():
        """Compatibility health endpoint (redirects to /health)"""
        return await get_health()
    
    @app.get("/api/status")
    async def api_status():
        """Status endpoint for UI compatibility"""
        chromadb_ready = chroma_client is not None and collection is not None
        openai_ready = openai_client is not None
        google_ready = google_credentials is not None
        
        # Get unique document count (not vector chunks)
        doc_count = 0
        if chromadb_ready:
            try:
                # Get all documents from collection to count unique filenames
                results = collection.get()
                unique_docs = set()
                
                if results and 'metadatas' in results and results['metadatas']:
                    for metadata in results['metadatas']:
                        if metadata and 'filename' in metadata:
                            unique_docs.add(metadata['filename'])
                
                doc_count = len(unique_docs)
            except Exception:
                pass
        
        return {
            "status": "ok",
            "backend": "minimal",
            "openai": {
                "available": openai_ready,
                "status": "ready" if openai_ready else "unavailable"
            },
            "google": {
                "available": google_ready,
                "status": "ready" if google_ready else "unavailable"
            },
            "chromadb": {
                "available": chromadb_ready,
                "status": "ready" if chromadb_ready else "unavailable"
            },
            "documents": {
                "count": doc_count,
                "available": chromadb_ready
            },
            "services": {
                "openai": openai_ready,
                "google": google_ready,
                "chromadb": chromadb_ready
            }
        }
        
    @app.get("/api/diagnostics")
    async def get_diagnostics():
        """Get diagnostic information about the API server"""
        return {
            "status": "ok",
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "chromadb_available": AI_CHROMADB_AVAILABLE,
            "openai_available": OPENAI_AVAILABLE,
            "google_api_available": GOOGLE_API_AVAILABLE,
            "contract_intelligence_available": CONTRACT_INTELLIGENCE_AVAILABLE,
            "services": {
                "chromadb": chroma_client is not None,
                "collection": collection is not None,
                "openai": openai_client is not None,
                "contract_intelligence": contract_intelligence_engine is not None,
                "google": google_credentials is not None
            }
        }

    # Settings endpoints
    @app.get("/api/settings/openai")
    async def get_openai_settings():
        """Get OpenAI specific settings"""
        global openai_client, app_settings
        
        # Get OpenAI API key (redacted for security)
        api_key = app_settings.get("openai_api_key")
        redacted_key = "sk-..." + api_key[-4:] if api_key and len(api_key) > 8 else None
        
        # Check if OpenAI is configured and available
        openai_configured = openai_client is not None
        
        return {
            "configured": openai_configured,
            "api_key": redacted_key,
            "embedding_model": "text-embedding-ada-002",
            "models_available": True if openai_configured else False
        }
        
    @app.post("/api/settings/openai", response_model=None)
    async def update_openai_settings(request: Request):
        """Update OpenAI settings with a new API key"""
        global openai_client, app_settings
        
        try:
            # Parse the request body
            data = await request.json()
            api_key = data.get("api_key")
            
            if not api_key:
                raise HTTPException(status_code=400, detail="API key is required")
                
            if not OPENAI_AVAILABLE:
                raise HTTPException(status_code=503, detail="OpenAI module not available")
            
            # Validate the key with a simple test
            try:
                test_client = openai.OpenAI(api_key=api_key)
                test_client.models.list()
                
                # Save valid key to settings
                app_settings["openai_api_key"] = api_key
                save_settings()
                
                # Update global client
                os.environ["OPENAI_API_KEY"] = api_key
                openai_client = test_client
                
                # Reinitialize components that depend on the API key
                global chroma_client, collection
                chroma_client, collection = init_chromadb()
                
                return {
                    "success": True,
                    "message": "OpenAI API key updated successfully"
                }
                
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Invalid OpenAI API key: {str(e)}")
                
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON in request body")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update OpenAI settings: {str(e)}")
    
    @app.post("/api/settings/openai/validate", response_model=None)
    async def validate_openai_api_key(api_key: str = Form(...)):
        """Validate an OpenAI API key and save it if valid"""
        global openai_client, app_settings
        
        if not OPENAI_AVAILABLE:
            raise HTTPException(status_code=503, detail="OpenAI module not available")
        
        # Simple validation - just check if we can list models
        try:
            # Create a temporary client to test the key
            test_client = openai.OpenAI(api_key=api_key)
            test_client.models.list()
            
            # Save the API key to settings
            app_settings["openai_api_key"] = api_key
            save_settings()
            
            # Update global client and reinitialize components
            os.environ["OPENAI_API_KEY"] = api_key
            openai_client = test_client
            
            # Reinitialize ChromaDB with the new key
            global chroma_client, collection
            chroma_client, collection = init_chromadb()
            
            return {
                "success": True,
                "message": "OpenAI API key validated and saved",
                "key_saved": True
            }
            
        except Exception as e:
            error_message = str(e)
            if "auth" in error_message.lower() or "key" in error_message.lower():
                raise HTTPException(status_code=400, detail="Invalid OpenAI API key")
            elif "rate" in error_message.lower() or "limit" in error_message.lower():
                raise HTTPException(status_code=400, detail="OpenAI rate limit exceeded")
            elif "connect" in error_message.lower() or "network" in error_message.lower():
                raise HTTPException(status_code=400, detail="Cannot connect to OpenAI - check internet connection")
            else:
                raise HTTPException(status_code=500, detail=f"Failed to validate API key: {error_message}")

    @app.post("/api/settings/google/upload-credentials", response_model=None)
    async def upload_google_credentials(file: UploadFile = File(...)):
        """Upload Google Service Account credentials file"""
        global google_credentials
        
        if not GOOGLE_API_AVAILABLE:
            raise HTTPException(status_code=503, detail="Google API module not available")
            
        try:
            if not file.filename.endswith('.json'):
                raise HTTPException(status_code=400, detail="Please upload a JSON credentials file")
            
            # Read and validate the credentials file
            content = await file.read()
            credentials_data = json.loads(content)
            
            # Basic validation for service account credentials
            if "type" not in credentials_data or credentials_data.get("type") != "service_account":
                raise HTTPException(status_code=400, detail="Invalid service account credentials. Please upload a service account JSON file.")
            
            # Ensure it has required fields
            required_fields = ["project_id", "private_key_id", "private_key", "client_email"]
            for field in required_fields:
                if field not in credentials_data:
                    raise HTTPException(status_code=400, detail=f"Invalid service account credentials: missing required field '{field}'")
                    
            # Save credentials file
            credentials_path = "google_service_account.json"
            with open(credentials_path, 'wb') as f:
                f.write(content)
            
            # Load the credentials immediately
            try:
                # Load with all scopes explicitly
                google_credentials = service_account.Credentials.from_service_account_file(
                    credentials_path, scopes=SCOPES)
                
                # Test credentials by validating both Drive AND Vision API
                # This ensures we have proper permissions for OCR
                drive_service = build('drive', 'v3', credentials=google_credentials)
                drive_service.about().get(fields="user").execute()
                
                # Also verify Vision API access specifically
                try:
                    vision_service = build('vision', 'v1', credentials=google_credentials)
                    # Basic request to validate API availability (doesn't make an actual API call)
                    print(f"[INFO] Vision API service initialized successfully")
                except Exception as vision_error:
                    print(f"[WARNING] Vision API service initialization check failed: {str(vision_error)}")
                    # Continue anyway, but notify user
                    return {
                        "success": True,
                        "warning": True,
                        "message": "Credentials saved but Vision API access may be limited. Please ensure the service account has Cloud Vision API enabled.",
                        "services": ["Drive", "Gmail"],
                        "project_id": credentials_data["project_id"]
                    }
                
                # Save the path to settings
                app_settings["google_credentials_path"] = credentials_path
                save_settings()
                
                return {
                    "success": True,
                    "message": "Google service account credentials verified and activated with OCR capability",
                    "services": ["OCR", "Drive", "Gmail"],
                    "project_id": credentials_data["project_id"]
                }
            except Exception as cred_error:
                # Clean up the file if validation fails
                if os.path.exists(credentials_path):
                    os.remove(credentials_path)
                raise HTTPException(status_code=400, detail=f"Invalid service account credentials: {str(cred_error)}")
            
        except json.JSONDecodeError:
            raise HTTPException(status_code=400, detail="Invalid JSON file")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to upload credentials: {str(e)}")

    @app.get("/api/settings/google")
    async def get_google_settings():
        """Get Google specific settings"""
        global google_credentials
        
        project_id = None
        creds_path = app_settings.get("google_credentials_path")
        if creds_path and os.path.exists(creds_path):
            try:
                with open(creds_path, 'r') as f:
                    creds_data = json.load(f)
                    project_id = creds_data.get("project_id")
            except:
                pass
                
        return {
            "configured": google_credentials is not None,
            "project_id": project_id,
            "services_available": ["OCR", "Drive", "Gmail"] if google_credentials else []
        }
        
    @app.get("/api/settings/google/status")
    async def get_google_auth_status():
        """Get Google authentication status"""
        global google_credentials
        
        status = "not_configured"
        services = []
        project_id = None
        
        if google_credentials:
            status = "authenticated"
            services = ["OCR", "Drive", "Gmail"]
            
            # Get project ID if available
            creds_path = app_settings.get("google_credentials_path")
            if creds_path and os.path.exists(creds_path):
                try:
                    with open(creds_path, 'r') as f:
                        creds_data = json.load(f)
                        project_id = creds_data.get("project_id")
                except:
                    pass
        
        return {
            "status": status,
            "services_available": services,
            "credentials_file_uploaded": bool(app_settings.get("google_credentials_path")),
            "project_id": project_id
        }
        
    @app.get("/api/settings")
    async def get_settings():
        """Get all system settings (with sensitive data redacted)"""
        global app_settings, google_credentials, openai_client, chroma_client
        
        # Check component status
        openai_configured = openai_client is not None
        google_configured = google_credentials is not None
        chromadb_configured = chroma_client is not None and collection is not None
        
        # Get OpenAI API key (redacted)
        api_key = app_settings.get("openai_api_key")
        redacted_key = "sk-..." + api_key[-4:] if api_key and len(api_key) > 8 else None
        
        # Get Google project info
        project_id = None
        creds_path = app_settings.get("google_credentials_path")
        if creds_path and os.path.exists(creds_path):
            try:
                with open(creds_path, 'r') as f:
                    creds_data = json.load(f)
                    project_id = creds_data.get("project_id")
            except Exception:
                pass
        
        return {
            "openai": {
                "configured": openai_configured,
                "api_key": redacted_key,
                "models_available": True if openai_configured else False
            },
            "google": {
                "configured": google_configured,
                "project_id": project_id,
                "services_available": ["OCR", "Drive", "Gmail"] if google_configured else []
            },
            "chromadb": {
                "configured": chromadb_configured,
                "persistent_dir": app_settings.get("chromadb_dir")
            },
            "system": {
                "version": "1.5.60",
                "backend": "minimal"
            }
        }
        
    @app.put("/api/settings/chromadb", response_model=None)
    async def update_chromadb_settings(settings: Dict[str, Any]):
        """Update ChromaDB settings"""
        global app_settings, chroma_client, collection
        
        # Update ChromaDB directory if provided
        if "persistent_dir" in settings:
            app_settings["chromadb_dir"] = settings["persistent_dir"]
            
        # Save settings
        if save_settings():
            # Reinitialize ChromaDB with new settings if needed
            if "persistent_dir" in settings:
                chroma_client, collection = init_chromadb()
                
            return {"success": True, "message": "ChromaDB settings updated"}  
        else:
            raise HTTPException(status_code=500, detail="Failed to save settings")
    
    @app.get("/api/config/check-setup")
    async def check_setup_status():
        """Check if initial setup is needed"""
        global app_settings, google_credentials
        
        openai_configured = bool(app_settings.get("openai_api_key") or os.getenv("OPENAI_API_KEY"))
        google_configured = google_credentials is not None
        
        return {
            "setup_needed": not openai_configured,  # Require at least OpenAI
            "openai_configured": openai_configured,
            "google_configured": google_configured,
            "chromadb_available": chroma_client is not None and collection is not None
        }
        
    @app.get("/api/documents")
    async def get_documents():
        """Get list of documents"""
        global chroma_client, collection
        
        # Check if ChromaDB is initialized
        if not chroma_client or not collection:
            raise HTTPException(status_code=503, detail="ChromaDB not initialized")
        
        try:
            # Get all documents from the collection
            results = collection.get()
            
            # Extract document names from metadata
            documents = []
            documents_by_folder = {"General": []}  # Default folder
            
            # Process documents if any exist
            if results and 'metadatas' in results and results['metadatas']:
                for metadata in results['metadatas']:
                    if metadata and 'filename' in metadata:  # Changed 'source' to 'filename'
                        doc_name = metadata['filename']
                        # Handle duplicates by checking if document is already in the list
                        if doc_name not in documents:
                            documents.append(doc_name)
                            
                            # Add to appropriate folder
                            folder = metadata.get('folder', 'General')
                            if folder not in documents_by_folder:
                                documents_by_folder[folder] = []
                            
                            if doc_name not in documents_by_folder[folder]:
                                documents_by_folder[folder].append(doc_name)
            
            return {
                "documents": documents,
                "documents_by_folder": documents_by_folder,
                "total_count": len(documents)
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
            
    @app.post("/api/documents/upload")
    async def upload_document(
        file: UploadFile = File(...),
        folder: str = Form("General"),
        use_ocr: bool = Form(False)
    ):
        """Upload and process document
        
        This endpoint handles:
        1. Text extraction from documents
        2. OCR processing if needed via Google Vision
        3. Smart chunking using the Contract Intelligence engine
        4. Embedding via OpenAI
        5. Storing vectors in ChromaDB
        """
        global collection, openai_client, contract_intelligence_engine, google_credentials
        
        # Verify required services
        if not collection or not openai_client:
            raise HTTPException(
                status_code=503, 
                detail="Required services not available. Please ensure OpenAI API key is configured."
            )
        
        # Check if OCR is requested but Google credentials are missing
        if use_ocr and not google_credentials:
            raise HTTPException(
                status_code=400,
                detail="OCR requested but Google credentials not configured. Please set up Google credentials first."
            )
        
        try:
            # Read file
            content = await file.read()
            filename = file.filename
            
            # Get file extension
            file_ext = filename.split(".")[-1].lower()
            
            # Extract text from file
            text = None
            page_content = []
            document_metadata = {
                "filename": filename,
                "folder": folder,
                "upload_date": datetime.now().isoformat(),
                "file_type": file_ext,
                "processed_with_ocr": False
            }
            
            # Process based on file type
            if file_ext in ["pdf", "docx", "doc", "txt"]:
                # Save to temporary file
                temp_path = f"temp_{uuid.uuid4()}.{file_ext}"
                with open(temp_path, "wb") as f:
                    f.write(content)
                
                try:
                    # Basic text extraction
                    if file_ext == "pdf":
                        try:
                            import fitz  # PyMuPDF
                            pdf = fitz.open(temp_path)
                            for page_num in range(len(pdf)):
                                page = pdf[page_num]
                                page_text = page.get_text()
                                if page_text.strip():
                                    page_content.append({
                                        "page": page_num + 1,
                                        "text": page_text
                                    })
                        except Exception as e:
                            print(f"PDF extraction failed: {e}")
                            # If text extraction fails, try OCR if enabled
                            if use_ocr:
                                page_content = await process_with_ocr(temp_path, document_metadata)
                    
                    elif file_ext in ["docx", "doc"]:
                        try:
                            import docx
                            doc = docx.Document(temp_path)
                            text = "\n".join([p.text for p in doc.paragraphs])
                            page_content.append({
                                "page": 1,
                                "text": text
                            })
                        except Exception as e:
                            print(f"DOCX extraction failed: {e}")
                            # If text extraction fails, try OCR if enabled
                            if use_ocr:
                                page_content = await process_with_ocr(temp_path, document_metadata)
                    
                    elif file_ext == "txt":
                        with open(temp_path, "r", encoding="utf-8", errors="ignore") as f:
                            text = f.read()
                            page_content.append({
                                "page": 1,
                                "text": text
                            })
                    
                    # If no text extracted, try OCR automatically
                    if not page_content or all(not p["text"].strip() for p in page_content):
                        # Try OCR if explicitly requested or as automatic fallback
                        try:
                            page_content = await process_with_ocr(temp_path, document_metadata)
                        except Exception as ocr_error:
                            print(f"Automatic OCR fallback failed: {str(ocr_error)}")
                            # Continue with empty page_content if OCR fails
                
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except PermissionError:
                            print(f"Note: Could not delete temporary file {temp_path} - file is in use")
            
            # Process images always with OCR - they're inherently image-based and need OCR
            elif file_ext in ["jpg", "jpeg", "png", "tiff", "bmp", "gif"]:
                # Save to temporary file
                temp_path = f"temp_{uuid.uuid4()}.{file_ext}"
                with open(temp_path, "wb") as f:
                    f.write(content)
                
                try:
                    # Always process images with OCR regardless of checkbox
                    page_content = await process_with_ocr(temp_path, document_metadata)
                except Exception as ocr_error:
                    print(f"OCR for image failed: {str(ocr_error)}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Failed to process image with OCR. Please check that Google credentials are configured correctly."
                    )
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        try:
                            os.remove(temp_path)
                        except PermissionError:
                            print(f"Note: Could not delete temporary file {temp_path} - file is in use")
            
            else:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unsupported file type: {file_ext}. Only PDF, DOC, DOCX, TXT, and common image formats are supported."
                )
            
            # If no text could be extracted after all attempts (including automatic OCR)
            if not page_content or all(not p["text"].strip() for p in page_content):
                raise HTTPException(
                    status_code=400,
                    detail="Failed to extract text from document even with automatic OCR. The document might be encrypted, damaged, or contains no extractable text."
                )
            
            # Combine all text for processing
            full_text = "\n\n".join([p["text"] for p in page_content])
            
            # Use contract intelligence engine for smart chunking if available
            chunks = []
            if contract_intelligence_engine and len(full_text) > 100:  # Only use for substantial text
                try:
                    # Use contract intelligence for smart chunking
                    contract_chunks = contract_intelligence_engine.chunk_contract(
                        full_text, chunk_size=1000, chunk_overlap=100
                    )
                    chunks = contract_chunks
                except Exception as e:
                    print(f"Contract intelligence chunking failed: {e}")
                    # Fallback to simple chunking
                    chunks = simple_text_chunker(full_text)
            else:
                # Use simple chunking
                chunks = simple_text_chunker(full_text)
            
            # Prepare for embedding and storage
            texts = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{filename}_{i}_{uuid.uuid4()}"
                chunk_metadata = document_metadata.copy()
                chunk_metadata.update({
                    "chunk_index": i,
                    "chunk_total": len(chunks),
                    "chunk_id": chunk_id,
                })
                
                texts.append(chunk)
                metadatas.append(chunk_metadata)
                ids.append(chunk_id)
            
            # Store in ChromaDB with embeddings
            if texts:
                collection.add(
                    documents=texts,
                    metadatas=metadatas,
                    ids=ids
                )
                
                return {
                    "success": True,
                    "message": f"Document processed successfully: {filename}",
                    "chunks": len(chunks),
                    "folder": folder
                }
            else:
                raise HTTPException(
                    status_code=400,
                    detail="Failed to process document: No valid text chunks were created."
                )
                
        except HTTPException:
            raise
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Document processing failed: {str(e)}")
    
    @app.delete("/api/documents/{document_name}")
    async def delete_document(document_name: str):
        """Delete a document and all its chunks"""
        global collection
        
        if not collection:
            raise HTTPException(status_code=503, detail="ChromaDB not initialized")
            
        try:
            # Get all chunks for this document
            results = collection.get(
                where={"filename": document_name}
            )
            
            # Check if document exists
            if not results or not results["ids"]:
                raise HTTPException(status_code=404, detail=f"Document not found: {document_name}")
                
            # Delete all chunks for this document
            collection.delete(ids=results["ids"])
            
            return {
                "success": True,
                "message": f"Document deleted: {document_name}",
                "chunks_deleted": len(results["ids"])
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")
    
    @app.delete("/api/folders/{folder_name}")
    async def delete_folder(folder_name: str):
        """Delete a folder and all documents in it"""
        global collection
        
        if not collection:
            raise HTTPException(status_code=503, detail="ChromaDB not initialized")
            
        try:
            # Get all chunks for this folder
            results = collection.get(
                where={"folder": folder_name}
            )
            
            # Check if folder exists and has documents
            if not results or not results["ids"]:
                raise HTTPException(status_code=404, detail=f"Folder not found or empty: {folder_name}")
                
            # Delete all chunks for this folder
            collection.delete(ids=results["ids"])
            
            return {
                "success": True,
                "message": f"Folder deleted: {folder_name}",
                "chunks_deleted": len(results["ids"])
            }
            
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to delete folder: {str(e)}")
            
    @app.post("/api/chat")
    async def chat_query(request: Request):
        """Process a chat query against the document collection"""
        global collection, openai_client
        
        # Check if required services are available
        if not collection or not openai_client:
            raise HTTPException(
                status_code=503, 
                detail="Required services not available. Please ensure OpenAI API key is configured."
            )
            
        try:
            # Parse request
            data = await request.json()
            query = data.get("query")
            target_documents = data.get("target_documents")
            target_folder = data.get("target_folder")
            
            if not query:
                raise HTTPException(status_code=400, detail="Query is required")
                
            # Build search filters
            where_clause = {}
            if target_documents and len(target_documents) > 0 and target_documents[0] != "all":
                where_clause["filename"] = target_documents[0]
                
            if target_folder and target_folder != "all":
                where_clause["folder"] = target_folder
                
            # Search for similar chunks
            results = None
            if where_clause:
                results = collection.query(
                    query_texts=[query],
                    where=where_clause,
                    n_results=5
                )
            else:
                # Search all documents
                results = collection.query(
                    query_texts=[query],
                    n_results=5
                )
                
            if not results or not results["documents"] or not results["documents"][0]:
                return {
                    "success": True,
                    "answer": "I couldn't find any relevant information in the documents.",
                    "source_info": []
                }
                
            # Prepare source context
            contexts = []
            source_info = []
            
            for i, doc in enumerate(results["documents"][0]):
                metadata = results["metadatas"][0][i]
                similarity = float(results["distances"][0][i]) if "distances" in results else 0
                
                # Transform similarity score (lower distance means higher similarity)
                similarity_score = max(0, 1 - similarity)
                
                # Get chunk information safely with fallbacks for backward compatibility
                chunk_index = metadata.get('chunk_index', 0) + 1  # +1 for display (1-indexed)
                chunk_total = metadata.get('chunk_total', 1)      # Default to 1 if not available
                
                context = f"Context from '{metadata['filename']}', chunk {chunk_index}/{chunk_total}:\n{doc}"
                contexts.append(context)
                
                # Create source info for frontend with backward compatibility
                source_info.append({
                    "filename": metadata["filename"],
                    "folder": metadata.get("folder", "General"),
                    "chunk_index": chunk_index,                   # Already adjusted above (+1)
                    "total_chunks": chunk_total,                  # Safe value from above
                    "similarity": similarity_score,
                    # Add language information if available
                    "language": metadata.get("language_detected", "unknown")
                })
                
            # Combine contexts
            full_context = "\n\n" + "\n\n".join(contexts)
            
            # Prepare prompt for OpenAI
            prompt = [
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant that answers questions based on the provided document context. "
                              "Only use information from the context to answer questions. If the information isn't in the "
                              "context, say you don't know. Be concise but thorough."
                },
                {
                    "role": "user",
                    "content": f"Based on the following document contexts, please answer this question: {query}\n\n{full_context}"
                }
            ]
            
            # Generate response from OpenAI
            response = openai_client.chat.completions.create(
                model="gpt-4-turbo",  # Use appropriate model
                messages=prompt,
                temperature=0,
                max_tokens=800
            )
            
            answer = response.choices[0].message.content
            
            return {
                "success": True,
                "answer": answer,
                "source_info": source_info
            }
            
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Chat query failed: {str(e)}")

async def process_with_ocr(file_path, metadata):
    """Process document with OCR using Google Vision API
    
    This function is optimized for Hebrew document processing but works for all languages.
    It uses DOCUMENT_TEXT_DETECTION to better handle multi-page PDF documents with complex layouts.
    """
    global google_credentials
    
    if not google_credentials:
        print("[ERROR] Google credentials not available for OCR")
        raise ValueError("Google credentials not available")
    
    # Detect if file is PDF or image for proper handling
    is_pdf = file_path.lower().endswith('.pdf')
    page_results = []
    
    try:
        # Build vision service with fresh credentials
        vision_service = build('vision', 'v1', credentials=google_credentials)
        
        # For PDFs, we may need to extract individual pages first (simplified implementation)
        if is_pdf:
            try:
                import fitz  # PyMuPDF
                pdf = fitz.open(file_path)
                
                # Process each page (up to 5 pages for demonstration)
                for page_num in range(min(len(pdf), 5)):  # Limit to 5 pages for now
                    page = pdf[page_num]
                    
                    # Create a temporary image file for this page
                    img_path = f"{file_path}_page_{page_num}.png"
                    try:
                        pix = page.get_pixmap(dpi=300)  # Higher resolution for better OCR
                        pix.save(img_path)
                        
                        # Process this page image
                        page_text = await _perform_ocr_request(img_path)
                        if page_text:
                            page_results.append({"page": page_num + 1, "text": page_text})
                            
                    except Exception as page_error:
                        print(f"Error processing PDF page {page_num}: {str(page_error)}")
                    finally:
                        # Clean up temporary page image
                        if os.path.exists(img_path):
                            try:
                                os.remove(img_path)
                            except:
                                pass
                
                # Close the PDF properly
                pdf.close()
                
            except ImportError:
                print("PyMuPDF not available, using direct PDF processing")
                # Fall back to single-page processing of the PDF
                page_text = await _perform_ocr_request(file_path)
                if page_text:
                    page_results.append({"page": 1, "text": page_text})
        else:
            # Regular image processing
            page_text = await _perform_ocr_request(file_path)
            if page_text:
                page_results.append({"page": 1, "text": page_text})
        
        # Update metadata
        metadata["processed_with_ocr"] = True
        metadata["language_detected"] = "he" if _contains_hebrew(str(page_results)) else "en"
        
        return page_results
        
    except Exception as e:
        print(f"OCR processing failed: {e}")
        traceback.print_exc()
        raise ValueError(f"OCR processing failed: {str(e)}")

async def _perform_ocr_request(file_path):
    """Helper function to perform the actual OCR request to Vision API"""
    global google_credentials
    
    try:
        # Build vision service
        vision_service = build('vision', 'v1', credentials=google_credentials)
        
        # Read file content
        with open(file_path, "rb") as image_file:
            content = image_file.read()
        
        # Encode image content
        encoded_content = base64.b64encode(content).decode("utf-8")
        
        # Prepare request with enhanced options for Hebrew support
        batch_request = {
            'requests': [
                {
                    'image': {
                        'content': encoded_content
                    },
                    'features': [
                        {
                            'type': 'DOCUMENT_TEXT_DETECTION',  # Better for documents than TEXT_DETECTION
                            'maxResults': 1
                        }
                    ],
                    'imageContext': {
                        'languageHints': ['he', 'en', 'iw']  # Hebrew, English hints (iw is older code for Hebrew)
                    }
                }
            ]
        }
        
        # Make API request
        response = vision_service.images().annotate(body=batch_request).execute()
        
        # Extract text annotations (full text is in textAnnotations[0])
        if not response['responses'] or 'fullTextAnnotation' not in response['responses'][0]:
            print("No text detected in the document")
            return ""
            
        # Get full text from the full text annotation (better for document structure)
        text = response['responses'][0]['fullTextAnnotation']['text']
        return text
        
    except Exception as e:
        print(f"OCR request failed: {e}")
        return ""
        
def _contains_hebrew(text):
    """Check if text contains Hebrew characters"""
    # Hebrew Unicode range: \u0590-\u05FF
    return bool(re.search(r'[\u0590-\u05FF]', text))

def simple_text_chunker(text, chunk_size=1000, chunk_overlap=100):
    """Simple text chunking algorithm that respects paragraph boundaries where possible"""
    if not text:
        return []
        
    # Split text into paragraphs
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Initialize chunks
    chunks = []
    current_chunk = []
    current_size = 0
    
    for paragraph in paragraphs:
        paragraph_size = len(paragraph)
        
        # If single paragraph is too big, split it by sentences
        if paragraph_size > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', paragraph)
            for sentence in sentences:
                sentence_size = len(sentence)
                
                # If current chunk plus this sentence would be too big, store current chunk
                if current_size + sentence_size > chunk_size and current_chunk:
                    chunks.append(" ".join(current_chunk))
                    # Keep some overlap by retaining the last sentence if possible
                    overlap_content = current_chunk[-1] if current_chunk else ""
                    current_chunk = [overlap_content] if overlap_content and len(overlap_content) < chunk_overlap else []
                    current_size = len(overlap_content) if current_chunk else 0
                
                # Add sentence to current chunk
                current_chunk.append(sentence)
                current_size += sentence_size
        else:
            # If current chunk plus this paragraph would be too big, store current chunk
            if current_size + paragraph_size > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                # Keep some overlap
                overlap_size = 0
                overlap_content = []
                for p in reversed(current_chunk):
                    if overlap_size + len(p) <= chunk_overlap:
                        overlap_content.insert(0, p)
                        overlap_size += len(p)
                    else:
                        break
                current_chunk = overlap_content
                current_size = overlap_size
            
            # Add paragraph to current chunk
            current_chunk.append(paragraph)
            current_size += paragraph_size
    
    # Add final chunk if it's not empty
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    # Handle edge case of no chunks created
    if not chunks and text:
        # Fallback to simple character-based chunking
        chunks = []
        for i in range(0, len(text), chunk_size - chunk_overlap):
            chunks.append(text[i:i + chunk_size])
    
    return chunks

def find_available_port(start_port=8503):
    """Find an available port starting from start_port"""
    import socket
    
    # Only try port 8503, no fallbacks
    port = start_port
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('127.0.0.1', port))
            return port
    except OSError:
        print(f"[ERROR] Port {port} is already in use. Please close any other instances.")
        return None

if __name__ == "__main__":
    import time
    
    # Check for test-only mode (for CI/CD environments)
    test_mode = "--test-only" in sys.argv
    verbose_mode = "--verbose" in sys.argv
    
    if verbose_mode:
        print("[DEBUG] Running in verbose diagnostic mode")
        print(f"[DEBUG] Python version: {sys.version}")
        print(f"[DEBUG] Platform: {platform.platform()}")
        print(f"[DEBUG] Executable: {sys.executable}")
        print(f"[DEBUG] Current working directory: {os.getcwd()}")
        print(f"[DEBUG] sys.path: {sys.path}")
        
        # Print available modules for diagnosis
        print("[DEBUG] Checking critical dependency availability:")
        dependencies = [
            "fastapi", "uvicorn", "openai", "chromadb", "google.oauth2", 
            "googleapiclient", "starlette", "pydantic", "tiktoken"
        ]
        for dep in dependencies:
            try:
                if dep == "openai":
                    import openai
                    print(f"[DEBUG] {dep} version: {openai.__version__}")
                elif dep == "fastapi":
                    import fastapi
                    print(f"[DEBUG] {dep} version: {fastapi.__version__}")
                elif dep == "uvicorn":
                    import uvicorn
                    print(f"[DEBUG] {dep} version: {uvicorn.__version__}")
                elif dep == "chromadb":
                    import chromadb
                    print(f"[DEBUG] {dep} available")
                elif dep == "google.oauth2":
                    from google.oauth2 import service_account
                    print(f"[DEBUG] {dep} available")
                elif dep == "googleapiclient":
                    import googleapiclient
                    print(f"[DEBUG] {dep} available")
                elif dep == "starlette":
                    import starlette
                    print(f"[DEBUG] {dep} version: {starlette.__version__}")
                elif dep == "pydantic":
                    import pydantic
                    print(f"[DEBUG] {dep} version: {pydantic.__version__}")
                elif dep == "tiktoken":
                    import tiktoken
                    print(f"[DEBUG] {dep} available")
                else:
                    print(f"[DEBUG] {dep} imported successfully")
            except ImportError as e:
                print(f"[DEBUG] {dep} IMPORT ERROR: {e}")
            except Exception as e:
                print(f"[DEBUG] {dep} ERROR: {e}")
                
        # Check if we're in a PyInstaller bundle
        if getattr(sys, 'frozen', False):
            print("[DEBUG] Running from PyInstaller bundle")
            print(f"[DEBUG] sys._MEIPASS: {getattr(sys, '_MEIPASS', 'Not available')}")
            
            # Check for any file access issues in the bundle
            try:
                bundle_dir = getattr(sys, '_MEIPASS', os.path.dirname(sys.executable))
                print(f"[DEBUG] Bundle directory: {bundle_dir}")
                files = os.listdir(bundle_dir)
                print(f"[DEBUG] First 10 files in bundle: {files[:10]}")
            except Exception as e:
                print(f"[DEBUG] Error listing bundle files: {e}")
    
    print("[INFO] Starting Contract Intelligence Minimal Backend Server...")
    
    # Use port 8503 only - no fallbacks
    port = find_available_port()
    if port is None:
        print("[ERROR] Port 8503 is not available. Please close other instances before continuing.")
        exit(1)
    
    print(f"[INFO] Server will be available at: http://127.0.0.1:{port}")
    
    # Test mode for CI/CD environments - start server briefly then exit
    if test_mode:
        print("[INFO] Running in test-only mode for CI/CD environment")
        if verbose_mode:
            # Perform basic initialization to test dependency loading
            print("[DEBUG] Test-only mode with initialization check")
            try:
                initialize()
                print("[DEBUG] Initialization successful in test-only mode")
            except Exception as e:
                print(f"[ERROR] Initialization failed in test-only mode: {e}")
                traceback.print_exc()
                sys.exit(1)
        sys.exit(0)
    
    # Normal operation mode - run server continuously
    if FASTAPI_AVAILABLE:
        try:
            print("[INFO] Starting FastAPI server...")
            uvicorn.run(
                app,
                host="127.0.0.1",
                port=port,
                log_level="debug" if verbose_mode else "info",
                access_log=verbose_mode
            )
        except Exception as e:
            print(f"[ERROR] Failed to start FastAPI server: {e}")
            traceback.print_exc()
            exit(1)
    else:
        print("[ERROR] FastAPI not available. Cannot start server.")
        print("[DEBUG] Available modules:")
        for name, module in sys.modules.items():
            if not name.startswith('_'):
                print(f"  - {name}")
        exit(1)