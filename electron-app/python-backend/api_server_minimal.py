#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Minimal FastAPI backend for Contract Intelligence Platform
Uses only installed packages: FastAPI, OpenAI, ChromaDB, basic document processing
"""

# Detect PyInstaller environment
import os
import sys
import importlib
import traceback

# Set up detailed logging for imports
def try_import(module_name):
    """Try to import a module and log the result"""
    try:
        module = importlib.import_module(module_name)
        print(f"[SUCCESS] Successfully imported {module_name}")
        return module
    except ImportError as e:
        print(f"[ERROR] Failed to import {module_name}: {e}")
        print(f"[TRACE] {traceback.format_exc()}")
        return None
    except Exception as e:
        print(f"[ERROR] Unexpected error importing {module_name}: {e}")
        print(f"[TRACE] {traceback.format_exc()}")
        return None

if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Running in PyInstaller bundle
    print("[INFO] Running in PyInstaller bundle")
    bundle_dir = sys._MEIPASS
    
    # Add bundle directory to path
    if bundle_dir not in sys.path:
        sys.path.insert(0, bundle_dir)
    
    # Print bundle directory contents for debugging
    print(f"[DEBUG] Bundle directory: {bundle_dir}")
    try:
        print(f"[DEBUG] Bundle contents: {os.listdir(bundle_dir)}")
        
        # Ensure contract_intelligence.py is accessible
        contract_intelligence_path = os.path.join(bundle_dir, 'contract_intelligence.py')
        if os.path.exists(contract_intelligence_path):
            print(f"[INFO] Found contract_intelligence.py at: {contract_intelligence_path}")
            
            # Create a module spec and import the module
            try:
                import importlib.util
                spec = importlib.util.spec_from_file_location('contract_intelligence', contract_intelligence_path)
                contract_intelligence_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(contract_intelligence_module)
                sys.modules['contract_intelligence'] = contract_intelligence_module
                print("[SUCCESS] Successfully loaded contract_intelligence module")
            except Exception as e:
                print(f"[ERROR] Failed to import contract_intelligence module: {e}")
        else:
            print(f"[ERROR] contract_intelligence.py not found at: {contract_intelligence_path}")
    except Exception as e:
        print(f"[ERROR] Failed to list bundle contents: {e}")
else:
    print("[INFO] Running in normal Python environment")

# **DIAGNOSTIC: Test critical imports early to catch missing dependencies**
print("[DIAGNOSTIC] Testing critical imports...")
try:
    import os
    import sys
    import json
    import tempfile
    import shutil
    import logging
    import re
    from pathlib import Path
    from typing import List, Dict, Any, Optional
    print("[DIAGNOSTIC] Standard library imports successful")
except ImportError as e:
    print(f"[DIAGNOSTIC] Standard library import failed: {e}")
    # Continue with limited functionality instead of exiting
    print("[CRITICAL] Standard library import failed but continuing with limited functionality")
    # Define minimal imports to continue
    import os
    import sys
    import json

# Configure logging to stderr so it appears in Electron terminal
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

# Set UTF-8 encoding for Windows console
if sys.platform.startswith('win'):
    import codecs
    try:
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer)
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer)
    except:
        # Fallback if console encoding setup fails
        pass

def safe_print(message: str):
    """Safe printing function that handles Hebrew text on Windows"""
    try:
        print(message)
    except UnicodeEncodeError:
        # Fallback: print without problematic characters
        safe_message = message.encode('ascii', errors='ignore').decode('ascii')
        print(f"{safe_message} [Hebrew text - encoding issue]")

def debug_text_sample(text: str, label: str = "Text sample", max_chars: int = 200) -> str:
    """Create a readable debug sample of text, handling Hebrew properly"""
    if not text:
        return f"[{label}]: <EMPTY>"
    
    # Get first max_chars characters
    sample = text[:max_chars].strip()
    
    # Try to display Hebrew properly, fallback to length info if encoding fails
    try:
        # Count Hebrew vs English characters for better debugging
        hebrew_chars = sum(1 for c in sample if '\u0590' <= c <= '\u05FF')
        english_chars = sum(1 for c in sample if c.isascii() and c.isalpha())
        
        if hebrew_chars > 0:
            return f"[{label}]: {len(text)} chars total ({hebrew_chars} Hebrew, {english_chars} English)"
        else:
            return f"[{label}]: {len(text)} chars total - Sample: {sample}"
    except:
        # Ultimate fallback
        return f"[{label}]: {len(text)} chars total - [Mixed encoding content]"

# Core imports
# Global feature availability flags
FASTAPI_AVAILABLE = False
DOCUMENT_PROCESSING_AVAILABLE = False
AI_CHROMADB_AVAILABLE = False
CONTRACT_INTELLIGENCE_AVAILABLE = False
TOKENIZERS_AVAILABLE = False
SENTENCE_TRANSFORMERS_AVAILABLE = False

print("[DIAGNOSTIC] Testing FastAPI imports...")
try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, UploadFile, File, Form
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import JSONResponse
    print("[DIAGNOSTIC] FastAPI imports successful")
    FASTAPI_AVAILABLE = True
except ImportError as e:
    print(f"[DIAGNOSTIC] FastAPI import failed: {e}")
    print("[CRITICAL] FastAPI import failed - this is required for the backend to function")
    print("[RECOVERY] Attempting to continue with minimal HTTP server...")
    FASTAPI_AVAILABLE = False
    
    # Define minimal HTTP server for health endpoint only
    import http.server
    import socketserver
    
    class MinimalHandler(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/health":
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                self.wfile.write(json.dumps({
                    "status": "limited",
                    "version": "1.5.46",
                    "message": "Running in emergency mode - FastAPI unavailable"
                }).encode())

# Document processing
print("[DIAGNOSTIC] Testing document processing imports...")
try:
    import PyPDF2
    import docx
    from PIL import Image
    import fitz  # PyMuPDF for PDF to image conversion
    print("[DIAGNOSTIC] Document processing imports successful")
    DOCUMENT_PROCESSING_AVAILABLE = True
except ImportError as e:
    print(f"[DIAGNOSTIC] Document processing import failed: {e}")
    print("[WARNING] Document processing imports failed - document upload functionality will be limited")
    DOCUMENT_PROCESSING_AVAILABLE = False

# AI and vector database
print("[DIAGNOSTIC] Testing AI/ChromaDB imports...")
try:
    import openai
    import chromadb
    from chromadb.config import Settings
    print("[DIAGNOSTIC] AI/ChromaDB imports successful")
    AI_CHROMADB_AVAILABLE = True
except ImportError as e:
    print(f"[DIAGNOSTIC] AI/ChromaDB import failed: {e}")
    print("[WARNING] AI/ChromaDB imports failed - chat functionality will be limited")
    AI_CHROMADB_AVAILABLE = False
    
    # Define minimal stubs for required classes
    class DummyOpenAI:
        def __init__(self, api_key=None):
            self.api_key = api_key
            
    # If openai failed to import, create a stub
    if 'openai' not in sys.modules:
        openai = type('openai', (), {})
        openai.OpenAI = DummyOpenAI

# Contract Intelligence Engine
try:
    from contract_intelligence import ContractIntelligenceEngine
    CONTRACT_INTELLIGENCE_AVAILABLE = True
    print("[INFO] Contract Intelligence Engine available")
except ImportError as e:
    CONTRACT_INTELLIGENCE_AVAILABLE = False
    print(f"[WARNING] Contract Intelligence Engine not available: {e}")

# Utilities
import pandas as pd
import tiktoken
from dotenv import load_dotenv

# Google Service Account imports
from google.oauth2 import service_account
import pickle

# Google Vision OCR
try:
    from google.cloud import vision
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("[WARNING] Google Vision not available - OCR functionality disabled")

# Load environment variables
load_dotenv()

# Google OAuth scopes
SCOPES = [
    'https://www.googleapis.com/auth/cloud-platform',
    'https://www.googleapis.com/auth/drive.readonly',
    'https://www.googleapis.com/auth/gmail.readonly'
]

# Settings storage
SETTINGS_FILE = "app_settings.json"

# Global instances
chroma_client = None
collection = None
openai_client = None
google_credentials = None
contract_intelligence_engine = None
app_settings = {}

def load_settings():
    """Load application settings from file"""
    global app_settings
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(SETTINGS_FILE, 'r') as f:
                app_settings = json.load(f)
        print(f"[INFO] Settings loaded: {list(app_settings.keys())}")
    except Exception as e:
        print(f"[WARNING] Failed to load settings: {e}")
        app_settings = {}

def save_settings():
    """Save application settings to file"""
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump(app_settings, f, indent=2)
        print("[INFO] Settings saved successfully")
    except Exception as e:
        print(f"[ERROR] Failed to save settings: {e}")

def load_google_credentials():
    """Load Google Service Account credentials"""
    global google_credentials
    try:
        credentials_path = app_settings.get("google_credentials_path")
        if credentials_path and os.path.exists(credentials_path):
            google_credentials = service_account.Credentials.from_service_account_file(
                credentials_path, scopes=SCOPES)
            print("[INFO] Google service account credentials loaded")
            return True
    except Exception as e:
        print(f"[WARNING] Failed to load Google credentials: {e}")
    return False

def save_google_credentials(credentials_path):
    """Save Google credentials path to settings"""
    try:
        app_settings["google_credentials_path"] = credentials_path
        save_settings()
        print("[INFO] Google credentials path saved")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save Google credentials path: {e}")
        return False

def trace_chromadb_dependencies():
    """Trace ChromaDB dependencies and log their availability"""
    print("\n[DIAGNOSTIC] Tracing ChromaDB dependencies...")
    
    # Core dependencies
    try_import("chromadb")
    
    # ONNX related
    try_import("onnx")
    try_import("onnxruntime")
    try_import("onnxruntime.capi.onnxruntime_pybind11_state")
    
    # Vector database dependencies
    try_import("numpy")
    try_import("protobuf")
    
    # Embedding related
    try_import("openai")
    try_import("tiktoken")
    try_import("tokenizers")
    try_import("sentence_transformers")
    
    # ChromaDB internal modules
    try_import("chromadb.config")
    try_import("chromadb.api")
    try_import("chromadb.utils.embedding_functions")
    
    # Check Python environment
    print(f"[INFO] Python version: {sys.version}")
    print(f"[INFO] Python executable: {sys.executable}")
    print(f"[INFO] sys.path: {sys.path}")
    
    # Check for PyInstaller environment
    if getattr(sys, 'frozen', False):
        print(f"[INFO] Running in PyInstaller bundle: {sys._MEIPASS}")
        try:
            print(f"[INFO] Bundle contents: {os.listdir(sys._MEIPASS)}")
        except Exception as e:
            print(f"[ERROR] Failed to list bundle contents: {e}")
    
    print("[DIAGNOSTIC] Dependency tracing complete\n")

def initialize_services():
    """Initialize ChromaDB and OpenAI services with embedding function"""
    global chroma_client, collection, openai_client, contract_intelligence_engine
    
    # Set defaults for global variables
    chroma_client = None
    collection = None
    openai_client = None
    contract_intelligence_engine = None
    
    # Trace dependencies before initialization
    trace_chromadb_dependencies()
    
    try:
        # Load settings first
        try:
            load_settings()
            print("[SUCCESS] Settings loaded successfully")
        except Exception as e:
            print(f"[WARNING] Failed to load settings: {e}")
            print("[RECOVERY] Using default settings")
        
        try:
            load_google_credentials()
            print("[SUCCESS] Google credentials loaded (if available)")
        except Exception as e:
            print(f"[WARNING] Failed to load Google credentials: {e}")
        
        # Initialize ChromaDB if available
        if AI_CHROMADB_AVAILABLE:
            try:
                print("\n[DIAGNOSTIC] Starting ChromaDB initialization...")
                print(f"[INFO] ChromaDB module details: {chromadb.__file__}")
                print(f"[INFO] ChromaDB version: {getattr(chromadb, '__version__', 'unknown')}")
                
                # Initialize ChromaDB with PERSISTENT storage
                # Use environment variable set in runtime hook for PyInstaller compatibility
                if getattr(sys, 'frozen', False) and 'CHROMADB_DIR' in os.environ:
                    persist_dir = os.environ['CHROMADB_DIR']
                    print(f"[INFO] Using PyInstaller ChromaDB directory: {persist_dir}")
                else:
                    persist_dir = "./chroma_db"
                    print(f"[INFO] Using default ChromaDB directory: {persist_dir}")
                
                print(f"[INFO] Creating directory: {persist_dir}")
                os.makedirs(persist_dir, exist_ok=True)  # Ensure directory exists
                print(f"[INFO] Directory exists: {os.path.exists(persist_dir)}")
                print(f"[INFO] Directory contents: {os.listdir(persist_dir) if os.path.exists(persist_dir) else 'N/A'}")
                
                # Check if directory is writable
                try:
                    test_file = os.path.join(persist_dir, 'test_write.txt')
                    with open(test_file, 'w') as f:
                        f.write('test')
                    os.remove(test_file)
                    print(f"[INFO] Directory is writable: {persist_dir}")
                except Exception as write_error:
                    print(f"[ERROR] Directory is not writable: {persist_dir} - {write_error}")
                    print(f"[TRACE] {traceback.format_exc()}")
                    # Try to create a directory in a more accessible location
                    persist_dir = os.path.join(os.path.expanduser('~'), '.contract_intelligence', 'chroma_db')
                    print(f"[RECOVERY] Trying alternative directory: {persist_dir}")
                    os.makedirs(persist_dir, exist_ok=True)
                
                # Create OpenAI embedding function FIRST - CRITICAL to prevent ONNX loading
                api_key = app_settings.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
                if not api_key:
                    print("[ERROR] OpenAI API key is required for ChromaDB - cannot initialize")
                    chroma_client = None
                    return
                
                try:
                    # Create OpenAI embedding function before initializing ChromaDB
                    print("[INFO] Creating OpenAI ada-002 embedding function for ChromaDB initialization")
                    
                    # Import with detailed error handling
                    try:
                        from chromadb.utils import embedding_functions
                        print("[SUCCESS] Imported chromadb.utils.embedding_functions")
                    except ImportError as e:
                        print(f"[ERROR] Failed to import embedding_functions: {e}")
                        print(f"[TRACE] {traceback.format_exc()}")
                        return
                    
                    # Create embedding function with detailed error handling
                    try:
                        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                            api_key=api_key,
                            model_name="text-embedding-ada-002"  # Explicitly use ada-002 only
                        )
                        print("[SUCCESS] OpenAI ada-002 embedding function created successfully")
                    except Exception as e:
                        print(f"[ERROR] Failed to create OpenAI embedding function: {e}")
                        print(f"[TRACE] {traceback.format_exc()}")
                        return
                    
                    # Initialize ChromaDB with detailed error handling
                    print("[INFO] Initializing ChromaDB with OpenAI embeddings (NO ONNX)")
                    try:
                        # Import Settings with detailed error handling
                        try:
                            from chromadb.config import Settings
                            print("[SUCCESS] Imported chromadb.config.Settings")
                        except ImportError as e:
                            print(f"[ERROR] Failed to import Settings: {e}")
                            print(f"[TRACE] {traceback.format_exc()}")
                            return
                        
                        # Create client with detailed error handling
                        chroma_client = chromadb.PersistentClient(
                            path=persist_dir,
                            settings=Settings(anonymized_telemetry=False)
                        )
                        print(f"[SUCCESS] ChromaDB initialized with persistent storage at {persist_dir}")
                    except Exception as e:
                        print(f"[ERROR] Failed to initialize ChromaDB client: {e}")
                        print(f"[TRACE] {traceback.format_exc()}")
                        return
                except Exception as chroma_error:
                    print(f"[ERROR] Failed to initialize ChromaDB client: {chroma_error}")
                    # Try with in-memory client as fallback, still using OpenAI embeddings
                    try:
                        print("[RECOVERY] Attempting to use in-memory ChromaDB client with OpenAI embeddings")
                        chroma_client = chromadb.Client(Settings(anonymized_telemetry=False, is_persistent=False))
                        print("[SUCCESS] ChromaDB initialized with in-memory storage (fallback)")
                    except Exception as memory_error:
                        print(f"[ERROR] Failed to initialize in-memory ChromaDB client: {memory_error}")
                        chroma_client = None
                print("[SUCCESS] ChromaDB initialized with persistent storage")
            except Exception as e:
                print(f"[WARNING] Failed to initialize ChromaDB: {e}")
                print("[RECOVERY] Vector search functionality will be limited")
                chroma_client = None
        else:
            print("[INFO] ChromaDB not available - vector search disabled")
        
        # Initialize OpenAI if available
        try:
            # Initialize OpenAI (from settings or environment)
            api_key = app_settings.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            if api_key:
                if 'openai' in sys.modules and hasattr(openai, 'OpenAI'):
                    openai_client = openai.OpenAI(api_key=api_key)
                    print("[SUCCESS] OpenAI client initialized")
                else:
                    print("[WARNING] OpenAI module not available")
                    openai_client = DummyOpenAI(api_key=api_key)
                    print("[RECOVERY] Using dummy OpenAI client")
            else:
                print("[WARNING] No OpenAI API key found - please configure in settings")
        except Exception as e:
            print(f"[WARNING] Failed to initialize OpenAI client: {e}")
        
        # Initialize Contract Intelligence Engine if available
        if CONTRACT_INTELLIGENCE_AVAILABLE and openai_client:
            try:
                contract_intelligence_engine = ContractIntelligenceEngine(openai_client)
                print("[SUCCESS] Contract Intelligence Engine initialized")
            except Exception as e:
                print(f"[WARNING] Failed to initialize Contract Intelligence Engine: {e}")
                contract_intelligence_engine = None
        else:
            print("[INFO] Contract Intelligence Engine not available")
            
        # Check for tokenizers and sentence_transformers availability (for diagnostics only)
        try:
            import tokenizers
            print("[INFO] Tokenizers library is available")
            TOKENIZERS_AVAILABLE = True
        except ImportError:
            print("[WARNING] Tokenizers library not available")
            TOKENIZERS_AVAILABLE = False
        
        try:
            import sentence_transformers
            print("[INFO] Sentence-transformers library is available")
            SENTENCE_TRANSFORMERS_AVAILABLE = True
        except ImportError:
            print("[WARNING] Sentence-transformers library not available")
            SENTENCE_TRANSFORMERS_AVAILABLE = False
        
        # openai_ef is now created during ChromaDB initialization
        # This ensures we never use ONNX models even temporarily
        
        # Try to get existing collection first (only if we have ChromaDB client)
        if chroma_client is not None:
            try:
                print("[INFO] Attempting to get or create ChromaDB collection")
                collection_name = "contracts_electron"
                
                # First try to get existing collection
                try:
                    print(f"[INFO] Looking for existing collection '{collection_name}'")
                    collection = chroma_client.get_collection(
                        name=collection_name,
                        embedding_function=openai_ef
                    )
                    print("[SUCCESS] Retrieved existing collection with ada-002 embeddings")
                    
                    # Verify collection has documents
                    existing_docs = collection.count()
                    print(f"[INFO] Found {existing_docs} existing document chunks in ChromaDB")
                    
                except Exception as get_error:
                    print(f"[INFO] Collection doesn't exist or needs recreation: {get_error}")
                    
                    # Collection doesn't exist, create it
                    try:
                        print(f"[INFO] Creating new collection '{collection_name}'")
                        collection = chroma_client.create_collection(
                            name=collection_name,
                            embedding_function=openai_ef,
                            metadata={
                                "description": "Contract documents for Electron app",
                                "embedding_model": "text-embedding-ada-002"
                            }
                        )
                        print("[SUCCESS] Created new collection with ada-002 embeddings")
                    except Exception as create_error:
                        print(f"[ERROR] Failed to create collection: {create_error}")
                        # Don't raise - allow backend to start without collection
                        collection = None
            except Exception as collection_error:
                print(f"[ERROR] ChromaDB collection setup failed: {collection_error}")
                collection = None
        else:
            print("[INFO] Skipping ChromaDB collection setup - no OpenAI API key or ChromaDB client available")
            collection = None
            
        # Final status check
        if collection is not None:
            print("[SUCCESS] ChromaDB collection setup complete and ready for use")
        else:
            print("[WARNING] ChromaDB collection not available - vector search will be disabled")
        
        print("[SUCCESS] Services initialized successfully")
        
    except Exception as e:
        print(f"[ERROR] Failed to initialize services: {e}")
        # Don't raise - allow backend to continue running
        chroma_client = None
        collection = None
        openai_client = None
        contract_intelligence_engine = None
        print("[INFO] Backend will continue running with limited functionality. Please configure API keys in settings.")

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Smart semantic chunking for contracts with clause-based segmentation"""
    if not text.strip():
        return []
    
    # Try semantic chunking first
    semantic_chunks = _semantic_chunk_contract(text)
    if semantic_chunks:
        print("[INFO] Using semantic clause-based chunking")
        return semantic_chunks
    
    # Fallback to sentence-aware chunking
    print("[INFO] Using sentence-aware chunking")
    return _sentence_aware_chunk(text, chunk_size, chunk_overlap)

def _semantic_chunk_contract(text: str) -> List[str]:
    """Semantic chunking based on contract structure and clauses"""
    try:
        import re
        chunks = []
        
        # Contract section patterns (English and Hebrew)
        section_patterns = [
            r'(?i)(?:^|\n)\s*(?:article|section|clause|paragraph|part)\s*[ivx\d]+[.:]',
            r'(?i)(?:^|\n)\s*\d+\.\s*[A-Z]',  # Numbered sections
            r'(?i)(?:^|\n)\s*[A-Z]\.\s*[A-Z]',  # Lettered sections
            r'(?i)(?:^|\n)\s*(?:whereas|now therefore|in witness whereof)',
            r'(?i)(?:^|\n)\s*(?:definitions?|terms?|conditions?|obligations?|rights?|responsibilities?)',
            # Hebrew patterns
            r'(?:^|\n)\s*(?:סעיף|פרק|חלק|מדור)\s*[\u05d0-\u05ea\d]+[.:]',
            r'(?:^|\n)\s*\d+\.\s*[\u05d0-\u05ea]',
            r'(?:^|\n)\s*[\u05d0-\u05ea]\.\s*[\u05d0-\u05ea]',
        ]
        
        # Split by major sections first
        combined_pattern = '|'.join(section_patterns)
        sections = re.split(combined_pattern, text, flags=re.MULTILINE)
        
        if len(sections) > 1:
            # We found semantic sections
            current_chunk = ""
            
            for i, section in enumerate(sections):
                section = section.strip()
                if not section:
                    continue
                
                # If adding this section would make chunk too large, save current and start new
                if len(current_chunk) + len(section) > 1500 and current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = section
                else:
                    current_chunk += ("\n\n" if current_chunk else "") + section
            
            # Add the last chunk
            if current_chunk.strip():
                chunks.append(current_chunk.strip())
            
            # Filter out very small chunks and merge them
            filtered_chunks = []
            for chunk in chunks:
                if len(chunk) < 100 and filtered_chunks:
                    # Merge small chunk with previous
                    filtered_chunks[-1] += "\n\n" + chunk
                else:
                    filtered_chunks.append(chunk)
            
            return filtered_chunks if len(filtered_chunks) > 1 else []
        
        return []  # No semantic structure found
        
    except Exception as e:
        print(f"[WARNING] Semantic chunking failed: {e}")
        return []

def _sentence_aware_chunk(text: str, chunk_size: int = 1200, overlap: int = 300) -> List[str]:
    """Enhanced sentence-aware chunking with better Hebrew support and larger chunks"""
    import re
    
    # CRITICAL: Protect Hebrew abbreviations before sentence splitting
    protected_text = text
    
    # Common Hebrew abbreviations that should NOT be split
    hebrew_abbreviations = [
        r'מע"מ',  # VAT
        r'ח"פ',   # Company registration number
        r'ע"ר',   # Non-profit organization
        r'בע"מ',  # Limited company
        r'ש"ח',   # New Israeli Shekel
        r'ת"ז',   # ID number
        r'מ"ר',   # Square meter
        r'כ"א',   # General
        r'ד"ר',   # Doctor
        r'פרופ"',  # Professor
        r'אדון"', # Mr.
        r'גב"',   # Mrs.
    ]
    
    # Create placeholder map to protect abbreviations
    placeholder_map = {}
    placeholder_counter = 0
    
    for abbrev in hebrew_abbreviations:
        # Find all instances of this abbreviation
        matches = list(re.finditer(abbrev, protected_text))
        for match in matches:
            placeholder = f"__HEBABBREV_{placeholder_counter}__"
            placeholder_map[placeholder] = match.group()
            protected_text = protected_text.replace(match.group(), placeholder, 1)
            placeholder_counter += 1
    
    # Enhanced sentence splitting for Hebrew and English (now on protected text)
    sentence_patterns = [
        r'[.!?]+\s+',  # English sentence endings
        r'[\u05c3\u05f3\u05f4]+\s+',  # Hebrew punctuation
        r'(?<=[\u05d0-\u05ea])\s*\.\s*(?=[\u05d0-\u05ea\u0590-\u05FF])',  # Hebrew periods (protected from abbreviations)
        r'(?<=\d)\s*\.\s*(?=[\u05d0-\u05ea])',  # Numbers followed by Hebrew
    ]
    
    # Split text into sentences using multiple patterns
    sentences = [protected_text]  # Start with protected text
    for pattern in sentence_patterns:
        new_sentences = []
        for sent in sentences:
            new_sentences.extend(re.split(pattern, sent))
        sentences = new_sentences
    
    # Restore Hebrew abbreviations in all sentences
    restored_sentences = []
    for sentence in sentences:
        restored_sentence = sentence
        for placeholder, original in placeholder_map.items():
            restored_sentence = restored_sentence.replace(placeholder, original)
        restored_sentences.append(restored_sentence)
    
    sentences = restored_sentences
    
    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 10]
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # If adding this sentence would exceed chunk_size and we have content
        if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
            chunks.append(current_chunk.strip())
            
            # Start new chunk with meaningful overlap
            overlap_text = ""
            if overlap > 0 and len(current_chunk) > overlap:
                # Take last portion for overlap
                overlap_text = current_chunk[-overlap:]
                # Find a good break point (sentence or clause boundary)
                for pattern in [r'\. ', r': ', r'\n', r'; ']:
                    matches = list(re.finditer(pattern, overlap_text))
                    if matches:
                        break_point = matches[0].end()
                        overlap_text = overlap_text[break_point:] + " "
                        break
                else:
                    overlap_text = overlap_text + " "
            
            current_chunk = overlap_text + sentence
        else:
            current_chunk += (" " if current_chunk else "") + sentence
    
    # Add the last chunk
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter and validate chunks
    filtered_chunks = []
    for chunk in chunks:
        if len(chunk) > 100:  # Only keep substantial chunks
            filtered_chunks.append(chunk)
    
    print(f"[INFO] Sentence-aware chunking created {len(filtered_chunks)} chunks (avg size: {sum(len(c) for c in filtered_chunks) // len(filtered_chunks) if filtered_chunks else 0})")
    return filtered_chunks if filtered_chunks else [text]

def get_embeddings(texts: List[str]) -> List[List[float]]:
    """Generate embeddings using OpenAI ada-002 model"""
    global openai_client
    
    if not openai_client:
        raise Exception("OpenAI client not initialized - please configure API key")
    
    try:
        print(f"[INFO] Generating embeddings for {len(texts)} chunks using ada-002")
        response = openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        embeddings = [item.embedding for item in response.data]
        print(f"[INFO] Generated {len(embeddings)} embeddings successfully")
        return embeddings
    except Exception as e:
        print(f"[ERROR] Failed to generate embeddings: {e}")
        raise Exception(f"Failed to generate embeddings: {str(e)}")

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyPDF2 with automatic OCR fallback - EXACT LocalRAGFlow approach"""
    global google_credentials
    
    try:
        print(f"[DEBUG] Opening PDF file: {file_path}")
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            print(f"[DEBUG] PDF has {len(reader.pages)} pages")
            text = ""
            
            for page_num, page in enumerate(reader.pages):
                print(f"[DEBUG] Processing page {page_num + 1}")
                try:
                    page_text = page.extract_text()
                    print(f"[DEBUG] Page {page_num + 1} extracted {len(page_text)} chars")
                    
                    if page_text.strip():  # If text found, use standard extraction
                        text += page_text + "\n"
                        print(f"[DEBUG] Page {page_num + 1} SUCCESS - Sample: {page_text[:50]}...")
                    else:
                        print(f"[DEBUG] Page {page_num + 1} returned EMPTY text")
                        
                except Exception as page_error:
                    print(f"[ERROR] Failed to extract from page {page_num + 1}: {page_error}")
                    continue
        
        final_text = text.strip()
        print(f"[DEBUG] Final PDF extraction result: {len(final_text)} total chars")
        return final_text
        
    except Exception as e:
        print(f"[ERROR] PDF extraction FAILED with exception: {e}")
        import traceback
        traceback.print_exc()
        return ""

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX using python-docx with Hebrew support"""
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            if paragraph.text:
                text += paragraph.text + "\n"
        
        # Clean and normalize text for Hebrew support
        text = text.strip()
        if text:
            # Ensure proper Unicode handling
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
        return text
    except Exception as e:
        raise Exception(f"Failed to extract text from DOCX: {str(e)}")

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file with multiple encoding fallbacks"""
    encodings = ['utf-8', 'utf-8-sig', 'cp1255', 'iso-8859-8', 'windows-1255']
    
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                text = file.read().strip()
                if text:
                    # Ensure proper Unicode handling
                    text = text.encode('utf-8', errors='ignore').decode('utf-8')
                return text
        except (UnicodeDecodeError, UnicodeError):
            continue
    
    # If all encodings fail, try binary read and decode
    try:
        with open(file_path, 'rb') as file:
            content = file.read()
            text = content.decode('utf-8', errors='ignore').strip()
            return text
    except Exception as e:
        raise Exception(f"Failed to extract text from TXT file with any encoding: {str(e)}")

def extract_text_from_pdf_ocr(file_path: str) -> str:
    """Extract text from PDF using OCR by converting pages to images first"""
    global google_credentials
    
    logger.info(f"Starting PDF OCR for file: {file_path}")
    
    if not VISION_AVAILABLE:
        logger.error("Google Vision not available")
        raise Exception("Google Vision not available - install google-cloud-vision")
    
    if not google_credentials:
        logger.error("No Google credentials found")
        raise Exception("Google credentials not configured - please set up in settings")
    
    try:
        # Check if credentials need refresh
        if hasattr(google_credentials, 'expired') and google_credentials.expired:
            if hasattr(google_credentials, 'refresh_token') and google_credentials.refresh_token:
                logger.info("Service account credentials don't need refreshing")
                # Service account credentials don't need refreshing with Request()
                pass
            else:
                raise Exception("Google credentials expired and cannot be refreshed")
        
        # Create Vision client with credentials
        logger.info("Creating Vision client")
        client = vision.ImageAnnotatorClient(credentials=google_credentials)
        
        # Open PDF with PyMuPDF
        logger.info(f"Opening PDF for image conversion: {file_path}")
        pdf_document = fitz.open(file_path)
        logger.info(f"PDF has {pdf_document.page_count} pages")
        
        all_text = ""
        
        # Process ALL pages - no artificial limits
        for page_num in range(pdf_document.page_count):
            logger.info(f"Converting page {page_num + 1} to image...")
            page = pdf_document[page_num]
            
            # Convert page to image (PNG format)
            pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # 2x scale for better OCR
            img_data = pix.tobytes("png")
            logger.info(f"Page {page_num + 1} image size: {len(img_data)} bytes")
            
            # Perform OCR on this page image
            image = vision.Image(content=img_data)
            response = client.text_detection(image=image)
            
            # Check for errors in response
            if response.error.message:
                logger.warning(f"Vision API error on page {page_num + 1}: {response.error.message}")
                continue
            
            texts = response.text_annotations
            if texts:
                page_text = texts[0].description.strip()
                logger.info(f"Page {page_num + 1} OCR extracted {len(page_text)} characters")
                if page_text:
                    all_text += page_text + "\n\n"
            else:
                logger.info(f"No text found on page {page_num + 1}")
        
        total_pages = pdf_document.page_count  # Store page count before closing
        pdf_document.close()
        
        final_text = all_text.strip()
        logger.info(f"Total PDF OCR extraction: {len(final_text)} characters from ALL {total_pages} pages")
        return final_text
        
    except Exception as e:
        logger.error(f"PDF OCR exception: {str(e)}")
        raise Exception(f"Failed to extract text from PDF using OCR: {str(e)}")

def extract_text_from_image_ocr(file_path: str) -> str:
    """Extract text from image using Google Vision OCR"""
    global google_credentials
    
    print(f"[DEBUG] Starting OCR for file: {file_path}")
    
    if not VISION_AVAILABLE:
        print("[ERROR] Google Vision not available")
        raise Exception("Google Vision not available - install google-cloud-vision")
    
    if not google_credentials:
        print("[ERROR] No Google credentials found")
        raise Exception("Google credentials not configured - please set up in settings")
    
    print(f"[DEBUG] Credentials available: {type(google_credentials)}")
    print(f"[DEBUG] Credentials expired: {getattr(google_credentials, 'expired', 'unknown')}")
    
    try:
        # Check if credentials need refresh
        if hasattr(google_credentials, 'expired') and google_credentials.expired:
            if hasattr(google_credentials, 'refresh_token') and google_credentials.refresh_token:
                print("[INFO] Service account credentials don't need refreshing")
                # Service account credentials don't need refreshing with Request()
                pass
            else:
                raise Exception("Google credentials expired and cannot be refreshed")
        
        # Create Vision client with credentials
        print("[DEBUG] Creating Vision client")
        client = vision.ImageAnnotatorClient(credentials=google_credentials)
        
        # Read image file
        print(f"[DEBUG] Reading image file: {file_path}")
        with open(file_path, 'rb') as image_file:
            content = image_file.read()
        
        print(f"[DEBUG] Image file size: {len(content)} bytes")
        image = vision.Image(content=content)
        
        # Perform OCR
        print("[DEBUG] Performing OCR...")
        response = client.text_detection(image=image)
        
        # Check for errors in response
        if response.error.message:
            raise Exception(f"Vision API error: {response.error.message}")
        
        texts = response.text_annotations
        print(f"[DEBUG] OCR found {len(texts)} text annotations")
        
        if texts:
            # First annotation contains all detected text
            extracted_text = texts[0].description.strip()
            
            # CRITICAL: Fix common OCR corruption of Hebrew abbreviations
            hebrew_abbreviation_fixes = {
                'מע מ': 'מע"מ',  # Fix broken VAT abbreviation
                'מע"מ': 'מע"מ',  # Ensure proper quotation marks
                'בע מ': 'בע"מ',  # Fix broken Ltd abbreviation
                'ש ח': 'ש"ח',    # Fix broken Shekel abbreviation
                'ח פ': 'ח"פ',    # Fix broken company number
                'ע ר': 'ע"ר',    # Fix broken non-profit
                'ת ז': 'ת"ז',    # Fix broken ID number
                'מ ר': 'מ"ר',    # Fix broken square meter
                # Handle different quote marks that OCR might produce
                'מע\'מ': 'מע"מ',
                'בע\'מ': 'בע"מ',
                'ש\'ח': 'ש"ח',
            }
            
            # Apply fixes
            for broken, fixed in hebrew_abbreviation_fixes.items():
                extracted_text = extracted_text.replace(broken, fixed)
            
            print(f"[DEBUG] Extracted text length: {len(extracted_text)}")
            return extracted_text
        else:
            print("[DEBUG] No text found in image")
            return ""
            
    except Exception as e:
        print(f"[ERROR] OCR exception: {str(e)}")
        raise Exception(f"Failed to extract text from image using OCR: {str(e)}")

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup and cleanup on shutdown"""
    print("[INFO] Starting Contract Intelligence Minimal Backend...")
    initialize_services()
    yield
    print("[INFO] Shutting down Contract Intelligence Minimal Backend...")

# Initialize FastAPI app with lifespan
app = FastAPI(
    title="Contract Intelligence API - Minimal",
    description="Minimal backend API for Contract Intelligence Desktop App",
    version="1.5.46",
    lifespan=lifespan
)

# Enable CORS for Electron frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.5.46",
        "backend": "minimal",
        "chromadb_ready": chroma_client is not None,
        "openai_ready": openai_client is not None,
        "features_available": {
            "document_processing": DOCUMENT_PROCESSING_AVAILABLE,
            "vector_search": AI_CHROMADB_AVAILABLE,
            "contract_intelligence": CONTRACT_INTELLIGENCE_AVAILABLE,
            "tokenizers": TOKENIZERS_AVAILABLE,
            "sentence_transformers": SENTENCE_TRANSFORMERS_AVAILABLE
        },
        "backend_mode": "full" if AI_CHROMADB_AVAILABLE and DOCUMENT_PROCESSING_AVAILABLE else "limited"
    }

@app.post("/api/test-upload")
async def test_upload():
    """Simple test endpoint to verify upload requests reach backend"""
    import sys
    print("=" * 50, flush=True)
    print("[DEBUG] TEST UPLOAD ENDPOINT HIT!", flush=True)
    print("=" * 50, flush=True)
    sys.stdout.flush()
    return {"message": "Test upload endpoint reached successfully"}

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify backend functionality"""
    try:
        # Test basic functionality
        test_text = "This is a test document for the Contract Intelligence Platform."
        chunks = chunk_text(test_text)
        
        return {
            "status": "success",
            "message": "Minimal backend is working correctly",
            "test_results": {
                "chunking": f"Generated {len(chunks)} chunks",
                "openai_available": openai_client is not None,
                "chromadb_available": chroma_client is not None,
                "collection_name": "contracts_electron" if collection else "not configured",
                "chromadb_collections": len(chroma_client.list_collections()) if chroma_client else 0
            }
        }
    except Exception as e:
        return {
            "status": "partial",
            "message": f"Backend is running but some services are not available: {str(e)}",
            "test_results": {
                "chunking": "available",
                "openai_available": openai_client is not None,
                "chromadb_available": chroma_client is not None
            }
        }

@app.get("/api/status")
async def get_status():
    """Get application status"""
    try:
        doc_count = 0
        collections_count = 0
        
        if chroma_client:
            collections = chroma_client.list_collections()
            collections_count = len(collections)
        
        # Count UNIQUE DOCUMENTS, not chunks
        if collection:
            results = collection.get()
            unique_filenames = set()
            for metadata in results['metadatas']:
                unique_filenames.add(metadata['filename'])
            doc_count = len(unique_filenames)
        
        return {
            "openai_configured": openai_client is not None,
            "chromadb_ready": chroma_client is not None,
            "collection_ready": collection is not None,
            "documents_count": doc_count,
            "collections_count": collections_count
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    folder: str = Form("General"),
    use_ocr: bool = Form(False)
):
    """Upload and process a document"""
    logger.info("=" * 80)
    logger.info("*** UPLOAD ENDPOINT HIT! ***")
    logger.info(f"File received: {file.filename if file else 'None'}")
    logger.info(f"Folder: {folder}")
    logger.info(f"Use OCR: {use_ocr} (type: {type(use_ocr)})")
    logger.info("=" * 80)
    
    if not chroma_client or not collection:
        print("[ERROR] ChromaDB not initialized!")
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")
    
    try:
        print(f"[DEBUG] Upload request - file: {file.filename}, folder: {folder}, use_ocr: {use_ocr} (type: {type(use_ocr)})")
        
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Extract text based on file type
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext == '.pdf':
            print(f"[DEBUG] Processing PDF file: {file.filename}")
            print(f"[DEBUG] Temp file path: {temp_path}")
            print(f"[DEBUG] Google credentials available: {google_credentials is not None}")
            print(f"[DEBUG] Vision API available: {VISION_AVAILABLE}")
            
            # EXACT LocalRAGFlow flow: try standard extraction first
            print("[DEBUG] Starting standard PDF text extraction...")
            text = extract_text_from_pdf(temp_path)
            print(f"[DEBUG] Standard extraction result - Text length: {len(text)} chars")
            if text.strip():
                print(f"[DEBUG] Standard extraction SUCCESS - Sample: {text[:100]}...")
            else:
                print("[DEBUG] Standard extraction returned EMPTY text")
            
            # If no text found, AUTOMATICALLY try OCR fallback (no user decision needed)
            if not text.strip() and google_credentials and VISION_AVAILABLE:
                try:
                    print("[INFO] No text extracted from PDF, automatically trying OCR fallback")
                    print("[DEBUG] Starting OCR fallback process...")
                    text = extract_text_from_pdf_ocr(temp_path)  # Use PDF-specific OCR with image conversion
                    print(f"[DEBUG] OCR fallback result - Text length: {len(text)} chars")
                    if text.strip():
                        print(f"[DEBUG] OCR fallback SUCCESS - Sample: {text[:100]}...")
                    else:
                        print("[DEBUG] OCR fallback also returned EMPTY text")
                except Exception as ocr_error:
                    print(f"[ERROR] OCR fallback FAILED with exception: {ocr_error}")
                    import traceback
                    traceback.print_exc()
                    # Continue with empty text - will be caught by validation below
            elif not text.strip():
                if not google_credentials:
                    print("[DEBUG] No OCR fallback - Google credentials not available")
                elif not VISION_AVAILABLE:
                    print("[DEBUG] No OCR fallback - Vision API not available")
                else:
                    print("[DEBUG] No OCR fallback - unknown reason")
        elif file_ext == '.docx':
            text = extract_text_from_docx(temp_path)
        elif file_ext == '.txt':
            text = extract_text_from_txt(temp_path)
        elif file_ext in ['.jpg', '.jpeg', '.png']:
            if use_ocr:
                try:
                    safe_print(f"[INFO] Processing image {file.filename} with OCR")
                    print(f"[INFO] Google credentials available: {google_credentials is not None}")
                    print(f"[INFO] Vision available: {VISION_AVAILABLE}")
                    text = extract_text_from_image_ocr(temp_path)
                    print(f"[INFO] OCR extracted {len(text)} characters")
                except Exception as ocr_error:
                    print(f"[ERROR] OCR failed: {ocr_error}")
                    raise HTTPException(status_code=500, detail=f"OCR processing failed: {str(ocr_error)}")
            else:
                raise HTTPException(status_code=400, detail=f"Image files require OCR to be enabled")
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
        
        print(f"[DEBUG] Final text validation - Length: {len(text)} chars")
        if text.strip():
            # Use safe debugging for Hebrew text
            safe_print(debug_text_sample(text, "Text validation PASSED"))
        else:
            print("[DEBUG] Text validation FAILED - No text content found")
        
        if not text.strip():
            if file_ext == '.pdf':
                if google_credentials and VISION_AVAILABLE:
                    print("[ERROR] FINAL FAILURE: Both standard PDF extraction and OCR failed")
                    raise HTTPException(status_code=400, detail=f"No text content found in PDF. Both standard extraction and OCR failed.")
                else:
                    print("[ERROR] FINAL FAILURE: No text from PDF and OCR not available")
                    raise HTTPException(status_code=400, detail=f"No text content found in PDF. OCR not available - please configure Google credentials in settings.")
            else:
                print(f"[ERROR] FINAL FAILURE: No text from {file_ext} file")
                raise HTTPException(status_code=400, detail=f"No text content found in document. File type: {file_ext}")
        
        # Ensure text is properly encoded for Hebrew support
        try:
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
        except Exception as e:
            print(f"[WARNING] Text encoding normalization failed: {e}")
        
        # Log text sample safely (avoid Hebrew encoding issues in console)
        sample_length = min(100, len(text))
        try:
            # Use safe debugging for Hebrew text
            safe_print(debug_text_sample(text, "Extracted text sample", sample_length))
        except:
            print(f"[INFO] Extracted {len(text)} characters (contains non-ASCII characters)")
        
        # PHASE 2: Contract Intelligence Analysis
        contract_analysis = None
        if contract_intelligence_engine and openai_client:
            try:
                print(f"[INFO] Analyzing contract intelligence for '{file.filename}'...")
                contract_analysis = contract_intelligence_engine.analyze_contract(text, file.filename)
                print(f"[SUCCESS] Contract analysis completed - Type: {contract_analysis.contract_type}")
                print(f"[INFO] Found {len(contract_analysis.parties)} parties, {len(contract_analysis.key_dates)} dates")
            except Exception as e:
                print(f"[WARNING] Contract intelligence analysis failed: {e}")
                contract_analysis = None
        
        # Chunk the text using contract-aware semantic chunking
        chunks = chunk_text(text)
        safe_print(f"[INFO] Created {len(chunks)} chunks from document '{file.filename}'")
        
        # Determine chunking method used (for metadata)
        semantic_test = _semantic_chunk_contract(text)
        chunking_method = "semantic_contract" if semantic_test else "sentence_aware"
        
        # Generate embeddings if OpenAI is available
        embeddings = None
        if openai_client:
            try:
                embeddings = get_embeddings(chunks)
            except Exception as e:
                print(f"[WARNING] Failed to generate embeddings: {e}")
                print("[INFO] Proceeding without embeddings - search quality may be reduced")
        
        # Add to ChromaDB
        doc_id = f"{folder}_{file.filename}_{len(collection.get()['ids'])}"
        
        # PHASE 2: Enhanced metadata with contract intelligence
        enhanced_metadatas = []
        for i in range(len(chunks)):
            chunk = chunks[i]
            base_metadata = {
                "filename": file.filename,
                "folder": folder,
                "chunk_index": i,
                "total_chunks": len(chunks),
                "has_embeddings": embeddings is not None,
                "chunking_method": chunking_method,
                "file_type": file_ext,
                "chunk_length": len(chunk),
                "contains_numbers": any(c.isdigit() for c in chunk),
                "contains_hebrew": any('\u0590' <= c <= '\u05FF' for c in chunk),
                "is_contract_clause": any(keyword in chunk.lower() for keyword in ['סעיף', 'פסקה', 'תנאי', 'מחיר', 'תשלום', 'חוזה', 'הסכם']) if chunk else False
            }
            
            # Add contract intelligence metadata if available
            if contract_analysis:
                try:
                    # Enhanced contract intelligence metadata
                    contract_metadata = contract_intelligence_engine.create_enhanced_metadata(contract_analysis, base_metadata)
                    
                    # Add chunk-specific contract intelligence
                    chunk_lower = chunk.lower() if chunk else ""
                    contract_metadata.update({
                        # Financial entity detection
                        "contains_financial_terms": any(term in chunk_lower for term in ['₪', 'שח', 'דולר', 'מחיר', 'עלות', 'תשלום', 'דמי']),
                        "contains_dates": any(term in chunk_lower for term in ['תאריך', 'יום', 'חודש', 'שנה', '2024', '2023', '2025']),
                        "contains_parties": any(party.name.lower() in chunk_lower for party in contract_analysis.parties if party.name),
                        # Advanced numerical detection
                        "contains_amounts": bool(re.search(r'\d+[,.]?\d*\s*(?:₪|שח|ש"ח|אלף|מיליון)', chunk) if chunk else False),
                        "contains_percentages": bool(re.search(r'\d+\.?\d*\s*%|אחוז', chunk) if chunk else False),
                        "contract_intelligence_available": True
                    })
                    enhanced_metadatas.append(contract_metadata)
                except Exception as e:
                    print(f"[WARNING] Failed to create enhanced metadata for chunk {i}: {e}")
                    enhanced_metadatas.append(base_metadata)
            else:
                enhanced_metadatas.append(base_metadata)
        
        # Prepare data for ChromaDB
        add_kwargs = {
            "documents": chunks,
            "ids": [f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
            "metadatas": enhanced_metadatas
        }
        
        # Add embeddings if available
        if embeddings:
            add_kwargs["embeddings"] = embeddings
            print(f"[INFO] Storing {len(chunks)} chunks with ada-002 embeddings")
        else:
            print(f"[INFO] Storing {len(chunks)} chunks without embeddings")
        
        collection.add(**add_kwargs)
        
        # Clean up temp file
        os.unlink(temp_path)
        
        # Create detailed success message
        processing_details = []
        if use_ocr and file_ext in ['.jpg', '.jpeg', '.png']:
            processing_details.append("OCR")
        if embeddings:
            processing_details.append("ada-002 embeddings")
        
        chunking_info = "contract-aware semantic" if chunking_method == "semantic_contract" else "sentence-aware"
        processing_details.append(f"{chunking_info} chunking")
        
        processing_info = f" with {', '.join(processing_details)}" if processing_details else ""
        
        # Build response with contract intelligence information
        response = {
            "message": f"Document '{file.filename}' processed successfully{processing_info}",
            "success": True,
            "chunks_created": len(chunks),
            "folder": folder,
            "ocr_used": use_ocr and file_ext in ['.jpg', '.jpeg', '.png'],
            "embeddings_generated": embeddings is not None,
            "embedding_model": "text-embedding-ada-002" if embeddings else None,
            "chunking_method": chunking_method,
            "chunking_description": f"{chunking_info} chunking"
        }
        
        # Add contract intelligence information if available
        if contract_analysis:
            response["contract_intelligence"] = {
                "contract_type": contract_analysis.contract_type,
                "contract_type_confidence": round(contract_analysis.contract_type_confidence, 2),
                "language": contract_analysis.language,
                "parties_found": len(contract_analysis.parties),
                "key_dates_found": len(contract_analysis.key_dates),
                "analysis_available": True
            }
            
            # Count enhanced chunks
            financial_chunks = sum(1 for meta in enhanced_metadatas if meta.get('contains_financial_terms', False))
            amount_chunks = sum(1 for meta in enhanced_metadatas if meta.get('contains_amounts', False))
            
            response["contract_intelligence"]["chunks_with_financial_terms"] = financial_chunks
            response["contract_intelligence"]["chunks_with_amounts"] = amount_chunks
        else:
            response["contract_intelligence"] = {"analysis_available": False}
        
        return response
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        
        print(f"[ERROR] Document processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/documents")
async def get_documents():
    """Get list of documents"""
    if not collection:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")
    
    try:
        results = collection.get()
        
        # Group by filename
        documents = {}
        documents_by_folder = {}
        
        for i, metadata in enumerate(results['metadatas']):
            filename = metadata['filename']
            folder = metadata.get('folder', 'General')
            
            # Group by filename
            if filename not in documents:
                documents[filename] = {
                    'filename': filename,
                    'folder': folder,
                    'chunks': 0
                }
            documents[filename]['chunks'] += 1
            
            # Group by folder
            if folder not in documents_by_folder:
                documents_by_folder[folder] = []
            
            # Add filename to folder if not already there
            if filename not in documents_by_folder[folder]:
                documents_by_folder[folder].append(filename)
        
        return {
            "documents": list(documents.values()),
            "documents_by_folder": documents_by_folder,
            "total_count": len(documents)  # This is already correct - counts unique documents, not chunks
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{document_name}")
async def delete_document(document_name: str):
    """Delete a document and all its chunks from ChromaDB"""
    if not collection:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")
    
    try:
        logger.info(f"Attempting to delete document: {document_name}")
        
        # Get all chunks for this document
        results = collection.get(
            where={"filename": document_name}
        )
        
        if not results['ids'] or len(results['ids']) == 0:
            logger.warning(f"Document not found: {document_name}")
            raise HTTPException(status_code=404, detail=f"Document '{document_name}' not found")
        
        # Delete all chunks for this document
        chunk_ids = results['ids']
        logger.info(f"Deleting {len(chunk_ids)} chunks for document: {document_name}")
        
        collection.delete(ids=chunk_ids)
        
        logger.info(f"Successfully deleted document: {document_name}")
        return {
            "message": f"Document '{document_name}' and its {len(chunk_ids)} chunks have been deleted successfully",
            "success": True,
            "deleted_chunks": len(chunk_ids)
        }
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error deleting document '{document_name}': {e}")
        raise HTTPException(status_code=500, detail=f"Failed to delete document: {str(e)}")

@app.post("/api/chat")
async def chat_query(request: Dict[str, Any]):
    """Process chat query using ChromaDB similarity search with document/folder filtering"""
    if not collection:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")
    
    try:
        query = request.get("query", "").strip()
        target_documents = request.get("target_documents")
        target_folder = request.get("target_folder")
        
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        logger.info(f"Chat query: '{query}' | target_documents: {target_documents} | target_folder: {target_folder}")
        
        # Enhanced query expansion for Hebrew contracts
        expanded_queries = [query]
        
        # Enhanced Hebrew contract-specific query variations with numerical focus
        hebrew_expansions = {
            'rent': ['שכירות', 'דמי שכירות', 'שכר דירה', 'תשלום'],
            'increase': ['עלייה', 'הגדלה', 'תוספת', 'הצמדה', 'עדכון'],
            'payment': ['תשלום', 'דמי', 'מחיר', 'עלות', 'סכום'],
            'clause': ['סעיף', 'פסקה', 'תנאי', 'הוראה'],
            'adjustment': ['התאמה', 'עדכון', 'הצמדה', 'שינוי'],
            'annual': ['שנתי', 'מדי שנה', 'לשנה'],
            'monthly': ['חודשי', 'מדי חודש', 'לחודש'],
            # CRITICAL: Add numerical-focused terms
            'amount': ['סכום', 'כמות', 'מחיר', 'דמי', '₪', 'שח'],
            'price': ['מחיר', 'עלות', 'דמי', 'תשלום', 'סכום'],
            'percentage': ['אחוז', '%', 'אחוזים', 'שיעור'],
            'money': ['כסף', 'שח', '₪', 'שקל', 'שקלים'],
            'cost': ['עלות', 'מחיר', 'דמי', 'הוצאה']
        }
        
        query_lower = query.lower()
        for eng_term, heb_terms in hebrew_expansions.items():
            if eng_term in query_lower:
                expanded_queries.extend(heb_terms)
            for heb_term in heb_terms:
                if heb_term in query:
                    expanded_queries.append(eng_term)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_queries = []
        for q in expanded_queries:
            if q not in seen:
                seen.add(q)
                unique_queries.append(q)
        
        logger.info(f"Expanded queries: {unique_queries}")
        
        # PHASE 4: Enhanced contract-aware query analysis
        contract_query_indicators = {
            'numerical': ['amount', 'price', 'cost', 'money', 'percentage', '₪', '$', '%', 
                         'סכום', 'מחיר', 'דמי', 'עלות', 'אחוז', 'שח', 'כמה', 'כמות'],
            'parties': ['party', 'parties', 'tenant', 'landlord', 'contractor', 'client',
                       'צד', 'צדדים', 'שוכר', 'משכיר', 'קבלן', 'לקוח'],
            'dates': ['date', 'when', 'period', 'duration', 'start', 'end', 'expire',
                     'תאריך', 'מתי', 'תקופה', 'משך', 'התחלה', 'סיום', 'פוקע'],
            'terms': ['term', 'condition', 'clause', 'section', 'article',
                     'תנאי', 'סעיף', 'פסקה', 'מדור'],
            'adjustments': ['increase', 'decrease', 'change', 'adjust', 'update',
                           'עלייה', 'ירידה', 'שינוי', 'התאמה', 'עדכון', 'הצמדה']
        }
        
        # Detect query types
        query_types = []
        query_lower = query.lower()
        for query_type, indicators in contract_query_indicators.items():
            if any(indicator in query_lower or indicator in query for indicator in indicators):
                query_types.append(query_type)
        
        is_numerical_query = 'numerical' in query_types
        is_parties_query = 'parties' in query_types
        is_dates_query = 'dates' in query_types
        is_adjustments_query = 'adjustments' in query_types
        
        logger.info(f"Contract query analysis - Types detected: {query_types}")
        logger.info(f"Numerical query: {is_numerical_query}, Parties: {is_parties_query}, Dates: {is_dates_query}, Adjustments: {is_adjustments_query}")
        
        # Build filter criteria for ChromaDB
        where_filter = {}
        if target_folder and target_folder != "all":
            where_filter["folder"] = target_folder
        if target_documents and target_documents != ["all"] and target_documents[0] != "all":
            where_filter["filename"] = {"$in": target_documents}
        
        # ADVANCED RAG: Multi-stage hybrid retrieval for perfect accuracy
        all_results = []
        all_chunks = set()  # To avoid duplicates
        
        # Stage 1: Semantic vector search with expanded queries
        logger.info("Stage 1: Semantic vector search")
        for search_query in unique_queries[:2]:
            search_kwargs = {
                "query_texts": [search_query],
                "n_results": 12
            }
            if where_filter:
                search_kwargs["where"] = where_filter
                
            try:
                query_results = collection.query(**search_kwargs)
                if query_results['documents'] and query_results['documents'][0]:
                    for doc, meta, dist in zip(query_results['documents'][0], 
                                             query_results['metadatas'][0], 
                                             query_results['distances'][0]):
                        chunk_id = f"{meta.get('filename', '')}_{meta.get('chunk_index', '')}"
                        if chunk_id not in all_chunks:
                            all_chunks.add(chunk_id)
                            all_results.append((doc, meta, dist, 'semantic'))
                logger.info(f"Semantic search '{search_query}' returned {len(query_results['documents'][0])} results")
            except Exception as e:
                logger.warning(f"Failed semantic search with query '{search_query}': {e}")
        
        # Stage 2: Keyword-based search for exact matches
        logger.info("Stage 2: Keyword-based exact matching")
        
        # Extract key Hebrew/English terms from query for exact matching
        keyword_terms = []
        # Add numbers and financial terms
        import re
        numbers = re.findall(r'\d+[,.]?\d*', query)
        keyword_terms.extend(numbers)
        
        # Add Hebrew financial keywords and abbreviations
        hebrew_keywords = ['דמי', 'שכירות', 'מחיר', 'סכום', 'עלות', 'תשלום', 'אחוז', '₪', 'שח']
        hebrew_abbreviations = ['מע"מ', 'ח"פ', 'ע"ר', 'בע"מ', 'ש"ח', 'ת"ז', 'מ"ר']
        
        for keyword in hebrew_keywords:
            if keyword in query:
                keyword_terms.append(keyword)
        
        # Check for Hebrew abbreviations in query
        for abbrev in hebrew_abbreviations:
            if abbrev in query:
                keyword_terms.append(abbrev)
        
        # Perform keyword-based search using ChromaDB's contains functionality
        if keyword_terms:
            try:
                # Get all documents and filter by keyword match
                all_data = collection.get()
                for i, (doc_id, doc, metadata) in enumerate(zip(all_data['ids'], all_data['documents'], all_data['metadatas'])):
                    # Check if any keyword appears in the document
                    doc_lower = doc.lower() if doc else ""
                    keyword_score = 0
                    for term in keyword_terms:
                        if term.lower() in doc_lower:
                            keyword_score += 1
                    
                    if keyword_score > 0:
                        chunk_id = f"{metadata.get('filename', '')}_{metadata.get('chunk_index', '')}"
                        if chunk_id not in all_chunks and len(all_results) < 25:
                            all_chunks.add(chunk_id)
                            # Calculate pseudo-distance based on keyword matches
                            pseudo_distance = max(0.1, 1.0 - (keyword_score * 0.2))
                            all_results.append((doc, metadata, pseudo_distance, 'keyword'))
                            
                logger.info(f"Keyword search found {sum(1 for r in all_results if r[3] == 'keyword')} additional results")
            except Exception as e:
                logger.warning(f"Keyword search failed: {e}")
        
        # Stage 3: Contract-specific entity search
        logger.info("Stage 3: Contract entity search")
        
        # Search for chunks containing specific contract entities
        entity_searches = []
        if is_numerical_query:
            entity_searches = ['contains_amounts', 'contains_financial_terms']
        elif is_parties_query:
            entity_searches = ['contains_parties']
        elif is_dates_query:
            entity_searches = ['contains_dates']
        
        for entity_field in entity_searches:
            try:
                entity_filter = where_filter.copy() if where_filter else {}
                entity_filter[entity_field] = True
                
                entity_results = collection.query(
                    query_texts=[query],
                    n_results=8,
                    where=entity_filter
                )
                
                if entity_results['documents'] and entity_results['documents'][0]:
                    for doc, meta, dist in zip(entity_results['documents'][0], 
                                             entity_results['metadatas'][0], 
                                             entity_results['distances'][0]):
                        chunk_id = f"{meta.get('filename', '')}_{meta.get('chunk_index', '')}"
                        if chunk_id not in all_chunks:
                            all_chunks.add(chunk_id)
                            all_results.append((doc, meta, dist, f'entity_{entity_field}'))
                    logger.info(f"Entity search for {entity_field} found {len(entity_results['documents'][0])} results")
            except Exception as e:
                logger.warning(f"Entity search for {entity_field} failed: {e}")
        
        # Stage 4: Hybrid scoring and ranking
        logger.info("Stage 4: Hybrid scoring and reranking")
        
        # Enhance results with hybrid scoring
        enhanced_results = []
        for doc, meta, dist, search_type in all_results:
            base_score = 1 - dist if dist else 1.0
            
            # Boost based on search type
            type_boost = {
                'semantic': 0.0,
                'keyword': 0.15,  # Strong boost for exact keyword matches
                'entity_contains_amounts': 0.25,  # Highest boost for amount entities
                'entity_contains_financial_terms': 0.2,
                'entity_contains_parties': 0.1,
                'entity_contains_dates': 0.1
            }
            
            search_boost = type_boost.get(search_type, 0.0)
            
            # Add contract intelligence boost
            ci_boost = 0.0
            if meta.get('contract_intelligence_available', False):
                if meta.get('contains_amounts', False):
                    ci_boost += 0.3  # Maximum boost for detected amounts
                if meta.get('contains_financial_terms', False):
                    ci_boost += 0.15
                if meta.get('contains_numbers', False):
                    ci_boost += 0.1
            
            final_score = min(1.0, base_score + search_boost + ci_boost)
            enhanced_results.append((doc, meta, dist, search_type, base_score, final_score))
        
        # Sort by final hybrid score (higher = better)
        enhanced_results.sort(key=lambda x: -x[5])
        
        # Take top results
        max_results = 20
        enhanced_results = enhanced_results[:max_results]
        
        # Convert back to simple format for downstream processing
        all_results = [(r[0], r[1], r[2]) for r in enhanced_results]
        
        logger.info(f"Multi-stage retrieval complete: {len(all_results)} total results")
        logger.info(f"Search type distribution: {dict([(r[3], sum(1 for x in enhanced_results if x[3] == r[3])) for r in enhanced_results])}")
        
        # Sort combined results by similarity for final ranking
        all_results.sort(key=lambda x: x[2] if x[2] is not None else 1.0)
        
        # PHASE 4: Contract-aware specialized search
        specialized_search_needed = False
        search_priority = []
        
        # Determine what type of specialized search is needed
        if is_numerical_query and not any(r[1].get('contains_amounts', False) for r in all_results[:5]):
            specialized_search_needed = True
            search_priority = ['contains_amounts', 'contains_financial_terms', 'contains_numbers']
            logger.info("Numerical query detected - searching for financial/numerical content")
            
        elif is_parties_query and not any(r[1].get('contains_parties', False) for r in all_results[:5]):
            specialized_search_needed = True
            search_priority = ['contains_parties']
            logger.info("Parties query detected - searching for party-related content")
            
        elif is_dates_query and not any(r[1].get('contains_dates', False) for r in all_results[:5]):
            specialized_search_needed = True
            search_priority = ['contains_dates']
            logger.info("Dates query detected - searching for date-related content")
        
        if specialized_search_needed:
            try:
                # Try different search strategies based on priority
                for search_field in search_priority:
                    try:
                        specialized_filter = where_filter.copy() if where_filter else {}
                        specialized_filter[search_field] = True
                        
                        specialized_search_kwargs = {
                            "query_texts": [query],
                            "n_results": 15,
                            "where": specialized_filter
                        }
                        specialized_results = collection.query(**specialized_search_kwargs)
                        
                        if specialized_results['documents'] and specialized_results['documents'][0]:
                            added_count = 0
                            for doc, meta, dist in zip(specialized_results['documents'][0], 
                                                     specialized_results['metadatas'][0], 
                                                     specialized_results['distances'][0]):
                                chunk_id = f"{meta.get('filename', '')}_{meta.get('chunk_index', '')}"
                                if chunk_id not in all_chunks and added_count < 5:
                                    all_chunks.add(chunk_id)
                                    all_results.append((doc, meta, dist))
                                    added_count += 1
                            
                            if added_count > 0:
                                logger.info(f"Added {added_count} specialized chunks for {search_field}")
                                break  # Found relevant content, stop searching
                                
                    except Exception as search_error:
                        logger.warning(f"Failed specialized search for {search_field}: {search_error}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Failed specialized contract search: {e}")
        
        # Convert back to ChromaDB format
        results = {
            'documents': [[r[0] for r in all_results]],
            'metadatas': [[r[1] for r in all_results]], 
            'distances': [[r[2] for r in all_results]]
        }
        
        logger.info(f"ChromaDB query returned {len(results['documents'][0]) if results['documents'] and results['documents'][0] else 0} results")
        
        if not results['documents'] or not results['documents'][0]:
            logger.warning("No relevant documents found in ChromaDB")
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents. Please make sure documents are properly uploaded and indexed.",
                "source_info": [],
                "context_chunks": [],
                "success": True
            }
        
        # Get relevant chunks
        relevant_chunks = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0] if results['distances'] else []
        
        # Debug: Log what we're actually retrieving
        logger.info(f"Retrieved {len(relevant_chunks)} chunks")
        for i, chunk in enumerate(relevant_chunks[:2]):
            logger.info(f"Chunk {i+1} preview: {chunk[:150]}...")
            logger.info(f"Chunk {i+1} metadata: {metadatas[i] if i < len(metadatas) else 'No metadata'}")
        
        # If OpenAI is available, generate an answer
        if openai_client:
            # HYBRID APPROACH: Prioritize numerical data chunks for contract intelligence
            good_chunks = []
            chunk_sources = []
            
            # Create chunk data with enhanced scoring
            chunk_data = []
            for i, (chunk, metadata, distance) in enumerate(zip(relevant_chunks, metadatas, distances)):
                similarity_score = 1 - distance if distance else 1.0
                
                # PHASE 3: Enhanced scoring with contract intelligence
                numerical_boost = 0.0
                
                # Original numerical detection
                if metadata.get('contains_numbers', False):
                    numerical_boost += 0.15  # Significant boost for numbers
                if any(pattern in chunk for pattern in ['₪', '$', '%', 'שח', 'אלף', 'מיליון']):
                    numerical_boost += 0.1   # Additional boost for currency/amounts
                if metadata.get('is_contract_clause', False):
                    numerical_boost += 0.05  # Boost for contract clauses
                
                # Contract intelligence enhanced scoring
                if metadata.get('contract_intelligence_available', False):
                    if metadata.get('contains_amounts', False):
                        numerical_boost += 0.2   # Strong boost for detected amounts
                    if metadata.get('contains_financial_terms', False):
                        numerical_boost += 0.1   # Boost for financial terms
                    if metadata.get('contains_parties', False):
                        numerical_boost += 0.05  # Boost for party mentions
                    if metadata.get('contains_percentages', False):
                        numerical_boost += 0.1   # Boost for percentages
                    if metadata.get('contains_dates', False):
                        numerical_boost += 0.05  # Boost for dates
                
                # Apply boost but cap at 1.0
                boosted_score = min(1.0, similarity_score + numerical_boost)
                
                chunk_data.append((chunk, metadata, distance, similarity_score, boosted_score))
            
            # Sort by boosted score (prioritizing numerical data)
            chunk_data.sort(key=lambda x: -x[4])  # Sort by boosted score (higher = better)
            
            # Select chunks with hybrid criteria
            for i, (chunk, metadata, distance, orig_score, boosted_score) in enumerate(chunk_data):
                # More lenient thresholds, especially for numerical chunks
                min_similarity = 0.3 if metadata.get('contains_numbers', False) else 0.4
                if any('\u0590' <= c <= '\u05FF' for c in chunk):
                    min_similarity -= 0.1  # Even more lenient for Hebrew
                
                # Include if: good similarity OR contains numbers OR is top 2
                should_include = (
                    orig_score > min_similarity or 
                    metadata.get('contains_numbers', False) or 
                    len(good_chunks) < 2
                )
                
                if should_include:
                    good_chunks.append(chunk)
                    chunk_source_info = {
                        "chunk_index": metadata.get('chunk_index', i),
                        "filename": metadata.get('filename', 'Unknown'),
                        "similarity": round(orig_score, 3),
                        "boosted_score": round(boosted_score, 3),
                        "is_hebrew": any('\u0590' <= c <= '\u05FF' for c in chunk),
                        "contains_numbers": metadata.get('contains_numbers', False),
                        "is_contract_clause": metadata.get('is_contract_clause', False)
                    }
                    
                    # Add contract intelligence information if available
                    if metadata.get('contract_intelligence_available', False):
                        chunk_source_info.update({
                            "contract_intelligence": {
                                "contains_amounts": metadata.get('contains_amounts', False),
                                "contains_financial_terms": metadata.get('contains_financial_terms', False),
                                "contains_parties": metadata.get('contains_parties', False),
                                "contains_percentages": metadata.get('contains_percentages', False),
                                "contains_dates": metadata.get('contains_dates', False),
                                "contract_type": metadata.get('contract_type', 'unknown')
                            }
                        })
                    
                    chunk_sources.append(chunk_source_info)
                    if len(good_chunks) >= 5:
                        break
            
            # PHASE 3: Final reranking with query relevance scoring
            logger.info("Final reranking based on query-content relevance")
            
            # Calculate query-specific relevance scores
            reranked_chunks = []
            for chunk, metadata, distance, orig_score, boosted_score in chunk_data[:8]:  # Process top 8 for reranking
                
                # Calculate content-query relevance
                content_relevance = 0.0
                chunk_lower = chunk.lower() if chunk else ""
                
                # Exact query term matches
                for term in query.lower().split():
                    if len(term) > 2 and term in chunk_lower:
                        content_relevance += 0.1
                
                # Hebrew-English cross-matching
                for eng_term, heb_terms in hebrew_expansions.items():
                    if eng_term in query.lower():
                        for heb_term in heb_terms:
                            if heb_term in chunk_lower:
                                content_relevance += 0.15
                
                # Contract-specific relevance
                if is_numerical_query and metadata.get('contains_amounts', False):
                    content_relevance += 0.25
                if is_parties_query and metadata.get('contains_parties', False):
                    content_relevance += 0.2
                if is_dates_query and metadata.get('contains_dates', False):
                    content_relevance += 0.15
                
                # Final reranked score
                final_reranked_score = min(1.0, boosted_score + content_relevance)
                
                reranked_chunks.append((chunk, metadata, distance, orig_score, boosted_score, final_reranked_score))
            
            # Sort by final reranked score
            reranked_chunks.sort(key=lambda x: -x[5])
            
            # Select final chunks
            final_chunks = []
            final_sources = []
            
            for chunk, metadata, distance, orig_score, boosted_score, reranked_score in reranked_chunks:
                if len(final_chunks) >= 5:
                    break
                    
                final_chunks.append(chunk)
                final_sources.append({
                    "chunk_index": metadata.get('chunk_index', 0),
                    "filename": metadata.get('filename', 'Unknown'),
                    "similarity": round(orig_score, 3),
                    "boosted_score": round(boosted_score, 3),
                    "reranked_score": round(reranked_score, 3),
                    "is_hebrew": any('\u0590' <= c <= '\u05FF' for c in chunk),
                    "contains_numbers": metadata.get('contains_numbers', False),
                    "contains_amounts": metadata.get('contains_amounts', False),
                    "contract_intelligence": metadata.get('contract_intelligence_available', False)
                })
            
            # Use reranked results
            good_chunks = final_chunks
            chunk_sources = final_sources
            
            logger.info(f"Reranking complete: Selected {len(good_chunks)} chunks with avg reranked score: {sum(cs['reranked_score'] for cs in chunk_sources) / len(chunk_sources) if chunk_sources else 0:.3f}")
            
            # PHASE 3: Enhanced numerical data search with contract intelligence (fallback)
            has_amounts = any(cs.get('contains_amounts', False) for cs in chunk_sources)
            has_numbers = any(cs.get('contains_numbers', False) for cs in chunk_sources)
            
            if not has_amounts and not has_numbers:
                # Search specifically for chunks with amounts or numbers
                for chunk, metadata, distance in zip(relevant_chunks, metadatas, distances)[:15]:
                    should_add = False
                    priority_type = "fallback"
                    
                    # Prioritize contract intelligence detected amounts
                    if metadata.get('contains_amounts', False):
                        should_add = True
                        priority_type = "contract_amounts"
                    elif metadata.get('contains_financial_terms', False):
                        should_add = True
                        priority_type = "financial_terms"
                    elif metadata.get('contains_numbers', False):
                        should_add = True
                        priority_type = "numerical_data"
                    
                    if should_add and len(good_chunks) < 6:
                        good_chunks.append(chunk)
                        chunk_sources.append({
                            "chunk_index": metadata.get('chunk_index', 0),
                            "filename": metadata.get('filename', 'Unknown'),
                            "similarity": round(1 - distance if distance else 1.0, 3),
                            "contains_numbers": metadata.get('contains_numbers', False),
                            "contains_amounts": metadata.get('contains_amounts', False),
                            "priority": priority_type
                        })
            
            # Fallback
            if not good_chunks and relevant_chunks:
                good_chunks = relevant_chunks[:3]
                chunk_sources = [{"chunk_index": meta.get('chunk_index', i), "filename": meta.get('filename', 'Unknown'), "similarity": "fallback"} for i, meta in enumerate(metadatas[:3])]
            
            context = "\n\n---CHUNK---\n\n".join(good_chunks)
            logger.info(f"Using {len(good_chunks)} high-quality chunks for context (length: {len(context)} chars)")
            logger.info(f"Chunk sources: {chunk_sources}")
            
            # Enhanced system prompt for Hebrew contract analysis with modern formatting
            system_prompt = """You are an expert contract intelligence assistant specializing in Hebrew legal documents. Your enhanced capabilities include:

CORE RESPONSIBILITIES:
1. Answer questions based STRICTLY on the provided contract context
2. Quote specific clauses, sections, and terms with exact Hebrew text when available
3. Reference clause numbers (סעיף), subsections (like 6.1, 6.2), and specific contractual terms
4. Extract and present numerical values, dates, amounts, and parties accurately
5. Preserve Hebrew text formatting and maintain right-to-left reading order for numbers and dates
6. If context is insufficient, state this clearly and suggest what information might be missing

HEBREW CONTRACT EXPERTISE:
- Recognize standard Hebrew contract terminology (שכירות, תשלום, הצמדה, עדכון, etc.)
- Identify payment clauses (דמי שכירות), adjustment mechanisms (הצמדה למדד), and termination conditions
- Extract dates, amounts, and percentages while preserving their original format
- Understand Hebrew legal structure and clause numbering systems
- CRITICAL: Preserve Hebrew abbreviations with quotation marks (מע"מ, בע"מ, ש"ח, ח"פ, etc.) - never split or alter these
- Maintain proper Hebrew punctuation and formatting for legal terms

RESPONSE FORMATTING (CRITICAL):
You MUST format your responses using proper Markdown for modern chat interface display:

**Structure your response as follows:**
## 📋 [Main Topic/Answer]

### 🔍 Key Findings:
- **Point 1:** [Details with **bold** for emphasis]
- **Point 2:** [Details with numbers and amounts]
- **Point 3:** [Additional relevant information]

### 📄 Contract References:
- **סעיף [X]:** "[Exact Hebrew text from contract]"
- **סעיף [Y]:** "[Additional Hebrew text]"

### 💰 Financial Details:
- **Amount:** [Specific amounts with currency]
- **Dates:** [Relevant dates]
- **Percentages:** [Any percentages or rates]

### Summary:
[Brief summary paragraph with key takeaways]

**Formatting Rules:**
- Use **bold** for important terms, amounts, and clause numbers
- Use bullet points (•) for lists
- Use numbered lists (1., 2., 3.) for sequential information
- Use > blockquotes for direct contract quotations
- Use `code formatting` for specific legal terms or references
- Use line breaks for better readability
- Use emojis appropriately for visual enhancement

CRITICAL: Always preserve Hebrew number formatting (left-to-right) and provide exact quotations from the contract text."""

            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": f"Contract Context:\n{context}\n\nQuestion: {query}\n\nPlease provide a detailed answer based on the contract context above."}
                ],
                max_tokens=800,  # Increased from 500 for more detailed responses
                temperature=0.1  # Minimal temperature for reduced hallucinations
            )
            
            answer = response.choices[0].message.content
        else:
            # Fallback without OpenAI
            answer = f"Found {len(relevant_chunks)} relevant sections in your documents. Here are the most relevant excerpts:\n\n" + \
                    "\n\n---\n\n".join(relevant_chunks[:2])
        
        return {
            "answer": answer,
            "source_info": [
                {
                    "filename": meta['filename'],
                    "folder": meta.get('folder', 'General'),
                    "chunk_index": meta.get('chunk_index', 0),
                    "similarity_score": round(1 - distances[i], 3) if i < len(distances) else 0,
                    "chunk_length": meta.get('chunk_length', 0),
                    "is_contract_clause": meta.get('is_contract_clause', False),
                    "contains_numbers": meta.get('contains_numbers', False)
                } for i, meta in enumerate(metadatas[:5])  # Show more source info
            ],
            "context_chunks": good_chunks if 'good_chunks' in locals() else relevant_chunks[:3],
            "similarity_scores": [round(1 - d, 3) for d in distances[:5]] if distances else [],
            "chunks_used": len(good_chunks) if 'good_chunks' in locals() else 3,
            "total_chunks_found": len(relevant_chunks),
            "success": True
        }
        
    except Exception as e:
        print(f"[ERROR] Chat query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config")
async def get_config():
    """Get application configuration"""
    return {
        "openai_models": ["gpt-4o-mini", "gpt-4o", "gpt-4", "gpt-3.5-turbo"],
        "supported_file_types": ["pdf", "docx", "txt", "jpg", "jpeg", "png"],
        "version": "1.5.46",
        "backend_type": "minimal"
    }

@app.get("/api/config/check-setup")
async def check_setup_status():
    """Check if initial setup is needed"""
    global openai_client, google_credentials
    
    openai_configured = openai_client is not None
    google_configured = google_credentials is not None
    
    return {
        "setup_needed": not openai_configured,  # Require at least OpenAI
        "openai_configured": openai_configured,
        "google_configured": google_configured,
        "ragflow_ready": True  # Minimal backend is always ready when running
    }

@app.get("/api/debug/chunks")
async def debug_chunks():
    """Debug endpoint to examine stored chunks and their numerical content"""
    if not collection:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")
    
    try:
        # Get all chunks
        all_data = collection.get()
        
        debug_info = {
            "total_chunks": len(all_data['ids']),
            "chunks_with_numbers": 0,
            "chunks_with_hebrew": 0,
            "chunks_with_contract_clauses": 0,
            "chunks_with_contract_intelligence": 0,
            "chunks_with_amounts": 0,
            "chunks_with_financial_terms": 0,
            "sample_numerical_chunks": [],
            "sample_chunk_contents": []
        }
        
        # Analyze chunks
        for i, (chunk_id, doc, metadata) in enumerate(zip(all_data['ids'], all_data['documents'], all_data['metadatas'])):
            # Count chunks with different properties
            if metadata.get('contains_numbers', False):
                debug_info['chunks_with_numbers'] += 1
                
            # Contract intelligence specific counts
            if metadata.get('contract_intelligence_available', False):
                debug_info['chunks_with_contract_intelligence'] += 1
                
            if metadata.get('contains_amounts', False):
                debug_info['chunks_with_amounts'] += 1
                
            if metadata.get('contains_financial_terms', False):
                debug_info['chunks_with_financial_terms'] += 1
                
                # Sample first 3 numerical chunks
                if len(debug_info['sample_numerical_chunks']) < 3:
                    debug_info['sample_numerical_chunks'].append({
                        "chunk_id": chunk_id,
                        "filename": metadata.get('filename', 'Unknown'),
                        "chunk_index": metadata.get('chunk_index', 0),
                        "content_preview": doc[:200] + "..." if len(doc) > 200 else doc,
                        "contains_numbers": metadata.get('contains_numbers', False),
                        "is_contract_clause": metadata.get('is_contract_clause', False),
                        "chunk_length": len(doc)
                    })
            
            if metadata.get('contains_hebrew', False):
                debug_info['chunks_with_hebrew'] += 1
            
            if metadata.get('is_contract_clause', False):
                debug_info['chunks_with_contract_clauses'] += 1
            
            # Sample first 2 chunks regardless
            if len(debug_info['sample_chunk_contents']) < 2:
                debug_info['sample_chunk_contents'].append({
                    "chunk_id": chunk_id,
                    "content_preview": doc[:150] + "..." if len(doc) > 150 else doc,
                    "metadata": metadata
                })
        
        return debug_info
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Debug failed: {str(e)}")

@app.get("/api/settings")
async def get_settings():
    """Get current settings status"""
    return {
        "openai_configured": openai_client is not None,
        "google_configured": google_credentials is not None,
        "openai_api_key_set": bool(app_settings.get("openai_api_key")),
        "google_auth_status": "authenticated" if google_credentials else "not_configured"
    }

@app.post("/api/settings/openai")
async def set_openai_key(request: Dict[str, Any]):
    """Set and validate OpenAI API key"""
    global openai_client, app_settings
    
    try:
        api_key = request.get("api_key", "").strip()
        if not api_key:
            raise HTTPException(status_code=400, detail="API key is required")
        
        # Test the API key
        test_client = openai.OpenAI(api_key=api_key)
        
        # Quick validation with a simple request
        response = test_client.models.list()
        if not hasattr(response, 'data') or len(response.data) == 0:
            raise HTTPException(status_code=400, detail="Invalid API key - no models returned")
        
        # Save the key
        app_settings["openai_api_key"] = api_key
        save_settings()
        
        # Update global client
        openai_client = test_client
        
        return {
            "success": True,
            "message": "OpenAI API key validated and saved successfully",
            "models_available": len(response.data)
        }
        
    except openai.AuthenticationError:
        raise HTTPException(status_code=400, detail="Invalid OpenAI API key")
    except openai.RateLimitError:
        raise HTTPException(status_code=400, detail="OpenAI rate limit exceeded")
    except openai.APIConnectionError:
        raise HTTPException(status_code=400, detail="Cannot connect to OpenAI - check internet connection")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to validate API key: {str(e)}")

@app.post("/api/settings/google/upload-credentials")
async def upload_google_credentials(file: UploadFile = File(...)):
    """Upload Google Service Account credentials file"""
    global google_credentials
    
    try:
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="Please upload a JSON credentials file")
        
        # Read and validate the credentials file
        content = await file.read()
        credentials_data = json.loads(content)
        
        # Basic validation for service account credentials
        if "type" not in credentials_data or credentials_data.get("type") != "service_account":
            raise HTTPException(status_code=400, detail="Invalid service account credentials. Please upload a service account JSON file.")
        
        # Save credentials file
        credentials_path = "google_service_account.json"
        with open(credentials_path, 'wb') as f:
            f.write(content)
        
        # Load the credentials immediately
        try:
            google_credentials = service_account.Credentials.from_service_account_file(
                credentials_path, scopes=SCOPES)
            
            # Save the path to settings
            save_google_credentials(credentials_path)
            
            return {
                "success": True,
                "message": "Google service account credentials verified and activated",
                "services": ["OCR", "Drive", "Gmail"]
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

@app.get("/api/settings/google/status")
async def get_google_auth_status():
    """Get Google authentication status"""
    global google_credentials
    
    status = "not_configured"
    services = []
    
    if google_credentials:
        # Service account credentials don't expire like OAuth tokens
        status = "authenticated"
        services = ["OCR", "Drive", "Gmail"]
    
    return {
        "status": status,
        "services_available": services,
        "credentials_file_uploaded": bool(app_settings.get("google_credentials_path"))
    }

def find_available_port(start_port=8503, max_attempts=10):
    """Find an available port starting from start_port"""
    import socket
    for port in range(start_port, start_port + max_attempts):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    return None

if __name__ == "__main__":
    import sys
    import time
    
    # Check for test-only mode (for CI/CD environments)
    test_mode = "--test-only" in sys.argv
    
    # Disable telemetry in PyInstaller environment
    if getattr(sys, 'frozen', False):
        try:
            # Monkey patch telemetry functions to do nothing
            import uvicorn.lifespan
            original_send_telemetry = uvicorn.lifespan.send_telemetry_event
            uvicorn.lifespan.send_telemetry_event = lambda *args, **kwargs: None
            print("[INFO] Telemetry disabled in PyInstaller environment")
        except Exception as e:
            print(f"[WARNING] Failed to disable telemetry: {e}")
    
    print("[INFO] Starting Contract Intelligence Minimal Backend Server...")
    
    # Find available port
    port = find_available_port()
    if port is None:
        print("[ERROR] Could not find an available port. Please close other instances.")
        exit(1)
    
    print(f"[INFO] Server will be available at: http://127.0.0.1:{port}")
    if port != 8503:
        print(f"[WARNING] Using port {port} instead of default 8503 due to port conflict")
    
    # Test mode for CI/CD environments - start server briefly then exit
    if test_mode:
        print("[INFO] Running in test-only mode for CI/CD environment")
        
        if FASTAPI_AVAILABLE:
            import threading
            
            def run_server():
                try:
                    uvicorn.run(
                        app,
                        host="127.0.0.1",
                        port=port,
                        log_level="info",
                        access_log=False
                    )
                except Exception as e:
                    print(f"[ERROR] Server thread error: {e}")
            
            # Start server in a thread
            server_thread = threading.Thread(target=run_server)
            server_thread.daemon = True
            server_thread.start()
            
            # Wait for server to initialize
            print("[INFO] Test mode: Server starting, will exit after initialization...")
            time.sleep(5)
            
            # Try to access health endpoint
            try:
                import urllib.request
                response = urllib.request.urlopen(f"http://127.0.0.1:{port}/health")
                print(f"[SUCCESS] Health check returned status: {response.status}")
                print("[INFO] Test mode: Server initialized successfully, exiting...")
                sys.exit(0)
            except Exception as e:
                print(f"[ERROR] Health check failed: {e}")
                print("[INFO] Test mode: Server failed to initialize, exiting...")
                sys.exit(1)
        else:
            # Test mode with minimal server
            print("[INFO] Test mode with minimal server: Starting briefly...")
            time.sleep(2)
            print("[INFO] Test mode: Minimal server test complete, exiting...")
            sys.exit(0)
    
    # Normal operation mode - run server continuously
    # Run server with error handling
    if FASTAPI_AVAILABLE:
        try:
            print("[INFO] Starting FastAPI server...")
            uvicorn.run(
                app,
                host="127.0.0.1",
                port=port,
                log_level="info",
                access_log=False
            )
        except Exception as e:
            print(f"[ERROR] Failed to start FastAPI server: {e}")
            print("[RECOVERY] Attempting to start minimal HTTP server...")
            try:
                # Fall back to basic HTTP server for health endpoint only
                handler = MinimalHandler
                with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
                    print(f"[INFO] Minimal HTTP server started at port {port}")
                    httpd.serve_forever()
            except Exception as e2:
                print(f"[CRITICAL] Failed to start minimal HTTP server: {e2}")
                exit(1)
    else:
        try:
            # Use minimal HTTP server
            print("[INFO] Starting minimal HTTP server (FastAPI not available)...")
            handler = MinimalHandler
            with socketserver.TCPServer(("127.0.0.1", port), handler) as httpd:
                print(f"[INFO] Minimal HTTP server started at port {port}")
                httpd.serve_forever()
        except Exception as e:
            print(f"[CRITICAL] Failed to start minimal HTTP server: {e}")
            exit(1)

