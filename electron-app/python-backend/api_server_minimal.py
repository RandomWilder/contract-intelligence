#!/usr/bin/env python3
"""
Minimal FastAPI backend for Contract Intelligence Platform
Uses only installed packages: FastAPI, OpenAI, ChromaDB, basic document processing
"""

import os
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional

# Core imports
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

# Document processing
import PyPDF2
import docx
from PIL import Image

# AI and vector database
import openai
import chromadb
from chromadb.config import Settings

# Utilities
import pandas as pd
import tiktoken
from dotenv import load_dotenv

# Google OAuth imports
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
import pickle

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
GOOGLE_TOKEN_FILE = "google_token.pickle"

# Global instances
chroma_client = None
collection = None
openai_client = None
google_credentials = None
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
    """Load Google OAuth credentials"""
    global google_credentials
    try:
        if os.path.exists(GOOGLE_TOKEN_FILE):
            with open(GOOGLE_TOKEN_FILE, 'rb') as f:
                google_credentials = pickle.load(f)
            print("[INFO] Google credentials loaded")
            return True
    except Exception as e:
        print(f"[WARNING] Failed to load Google credentials: {e}")
    return False

def save_google_credentials(credentials):
    """Save Google OAuth credentials"""
    try:
        with open(GOOGLE_TOKEN_FILE, 'wb') as f:
            pickle.dump(credentials, f)
        print("[INFO] Google credentials saved")
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save Google credentials: {e}")
        return False

def initialize_services():
    """Initialize ChromaDB and OpenAI services"""
    global chroma_client, collection, openai_client
    
    # Load settings first
    load_settings()
    load_google_credentials()
    
    try:
        # Initialize ChromaDB
        chroma_client = chromadb.Client(Settings(
            persist_directory="./chroma_db",
            anonymized_telemetry=False
        ))
        
        # Get or create collection
        collection = chroma_client.get_or_create_collection(
            name="contracts_electron",
            metadata={"description": "Contract documents for Electron app"}
        )
        
        print("[SUCCESS] ChromaDB initialized successfully")
        
        # Initialize OpenAI (from settings or environment)
        api_key = app_settings.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
        if api_key:
            openai_client = openai.OpenAI(api_key=api_key)
            print("[SUCCESS] OpenAI client initialized")
        else:
            print("[WARNING] No OpenAI API key found - please configure in settings")
            
    except Exception as e:
        print(f"[ERROR] Failed to initialize services: {e}")

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
    """Simple text chunking function"""
    if len(text) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence endings near the chunk boundary
            sentence_end = max(
                text.rfind('.', start, end),
                text.rfind('!', start, end),
                text.rfind('?', start, end)
            )
            if sentence_end > start + chunk_size // 2:
                end = sentence_end + 1
        
        chunks.append(text[start:end].strip())
        start = end - chunk_overlap
        
        if start >= len(text):
            break
    
    return chunks

def extract_text_from_pdf(file_path: str) -> str:
    """Extract text from PDF using PyPDF2"""
    try:
        with open(file_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Failed to extract text from PDF: {str(e)}")

def extract_text_from_docx(file_path: str) -> str:
    """Extract text from DOCX using python-docx"""
    try:
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            text += paragraph.text + "\n"
        return text.strip()
    except Exception as e:
        raise Exception(f"Failed to extract text from DOCX: {str(e)}")

def extract_text_from_txt(file_path: str) -> str:
    """Extract text from TXT file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read().strip()
    except Exception as e:
        raise Exception(f"Failed to extract text from TXT: {str(e)}")

# Initialize FastAPI app
app = FastAPI(
    title="Contract Intelligence API - Minimal",
    description="Minimal backend API for Contract Intelligence Desktop App",
    version="1.5.13"
)

# Enable CORS for Electron frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    print("[INFO] Starting Contract Intelligence Minimal Backend...")
    initialize_services()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.5.13",
        "backend": "minimal",
        "chromadb_ready": chroma_client is not None,
        "openai_ready": openai_client is not None
    }

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify backend functionality"""
    if not chroma_client:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")
    
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
                "collection_name": "contracts_electron",
                "chromadb_collections": len(chroma_client.list_collections())
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend test failed: {str(e)}")

@app.get("/api/status")
async def get_status():
    """Get application status"""
    if not chroma_client:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")
    
    try:
        collections = chroma_client.list_collections()
        doc_count = collection.count() if collection else 0
        
        return {
            "openai_configured": openai_client is not None,
            "chromadb_ready": True,
            "documents_count": doc_count,
            "collections": len(collections)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    folder: str = Form("General")
):
    """Upload and process a document"""
    if not chroma_client or not collection:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Extract text based on file type
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext == '.pdf':
            text = extract_text_from_pdf(temp_path)
        elif file_ext == '.docx':
            text = extract_text_from_docx(temp_path)
        elif file_ext == '.txt':
            text = extract_text_from_txt(temp_path)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
        
        if not text.strip():
            raise HTTPException(status_code=400, detail="No text content found in document")
        
        # Chunk the text
        chunks = chunk_text(text)
        
        # Add to ChromaDB
        doc_id = f"{folder}_{file.filename}_{len(collection.get()['ids'])}"
        
        collection.add(
            documents=chunks,
            ids=[f"{doc_id}_chunk_{i}" for i in range(len(chunks))],
            metadatas=[{
                "filename": file.filename,
                "folder": folder,
                "chunk_index": i,
                "total_chunks": len(chunks)
            } for i in range(len(chunks))]
        )
        
        # Clean up temp file
        os.unlink(temp_path)
        
        return {
            "message": f"Document '{file.filename}' processed successfully",
            "success": True,
            "chunks_created": len(chunks),
            "folder": folder
        }
        
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
        for i, metadata in enumerate(results['metadatas']):
            filename = metadata['filename']
            folder = metadata.get('folder', 'General')
            
            if filename not in documents:
                documents[filename] = {
                    'filename': filename,
                    'folder': folder,
                    'chunks': 0
                }
            documents[filename]['chunks'] += 1
        
        return {
            "documents": list(documents.values()),
            "total_count": len(documents)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_query(request: Dict[str, Any]):
    """Process chat query using ChromaDB similarity search"""
    if not collection:
        raise HTTPException(status_code=503, detail="ChromaDB not initialized")
    
    try:
        query = request.get("query", "").strip()
        if not query:
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Search in ChromaDB
        results = collection.query(
            query_texts=[query],
            n_results=5
        )
        
        if not results['documents'] or not results['documents'][0]:
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents.",
                "source_info": [],
                "context_chunks": [],
                "success": True
            }
        
        # Get relevant chunks
        relevant_chunks = results['documents'][0]
        metadatas = results['metadatas'][0]
        distances = results['distances'][0] if results['distances'] else []
        
        # If OpenAI is available, generate an answer
        if openai_client:
            context = "\n\n".join(relevant_chunks[:3])  # Use top 3 chunks
            
            response = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that answers questions based on provided document context. If the context doesn't contain relevant information, say so clearly."},
                    {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
                ],
                max_tokens=500,
                temperature=0.3
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
                    "chunk_index": meta.get('chunk_index', 0)
                } for meta in metadatas[:3]
            ],
            "context_chunks": relevant_chunks[:3],
            "similarity_scores": [1 - d for d in distances[:3]] if distances else [],
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
        "supported_file_types": ["pdf", "docx", "txt"],
        "version": "1.5.13",
        "backend_type": "minimal"
    }

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
    """Upload Google OAuth credentials file"""
    try:
        if not file.filename.endswith('.json'):
            raise HTTPException(status_code=400, detail="Please upload a JSON credentials file")
        
        # Read and validate the credentials file
        content = await file.read()
        credentials_data = json.loads(content)
        
        # Basic validation
        if "installed" not in credentials_data and "web" not in credentials_data:
            raise HTTPException(status_code=400, detail="Invalid credentials file format")
        
        # Save credentials file
        credentials_path = "google_credentials.json"
        with open(credentials_path, 'wb') as f:
            f.write(content)
        
        app_settings["google_credentials_path"] = credentials_path
        save_settings()
        
        return {
            "success": True,
            "message": "Google credentials file uploaded successfully",
            "next_step": "authenticate"
        }
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload credentials: {str(e)}")

@app.post("/api/settings/google/authenticate")
async def authenticate_google():
    """Start Google OAuth authentication flow"""
    global google_credentials
    
    try:
        credentials_path = app_settings.get("google_credentials_path")
        if not credentials_path or not os.path.exists(credentials_path):
            raise HTTPException(status_code=400, detail="Please upload credentials file first")
        
        # Create OAuth flow
        flow = InstalledAppFlow.from_client_secrets_file(credentials_path, SCOPES)
        
        # Run local server flow (will open browser)
        credentials = flow.run_local_server(port=0)
        
        # Save credentials
        google_credentials = credentials
        save_google_credentials(credentials)
        
        return {
            "success": True,
            "message": "Google authentication completed successfully",
            "services": ["OCR", "Drive", "Gmail"]
        }
        
    except Exception as e:
        print(f"[ERROR] Google authentication failed: {e}")
        raise HTTPException(status_code=500, detail=f"Authentication failed: {str(e)}")

@app.get("/api/settings/google/status")
async def get_google_auth_status():
    """Get Google authentication status"""
    global google_credentials
    
    status = "not_configured"
    services = []
    
    if google_credentials:
        if google_credentials.expired:
            if google_credentials.refresh_token:
                try:
                    google_credentials.refresh(Request())
                    save_google_credentials(google_credentials)
                    status = "authenticated"
                    services = ["OCR", "Drive", "Gmail"]
                except:
                    status = "expired"
            else:
                status = "expired"
        else:
            status = "authenticated"
            services = ["OCR", "Drive", "Gmail"]
    
    return {
        "status": status,
        "services_available": services,
        "credentials_file_uploaded": bool(app_settings.get("google_credentials_path"))
    }

if __name__ == "__main__":
    print("[INFO] Starting Contract Intelligence Minimal Backend Server...")
    print("[INFO] Server will be available at: http://127.0.0.1:8503")
    
    # Run server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8503,
        log_level="info",
        access_log=False
    )

