#!/usr/bin/env python3
"""
Simple FastAPI backend for Contract Intelligence Platform
Wraps existing functionality in REST API
"""

import os
import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import uvicorn
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import tempfile
import shutil

# Add parent directory to path to import existing modules
sys.path.append(str(Path(__file__).parent.parent))

# Import your existing modules
from local_rag_app import LocalRAGFlow, GoogleAuthManager
from contract_intelligence import ContractIntelligenceEngine
from telemetry_client import get_telemetry_client

from contextlib import asynccontextmanager

# Global instances
rag_flow = None
telemetry = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize services on startup"""
    global rag_flow, telemetry
    
    print("[INFO] Starting Contract Intelligence Backend...")
    
    # Initialize telemetry (disabled for now to avoid connection errors)
    try:
        telemetry = get_telemetry_client()
        telemetry.disable()  # Disable to avoid connection errors
    except:
        telemetry = None
    
    # Initialize RAG with default model and separate collection for Electron
    try:
        rag_flow = LocalRAGFlow(chat_model="gpt-4o-mini", use_electron_collection=True)
        print("[SUCCESS] RAGFlow initialized successfully for Electron app")
        print(f"[INFO] Using separate ChromaDB collection: 'contracts_electron'")
    except Exception as e:
        print(f"[ERROR] Failed to initialize RAGFlow: {e}")
        # Continue without RAGFlow - let frontend handle the error
    
    yield
    
    # Cleanup
    if telemetry:
        telemetry.track_app_end()

# Initialize FastAPI app
app = FastAPI(
    title="Contract Intelligence API",
    description="Backend API for Contract Intelligence Desktop App",
    version="1.5.23",
    lifespan=lifespan
)

# Enable CORS for Electron frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.5.21",
        "ragflow_ready": rag_flow is not None
    }

@app.get("/api/test")
async def test_endpoint():
    """Test endpoint to verify backend functionality"""
    if not rag_flow:
        raise HTTPException(status_code=503, detail="RAGFlow not initialized")
    
    try:
        # Test basic functionality
        test_text = "This is a test document."
        chunks = rag_flow.chunk_text(test_text)
        
        return {
            "status": "success",
            "message": "Backend is working correctly",
            "test_results": {
                "chunking": f"Generated {len(chunks)} chunks",
                "openai_available": bool(os.getenv("OPENAI_API_KEY")),
                "collection_name": "contracts_electron"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Backend test failed: {str(e)}")

@app.get("/api/status")
async def get_status():
    """Get application status"""
    if not rag_flow:
        raise HTTPException(status_code=503, detail="RAGFlow not initialized")
    
    return {
        "openai_configured": bool(os.getenv("OPENAI_API_KEY")),
        "google_authenticated": rag_flow.is_google_authenticated(),
        "documents_count": len(rag_flow.list_documents()),
        "folders": rag_flow.list_folders()
    }

# **NEW: Configuration endpoint for first-run setup**
@app.post("/api/config/setup")
async def setup_config(request: Dict[str, Any]):
    """Setup API keys and credentials for first-run experience"""
    global rag_flow
    
    try:
        openai_key = request.get("openai_key", "").strip()
        google_creds_path = request.get("google_creds_path", "").strip()
        
        changes_made = []
        
        # Setup OpenAI API key
        if openai_key:
            # Validate the API key before setting it
            try:
                import openai
                test_client = openai.OpenAI(api_key=openai_key, timeout=10.0)
                # Quick validation with models.list
                models = test_client.models.list()
                if hasattr(models, 'data') and len(models.data) > 0:
                    os.environ['OPENAI_API_KEY'] = openai_key
                    changes_made.append("OpenAI API key configured")
                    
                    # Update RAGFlow with new key if it exists
                    if rag_flow:
                        rag_flow.openai_client = test_client
                else:
                    raise HTTPException(status_code=400, detail="Invalid OpenAI API key - no models returned")
            except openai.AuthenticationError:
                raise HTTPException(status_code=400, detail="Invalid OpenAI API key")
            except openai.RateLimitError:
                raise HTTPException(status_code=400, detail="OpenAI rate limit exceeded - check your usage limits")
            except openai.APIConnectionError:
                raise HTTPException(status_code=400, detail="Cannot connect to OpenAI - check your internet connection")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"OpenAI API key validation failed: {str(e)}")
        
        # Setup Google credentials
        if google_creds_path and os.path.exists(google_creds_path):
            try:
                from utils import get_google_credentials_path
                dest_path = get_google_credentials_path()
                
                # Ensure destination directory exists
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Copy credentials file
                shutil.copy2(google_creds_path, dest_path)
                os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = str(dest_path)
                changes_made.append("Google credentials configured")
                
                # Try to initialize Google services if RAGFlow exists
                if rag_flow:
                    try:
                        rag_flow.auth_manager = GoogleAuthManager()
                        if rag_flow.auth_manager.load_credentials():
                            from local_rag_app import GoogleVisionOCR
                            rag_flow.vision_ocr = GoogleVisionOCR(rag_flow.auth_manager.credentials)
                            changes_made.append("Google OCR services initialized")
                    except Exception as google_error:
                        print(f"[WARNING] Google services setup failed: {google_error}")
                        # Don't fail the whole setup for Google issues
                        
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Failed to setup Google credentials: {str(e)}")
        
        # Reinitialize RAGFlow if we have new credentials and it wasn't initialized before
        if not rag_flow and openai_key:
            try:
                rag_flow = LocalRAGFlow(chat_model="gpt-4o-mini", use_electron_collection=True)
                changes_made.append("RAGFlow system initialized")
            except Exception as e:
                print(f"[WARNING] RAGFlow initialization failed: {e}")
                # Don't fail setup for this
        
        return {
            "success": True, 
            "message": f"Configuration updated: {', '.join(changes_made) if changes_made else 'No changes made'}",
            "changes": changes_made
        }
        
    except HTTPException:
        # Re-raise HTTP exceptions (validation errors)
        raise
    except Exception as e:
        print(f"[ERROR] Configuration setup failed: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Configuration setup failed: {str(e)}")

@app.get("/api/config/check-setup")
async def check_setup_status():
    """Check if initial setup is needed"""
    openai_configured = bool(os.getenv("OPENAI_API_KEY"))
    
    from utils import get_google_credentials_path
    google_creds_exist = os.path.exists(get_google_credentials_path())
    
    return {
        "setup_needed": not (openai_configured),  # Require at least OpenAI
        "openai_configured": openai_configured,
        "google_configured": google_creds_exist,
        "ragflow_ready": rag_flow is not None
    }

@app.get("/api/documents")
async def get_documents():
    """Get list of documents"""
    if not rag_flow:
        raise HTTPException(status_code=503, detail="RAGFlow not initialized")
    
    try:
        documents = rag_flow.list_documents()
        documents_by_folder = rag_flow.list_documents_by_folder()
        
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
    """Upload and process a document"""
    if not rag_flow:
        raise HTTPException(status_code=503, detail="RAGFlow not initialized")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_path = temp_file.name
        
        # Process document
        result = rag_flow.add_document(
            temp_path,
            file.filename,
            use_ocr=use_ocr,
            folder=folder
        )
        
        # Clean up temp file
        os.unlink(temp_path)
        
        # Track telemetry
        if telemetry:
            telemetry.track_document_processed(
                file.content_type or "unknown",
                file.size or 0,
                0  # Processing time would need to be measured
            )
        
        return {"message": result, "success": True}
        
    except Exception as e:
        # Clean up temp file if it exists
        if 'temp_path' in locals():
            try:
                os.unlink(temp_path)
            except:
                pass
        
        # Enhanced error logging
        print(f"[ERROR] Document processing failed: {e}")
        import traceback
        traceback.print_exc()
        
        # Track error in telemetry
        if telemetry:
            telemetry.track_error("document_upload", str(e), "api_server")
        
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/documents/{document_name}")
async def delete_document(document_name: str):
    """Delete a document"""
    if not rag_flow:
        raise HTTPException(status_code=503, detail="RAGFlow not initialized")
    
    try:
        result = rag_flow.delete_document(document_name)
        return {"message": result, "success": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat")
async def chat_query(request: Dict[str, Any]):
    """Process chat query"""
    if not rag_flow:
        raise HTTPException(status_code=503, detail="RAGFlow not initialized")
    
    try:
        query = request.get("query", "")
        target_documents = request.get("target_documents")
        target_folder = request.get("target_folder")
        
        if not query.strip():
            raise HTTPException(status_code=400, detail="Query cannot be empty")
        
        # Process query
        results = rag_flow.query_documents(
            query,
            target_documents=target_documents,
            target_folder=target_folder
        )
        
        # Track telemetry
        if telemetry:
            telemetry.track_chat_interaction(
                model_used=rag_flow.chat_model,
                response_time=0,  # Would need to measure this
                tokens_used=None
            )
        
        return {
            "answer": results["answer"],
            "source_info": results["source_info"],
            "context_chunks": results["context_chunks"],
            "similarity_scores": results["similarity_scores"],
            "success": True
        }
        
    except Exception as e:
        if telemetry:
            telemetry.track_error("chat_query", str(e), "api_server")
        
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/google/auth-status")
async def get_google_auth_status():
    """Get Google authentication status"""
    if not rag_flow:
        raise HTTPException(status_code=503, detail="RAGFlow not initialized")
    
    # Check if credentials file exists
    from utils import get_google_credentials_path
    credentials_path = get_google_credentials_path()
    
    return {
        "authenticated": rag_flow.is_google_authenticated(),
        "services_available": ["vision", "drive"],
        "credentials_file_exists": os.path.exists(credentials_path),
        "credentials_path": credentials_path,
        "needs_setup": not os.path.exists(credentials_path)
    }

@app.post("/api/google/authenticate")
async def authenticate_google():
    """Authenticate with Google services"""
    if not rag_flow:
        raise HTTPException(status_code=503, detail="RAGFlow not initialized")
    
    try:
        success = rag_flow.auth_manager.authenticate()
        if success:
            # Initialize Vision OCR
            from local_rag_app import GoogleVisionOCR
            rag_flow.vision_ocr = GoogleVisionOCR(rag_flow.auth_manager.credentials)
            
        return {"success": success}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/google/setup-credentials")
async def setup_google_credentials(request: Dict[str, Any]):
    """Setup Google credentials by copying file to expected location"""
    if not rag_flow:
        raise HTTPException(status_code=503, detail="RAGFlow not initialized")
    
    try:
        source_path = request.get("source_path")
        if not source_path or not os.path.exists(source_path):
            raise HTTPException(status_code=400, detail="Invalid source file path")
        
        # Copy to expected location
        from utils import get_google_credentials_path
        dest_path = get_google_credentials_path()
        
        import shutil
        shutil.copy2(source_path, dest_path)
        
        return {
            "success": True, 
            "message": "Credentials file copied successfully",
            "path": dest_path
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/google/clear-credentials")
async def clear_google_credentials():
    """Clear Google credentials"""
    if not rag_flow:
        raise HTTPException(status_code=503, detail="RAGFlow not initialized")
    
    try:
        rag_flow.clear_google_credentials()
        return {"success": True, "message": "Credentials cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/config")
async def get_config():
    """Get application configuration"""
    return {
        "openai_models": [
            "gpt-4o-mini",
            "gpt-4o", 
            "gpt-4",
            "gpt-4-turbo",
            "gpt-3.5-turbo"
        ],
        "supported_file_types": ["pdf", "docx", "txt", "jpg", "jpeg", "png"],
        "version": "1.5.21"
    }

# Error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    print(f"Unhandled exception: {exc}")
    
    if telemetry:
        telemetry.track_error("unhandled_exception", str(exc), "api_server")
    
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "success": False}
    )

if __name__ == "__main__":
    print("[INFO] Starting Contract Intelligence Backend Server...")
    print("[INFO] Server will be available at: http://127.0.0.1:8503")
    
    # Run server
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=8503,
        log_level="info",
        access_log=False  # Reduce noise
    )