# local_rag_app.py
import os
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
import openai
from dotenv import load_dotenv
import PyPDF2
import docx
import hashlib
import keyring
import json
import requests
import base64
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import Flow
from google.auth.exceptions import RefreshError
import google.auth
from chromadb.config import Settings
from contract_intelligence import ContractIntelligenceEngine
from datetime import datetime

load_dotenv()

def get_google_credentials_path():
    """Get the path to Google credentials file in user's home directory"""
    import os
    home_dir = os.path.expanduser("~")
    credentials_dir = os.path.join(home_dir, ".contract_intelligence")
    os.makedirs(credentials_dir, exist_ok=True)
    new_path = os.path.join(credentials_dir, "google_credentials.json")
    
    # Migration: Check if old file exists and move it
    old_path = "google_credentials.json"
    if os.path.exists(old_path) and not os.path.exists(new_path):
        import shutil
        shutil.move(old_path, new_path)
        print(f"✅ Migrated Google credentials to secure location: {new_path}")
    
    return new_path

class GoogleAuthManager:
    """Handles Google OAuth authentication and credential management"""
    
    # OAuth2 scopes for contract analysis with Vision API
    SCOPES = [
        'https://www.googleapis.com/auth/cloud-platform',     # Full Google Cloud Platform access
        'https://www.googleapis.com/auth/cloud-vision',       # Specific Vision API access
        'https://www.googleapis.com/auth/drive.readonly',     # Google Drive (optional)
    ]
    
    # Full scopes (uncomment if you need all integrations)
    # SCOPES = [
    #     'https://www.googleapis.com/auth/cloud-platform',     # Vision API, Translate
    #     'https://www.googleapis.com/auth/drive.readonly',     # Google Drive
    #     'https://www.googleapis.com/auth/gmail.readonly',     # Gmail
    #     'https://www.googleapis.com/auth/calendar',           # Calendar
    #     'https://www.googleapis.com/auth/documents.readonly'  # Google Docs
    # ]
    
    def __init__(self):
        self.credentials = None
        self.client_config_path = get_google_credentials_path()  # User's home directory credentials
        
    def save_credentials(self, credentials):
        """Securely save credentials to system keyring"""
        cred_data = {
            'token': credentials.token,
            'refresh_token': credentials.refresh_token,
            'token_uri': credentials.token_uri,
            'client_id': credentials.client_id,
            'client_secret': credentials.client_secret,
            'scopes': credentials.scopes
        }
        keyring.set_password("contract_intelligence", "google_oauth", json.dumps(cred_data))
    
    def load_credentials(self):
        """Load credentials from system keyring"""
        try:
            cred_json = keyring.get_password("contract_intelligence", "google_oauth")
            if cred_json:
                cred_data = json.loads(cred_json)
                from google.oauth2.credentials import Credentials
                credentials = Credentials(
                    token=cred_data['token'],
                    refresh_token=cred_data['refresh_token'],
                    token_uri=cred_data['token_uri'],
                    client_id=cred_data['client_id'],
                    client_secret=cred_data['client_secret'],
                    scopes=cred_data['scopes']
                )
                
                # Refresh if needed
                if credentials.expired and credentials.refresh_token:
                    credentials.refresh(Request())
                    self.save_credentials(credentials)
                
                self.credentials = credentials
                return True
        except Exception as e:
            print(f"Error loading credentials: {e}")
        return False
    
    def authenticate(self):
        """Device flow for desktop authentication"""
        if not os.path.exists(self.client_config_path):
            raise FileNotFoundError(f"Google credentials file not found: {self.client_config_path}")
        
        from google_auth_oauthlib.flow import InstalledAppFlow
        
        flow = InstalledAppFlow.from_client_secrets_file(
            self.client_config_path,
            scopes=self.SCOPES
        )
        
        # Use device flow
        credentials = flow.run_local_server(port=0, open_browser=True)
        self.credentials = credentials
        self.save_credentials(credentials)
        return True
    
    def complete_auth(self, flow, authorization_response):
        """Complete OAuth flow with authorization response"""
        flow.fetch_token(authorization_response=authorization_response)
        self.credentials = flow.credentials
        self.save_credentials(self.credentials)
        return True
    
    def is_authenticated(self):
        """Check if user is authenticated with proper scopes"""
        if self.credentials is None:
            return False
        
        # Check if credentials are expired
        if self.credentials.expired:
            # Try to refresh if we have a refresh token
            if self.credentials.refresh_token:
                try:
                    self.credentials.refresh(Request())
                    self.save_credentials(self.credentials)
                except RefreshError:
                    return False
            else:
                return False
        
        # Validate that we have the required scopes
        if not self.credentials.scopes:
            return False
            
        required_scopes = set(self.SCOPES)
        granted_scopes = set(self.credentials.scopes)
        
        # Check if all required scopes are granted
        return required_scopes.issubset(granted_scopes)
    
    def clear_credentials(self):
        """Clear stored credentials to force re-authentication"""
        try:
            keyring.delete_password("contract_intelligence", "google_oauth")
            self.credentials = None
            print("Credentials cleared successfully")
        except Exception as e:
            print(f"Error clearing credentials: {e}")

class GoogleVisionOCR:
    """Google Vision API OCR integration using OAuth2 with proper scopes"""
    
    def __init__(self, credentials):
        self.credentials = credentials
        self.vision_api_url = "https://vision.googleapis.com/v1/images:annotate"
        # Check if credentials have the right scopes
        self._validate_scopes()
    
    def _validate_scopes(self):
        """Validate that credentials have the required scopes for Vision API"""
        required_scopes = [
            'https://www.googleapis.com/auth/cloud-platform',
            'https://www.googleapis.com/auth/cloud-vision'
        ]
        
        if hasattr(self.credentials, 'scopes') and self.credentials.scopes:
            granted_scopes = set(self.credentials.scopes)
            has_cloud_platform = 'https://www.googleapis.com/auth/cloud-platform' in granted_scopes
            has_cloud_vision = 'https://www.googleapis.com/auth/cloud-vision' in granted_scopes
            
            if not (has_cloud_platform or has_cloud_vision):
                print("Warning: Credentials may not have the required scopes for Vision API")
                print(f"Current scopes: {self.credentials.scopes}")
                print(f"Required scopes: {required_scopes}")
    
    def _get_access_token(self):
        """Get valid access token, refreshing if necessary"""
        if self.credentials.expired and self.credentials.refresh_token:
            try:
                self.credentials.refresh(Request())
            except Exception as e:
                print(f"Token refresh error: {e}")
                return None
        return self.credentials.token
    
    def _make_vision_request(self, image_content):
        """Make a request to Google Vision API using REST API with proper OAuth2 token"""
        try:
            # Encode image content to base64
            image_base64 = base64.b64encode(image_content).decode('utf-8')
            
            # Prepare the request payload
            request_payload = {
                "requests": [
                    {
                        "image": {
                            "content": image_base64
                        },
                        "features": [
                            {
                                "type": "TEXT_DETECTION"
                            }
                        ]
                    }
                ]
            }
            
            # Get access token
            access_token = self._get_access_token()
            if not access_token:
                print("Failed to get access token")
                return ""
            
            # Make the API request with proper headers
            headers = {
                "Authorization": f"Bearer {access_token}",
                "Content-Type": "application/json",
                "X-Goog-User-Project": self._get_project_id()
            }
            
            response = requests.post(
                self.vision_api_url,
                headers=headers,
                json=request_payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if "responses" in result and len(result["responses"]) > 0:
                    response_data = result["responses"][0]
                    if "textAnnotations" in response_data and len(response_data["textAnnotations"]) > 0:
                        return response_data["textAnnotations"][0]["description"]
                return ""
            else:
                print(f"Vision API Error: {response.status_code} - {response.text}")
                # If it's a scope error, provide helpful message
                if response.status_code == 403:
                    print("This might be a scope issue. Ensure your OAuth2 credentials have 'https://www.googleapis.com/auth/cloud-platform' scope.")
                return ""
                
        except Exception as e:
            print(f"Vision API Request Error: {e}")
            return ""
    
    def _get_project_id(self):
        """Extract project ID from OAuth2 credentials"""
        try:
            # Try to get project from token info
            if hasattr(self.credentials, 'client_id'):
                # Extract project ID from client ID format
                client_id = self.credentials.client_id
                if '-' in client_id:
                    # Client ID format is usually: PROJECT_NUMBER-RANDOM_STRING.apps.googleusercontent.com
                    # But we need the project ID, let's try to get it from the credentials file
                    return self._extract_project_from_credentials_file()
            return None
        except Exception:
            return None
    
    def _extract_project_from_credentials_file(self):
        """Extract project ID from the credentials file"""
        try:
            credentials_path = get_google_credentials_path()
            with open(credentials_path, 'r') as f:
                cred_data = json.load(f)
                return cred_data.get('installed', {}).get('project_id')
        except Exception:
            return None
    
    def extract_text_from_image(self, image_path: str) -> str:
        """Extract text from image using Google Vision API REST API"""
        try:
            with open(image_path, 'rb') as image_file:
                content = image_file.read()
            
            return self._make_vision_request(content)
            
        except Exception as e:
            print(f"OCR Error: {e}")
            return ""
    
    def extract_text_from_pdf_pages(self, pdf_path: str) -> str:
        """Convert PDF pages to images and extract text via OCR"""
        try:
            import fitz  # PyMuPDF for PDF to image conversion
            doc = fitz.open(pdf_path)
            extracted_text = ""
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                pix = page.get_pixmap()
                img_data = pix.tobytes("png")
                
                # Use REST API to extract text
                page_text = self._make_vision_request(img_data)
                if page_text:
                    extracted_text += f"\n--- Page {page_num + 1} ---\n"
                    extracted_text += page_text + "\n"
            
            doc.close()
            return extracted_text
        except ImportError:
            print("PyMuPDF not installed. Install with: pip install PyMuPDF")
            return ""
        except Exception as e:
            print(f"PDF OCR Error: {e}")
            return ""

class LocalRAGFlow:
    def __init__(self, chat_model="gpt-4o-mini"):
        self.openai_client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.chat_model = chat_model
        
        # Initialize ChromaDB with local persistence
        self.chroma_client = chromadb.PersistentClient(
            path="./data/chroma_db",
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        self.collection = self.chroma_client.get_or_create_collection(
            name="contracts",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Initialize Google services
        self.auth_manager = GoogleAuthManager()
        self.vision_ocr = None
        
        # Initialize Contract Intelligence Engine
        self.contract_intelligence = ContractIntelligenceEngine(self.openai_client)
        
        # Try to load existing credentials
        if self.auth_manager.load_credentials():
            self.vision_ocr = GoogleVisionOCR(self.auth_manager.credentials)
    
    def is_google_authenticated(self) -> bool:
        """Check if Google services are available"""
        return self.auth_manager.is_authenticated() and self.vision_ocr is not None
    
    def clear_google_credentials(self):
        """Clear Google credentials to force re-authentication"""
        self.auth_manager.clear_credentials()
        self.vision_ocr = None
    
    def get_google_auth_url(self):
        """Get Google OAuth URL for authentication"""
        return self.auth_manager.authenticate()
    
    def complete_google_auth(self, auth_url):
        """Complete Google authentication"""
        try:
            self.auth_manager.complete_auth(auth_url)
            self.vision_ocr = GoogleVisionOCR(self.auth_manager.credentials)
            return True
        except Exception as e:
            print(f"Auth completion error: {e}")
            return False
    
    def extract_text_from_file(self, file_path: str, use_ocr: bool = False) -> str:
        """Extract text from PDF, DOCX, or image files with optional OCR"""
        file_path = Path(file_path)
        text = ""
        
        if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
            # Image files - use OCR
            if self.vision_ocr and use_ocr:
                text = self.vision_ocr.extract_text_from_image(str(file_path))
            else:
                raise ValueError("OCR not available. Please authenticate with Google first.")
        
        elif file_path.suffix.lower() == '.pdf':
            if use_ocr and self.vision_ocr:
                # Use OCR for scanned PDFs
                text = self.vision_ocr.extract_text_from_pdf_pages(str(file_path))
            else:
                # Standard PDF text extraction
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    for page in pdf_reader.pages:
                        page_text = page.extract_text()
                        if page_text.strip():  # If text found, use standard extraction
                            text += page_text + "\n"
                        elif self.vision_ocr:  # If no text but OCR available, fall back to OCR
                            text += self.vision_ocr.extract_text_from_pdf_pages(str(file_path))
                            break
        
        elif file_path.suffix.lower() == '.docx':
            doc = docx.Document(file_path)
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
        
        elif file_path.suffix.lower() == '.txt':
            with open(file_path, 'r', encoding='utf-8') as file:
                text = file.read()
        
        return text
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings from OpenAI"""
        response = self.openai_client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts
        )
        return [item.embedding for item in response.data]
    
    def add_document(self, file_path: str, document_name: str = None, use_ocr: bool = False, folder: str = "General"):
        """Add a document to the vector database with optional OCR and contract intelligence"""
        # Extract text
        text = self.extract_text_from_file(file_path, use_ocr=use_ocr)
        if not text.strip():
            raise ValueError("No text could be extracted from the file")
        
        # Create document ID
        doc_id = document_name or Path(file_path).stem
        
        # Analyze contract with intelligence engine
        print(f"🧠 Analyzing contract intelligence for {doc_id}...")
        contract_analysis = self.contract_intelligence.analyze_contract(text, doc_id)
        
        # Create chunks
        chunks = self.chunk_text(text)
        
        # Generate embeddings
        embeddings = self.get_embeddings(chunks)
        
        # Prepare enhanced metadata with contract intelligence
        base_metadata = {
            "document_name": doc_id,
            "file_path": str(file_path),
            "extraction_method": "ocr" if use_ocr else "standard",
            "folder": folder
        }
        
        # Create enhanced metadata for each chunk
        ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
        metadatas = []
        
        for i, chunk in enumerate(chunks):
            chunk_metadata = base_metadata.copy()
            chunk_metadata["chunk_index"] = i
            chunk_metadata["chunk_size"] = len(chunk)
            
            # Add contract intelligence to metadata
            enhanced_metadata = self.contract_intelligence.create_enhanced_metadata(
                contract_analysis, chunk_metadata
            )
            metadatas.append(enhanced_metadata)
        
        # Add to ChromaDB
        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas
        )
        
        # Create detailed return message
        intelligence_summary = (
            f"📊 Contract Analysis Results:\n"
            f"   • Type: {contract_analysis.contract_type} "
            f"(confidence: {contract_analysis.contract_type_confidence:.2f})\n"
            f"   • Language: {contract_analysis.language}\n"
            f"   • Parties found: {len(contract_analysis.parties)}\n"
            f"   • Key dates: {len(contract_analysis.key_dates)}\n"
        )
        
        if contract_analysis.parties:
            party_names = [p.name for p in contract_analysis.parties[:3] if p.name and p.name != "Unknown"]
            if party_names:
                intelligence_summary += f"   • Main parties: {', '.join(party_names)}\n"
        
        return (
            f"✅ Added {len(chunks)} chunks from {doc_id} using "
            f"{'OCR' if use_ocr else 'standard'} extraction\n\n{intelligence_summary}"
        )
    
    def query_documents(self, query: str, n_results: int = 5, target_documents: List[str] = None, target_folder: str = None) -> Dict[str, Any]:
        """Query the document collection with optional document filtering"""
        # Get query embedding
        query_embedding = self.get_embeddings([query])[0]
        
        # Build where clause for document filtering
        where_clause = None
        if target_folder and target_documents:
            # Both folder and specific documents specified
            if len(target_documents) == 1:
                where_clause = {"$and": [{"document_name": target_documents[0]}, {"folder": target_folder}]}
            else:
                where_clause = {"$and": [{"document_name": {"$in": target_documents}}, {"folder": target_folder}]}
        elif target_folder:
            # Only folder specified
            where_clause = {"folder": target_folder}
        elif target_documents:
            # Only specific documents specified
            if len(target_documents) == 1:
                where_clause = {"document_name": target_documents[0]}
            else:
                where_clause = {"document_name": {"$in": target_documents}}
        
        # Search ChromaDB with optional filtering
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
            where=where_clause
        )
        
        # Check if we got any results
        if not results["documents"][0]:
            # If no results with filtering, try without filtering to see if documents exist
            fallback_results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            if fallback_results["documents"][0]:
                print(f"⚠️ No results found with filtering. Available documents:")
                for meta in fallback_results["metadatas"][0][:3]:
                    print(f"   - {meta.get('document_name', 'Unknown')} in folder '{meta.get('folder', 'Unknown')}'")
            return {
                "answer": "I couldn't find any relevant content in the specified document/folder. Please check if the document exists and contains the information you're looking for.",
                "context_chunks": [],
                "source_info": [],
                "similarity_scores": [],
                "search_scope": search_scope,
                "target_documents": target_documents,
                "target_folder": target_folder
            }
        
        # Prepare context for LLM
        context_chunks = results["documents"][0]
        context = "\n\n".join(context_chunks)
        
        # Generate response with OpenAI
        # Add context about which documents were searched
        if target_folder and target_documents:
            search_scope = f"contract(s) in folder '{target_folder}': {', '.join(target_documents)}"
        elif target_folder:
            folder_docs = self.list_documents_by_folder().get(target_folder, [])
            search_scope = f"contracts in folder '{target_folder}' ({len(folder_docs)} documents)"
        elif target_documents:
            search_scope = f"contract(s): {', '.join(target_documents)}"
        else:
            search_scope = "all contracts"
        
        response = self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=[
                {
                    "role": "system",
                    "content": f"""You are an expert legal contract analysis assistant with extensive experience in contract law. 

Today's date is: \u202D{datetime.now().strftime('%Y-%m-%d')}\u202C (\u202D{datetime.now().strftime('%A, %B %d, %Y')}\u202C)

You are currently analyzing {search_scope}.

When analyzing contracts, provide:
- Comprehensive yet concise responses (2-4 paragraphs when appropriate)
- Professional but approachable tone
- Clear structure with key points highlighted
- Specific references to relevant contract sections when available
- Practical implications and potential risks when relevant
- If analyzing specific contracts, mention which contract(s) the information comes from
- When relevant, reference dates in relation to today's date (e.g., "expired 2 years ago", "expires in 6 months")
- For contract status analysis, consider today's date when determining if contracts are active, expired, or upcoming
- CRITICAL: For Hebrew text, ensure all numbers (dates, amounts, IDs) maintain their original LTR direction and are not reversed

If information isn't available in the provided context, clearly state this and suggest what additional information might be needed. Always prioritize accuracy over completeness."""
                },
                {
                    "role": "user",
                    "content": f"Context from documents:\n{context}\n\nQuestion: {query}"
                }
            ],
            temperature=0.1
        )
        
        return {
            "answer": response.choices[0].message.content,
            "context_chunks": context_chunks,
            "source_info": results["metadatas"][0],
            "similarity_scores": [1 - d for d in results["distances"][0]],  # Convert distance to similarity
            "search_scope": search_scope,
            "target_documents": target_documents,
            "target_folder": target_folder
        }
    
    def list_documents(self):
        """List all documents in the collection"""
        results = self.collection.get()
        documents = set()
        for metadata in results["metadatas"]:
            documents.add(metadata["document_name"])
        return list(documents)
    
    def list_documents_by_folder(self):
        """List all documents grouped by folder"""
        results = self.collection.get()
        folders = {}
        for metadata in results["metadatas"]:
            folder = metadata.get("folder", "General")
            if folder not in folders:
                folders[folder] = set()
            folders[folder].add(metadata["document_name"])
        
        # Convert sets to sorted lists
        for folder in folders:
            folders[folder] = sorted(list(folders[folder]))
        
        return folders
    
    def list_folders(self):
        """List all folders"""
        results = self.collection.get()
        folders = set()
        for metadata in results["metadatas"]:
            folder = metadata.get("folder", "General")
            folders.add(folder)
        return sorted(list(folders))
    
    def debug_document_storage(self, document_name: str = None):
        """Debug method to inspect document storage"""
        if document_name:
            results = self.collection.get(where={"document_name": document_name})
        else:
            results = self.collection.get()
        
        print(f"📊 Database contains {len(results['ids'])} chunks")
        for i, (doc_id, metadata) in enumerate(zip(results['ids'][:5], results['metadatas'][:5])):
            print(f"   {i+1}. ID: {doc_id}")
            print(f"      Document: {metadata.get('document_name', 'Unknown')}")
            print(f"      Folder: {metadata.get('folder', 'Unknown')}")
            print(f"      Content length: {len(results['documents'][i]) if i < len(results['documents']) else 'N/A'}")
            print("---")
        return results
    
    def migrate_existing_documents_to_folders(self):
        """Add folder field to existing documents that don't have it"""
        results = self.collection.get()
        updated_count = 0
        
        for i, (doc_id, metadata) in enumerate(zip(results['ids'], results['metadatas'])):
            if 'folder' not in metadata:
                # Update metadata to include folder
                metadata['folder'] = 'General'
                
                # Update the document in ChromaDB
                self.collection.update(
                    ids=[doc_id],
                    metadatas=[metadata]
                )
                updated_count += 1
        
        if updated_count > 0:
            print(f"✅ Migrated {updated_count} document chunks to include folder field")
        else:
            print("ℹ️ All documents already have folder field")
        
        return updated_count
    
    def delete_document(self, document_name: str):
        """Delete a document and all its chunks"""
        # Get all chunk IDs for this document
        results = self.collection.get(
            where={"document_name": document_name}
        )
        
        if results["ids"]:
            self.collection.delete(ids=results["ids"])
            return f"Deleted {len(results['ids'])} chunks from {document_name}"
        else:
            return f"No document found with name: {document_name}"
    
    def clear_all_documents(self):
        """Clear all documents from the database"""
        try:
            # Get all document IDs first
            results = self.collection.get()
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                return f"✅ Cleared {len(results['ids'])} document chunks from database"
            else:
                return "ℹ️ Database is already empty"
        except Exception as e:
            return f"❌ Error clearing database: {str(e)}"
    
    def reprocess_all_documents(self):
        """Re-process all documents with RTL number fixes"""
        try:
            # Get all unique documents
            results = self.collection.get()
            if not results['ids']:
                return "ℹ️ No documents to reprocess"
            
            # Extract unique file paths from metadata
            file_paths = set()
            for metadata in results['metadatas']:
                if metadata and 'file_path' in metadata:
                    file_paths.add(metadata['file_path'])
            
            if not file_paths:
                return "❌ No file paths found in metadata"
            
            # Clear existing data
            self.clear_all_documents()
            
            # Re-process each document
            processed_count = 0
            for file_path in file_paths:
                try:
                    if Path(file_path).exists():
                        doc_name = Path(file_path).stem
                        self.add_document(file_path, doc_name)
                        processed_count += 1
                        print(f"✅ Re-processed: {doc_name}")
                    else:
                        print(f"⚠️ File not found: {file_path}")
                except Exception as e:
                    print(f"❌ Error processing {file_path}: {str(e)}")
            
            return f"✅ Re-processed {processed_count} documents with RTL number fixes"
            
        except Exception as e:
            return f"❌ Error during reprocessing: {str(e)}"