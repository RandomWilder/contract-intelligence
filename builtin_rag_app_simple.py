# builtin_rag_app_simple.py - Simplified RAGFlow Backend
import os
import sys
import chromadb
from pathlib import Path
from typing import List, Dict, Any, Optional
import hashlib
import json
from datetime import datetime
from chromadb.config import Settings
import numpy as np
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Optional OpenCV import - handle gracefully if not available
try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Simple document processing imports
import PyPDF2
import docx
from openai import OpenAI

# Use our simple OCR instead of complex RAGFlow dependencies
OCR_AVAILABLE = False
try:
    from simple_ocr import SimpleOCR
    OCR_AVAILABLE = True
    print("✅ Simple OCR available")
except ImportError as e:
    print(f"ℹ️ Simple OCR not available: {e}")

class RAGFlowBuiltinEngine:
    """Simplified RAGFlow engine - built-in OCR when available, simple parsing otherwise"""
    
    def __init__(self, chat_model="gpt-4o-mini"):
        self.chat_model = chat_model
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize simple OCR if available (supports Hebrew/RTL)
        if OCR_AVAILABLE:
            try:
                self.ocr_engine = SimpleOCR()
                if self.ocr_engine.is_available():
                    print("✅ Simple OCR initialized (supports Hebrew/RTL)")
                else:
                    print("⚠️ Simple OCR failed to initialize")
                    self.ocr_engine = None
            except Exception as e:
                print(f"⚠️ Simple OCR initialization failed: {e}")
                self.ocr_engine = None
        else:
            self.ocr_engine = None
        
        # Initialize ChromaDB with separate collection for built-in app
        self.chroma_client = chromadb.PersistentClient(
            path="./data/chroma_builtin",  # Separate DB to avoid conflicts
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create collection for built-in app
        self.collection = self.chroma_client.get_or_create_collection(
            name="contracts_builtin",
            metadata={"hnsw:space": "cosine"}
        )
        
        # Use simple OpenAI client (reliable and works)
        self.openai_client = OpenAI(api_key=self.openai_api_key)
        
        print("✅ Simplified RAGFlow Engine initialized")
        print(f"   - OCR: {'Available' if self.ocr_engine else 'Simple text extraction only'}")
        print(f"   - Parsers: Simple PDF/DOCX (PyPDF2, python-docx)")
        print(f"   - Database: ./data/chroma_builtin")
        print(f"   - LLM: OpenAI {chat_model}")
    
    def extract_text_from_file(self, file_path: str, use_advanced_parsing: bool = False) -> str:
        """Extract text using simple parsers and built-in OCR when available"""
        file_path = Path(file_path)
        text = ""
        
        try:
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                # Image files - use SimpleOCR
                if self.ocr_engine:
                    ocr_results = self.ocr_engine.extract_text_from_image(str(file_path))
                    if ocr_results:
                        text = "\\n".join([result[0] for result in ocr_results if result[1] > 0.5])
                        if not text.strip():
                            text = "No text detected in image"
                    else:
                        text = "No text detected in image"
                else:
                    text = "Error: OCR not available for image processing"
            
            elif file_path.suffix.lower() == '.pdf':
                # Follow local_rag_app.py flow: try PyPDF2 first, auto-fallback to OCR
                text = self._extract_text_from_pdf(file_path, use_advanced_parsing)
            
            elif file_path.suffix.lower() in ['.docx']:
                # Simple DOCX processing using python-docx
                try:
                    doc = docx.Document(file_path)
                    text_parts = []
                    for paragraph in doc.paragraphs:
                        if paragraph.text.strip():
                            text_parts.append(paragraph.text)
                    text = "\\n".join(text_parts)
                    if not text.strip():
                        text = "No text extracted from DOCX"
                except Exception as e:
                    text = f"Error processing DOCX: {str(e)}"
            
            elif file_path.suffix.lower() == '.txt':
                # Simple text file processing
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        text = file.read()
                except UnicodeDecodeError:
                    # Try different encodings
                    for encoding in ['latin1', 'cp1252', 'iso-8859-1']:
                        try:
                            with open(file_path, 'r', encoding=encoding) as file:
                                text = file.read()
                            break
                        except UnicodeDecodeError:
                            continue
                    else:
                        text = "Error: Could not decode text file"
                except Exception as e:
                    text = f"Error processing text file: {str(e)}"
            
            else:
                text = f"Unsupported file type: {file_path.suffix}"
                
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            text = f"Error processing file: {str(e)}"
        
        return text
    
    def _extract_text_from_pdf(self, file_path: Path, use_advanced_parsing: bool = False) -> str:
        """Extract text from PDF with automatic OCR fallback - following local_rag_app.py pattern"""
        try:
            # First try PyPDF2 for text-based PDFs
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text_parts = []
                has_meaningful_text = False
                
                for page_num, page in enumerate(pdf_reader.pages):
                    page_text = page.extract_text()
                    if page_text.strip() and len(page_text.strip()) > 20:  # Meaningful text threshold
                        text_parts.append(page_text)
                        has_meaningful_text = True
                        print(f"   ✅ PyPDF2 extracted {len(page_text)} chars from page {page_num + 1}")
                
                # If we found meaningful text, use it
                if has_meaningful_text:
                    text = "\\n".join(text_parts)
                    print(f"   📄 PyPDF2 extraction successful: {len(text)} total characters")
                    return text
                
                # Otherwise, fall back to OCR for scanned documents
                print(f"   📄 PyPDF2 found minimal text, falling back to OCR...")
                return self._extract_text_from_pdf_via_ocr(file_path)
                
        except Exception as e:
            print(f"   ⚠️ PyPDF2 failed: {e}, trying OCR...")
            return self._extract_text_from_pdf_via_ocr(file_path)
    
    def _extract_text_from_pdf_via_ocr(self, file_path: Path) -> str:
        """Extract text from PDF using SimpleOCR - following local_rag_app.py pattern"""
        if not self.ocr_engine:
            return "No text extracted from PDF (scanned document, OCR not available)"
        
        try:
            # Use SimpleOCR's built-in PDF processing
            text = self.ocr_engine.extract_text_from_pdf(str(file_path))
            return text
                
        except Exception as ocr_e:
            print(f"   ⚠️ OCR processing failed: {ocr_e}")
            return f"No text extracted from PDF (OCR failed: {str(ocr_e)})"
    
    def chunk_text(self, text: str, chunk_size: int = 800, chunk_overlap: int = 150) -> List[str]:
        """Improved text chunking optimized for contract documents"""
        if not text.strip():
            return []
        
        # First, try to split by obvious document sections (contracts often have numbered sections)
        section_patterns = [
            r'\n\s*\d+\.\s+',  # "1. ", "2. " etc.
            r'\n\s*[A-Z][A-Z\s]{10,}:\s*\n',  # "PAYMENT TERMS:" etc.
            r'\n\s*Article\s+\d+',  # "Article 1", "Article 2" etc.
            r'\n\s*Section\s+\d+',  # "Section 1", "Section 2" etc.
            r'\n\s*\([a-z]\)\s+',  # "(a) ", "(b) " etc.
        ]
        
        # Try section-based chunking first
        import re
        for pattern in section_patterns:
            sections = re.split(pattern, text)
            if len(sections) > 2:  # Found meaningful sections
                print(f"   📋 Found {len(sections)} document sections using pattern")
                # Process each section, but still apply size limits
                chunks = []
                for i, section in enumerate(sections):
                    if section.strip():
                        # If section is too large, sub-chunk it
                        if len(section) > chunk_size * 1.5:
                            sub_chunks = self._chunk_by_size(section, chunk_size, chunk_overlap)
                            chunks.extend(sub_chunks)
                        else:
                            chunks.append(section.strip())
                
                if chunks:
                    print(f"   ✅ Section-based chunking produced {len(chunks)} chunks")
                    return [chunk for chunk in chunks if chunk.strip() and len(chunk) > 20]
        
        # Fallback to size-based chunking with better boundaries
        print(f"   📄 Using size-based chunking (no clear sections found)")
        return self._chunk_by_size(text, chunk_size, chunk_overlap)
    
    def _chunk_by_size(self, text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
        """Size-based chunking with smart boundary detection"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Try to end at a good boundary
            if end < len(text):
                # Look for boundaries in order of preference
                boundaries = [
                    (chunk.rfind('\\n\\n'), 2),  # Double newline (paragraph break)
                    (chunk.rfind('.\\n'), 2),    # Sentence end with newline  
                    (chunk.rfind('. '), 2),      # Sentence end with space
                    (chunk.rfind('\\n'), 1),     # Single newline
                    (chunk.rfind('; '), 2),      # Semicolon (common in contracts)
                    (chunk.rfind(', '), 2),      # Comma
                ]
                
                best_boundary = -1
                best_offset = 0
                
                for boundary_pos, offset in boundaries:
                    # Only use boundary if it's not too early in the chunk
                    if boundary_pos > start + chunk_size // 3:
                        best_boundary = boundary_pos
                        best_offset = offset
                        break
                
                if best_boundary > -1:
                    chunk = text[start:start + best_boundary - start + best_offset]
                    end = start + best_boundary - start + best_offset
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            # Move start position with overlap
            start = end - chunk_overlap
            
            if start >= len(text):
                break
        
        return [chunk for chunk in chunks if chunk.strip() and len(chunk) > 20]
    
    def get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings using OpenAI"""
        try:
            response = self.openai_client.embeddings.create(
                input=texts,
                model="text-embedding-3-small"
            )
            return [embedding.embedding for embedding in response.data]
        except Exception as e:
            print(f"Error getting embeddings: {e}")
            return []
    
    def add_document(self, file_path: str, use_advanced_parsing: bool = False) -> bool:
        """Add document to the vector database"""
        try:
            # Convert to Path object for consistent handling
            file_path = Path(file_path)
            
            # Extract text
            text = self.extract_text_from_file(str(file_path), use_advanced_parsing)
            
            if not text or text.startswith("Error"):
                print(f"Failed to extract text from {file_path}: {text}")
                return False
            
            # Chunk text
            chunks = self.chunk_text(text)
            
            if not chunks:
                print(f"No chunks generated from {file_path}")
                return False
            
            # Get embeddings
            embeddings = self.get_embeddings(chunks)
            
            if not embeddings:
                print(f"Failed to get embeddings for {file_path}")
                return False
            
            # Generate document ID
            doc_id = hashlib.md5(str(file_path).encode()).hexdigest()
            
            # Prepare data for ChromaDB
            ids = [f"{doc_id}_{i}" for i in range(len(chunks))]
            metadatas = [
                {
                    "source": str(file_path),
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "timestamp": datetime.now().isoformat(),
                    "file_type": file_path.suffix.lower(),
                    "processing_method": "ocr" if "OCR processing complete" in text else "text_extraction"
                }
                for i in range(len(chunks))
            ]
            
            # Add to ChromaDB
            self.collection.add(
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Added {len(chunks)} chunks from {file_path}")
            return True
            
        except Exception as e:
            print(f"Error adding document {file_path}: {e}")
            return False
    
    def clear_collection(self) -> bool:
        """Clear all documents from the collection"""
        try:
            # Delete the collection and recreate it
            self.chroma_client.delete_collection(name="contracts_builtin")
            self.collection = self.chroma_client.create_collection(
                name="contracts_builtin",
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ Collection cleared successfully")
            return True
        except Exception as e:
            print(f"Error clearing collection: {e}")
            return False
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            count = self.collection.count()
            return {
                "document_count": count,
                "collection_name": "contracts_builtin", 
                "database_path": "./data/chroma_builtin"
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {
                "document_count": 0,
                "collection_name": "contracts_builtin",
                "database_path": "./data/chroma_builtin"
            }
    
    def search_documents(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        """Search documents using vector similarity"""
        try:
            # Get query embedding
            query_embedding = self.get_embeddings([query])[0]
            
            # Search ChromaDB
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            search_results = []
            for i in range(len(results["documents"][0])):
                search_results.append({
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "similarity": 1 - results["distances"][0][i]  # Convert distance to similarity
                })
            
            return search_results
            
        except Exception as e:
            print(f"Error searching documents: {e}")
            return []
    
    def generate_response(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Generate response using OpenAI with retrieved context"""
        try:
            # Prepare context
            context = "\\n\\n".join([doc["content"] for doc in context_docs])
            
            # Create prompt
            prompt = f"""Based on the following contract documents, please answer the question.
            
Context from contracts:
{context}

Question: {query}

Please provide a comprehensive answer based on the contract information provided. If the information is not sufficient to answer the question, please state that clearly."""
            
            # Generate response
            response = self.openai_client.chat.completions.create(
                model=self.chat_model,
                messages=[
                    {"role": "system", "content": "You are a helpful contract analysis assistant. Provide accurate, detailed responses based on the contract documents provided."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=1000
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error generating response: {e}")
            return f"Error generating response: {str(e)}"
    
    def get_collection_info(self) -> Dict[str, Any]:
        """Get information about the current collection"""
        try:
            count = self.collection.count()
            return {
                "document_count": count,
                "collection_name": "contracts_builtin",
                "database_path": "./data/chroma_builtin"
            }
        except Exception as e:
            print(f"Error getting collection info: {e}")
            return {"error": str(e)}
    
    def clear_database(self) -> bool:
        """Clear all documents from the database"""
        try:
            self.chroma_client.delete_collection("contracts_builtin")
            self.collection = self.chroma_client.get_or_create_collection(
                name="contracts_builtin",
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ Database cleared successfully")
            return True
        except Exception as e:
            print(f"Error clearing database: {e}")
            return False

if __name__ == "__main__":
    # Test the engine
    try:
        engine = RAGFlowBuiltinEngine()
        print("✅ Engine test successful")
    except Exception as e:
        print(f"❌ Engine test failed: {e}")
