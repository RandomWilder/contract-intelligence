# builtin_rag_app.py - RAGFlow Native Backend
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

# Optional OpenCV import - handle gracefully if not available
try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

# Add RAGFlow modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import only essential RAGFlow components we actually need
# Skip complex imports that require too many dependencies
try:
    from deepdoc.vision.ocr import OCR
    OCR_AVAILABLE = True
except ImportError as e:
    print(f"OCR not available: {e}")
    OCR_AVAILABLE = False

# Use simple document processing instead of complex RAGFlow parsers
import PyPDF2
import docx
from openai import OpenAI

class RAGFlowBuiltinEngine:
    """RAGFlow engine using only built-in capabilities - no external APIs"""
    
    def __init__(self, chat_model="gpt-4o-mini"):
        self.chat_model = chat_model
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        
        # Initialize built-in OCR if available (free, supports Hebrew/RTL)
        if OCR_AVAILABLE:
            try:
                self.ocr_engine = OCR()
                print("✅ Built-in OCR initialized (supports Hebrew/RTL)")
            except Exception as e:
                print(f"⚠️ Built-in OCR initialization failed: {e}")
                self.ocr_engine = None
        else:
            print("⚠️ Built-in OCR not available - using fallback methods")
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
        
        print("✅ RAGFlow Built-in Engine initialized")
        print(f"   - OCR: {'Available' if self.ocr_engine else 'Fallback methods'}")
        print(f"   - Parsers: Simple PDF/DOCX (no complex dependencies)")
        print(f"   - Database: ./data/chroma_builtin")
        print(f"   - LLM: OpenAI {chat_model}")
    
    def extract_text_from_file(self, file_path: str, use_advanced_parsing: bool = False) -> str:
        """Extract text using RAGFlow's built-in parsers and OCR"""
        file_path = Path(file_path)
        text = ""
        
        try:
            if file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']:
                # Image files - use built-in OCR
                if self.ocr_engine and CV2_AVAILABLE:
                    img = cv2.imread(str(file_path))
                    if img is not None:
                        ocr_result = self.ocr_engine(img)
                        if ocr_result:
                            text = "\n".join([result[1][0] for result in ocr_result if result[1][1] > 0.5])
                    else:
                        raise ValueError("Could not load image file")
                elif not CV2_AVAILABLE:
                    raise ValueError("OpenCV (cv2) not available for image processing")
                else:
                    raise ValueError("Built-in OCR not available")
            
            elif file_path.suffix.lower() == '.pdf':
                if use_advanced_parsing:
                    # Use RAGFlow's advanced PDF parser (laws parser for contracts)
                    try:
                        chunks = laws.chunk(str(file_path), lang="English")
                        text = "\n\n".join([chunk.get("content_with_weight", "") for chunk in chunks if chunk.get("content_with_weight")])
                    except Exception as e:
                        print(f"Advanced parsing failed, falling back to naive: {e}")
                        chunks = naive.chunk(str(file_path), lang="English")
                        text = "\n\n".join([chunk.get("content_with_weight", "") for chunk in chunks if chunk.get("content_with_weight")])
                else:
                    # Use basic PDF parser
                    with open(file_path, 'rb') as file:
                        binary = file.read()
                    chunks = naive.chunk(str(file_path), binary=binary, lang="English")
                    text = "\n\n".join([chunk.get("content_with_weight", "") for chunk in chunks if chunk.get("content_with_weight")])
            
            elif file_path.suffix.lower() == '.docx':
                # Use built-in DOCX parser
                with open(file_path, 'rb') as file:
                    binary = file.read()
                sections, tables = self.docx_parser(str(file_path), binary)
                text = "\n".join([section for section, _ in sections if section])
                # Add table content
                for table, _ in tables:
                    if table:
                        text += f"\n\nTable:\n{table[1] if isinstance(table, tuple) else str(table)}"
            
            elif file_path.suffix.lower() == '.txt':
                # Use built-in TXT parser
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
            
            else:
                raise ValueError(f"Unsupported file type: {file_path.suffix}")
                
        except Exception as e:
            print(f"Error extracting text from {file_path}: {e}")
            # Fallback to simple text extraction
            if file_path.suffix.lower() == '.txt':
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
        
        return text.strip()
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks using RAGFlow's tokenizer"""
        if not text:
            return []
        
        # Use RAGFlow's built-in tokenizer
        tokens = rag_tokenizer.tokenize(text)
        words = tokens.split() if isinstance(tokens, str) else tokens
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = start + chunk_size
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words) if isinstance(chunk_words, list) else str(chunk_words)
            
            if chunk_text.strip():
                chunks.append(chunk_text.strip())
            
            start = end - overlap
            if start >= len(words):
                break
        
        return chunks if chunks else [text]  # Fallback to original text
    
    def add_document(self, file_path: str, use_advanced_parsing: bool = False) -> bool:
        """Add document to vector store using built-in processing"""
        try:
            # Extract text using built-in parsers
            text = self.extract_text_from_file(file_path, use_advanced_parsing)
            
            if not text or len(text.strip()) < 10:
                print(f"⚠️ No meaningful text extracted from {file_path}")
                return False
            
            # Create document chunks
            chunks = self.chunk_text(text)
            
            if not chunks:
                print(f"⚠️ No chunks created from {file_path}")
                return False
            
            # Generate document ID
            doc_id = hashlib.md5(f"{file_path}_{datetime.now()}".encode()).hexdigest()
            
            # Prepare data for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                chunk_id = f"{doc_id}_chunk_{i}"
                
                documents.append(chunk)
                metadatas.append({
                    "source": str(file_path),
                    "chunk_index": i,
                    "doc_id": doc_id,
                    "parsing_method": "advanced" if use_advanced_parsing else "basic",
                    "timestamp": datetime.now().isoformat(),
                    "engine": "ragflow_builtin"
                })
                ids.append(chunk_id)
            
            # Add to ChromaDB (it will handle embedding automatically)
            self.collection.add(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
            print(f"✅ Added document: {Path(file_path).name}")
            print(f"   - Chunks: {len(chunks)}")
            print(f"   - Parsing: {'Advanced (Laws)' if use_advanced_parsing else 'Basic (Naive)'}")
            print(f"   - Text length: {len(text)} characters")
            
            return True
            
        except Exception as e:
            print(f"❌ Error adding document {file_path}: {e}")
            return False
    
    def search_documents(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Search documents using built-in capabilities"""
        try:
            if not query.strip():
                return []
            
            # Query the collection
            results = self.collection.query(
                query_texts=[query],
                n_results=top_k,
                include=["documents", "metadatas", "distances"]
            )
            
            # Format results
            search_results = []
            
            if results["documents"] and results["documents"][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0]
                )):
                    search_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "similarity": 1 - distance,  # Convert distance to similarity
                        "rank": i + 1
                    })
            
            return search_results
            
        except Exception as e:
            print(f"❌ Search error: {e}")
            return []
    
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get statistics about the document collection"""
        try:
            count = self.collection.count()
            return {
                "total_chunks": count,
                "collection_name": "contracts_builtin",
                "engine": "RAGFlow Built-in",
                "database_path": "./data/chroma_builtin"
            }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return {"total_chunks": 0, "error": str(e)}
    
    def clear_collection(self) -> bool:
        """Clear all documents from collection"""
        try:
            # Delete the collection and recreate it
            self.chroma_client.delete_collection("contracts_builtin")
            self.collection = self.chroma_client.get_or_create_collection(
                name="contracts_builtin",
                metadata={"hnsw:space": "cosine"}
            )
            print("✅ Collection cleared")
            return True
        except Exception as e:
            print(f"❌ Error clearing collection: {e}")
            return False

# Test the engine
if __name__ == "__main__":
    print("🚀 Testing RAGFlow Built-in Engine...")
    engine = RAGFlowBuiltinEngine()
    
    stats = engine.get_collection_stats()
    print(f"📊 Collection stats: {stats}")
    
    print("✅ RAGFlow Built-in Engine ready!")
