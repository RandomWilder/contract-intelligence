#!/usr/bin/env python3

import os
import chromadb
from chromadb.config import Settings

def check_chromadb_status():
    """Check current ChromaDB status"""
    print("🔍 Checking ChromaDB status...")
    
    # Explicitly disable ONNX models
    os.environ['ALLOW_RESET'] = 'TRUE'
    os.environ['ANONYMIZED_TELEMETRY'] = 'FALSE'
    
    # Initialize ChromaDB client with explicit settings to prevent ONNX model loading
    client = chromadb.PersistentClient(
        path="./data/chroma_db",
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True,
            chroma_server_nofile=True
        )
    )
    
    # Get collections
    collections = client.list_collections()
    print(f"📊 Collections found: {len(collections)}")
    
    for collection in collections:
        print(f"\n📁 Collection: {collection.name}")
        
        # Get all items in collection
        results = collection.get()
        total_chunks = len(results['ids']) if results['ids'] else 0
        print(f"   📄 Total chunks: {total_chunks}")
        
        if total_chunks > 0:
            # Show unique documents
            documents = set()
            for metadata in results['metadatas']:
                if metadata and 'document_name' in metadata:
                    documents.add(metadata['document_name'])
            
            print(f"   📋 Unique documents: {len(documents)}")
            for doc in sorted(documents):
                doc_chunks = sum(1 for m in results['metadatas'] if m and m.get('document_name') == doc)
                print(f"      • {doc}: {doc_chunks} chunks")
            
            # Show sample chunk content (first few characters to check for RTL issues)
            if results['documents']:
                sample_text = results['documents'][0][:200] + "..."
                print(f"   📝 Sample chunk text: {sample_text}")
                
            # Show sample metadata
            if results['metadatas'] and results['metadatas'][0]:
                print(f"   📋 Sample metadata keys: {list(results['metadatas'][0].keys())}")
        
        print()

if __name__ == "__main__":
    check_chromadb_status()
