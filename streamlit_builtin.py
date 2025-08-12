# streamlit_builtin.py - RAGFlow Built-in Capabilities UI
import streamlit as st
import os
from pathlib import Path
import re
import json
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from builtin_rag_app_simple import RAGFlowBuiltinEngine

# Configure Streamlit page
st.set_page_config(
    page_title="Contract Intelligence Platform - Built-in RAGFlow",
    page_icon="🚀",
    layout="wide"
)

# RTL Language Detection and Styling (same as your current app)
def detect_rtl(text):
    """Detect if text contains RTL characters (Hebrew, Arabic, etc.)"""
    rtl_chars = re.findall(r'[\u0590-\u05FF\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF\uFB50-\uFDFF\uFE70-\uFEFF]', text)
    return len(rtl_chars) > 0

def inject_rtl_css():
    """Inject CSS for RTL support and modern UI styling"""
    st.markdown("""
    <style>
    /* =========================
       MODERN UI OPTIMIZATIONS
       ========================= */
    
    /* Remove excessive padding and margins */
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
        max-width: 100% !important;
    }
    
    /* Optimize header spacing */
    .stApp > header {
        background: transparent !important;
        height: 0rem !important;
    }
    
    /* Reduce space between elements */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Modern button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1.2rem !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        height: auto !important;
        min-height: 2.5rem !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
        background: linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%) !important;
    }
    
    /* RTL Text Support */
    .rtl-text {
        direction: rtl !important;
        text-align: right !important;
        font-family: 'Arial', 'Tahoma', sans-serif !important;
        line-height: 1.6 !important;
        unicode-bidi: bidi-override !important;
    }
    
    .rtl-preserve-numbers {
        direction: rtl !important;
        text-align: right !important;
        unicode-bidi: embed !important;
    }
    
    /* Success messages */
    .success-box {
        background-color: #d4edda !important;
        border: 1px solid #c3e6cb !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        color: #155724 !important;
    }
    
    /* Warning messages */
    .warning-box {
        background-color: #fff3cd !important;
        border: 1px solid #ffeaa7 !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
        color: #856404 !important;
    }
    
    /* Built-in engine badge */
    .builtin-badge {
        background: linear-gradient(135deg, #28a745 0%, #20c997 100%) !important;
        color: white !important;
        padding: 0.3rem 0.8rem !important;
        border-radius: 20px !important;
        font-size: 0.8rem !important;
        font-weight: 600 !important;
        display: inline-block !important;
        margin-left: 0.5rem !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

def format_rtl_text(text, preserve_numbers=True):
    """Format text with RTL support while preserving number order"""
    if not text:
        return ""
    
    if detect_rtl(text):
        css_class = "rtl-preserve-numbers" if preserve_numbers else "rtl-text"
        return f'<div class="{css_class}">{text}</div>'
    return text

# Initialize the built-in RAGFlow engine
@st.cache_resource
def get_builtin_engine():
    """Initialize and cache the built-in RAGFlow engine"""
    return RAGFlowBuiltinEngine()

def main():
    inject_rtl_css()
    
    # Header with built-in badge
    st.markdown("""
    <div style="text-align: center; padding: 1rem 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 10px; margin-bottom: 2rem;">
        <h1 style="color: white; margin: 0; font-size: 2.2rem;">📄 Contract Intelligence Platform</h1>
        <p style="color: #e8f4f8; margin: 0.5rem 0 0 0; font-size: 1.1rem;">
            Built-in RAGFlow Engine 
            <span class="builtin-badge">🚀 No External APIs</span>
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize engine
    try:
        rag_engine = get_builtin_engine()
        st.success("✅ RAGFlow Built-in Engine initialized successfully!")
    except Exception as e:
        st.error(f"❌ Failed to initialize RAGFlow engine: {e}")
        st.stop()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## 🚀 Built-in RAGFlow")
        st.markdown("**Zero external API costs**")
        st.markdown("**Hebrew/RTL support included**")
        
        # Engine status
        stats = rag_engine.get_collection_info()
        st.markdown("### 📊 Collection Stats")
        st.info(f"**Documents**: {stats.get('document_count', 0)} chunks")
        st.info(f"**Collection**: {stats.get('collection_name', 'Unknown')}")
        st.info(f"**Database**: {stats.get('database_path', 'Unknown')}")
        
        # Advanced parsing toggle
        use_advanced_parsing = st.checkbox(
            "🧠 Advanced Parsing (Laws)",
            value=True,
            help="Use RAGFlow's specialized legal document parser"
        )
        
        # Clear collection button
        if st.button("🗑️ Clear Collection", help="Remove all documents"):
            if rag_engine.clear_collection():
                st.success("Collection cleared!")
                st.rerun()
            else:
                st.error("Failed to clear collection")
    
    # Main content area
    tab1, tab2, tab3 = st.tabs(["📁 Document Upload", "🔍 Search & Query", "📊 Analysis"])
    
    with tab1:
        st.markdown("### 📄 Upload Documents")
        st.markdown("**Supported formats**: PDF, DOCX, TXT, Images (JPG, PNG, etc.)")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Choose files",
            type=['pdf', 'docx', 'txt', 'jpg', 'jpeg', 'png', 'bmp', 'tiff'],
            accept_multiple_files=True,
            help="Upload contracts, legal documents, or images with text"
        )
        
        if uploaded_files:
            st.markdown(f"**{len(uploaded_files)} file(s) selected**")
            
            if st.button("🚀 Process Documents", type="primary"):
                progress_bar = st.progress(0)
                success_count = 0
                
                for i, uploaded_file in enumerate(uploaded_files):
                    # Save uploaded file temporarily
                    temp_path = Path("./data/temp") / uploaded_file.name
                    temp_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.read())
                    
                    # Process with built-in engine
                    with st.spinner(f"Processing {uploaded_file.name}..."):
                        if rag_engine.add_document(str(temp_path), use_advanced_parsing):
                            success_count += 1
                            st.success(f"✅ {uploaded_file.name}")
                        else:
                            st.error(f"❌ Failed: {uploaded_file.name}")
                    
                    # Clean up temp file
                    temp_path.unlink(missing_ok=True)
                    
                    # Update progress
                    progress_bar.progress((i + 1) / len(uploaded_files))
                
                st.markdown(f"""
                <div class="success-box">
                    <strong>Processing Complete!</strong><br>
                    Successfully processed: {success_count}/{len(uploaded_files)} files<br>
                    Engine: RAGFlow Built-in (No API costs)
                </div>
                """, unsafe_allow_html=True)
                
                if success_count > 0:
                    st.rerun()  # Refresh to update stats
    
    with tab2:
        st.markdown("### 🔍 Search & Query Documents")
        
        # Search interface
        query = st.text_input(
            "Enter your question or search term:",
            placeholder="e.g., What are the payment terms? מהם תנאי התשלום?",
            help="Supports Hebrew, Arabic, and other RTL languages"
        )
        
        col1, col2 = st.columns([3, 1])
        with col1:
            top_k = st.slider("Number of results", 1, 20, 5)
        with col2:
            search_button = st.button("🔍 Search", type="primary")
        
        if query and (search_button or query):
            with st.spinner("Searching with built-in capabilities..."):
                results = rag_engine.search_documents(query, top_k)
            
            if results:
                st.markdown(f"### 📋 Found {len(results)} results")
                
                for i, result in enumerate(results):
                    with st.expander(f"Result {i+1} - Similarity: {result['similarity']:.3f}", expanded=i<3):
                        content = result['content']
                        metadata = result['metadata']
                        
                        # Display content with RTL support
                        formatted_content = format_rtl_text(content, preserve_numbers=True)
                        st.markdown(formatted_content, unsafe_allow_html=True)
                        
                        # Metadata
                        st.markdown("**Document Info:**")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.text(f"Source: {Path(metadata['source']).name}")
                        with col2:
                            st.text(f"Chunk: {metadata['chunk_index']}")
                        with col3:
                            st.text(f"Method: {metadata.get('parsing_method', 'unknown')}")
            else:
                st.warning("No results found. Try different search terms or upload more documents.")
    
    with tab3:
        st.markdown("### 📊 Document Analysis")
        
        if stats.get('total_chunks', 0) == 0:
            st.info("Upload documents first to see analysis options.")
        else:
            st.markdown("#### 🎯 Quick Analysis")
            
            # Predefined analysis queries
            analysis_options = [
                "Summarize the main contract terms",
                "What are the payment obligations?", 
                "List all parties mentioned in the documents",
                "What are the termination conditions?",
                "Identify key dates and deadlines",
                "What are the liability limitations?"
            ]
            
            selected_analysis = st.selectbox(
                "Choose analysis type:",
                analysis_options,
                help="Select a predefined analysis or enter custom query above"
            )
            
            if st.button("🧠 Analyze", type="primary"):
                with st.spinner("Analyzing with built-in RAGFlow..."):
                    results = rag_engine.search_documents(selected_analysis, top_k=10)
                
                if results:
                    st.markdown("#### 📋 Analysis Results")
                    
                    # Combine top results for comprehensive analysis
                    combined_content = "\n\n".join([r['content'] for r in results[:5]])
                    
                    # Display formatted results
                    formatted_analysis = format_rtl_text(combined_content, preserve_numbers=True)
                    st.markdown(formatted_analysis, unsafe_allow_html=True)
                    
                    # Show source breakdown
                    st.markdown("#### 📚 Sources")
                    sources = {}
                    for result in results:
                        source = Path(result['metadata']['source']).name
                        if source not in sources:
                            sources[source] = 0
                        sources[source] += 1
                    
                    for source, count in sources.items():
                        st.text(f"📄 {source}: {count} relevant sections")
                else:
                    st.warning("No relevant content found for this analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; font-size: 0.9rem; padding: 1rem;">
        🚀 <strong>RAGFlow Built-in Engine</strong> | 
        No external API costs | 
        Hebrew/RTL support included | 
        Local processing only
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
