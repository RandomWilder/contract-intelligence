# streamlit_app.py
import streamlit as st
import os
from local_rag_app import LocalRAGFlow
from pathlib import Path
import webbrowser
from urllib.parse import urlparse, parse_qs
import re
import time
from telemetry_client import get_telemetry_client

# Initialize telemetry (respects user consent)
telemetry = get_telemetry_client()
telemetry.track_app_start()

# Configure Streamlit page
st.set_page_config(
    page_title="Contract Intelligence Platform",
    page_icon="📄",
    layout="wide"
)

# Available OpenAI Chat Models
OPENAI_CHAT_MODELS = [
    "gpt-4o-mini",
    "gpt-5",
    "gpt-4o", 
    "gpt-4",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
    "gpt-3.5-turbo-16k-0613"
]

# RTL Language Detection and Styling
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
    
    /* Compact sidebar */
    .css-1d391kg {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
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
    
    /* Primary button special styling */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%) !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #45a049 0%, #3d8b40 100%) !important;
    }
    
    /* Compact input fields */
    .stTextInput > div > div > input {
        padding: 0.6rem 0.8rem !important;
        border-radius: 6px !important;
        border: 1px solid #e0e0e0 !important;
        font-size: 0.9rem !important;
        height: 2.5rem !important;
    }
    
    /* Compact selectbox */
    .stSelectbox > div > div > div {
        padding: 0.6rem 0.8rem !important;
        border-radius: 6px !important;
        min-height: 2.5rem !important;
    }
    
    /* File uploader optimization */
    .stFileUploader > div {
        padding: 1rem !important;
        border: 2px dashed #ccc !important;
        border-radius: 8px !important;
        background: #fafafa !important;
    }
    
    .stFileUploader > div:hover {
        border-color: #667eea !important;
        background: #f8f9ff !important;
    }
    
    /* Compact expanders */
    .streamlit-expanderHeader {
        padding: 0.5rem 0.8rem !important;
        border-radius: 6px !important;
        background: #f8f9fa !important;
        font-size: 0.9rem !important;
    }
    
    .streamlit-expanderContent {
        padding: 0.8rem !important;
        border: 1px solid #e9ecef !important;
        border-radius: 0 0 6px 6px !important;
    }
    
    /* Status indicators styling */
    .stSuccess {
        background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%) !important;
        border: 1px solid #c3e6cb !important;
        border-radius: 6px !important;
        padding: 0.6rem 1rem !important;
        margin: 0.3rem 0 !important;
    }
    
    .stError {
        background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%) !important;
        border: 1px solid #f5c6cb !important;
        border-radius: 6px !important;
        padding: 0.6rem 1rem !important;
        margin: 0.3rem 0 !important;
    }
    
    .stWarning {
        background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%) !important;
        border: 1px solid #ffeaa7 !important;
        border-radius: 6px !important;
        padding: 0.6rem 1rem !important;
        margin: 0.3rem 0 !important;
    }
    
    .stInfo {
        background: linear-gradient(135deg, #d1ecf1 0%, #bee5eb 100%) !important;
        border: 1px solid #bee5eb !important;
        border-radius: 6px !important;
        padding: 0.6rem 1rem !important;
        margin: 0.3rem 0 !important;
    }
    
    /* Compact columns */
    .css-1kyxreq {
        gap: 0.5rem !important;
    }
    
    /* Modern card styling for sections */
    .stContainer {
        background: white !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
        padding: 1rem !important;
        margin: 0.5rem 0 !important;
    }
    
    /* Optimize text area */
    .stTextArea > div > div > textarea {
        padding: 0.6rem !important;
        border-radius: 6px !important;
        border: 1px solid #e0e0e0 !important;
        font-size: 0.85rem !important;
        line-height: 1.4 !important;
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        font-size: 0.9rem !important;
        padding-left: 0.3rem !important;
    }
    
    /* Sidebar optimization */
    .css-1d391kg {
        background: #f8f9fa !important;
        border-right: 1px solid #e9ecef !important;
    }
    
    /* Header styling */
    .stApp h1 {
        color: #2c3e50 !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
        font-size: 2rem !important;
    }
    
    .stApp h2 {
        color: #34495e !important;
        font-weight: 500 !important;
        margin-bottom: 0.5rem !important;
        font-size: 1.3rem !important;
    }
    
    .stApp h3 {
        color: #34495e !important;
        font-weight: 500 !important;
        margin-bottom: 0.3rem !important;
        font-size: 1.1rem !important;
    }
    
    /* Caption styling */
    .caption {
        font-size: 0.8rem !important;
        color: #6c757d !important;
        text-align: center !important;
        padding: 0.2rem !important;
    }
    
    /* Divider styling */
    hr {
        margin: 0.8rem 0 !important;
        border: none !important;
        height: 1px !important;
        background: linear-gradient(90deg, transparent, #e9ecef, transparent) !important;
    }
    
    /* =========================
       RTL LANGUAGE SUPPORT
       ========================= */
    
    /* RTL Input Field Styling */
    .rtl-input input {
        direction: rtl !important;
        text-align: right !important;
        unicode-bidi: bidi-override !important;
    }
    
    /* RTL Text Display */
    .rtl-text {
        direction: rtl !important;
        text-align: right !important;
        unicode-bidi: bidi-override !important;
        font-family: 'Segoe UI', Tahoma, Arial, sans-serif !important;
        line-height: 1.6 !important;
        background: #f8f9ff !important;
        padding: 0.8rem !important;
        border-radius: 6px !important;
        border: 1px solid #e9ecef !important;
    }
    
    /* RTL Markdown Content */
    .rtl-markdown {
        direction: rtl !important;
        text-align: right !important;
        unicode-bidi: bidi-override !important;
    }
    
    .rtl-markdown p, .rtl-markdown li, .rtl-markdown div {
        direction: rtl !important;
        text-align: right !important;
        unicode-bidi: bidi-override !important;
    }
    
    /* RTL Text Area */
    .rtl-textarea textarea {
        direction: rtl !important;
        text-align: right !important;
        unicode-bidi: bidi-override !important;
    }
    
    /* Mixed content support */
    .mixed-content {
        unicode-bidi: plaintext !important;
        text-align: start !important;
    }
    
    /* =========================
       RESPONSIVE DESIGN
       ========================= */
    
    @media (max-width: 768px) {
        .main .block-container {
            padding-left: 1rem !important;
            padding-right: 1rem !important;
        }
        
        .stButton > button {
            width: 100% !important;
            margin: 0.2rem 0 !important;
        }
    }
    
    /* =========================
       LOADING ANIMATIONS
       ========================= */
    
    .stSpinner {
        color: #667eea !important;
    }
    
    /* Custom loading animation */
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    .loading {
        animation: pulse 1.5s ease-in-out infinite !important;
    }
    
    /* =========================
       CHAT INTERFACE STYLING
       ========================= */
    
    /* Chat container */
    .chat-container {
        height: 500px !important;
        overflow-y: auto !important;
        padding: 1rem !important;
        border: 1px solid #e9ecef !important;
        border-radius: 8px !important;
        background: #f8f9fa !important;
        margin-bottom: 1rem !important;
    }
    
    /* Chat message bubbles */
    .stChatMessage {
        margin-bottom: 1rem !important;
    }
    
    /* User message styling */
    .stChatMessage[data-testid="user-message"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border-radius: 18px 18px 4px 18px !important;
        margin-left: 20% !important;
        margin-right: 0 !important;
    }
    
    /* Assistant message styling */
    .stChatMessage[data-testid="assistant-message"] {
        background: white !important;
        color: #2c3e50 !important;
        border: 1px solid #e9ecef !important;
        border-radius: 18px 18px 18px 4px !important;
        margin-left: 0 !important;
        margin-right: 20% !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }
    
    /* RTL chat message alignment */
    .rtl-chat-user {
        direction: rtl !important;
        text-align: right !important;
        margin-left: 0 !important;
        margin-right: 20% !important;
        border-radius: 18px 18px 18px 4px !important;
    }
    
    .rtl-chat-assistant {
        direction: rtl !important;
        text-align: right !important;
        margin-left: 20% !important;
        margin-right: 0 !important;
        border-radius: 18px 18px 4px 18px !important;
    }
    
    /* Chat input styling */
    .stChatInput > div {
        border-radius: 25px !important;
        border: 2px solid #667eea !important;
        background: white !important;
    }
    
    .stChatInput input {
        padding: 0.8rem 1.2rem !important;
        border-radius: 25px !important;
        font-size: 0.9rem !important;
    }
    
    /* Chat input RTL support */
    .rtl-chat-input input {
        direction: rtl !important;
        text-align: right !important;
        unicode-bidi: bidi-override !important;
    }
    
    /* Scroll to bottom animation */
    .chat-scroll-target {
        scroll-behavior: smooth !important;
    }
    
    </style>
    """, unsafe_allow_html=True)

def display_rtl_text(text, key=None):
    """Display text with proper RTL support"""
    if detect_rtl(text):
        st.markdown(f'<div class="rtl-text">{text}</div>', unsafe_allow_html=True)
    else:
        st.write(text)

def initialize_chat_history():
    """Initialize chat history in session state"""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_context" not in st.session_state:
        st.session_state.chat_context = {}

def add_chat_message(role, content, metadata=None):
    """Add a message to chat history"""
    message = {
        "role": role,
        "content": content,
        "timestamp": time.time(),
        "metadata": metadata or {}
    }
    st.session_state.chat_messages.append(message)

def display_chat_message(message):
    """Display a chat message with RTL support"""
    role = message["role"]
    content = message["content"]
    
    # Determine if content is RTL
    is_rtl = detect_rtl(content)
    
    with st.chat_message(role):
        if is_rtl:
            # Apply RTL styling for RTL text
            css_class = "rtl-chat-user" if role == "user" else "rtl-chat-assistant"
            st.markdown(f'<div class="{css_class}">{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(content)
        
        # Display metadata if available (for assistant messages)
        if role == "assistant" and message.get("metadata", {}).get("show_sources"):
            metadata = message["metadata"]
            with st.expander("📚 Source Information & Context", expanded=False):
                if metadata.get("search_scope"):
                    st.markdown(f"**🎯 Search Scope**: {metadata['search_scope']}")
                
                if metadata.get("source_contracts"):
                    contracts = metadata["source_contracts"]
                    if len(contracts) > 1:
                        st.markdown(f"**📊 Results found in**: {', '.join(sorted(contracts))}")
                    elif len(contracts) == 1:
                        st.markdown(f"**📊 Results found in**: {list(contracts)[0]}")
                
                st.divider()
                
                # Display source chunks
                if metadata.get("source_chunks"):
                    for i, chunk_info in enumerate(metadata["source_chunks"]):
                        st.write(f"**Source {i+1}** (Similarity: {chunk_info['score']:.2f})")
                        st.write(f"📄 Document: {chunk_info['document_name']}")
                        st.write(f"📁 Folder: {chunk_info.get('folder', 'General')}")
                        st.write(f"🔧 Extraction: {chunk_info.get('extraction_method', 'standard').title()}")
                        st.write(f"📝 Content Preview:")
                        
                        chunk_preview = chunk_info['content'][:500] + ("..." if len(chunk_info['content']) > 500 else "")
                        
                        if detect_rtl(chunk_preview):
                            st.markdown(f'<div class="rtl-text" style="background-color: #f0f2f6; padding: 10px; border-radius: 5px; border: 1px solid #ddd; height: 100px; overflow-y: auto;">{chunk_preview}</div>', unsafe_allow_html=True)
                        else:
                            st.text_area(
                                f"Context {i+1}:",
                                chunk_preview,
                                height=100,
                                key=f"chat_context_{message['timestamp']}_{i}"
                            )
                        st.divider()

# Initialize RAGFlow
@st.cache_resource
def initialize_ragflow(chat_model="gpt-4o-mini"):
    return LocalRAGFlow(chat_model=chat_model)

def handle_google_auth(rag):
    """Handle Google OAuth authentication using device flow"""
    from local_rag_app import get_google_credentials_path
    credentials_path = get_google_credentials_path()
    
    if not os.path.exists(credentials_path):
        st.error(f"""
        **Missing Google Credentials File**
        
        Please download your Google Cloud credentials and save as `google_credentials.json` in your home directory:
        `{credentials_path}`
        
        Steps:
        1. Go to Google Cloud Console
        2. Navigate to APIs & Services > Credentials  
        3. Create OAuth 2.0 Client ID (Desktop Application)
        4. Download JSON file and save to the path above
        """)
        return False
    
    if rag.is_google_authenticated():
        st.success("✅ Google services connected successfully!")
        st.info("You can now use OCR for scanned documents and access Google Drive.")
        
        # Add option to clear credentials for debugging
        with st.expander("🔧 Troubleshooting"):
            st.warning("If you're experiencing authentication issues, you can clear stored credentials and re-authenticate.")
            if st.button("🗑️ Clear Credentials", help="This will force re-authentication with fresh credentials"):
                rag.clear_google_credentials()
                st.success("Credentials cleared. Please re-authenticate below.")
                st.rerun()
        
        return True
    else:
        st.warning("Google services not connected. Click below to authenticate.")
        
        # Show scope information
        with st.expander("📋 Required Permissions"):
            st.info("""
            This application requires the following Google API permissions:
            • **Cloud Platform** - For Google Vision OCR (text extraction from images/PDFs)
            • **Drive (Read-only)** - To access Google Drive documents (optional)
            
            Make sure to approve all permissions in the browser window.
            
            **Note**: Simplified scope configuration for better compatibility.
            """)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔗 Connect Google Services", type="primary"):
                try:
                    with st.spinner("Authenticating with Google..."):
                        success = rag.auth_manager.authenticate()
                        if success:
                            from local_rag_app import GoogleVisionOCR
                            rag.vision_ocr = GoogleVisionOCR(rag.auth_manager.credentials)
                            st.success("🎉 Google authentication successful!")
                            st.rerun()
                        else:
                            st.error("Authentication failed. Please try again.")
                except Exception as e:
                    st.error(f"Authentication failed: {e}")
                    if "scope" in str(e).lower():
                        st.error("**Scope Error**: The authentication scopes have changed. Please clear credentials and try again.")
                    st.info("💡 Make sure you allow ALL permissions in the browser window that opens.")
        
        with col2:
            if st.button("🗑️ Clear & Retry", help="Clear existing credentials and try fresh authentication"):
                rag.clear_google_credentials()
                st.info("Credentials cleared. Click 'Connect Google Services' to re-authenticate.")
                st.rerun()
        
        return False

def main():
    # Inject RTL CSS and modern styling first
    inject_rtl_css()
    
    # Initialize chat history
    initialize_chat_history()
    
    # Centered header title and subtitle - pushed to top
    st.markdown("""
    <div style="text-align: center; margin-top: -2rem; margin-bottom: 1.5rem; padding-top: 0;">
        <h1 style="color: #2c3e50; font-weight: 600; margin-bottom: 0.3rem; margin-top: 0; font-size: 2.5rem;">
            📄 Contract Intelligence Platform
        </h1>
        <p style="color: #6c757d; font-size: 1.5rem; margin-bottom: 0; margin-top: 0;">
            Analyze. Recommend. Act.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize RAGFlow to get status information
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ OpenAI API Key Missing - Please set OPENAI_API_KEY in .env file")
        return
    
    # Initialize with default model first
    selected_model = "gpt-4o-mini"
    
    rag = initialize_ragflow(selected_model)
    
    # Get documents list for use across columns
    documents = rag.list_documents()
    documents_by_folder = rag.list_documents_by_folder()
    
    # Consolidated status indicators row - 5 columns
    status_col1, status_col2, status_col3, status_col4, status_col5 = st.columns([1, 1, 1, 1, 1])
    
    with status_col1:
        # Google OCR status
        google_authenticated = rag.is_google_authenticated()
        if google_authenticated:
            st.success("✅ Google OCR")
        else:
            st.warning("⚠️ Google OCR")
    
    with status_col2:
        # OpenAI API connection status
        if os.getenv("OPENAI_API_KEY"):
            st.success("✅ OpenAI Connected")
        else:
            st.error("❌ OpenAI Missing")
    
    with status_col3:
        # Document count and folder statistics
        total_docs = len(documents)
        total_folders = len(documents_by_folder)
        if total_folders > 1:
            st.info(f"📊 {total_docs} Docs ({total_folders} folders)")
        else:
            st.info(f"📊 {total_docs} Documents")
    
    with status_col4:
        # Current AI model display
        st.info(f"🤖 {selected_model}")
    
    with status_col5:
        # Model selection dropdown
        selected_model = st.selectbox(
            "AI Model:",
            options=OPENAI_CHAT_MODELS,
            index=OPENAI_CHAT_MODELS.index(selected_model) if selected_model in OPENAI_CHAT_MODELS else 0,
            help="Choose the OpenAI model for chat completion",
            key="main_model_selector"
        )
    
    # Create main layout with optimized columns
    main_col1, main_col2 = st.columns([1, 4])
    
    # Left column: Document Management & Google Auth
    with main_col1:
        # Compact Google Authentication Section
        google_authenticated = handle_google_auth(rag)
        
        st.subheader("📁 Document Management")
        
        # Upload document
        uploaded_file = st.file_uploader(
            "Upload Contract Document",
            type=["pdf", "docx", "txt", "jpg", "jpeg", "png"],
            help="Supported formats: PDF, DOCX, TXT, Images (JPG, PNG)"
        )
        
        if uploaded_file:
            # Save uploaded file
            uploads_dir = Path("./data/uploads")
            uploads_dir.mkdir(exist_ok=True)
            
            file_path = uploads_dir / uploaded_file.name
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            
            # Folder selection
            existing_folders = rag.list_folders()
            folder_options = ["+ Create New Folder"] + existing_folders
            
            selected_folder_option = st.selectbox(
                "📁 Select Folder:",
                options=folder_options,
                index=1 if existing_folders else 0,
                help="Choose existing folder or create new one"
            )
            
            # Handle folder selection
            if selected_folder_option == "+ Create New Folder":
                folder_name = st.text_input(
                    "New Folder Name:",
                    placeholder="Enter folder name (e.g., 'Rental Agreements')",
                    help="Create a new folder for better organization"
                )
                if not folder_name.strip():
                    folder_name = "General"
            else:
                folder_name = selected_folder_option
            
            # OCR option for images and scanned PDFs
            use_ocr = False
            if uploaded_file.type.startswith('image/') or uploaded_file.name.lower().endswith('.pdf'):
                if google_authenticated:
                    use_ocr = st.checkbox(
                        "🔍 Use OCR", 
                        value=uploaded_file.type.startswith('image/'),
                        help="Extract text using Google Vision API OCR"
                    )
                else:
                    st.info("💡 Connect Google services to enable OCR")
            
            if st.button("📤 Process Document", type="primary"):
                with st.spinner("Processing document..."):
                    try:
                        result = rag.add_document(
                            str(file_path), 
                            uploaded_file.name,
                            use_ocr=use_ocr,
                            folder=folder_name
                        )
                        st.success(result)
                        
                        # Show extraction method used
                        if use_ocr:
                            st.info("📸 Used Google Vision OCR")
                        else:
                            st.info("📝 Used standard extraction")
                        
                        # Refresh the interface to update document list and clear upload area
                        st.rerun()
                            
                    except Exception as e:
                        st.error(f"Error: {e}")
                        if "OCR not available" in str(e):
                            st.info("💡 Connect Google services to enable OCR")
        
        # Document list grouped by folders
        st.subheader("📋 Documents")
        documents_by_folder = rag.list_documents_by_folder()
        
        if documents_by_folder:
            for folder, docs in documents_by_folder.items():
                with st.expander(f"📁 {folder} ({len(docs)} documents)", expanded=True):
                    for doc in docs:
                        doc_col1, doc_col2 = st.columns([4, 1])
                        with doc_col1:
                            st.caption(f"📄 {doc}")
                        with doc_col2:
                            if st.button("🗑️", key=f"del_{doc}", help=f"Delete {doc}"):
                                result = rag.delete_document(doc)
                                st.success(result)
                                st.rerun()
        else:
            st.info("No documents yet")
    
    # Right column: Chat Interface
    with main_col2:
        st.subheader("💬 Chat with Your Contracts")
        
        # Contract Selection Interface - Compact layout above chat
        if documents:
            # Create two columns for side-by-side selection
            selection_col1, selection_col2 = st.columns(2)
            
            with selection_col1:
                # Folder selection first
                folder_options = ["All Folders"] + list(documents_by_folder.keys())
                selected_folder = st.selectbox(
                    "📁 Select Folder:",
                    options=folder_options,
                    index=0,
                    help="Choose folder to search in, or search all folders"
                )
            
            with selection_col2:
                # Contract selection based on folder
                if selected_folder == "All Folders":
                    contract_options = ["All Contracts"] + documents
                    available_docs = documents
                else:
                    folder_docs = documents_by_folder[selected_folder]
                    contract_options = ["All in Folder"] + folder_docs
                    available_docs = folder_docs
                
                selected_contract = st.selectbox(
                    "🎯 Target Contract:",
                    options=contract_options,
                    index=0,
                    help="Select a specific contract or search all in selected scope"
                )
            
            # Show selection status
            if selected_folder == "All Folders":
                if selected_contract == "All Contracts":
                    st.caption(f"🔍 Search scope: All {len(documents)} contracts")
                else:
                    st.caption(f"🔍 Search scope: {selected_contract} only")
            else:
                if selected_contract == "All in Folder":
                    st.caption(f"🔍 Search scope: All {len(available_docs)} contracts in '{selected_folder}' folder")
                else:
                    st.caption(f"🔍 Search scope: {selected_contract} in '{selected_folder}' folder")
        else:
            selected_contract = "All Contracts"
            selected_folder = "All Folders"
            st.info("📤 Upload documents to start querying")
        
        # Chat controls
        chat_controls_col1, chat_controls_col2 = st.columns([3, 1])
        with chat_controls_col1:
            # Quick Questions (compact)
            with st.expander("📝 Quick Questions"):
                example_queries = [
                    "Key terms and conditions?",
                    "Who are the parties?",
                    "Contract duration?",
                    "Payment terms?",
                    "Termination clauses?",
                    "Penalty clauses?",
                    "IP terms?",
                    "Governing law?",
                    "Confidentiality?",
                    "Force majeure?"
                ]
                
                # Create buttons in rows of 2 with compact layout
                for i in range(0, len(example_queries), 2):
                    q_col1, q_col2 = st.columns(2)
                    
                    with q_col1:
                        if i < len(example_queries):
                            if st.button(example_queries[i], key=f"example_{i}"):
                                # Add user message to chat
                                add_chat_message("user", example_queries[i])
                                st.rerun()
                    
                    with q_col2:
                        if i + 1 < len(example_queries):
                            if st.button(example_queries[i + 1], key=f"example_{i+1}"):
                                # Add user message to chat
                                add_chat_message("user", example_queries[i + 1])
                                st.rerun()
        
        with chat_controls_col2:
            st.write("")  # Spacer
            if st.button("🗑️ Clear Chat", help="Clear chat history"):
                st.session_state.chat_messages = []
                st.rerun()
        
        # Chat History Container
        chat_container = st.container()
        with chat_container:
            if st.session_state.chat_messages:
                for message in st.session_state.chat_messages:
                    display_chat_message(message)
            else:
                # Welcome message
                with st.chat_message("assistant"):
                    st.markdown("👋 Hello! I'm your Contract Intelligence Assistant. Ask me anything about your uploaded contracts!")
        
        # Apply RTL styling to chat input if needed (detect from last user message)
        if st.session_state.chat_messages:
            last_user_messages = [msg for msg in st.session_state.chat_messages if msg["role"] == "user"]
            if last_user_messages and detect_rtl(last_user_messages[-1]["content"]):
                st.markdown("""
                <style>
                div[data-testid="stChatInput"] input {
                    direction: rtl !important;
                    text-align: right !important;
                    unicode-bidi: bidi-override !important;
                }
                </style>
                """, unsafe_allow_html=True)
        
        # Chat Input (sticky to bottom)
        if prompt := st.chat_input("Ask me about your contracts...", key="chat_input"):
            if not documents:
                st.warning("Please upload and process at least one document first.")
            else:
                # Add user message to chat history
                add_chat_message("user", prompt)
                
                # Process the query
                with st.spinner("Analyzing contracts..."):
                    try:
                        # Determine target documents and folder based on selection
                        target_folder = None if selected_folder == "All Folders" else selected_folder
                        
                        if selected_contract in ["All Contracts", "All in Folder"]:
                            target_documents = None
                        else:
                            target_documents = [selected_contract]
                        
                        # Query with contract and folder filtering
                        results = rag.query_documents(prompt, target_documents=target_documents, target_folder=target_folder)
                        
                        # Prepare metadata for assistant message
                        search_scope = ""
                        if target_folder and target_documents:
                            search_scope = f"{selected_contract} in folder '{target_folder}'"
                        elif target_folder:
                            folder_doc_count = len(documents_by_folder[target_folder])
                            search_scope = f"All {folder_doc_count} contracts in folder '{target_folder}'"
                        elif target_documents:
                            search_scope = f"{selected_contract} only"
                        else:
                            search_scope = f"All {len(documents)} contracts"
                        
                        # Prepare source information
                        source_contracts = set(metadata['document_name'] for metadata in results["source_info"])
                        source_chunks = []
                        
                        for i, (chunk, metadata, score) in enumerate(zip(
                            results["context_chunks"],
                            results["source_info"],
                            results["similarity_scores"]
                        )):
                            source_chunks.append({
                                'content': chunk,
                                'document_name': metadata['document_name'],
                                'folder': metadata.get('folder', 'General'),
                                'extraction_method': metadata.get('extraction_method', 'standard'),
                                'score': score
                            })
                        
                        # Add assistant response to chat history
                        assistant_metadata = {
                            "show_sources": True,
                            "search_scope": search_scope,
                            "source_contracts": source_contracts,
                            "source_chunks": source_chunks
                        }
                        
                        add_chat_message("assistant", results["answer"], assistant_metadata)
                        
                    except Exception as e:
                        add_chat_message("assistant", f"❌ Error analyzing contracts: {e}")
                
                # Rerun to display new messages
                st.rerun()
    
    # Compact footer
    st.divider()
    footer_col1, footer_col2, footer_col3 = st.columns(3)
    with footer_col1:
        st.caption(f"🤖 OpenAI {selected_model}")
    with footer_col2:
        st.caption("🔍 Google Vision OCR")
    with footer_col3:
        st.caption("💾 ChromaDB Storage")

if __name__ == "__main__":
    main()