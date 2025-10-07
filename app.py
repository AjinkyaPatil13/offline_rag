import os
import tempfile
import threading
import time
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

# Suppress pkg_resources deprecation warning from ctranslate2
import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

import streamlit as st
from PIL import Image
import re
import speech_recognition as sr
import threading
from io import BytesIO
import queue
import time
from streamlit_js_eval import streamlit_js_eval
import json
import vosk
import sounddevice as sd
import numpy as np

from rag.models import ClipMultiModal
from rag.ingestion import ingest_pdf, ingest_docx, ingest_image, ingest_audio, ingest_path, Record
from rag.indexing import MultiModalIndex
from rag.retrieval import Retriever
from rag.ui_utils import build_context_with_citations, format_citation

# Optional: LangChain + Ollama for grounded generation
try:
    from langchain_ollama import ChatOllama
    from langchain.schema import HumanMessage, SystemMessage
    OLLAMA_AVAILABLE = True
except Exception:
    OLLAMA_AVAILABLE = False

BASE_DIR = Path(__file__).parent.resolve()
INDEX_DIR = BASE_DIR / "index"
INDEX_DIR.mkdir(exist_ok=True)

# Permanent uploads directory (not temp - temp gets cleaned up!)
UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)

# TTL for uploads cleanup (minutes)
UPLOAD_TTL_MINUTES = int(os.getenv("UPLOAD_TTL_MINUTES", "60"))

st.set_page_config(page_title="Multimodal RAG (Streamlit + LangChain + Ollama)", layout="wide")

# Custom CSS for better styling
st.markdown("""
<style>
    /* AGGRESSIVE Streamlit override */
    #MainMenu {visibility: hidden !important;}
    footer {visibility: hidden !important;}
    header {visibility: hidden !important;}
    .stDeployButton {visibility: hidden !important;}
    
    /* Target specific Streamlit containers */
    .stApp {
        background-color: white;
        height: 100vh !important;
        overflow: hidden !important;
    }
    
    .main .block-container {
        padding: 0 !important;
        margin: 0 !important;
        max-width: 100% !important;
        height: 100vh !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    /* Center container - shifted down from header */
    .center-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        gap: 20px;
        width: 100%;
        max-width: 600px;
        padding: 20px;
        margin-top: 80px;
        transform: translateY(20px);
    }
    
    /* Query interface wrapper */
    .query-wrapper {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 15px;
        max-width: 600px;
        width: 100%;
        margin: 0 auto;
    }
    
    /* Plus button styling - circular and compact */
    .stButton > button[key="upload_btn"] {
        background-color: #4CAF50;
        color: white;
        border-radius: 50%;
        width: 45px;
        height: 45px;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
        font-size: 18px;
        cursor: pointer;
        transition: all 0.2s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        outline: none !important;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .stButton > button[key="upload_btn"]:hover {
        background-color: #45a049;
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }
    
    /* Remove default button container styling */
    .stButton:has(button[key="upload_btn"]) {
        margin: 0 !important;
        padding: 0 !important;
    }
    
    /* Style the text input to be more prominent */
    .stTextInput > div > div > input {
        padding: 12px 20px;
        border-radius: 25px;
        border: 2px solid #e0e0e0;
        font-size: 16px;
        background-color: #f8f9fa;
        transition: all 0.3s;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #4CAF50;
        background-color: white;
        box-shadow: 0 0 0 3px rgba(76, 175, 80, 0.1);
    }
    
    /* Hide text input label */
    .stTextInput > label {
        display: none;
    }
    
    /* Style file uploader */
    .stFileUploader > div {
        border-radius: 8px;
        border: 2px solid #e0e0e0;
        padding: 8px 12px;
        background-color: #f8f9fa;
    }
    
    .stFileUploader > div:hover {
        border-color: #4CAF50;
    }
    
    /* Search button styling */
    .search-button .stButton > button {
        background-color: #4CAF50;
        color: white;
        border-radius: 25px;
        padding: 12px 30px;
        font-size: 16px;
        font-weight: 500;
        border: none;
        transition: all 0.2s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .search-button .stButton > button:hover {
        background-color: #45a049;
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
        transform: translateY(-1px);
    }
    
    /* Upload modal styling */
    .uploadedFile {
        border: 1px dashed #ccc;
        border-radius: 8px;
        padding: 10px;
        margin: 5px 0;
    }
    
    /* Search arrow button styling */
    .stButton > button[key="search_arrow"] {
        background-color: #4CAF50;
        color: white;
        border-radius: 50%;
        width: 45px;
        height: 45px;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
        font-size: 20px;
        cursor: pointer;
        transition: all 0.2s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        outline: none !important;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .stButton > button[key="search_arrow"]:hover {
        background-color: #45a049;
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }
    
    /* Microphone button styling - Vosk offline version */
    .stButton > button[key="mic_btn"] {
        background-color: #ff4444;
        color: white;
        border-radius: 50%;
        width: 45px;
        height: 45px;
        border: none !important;
        margin: 0 !important;
        padding: 0 !important;
        font-size: 16px;
        cursor: pointer;
        transition: all 0.2s;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
        outline: none !important;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .stButton > button[key="mic_btn"]:hover {
        background-color: #cc3333;
        transform: scale(1.05);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15) !important;
    }
    
    /* Top title styling */
    .top-header {
        position: fixed;
        top: 10px;
        left: 0;
        right: 0;
        text-align: center;
        z-index: 1000;
        background: rgba(255, 255, 255, 0.95);
        padding: 8px;
        backdrop-filter: blur(5px);
    }
    
    /* Page structure */
    .page-container {
        display: flex;
        flex-direction: column;
        height: 100vh;
        max-width: 1000px;
        margin: 0 auto;
        background-color: white;
        box-shadow: 0 0 20px rgba(0,0,0,0.1);
    }
    
    
    /* Results section styling - simple approach */
    .results-section {
        width: 100%;
        max-width: 1000px;
        margin-top: 30px;
        padding: 20px;
        background-color: #f8f9fa;
        border-radius: 10px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
</style>

""", unsafe_allow_html=True)

# Note: Files are now stored permanently in uploads/ directory
# No cleanup needed since we want to keep uploaded files for retrieval

def clean_llm_response(response_text: str) -> str:
    """
    Clean LLM response by removing <think> tags and their content.
    This is useful for reasoning models like DeepSeek R1 that include thinking steps.
    """
    # Remove <think>...</think> blocks (including multiline)
    cleaned = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL)
    # Clean up extra whitespace
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned.strip())
    return cleaned

# Configuration settings - optimized for available system memory
# Use DeepSeek R1 1.5B - fits in 4.6GB RAM and great for reasoning
ollama_model = os.getenv("OLLAMA_MODEL", "deepseek-r1:1.5b")  # Reasoning model that fits in available memory
top_k = 5  # More results for better context
min_score_threshold = 0.1  # Keep low threshold

# Optimized for DeepSeek R1 1.5B within memory constraints
max_context_length = 2500  # Balanced context length for 1.5B model
max_answer_tokens = 600    # Good length responses within memory limits

# Initialize Vosk offline speech recognition
@st.cache_resource(show_spinner=False)
def get_vosk_model():
    """Load Vosk model for offline speech recognition"""
    model_path = Path("models/vosk-model-small-en-us-0.15")
    if not model_path.exists():
        st.error(f"❌ Vosk model not found at {model_path}. Please run setup_vosk.py first.")
        return None
    
    try:
        # Set Vosk log level to avoid verbose output
        vosk.SetLogLevel(-1)
        model = vosk.Model(str(model_path))
        return model
    except Exception as e:
        st.error(f"❌ Failed to load Vosk model: {e}")
        return None

def check_microphone_available():
    """Check if microphone is available"""
    try:
        # Test sounddevice microphone access
        devices = sd.query_devices()
        input_devices = [d for d in devices if d['max_input_channels'] > 0]
        return len(input_devices) > 0
    except Exception:
        return False

# JavaScript injection function removed - using session state approach instead

def record_audio_offline():
    """Record audio and convert to text using offline Vosk"""
    # Check microphone availability first
    if not check_microphone_available():
        st.error("🎤 Microphone not available. Please check your microphone connection and permissions.")
        return None
    
    # Load Vosk model
    model = get_vosk_model()
    if model is None:
        return None
    
    # Create placeholder for status updates
    status_placeholder = st.empty()
    
    try:
        # Vosk recognizer configuration
        sample_rate = 16000  # Vosk works best with 16kHz
        rec = vosk.KaldiRecognizer(model, sample_rate)
        rec.SetWords(True)  # Enable word-level timestamps
        
        # Recording parameters
        duration = 5  # seconds
        channels = 1  # mono
        
        status_placeholder.info(f"🎤 Listening for {duration} seconds... Speak clearly!")
        
        # Record audio using sounddevice
        audio_data = sd.rec(
            frames=int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            dtype=np.int16
        )
        sd.wait()  # Wait for recording to complete
        
        status_placeholder.info("🔄 Processing speech with Vosk... (Fast offline recognition)")
        
        # Convert audio to bytes and process with Vosk
        audio_bytes = audio_data.tobytes()
        
        # Process audio in chunks for better recognition
        chunk_size = 4000  # bytes
        results = []
        
        for i in range(0, len(audio_bytes), chunk_size):
            chunk = audio_bytes[i:i + chunk_size]
            if rec.AcceptWaveform(chunk):
                result = json.loads(rec.Result())
                if result.get('text', '').strip():
                    results.append(result['text'].strip())
        
        # Get final result
        final_result = json.loads(rec.FinalResult())
        if final_result.get('text', '').strip():
            results.append(final_result['text'].strip())
        
        # Combine all recognized text
        recognized_text = ' '.join(results).strip()
        
        if recognized_text:
            status_placeholder.success("✅ Speech recognized with Vosk (offline)!")
            time.sleep(1)  # Brief success message
            status_placeholder.empty()  # Clear status
            return recognized_text
        else:
            status_placeholder.warning("🤔 No speech detected. Please try speaking more clearly and closer to the microphone.")
            time.sleep(2)
            status_placeholder.empty()
            return None
            
    except Exception as e:
        status_placeholder.error(f"⚠️ Error with Vosk speech recognition: {e}")
        time.sleep(2)
        status_placeholder.empty()
        return None

@st.cache_resource(show_spinner=False)
def load_clip() -> ClipMultiModal:
    return ClipMultiModal(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")

@st.cache_resource(show_spinner=False)
def get_index(persistent: bool) -> MultiModalIndex:
    if persistent:
        return MultiModalIndex(persist_dir=INDEX_DIR, persistent=True)
    else:
        return MultiModalIndex(persistent=False)

@st.cache_resource(show_spinner=False)
def get_quality_llm():
    """Get a cached, quality-optimized LLM instance for code explanations"""
    if OLLAMA_AVAILABLE:
        return ChatOllama(
            model=ollama_model, 
            temperature=0.3,  # Higher temperature for more detailed explanations
            num_predict=max_answer_tokens,
            top_p=0.95,
            repeat_penalty=1.1,
            num_ctx=8192  # Larger context window for code analysis
        )
    return None

clip = load_clip()

# Index mode configuration
ephemeral_mode = False  # Use persistent index to maintain uploaded data

index = get_index(persistent=not ephemeral_mode)
retriever = Retriever(index=index, clip=clip)

# Check current index status on startup
try:
    current_index_size = index.count()
    if current_index_size > 0:
        print(f"[INFO] Loaded persistent index with {current_index_size} existing documents")
except Exception as e:
    print(f"[WARNING] Could not check index status: {e}")

# Initialize session state for upload modal and query
if 'show_upload_modal' not in st.session_state:
    st.session_state.show_upload_modal = False
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {'pdfs': [], 'docxs': [], 'imgs': [], 'auds': []}
if 'query_text' not in st.session_state:
    st.session_state.query_text = ""


# ----------------------- Main Interface -----------------------

# Title at the top
st.markdown('<div class="top-header"><h1 style="margin: 0; color: #333; font-size: 1.8rem; font-weight: 600;">🤖 Multimodal RAG</h1><p style="margin: 2px 0 0 0; color: #666; font-size: 0.9rem;">Search with Text, Voice & Images</p></div>', unsafe_allow_html=True)

# ----------------------- Centered Query Interface -----------------------

# Create main centered container
st.markdown('<div class="center-container">', unsafe_allow_html=True)

# Query interface wrapper
st.markdown('<div class="query-wrapper">', unsafe_allow_html=True)

# Initialize session states
if 'voice_text' not in st.session_state:
    st.session_state.voice_text = ""
if 'speech_result' not in st.session_state:
    st.session_state.speech_result = ""
if 'update_input' not in st.session_state:
    st.session_state.update_input = False

# Create columns for plus button, query input, search arrow, and microphone
col_plus, col_query, col_search, col_mic = st.columns([1, 7, 1, 1])

with col_plus:
    if st.button("➕", key="upload_btn", help="Add files"):
        st.session_state.show_upload_modal = not st.session_state.show_upload_modal

# Check if we need to update input with speech result BEFORE creating widgets
if st.session_state.update_input and st.session_state.speech_result:
    # Set the input value directly in session state for the widget
    st.session_state.query_bar = st.session_state.speech_result
    st.session_state.update_input = False  # Reset flag

with col_query:
    user_query = st.text_input(
        "Search Query", 
        placeholder="Ask questions, search for 'email screenshot', 'code diagram'...", 
        key="query_bar", 
        label_visibility="collapsed",
        help="💡 Cross-modal search: Use text to find images! Try 'email screenshot', 'interface', 'diagram', etc."
    )
    
    # Keep session state in sync with text input
    st.session_state.query_text = user_query
    st.session_state.voice_text = user_query

with col_search:
    # Search arrow button
    run_query = st.button("➤", key="search_arrow", help="Search", type="primary")

with col_mic:
    # Offline speech recognition button using Vosk
    if st.button("🎤", key="mic_btn", help="Click to speak your query (Vosk offline - 5 seconds)"):
        # Record and transcribe audio
        transcribed_text = record_audio_offline()
        
        # If speech was recognized, update session state and trigger rerun
        if transcribed_text:
            st.success(f"✅ Speech recognized: '{transcribed_text}'")
            
            # Set session state for input update
            st.session_state.speech_result = transcribed_text
            st.session_state.update_input = True
            st.rerun()
        else:
            st.warning("🤔 No speech detected. Please try again.")

st.markdown('</div>', unsafe_allow_html=True)  # Close query-wrapper

# Close the center container
st.markdown('</div>', unsafe_allow_html=True)  # Close center-container

# Compact Image Upload Section
with st.expander("🖼️ Upload Image to Search", expanded=False):
    query_image = st.file_uploader(
        "Choose an image file", 
        type=["png", "jpg", "jpeg", "bmp", "gif", "webp"], 
        accept_multiple_files=False,
        help="Upload an image to search for similar visual content"
    )
    
    if query_image is not None:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.image(query_image, width=150)
        with col2:
            st.success("✅ Image ready for search!")
            st.caption(f"**File:** {query_image.name}")
            st.caption(f"**Size:** {len(query_image.getbuffer())/1024:.1f} KB")
            st.info("🔍 Click the Search button below to search with this image")

# Upload Modal/Expander
if st.session_state.show_upload_modal:
    with st.expander("📁 Add Files to Knowledge Base", expanded=True):
        upload_tabs = st.tabs(["📄 Documents", "🖼️ Images", "🎵 Audio", "⚙️ Actions"])
        
        with upload_tabs[0]:  # Documents tab
            col1, col2 = st.columns(2)
            with col1:
                pdfs = st.file_uploader("📄 PDF Files", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")
            with col2:
                docxs = st.file_uploader("📝 Word Documents", type=["docx"], accept_multiple_files=True, key="docx_uploader")
        
        with upload_tabs[1]:  # Images tab
            imgs = st.file_uploader("🖼️ Images", type=["png", "jpg", "jpeg", "bmp", "gif", "webp"], accept_multiple_files=True, key="img_uploader")
        
        with upload_tabs[2]:  # Audio tab
            st.info("⚠️ Audio processing is **very slow** (5-10 minutes per file) as it uses AI transcription.")
            
            # Audio processing options
            audio_mode = st.radio(
                "Audio Processing Mode:",
                ["Fast (Tiny Whisper model - less accurate)", "Skip audio processing"],
                index=0,
                help="Fast mode uses a smaller AI model for quicker transcription but lower accuracy."
            )
            
            if audio_mode != "Skip audio processing":
                auds = st.file_uploader("🎵 Audio Files", type=["mp3", "wav", "m4a", "flac", "ogg"], accept_multiple_files=True, key="aud_uploader")
            else:
                auds = []
                st.info("🚫 Audio processing is disabled.")
        
        with upload_tabs[3]:  # Actions tab
            # Show current index status
            try:
                current_count = index.count()
                if current_count > 0:
                    st.success(f"📁 Knowledge Base Status: {current_count} documents indexed")
                else:
                    st.info("📁 Knowledge Base Status: Empty - upload documents to get started")
            except Exception as e:
                st.error(f"⚠️ Could not check index status: {e}")
            
            st.markdown("---")
            
            col_action1, col_action2 = st.columns(2)
            with col_action1:
                ingest_btn = st.button("🔄 Process & Index Files", type="primary")
            with col_action2:
                clear_btn = st.button("🗑️ Clear All Data", type="secondary")
                
        # Process files immediately when button is clicked
        if ingest_btn:
            new_records: List[Record] = []
            with st.spinner("Processing and indexing files..."):
                # Save uploads to data dir and ingest
                import hashlib

                def sha256_file(p: Path) -> str:
                    h = hashlib.sha256()
                    with open(p, "rb") as f:
                        for chunk in iter(lambda: f.read(8192), b""):
                            h.update(chunk)
                    return h.hexdigest()

                def save_and_path(upload, subdir: str) -> Path:
                    target_dir = UPLOADS_DIR / subdir
                    target_dir.mkdir(parents=True, exist_ok=True)
                    # Write bytes and ensure unique name by appending short hash
                    data = upload.getbuffer()
                    import hashlib as _hashlib
                    h = _hashlib.sha256(data).hexdigest()[:8]
                    name = upload.name
                    base = Path(name).stem
                    suf = Path(name).suffix
                    p = target_dir / f"{base}_{h}{suf}"
                    with open(p, "wb") as f:
                        f.write(data)
                    return p

                # PDFs
                if pdfs:
                    st.info(f"Processing {len(pdfs)} PDF files...")
                    for up in pdfs:
                        try:
                            p = save_and_path(up, "pdf")
                            sid = f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}:{sha256_file(p)[:12]}"
                            records = ingest_pdf(p, clip, source_id=sid)
                            new_records.extend(records)
                            st.success(f"✅ Processed {up.name} - {len(records)} chunks created")
                        except Exception as e:
                            st.error(f"❌ Failed to process {up.name}: {e}")
                
                # DOCX
                if docxs:
                    st.info(f"Processing {len(docxs)} DOCX files...")
                    for up in docxs:
                        try:
                            p = save_and_path(up, "docx")
                            sid = f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}:{sha256_file(p)[:12]}"
                            records = ingest_docx(p, clip, source_id=sid)
                            new_records.extend(records)
                            st.success(f"✅ Processed {up.name} - {len(records)} chunks created")
                        except Exception as e:
                            st.error(f"❌ Failed to process {up.name}: {e}")
                
                # Images
                if imgs:
                    st.info(f"Processing {len(imgs)} image files...")
                    for up in imgs:
                        try:
                            p = save_and_path(up, "images")
                            sid = f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}:{sha256_file(p)[:12]}"
                            records = ingest_image(p, clip, source_id=sid)
                            new_records.extend(records)
                            st.success(f"✅ Processed {up.name} - {len(records)} items created")
                        except Exception as e:
                            st.error(f"❌ Failed to process {up.name}: {e}")
                
                # Audio
                if auds:
                    st.info(f"Processing {len(auds)} audio files...")
                    st.warning("⏱️ Audio processing uses AI transcription and may take several minutes for long files.")
                    
                    for i, up in enumerate(auds):
                        try:
                            # Show file size warning
                            file_size_mb = len(up.getbuffer()) / (1024 * 1024)
                            if file_size_mb > 10:
                                st.warning(f"⚠️ Large file ({file_size_mb:.1f}MB): {up.name} - this may take 5-10 minutes...")
                            
                            p = save_and_path(up, "audio")
                            sid = f"{p.name}:{p.stat().st_size}:{int(p.stat().st_mtime)}:{sha256_file(p)[:12]}"
                            
                            # Create progress tracker
                            progress_placeholder = st.empty()
                            def progress_update(msg):
                                progress_placeholder.info(f"🎵 [{i+1}/{len(auds)}] {up.name}: {msg}")
                            
                            progress_update("Starting transcription...")
                            records = ingest_audio(p, clip, source_id=sid, progress_callback=progress_update)
                            progress_placeholder.empty()
                            
                            new_records.extend(records)
                            st.success(f"✅ Processed {up.name} - {len(records)} chunks created ({file_size_mb:.1f}MB)")
                        except Exception as e:
                            st.error(f"❌ Failed to process {up.name}: {e}")

                # Add to index
                if new_records:
                    st.info(f"Adding {len(new_records)} chunks to knowledge base...")
                    try:
                        index.add_records(new_records)
                        total_count = index.count()
                        st.success(f"🎉 Successfully indexed {len(new_records)} chunks from your documents!")
                        st.info(f"📊 Knowledge base now contains {total_count} total items")
                    except Exception as e:
                        st.error(f"❌ Failed to add to knowledge base: {e}")
                else:
                    st.warning("⚠️ No files were uploaded or processed. Please upload files first.")
                    
        # Handle clear button action within the modal
        if clear_btn:
            # Clear vector index and uploaded files
            try:
                with st.spinner("Clearing knowledge base..."):
                    index.reset()
                    
                    # Also clear the uploads directory to remove stored files
                    import shutil
                    if UPLOADS_DIR.exists():
                        shutil.rmtree(UPLOADS_DIR, ignore_errors=True)
                        UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
                        
                st.success("✅ Successfully cleared knowledge base and uploaded files!")
                st.info("🔄 Refresh the page to see the updated count.")
            except Exception as e:
                st.error(f"❌ Could not reset knowledge base: {e}")
else:
    # When modal is closed, set default values
    pdfs = []
    docxs = []
    imgs = []
    auds = []
    ingest_btn = False
    clear_btn = False


# Close the center container
st.markdown('</div>', unsafe_allow_html=True)

# Handle search button click and show results immediately
if run_query:
    st.markdown('<div class="results-section">', unsafe_allow_html=True)
    
    # Use the current query from session state (which includes speech input)
    current_query = st.session_state.query_text
    
    # Check if index has documents
    try:
        index_size = index.count()
        if index_size == 0:
            st.warning("⚠️ Knowledge base is empty. Upload and index documents first for better results.")
        else:
            st.info(f"📁 Searching {index_size} indexed items...")
    except Exception as e:
        st.warning(f"Could not check index status: {e}")
    
    results: List[Dict[str, Any]] = []
    
    # Fast search with immediate feedback
    search_placeholder = st.empty()
    search_placeholder.info("🔍 Searching knowledge base...")
    
    if query_image is not None:
        qdir = UPLOADS_DIR / "queries"
        qdir.mkdir(parents=True, exist_ok=True)
        data = query_image.getbuffer()
        import hashlib as _hashlib
        h = _hashlib.sha256(data).hexdigest()[:8]
        name = query_image.name
        base = Path(name).stem
        suf = Path(name).suffix
        img_path = qdir / f"{base}_{h}{suf}"
        with open(img_path, "wb") as f:
            f.write(data)
        hits = retriever.search_image(img_path, k=top_k)
    else:
        if not current_query.strip():
            st.error("Provide a text query or an image.")
            st.stop()
        
        # Check if this is a text-to-image search query
        image_search_terms = ['screenshot', 'image', 'picture', 'photo', 'diagram', 'chart', 'graph', 'interface', 'ui', 'screen']
        is_image_search = any(term in current_query.lower() for term in image_search_terms)
        
        if is_image_search:
            st.info(f"🖼️ Image search: Looking for images related to '{current_query}'")
            # Use regular search but apply smart boosting to relevant images
            hits = retriever.search_text(current_query, k=top_k * 3)
            
            # Apply smart boosting based on keyword matching
            boosted_hits = []
            for hit in hits:
                if (hit.metadata.get('modality') == 'image' or 
                    hit.metadata.get('type') in ['image', 'image_text']):
                    
                    # Get OCR text for relevance checking
                    ocr_text = hit.metadata.get('ocr_text', hit.metadata.get('text', '')).lower()
                    
                    # Smart relevance boost based on keyword matching
                    boost_factor = 1.0
                    
                    # Check for specific query keywords in OCR text
                    query_words = current_query.lower().split()
                    matching_words = sum(1 for word in query_words if word in ocr_text)
                    
                    if matching_words > 0:
                        # Boost based on how many query words match
                        boost_factor = 1.2 + (matching_words * 0.3)  # More conservative boost
                        
                        # Extra boost for exact phrase matches
                        if current_query.lower() in ocr_text:
                            boost_factor *= 1.5
                    
                    from rag.retrieval import RetrievalResult
                    boosted_score = min(0.99, hit.score * boost_factor)
                    boosted_hit = RetrievalResult(
                        score=boosted_score,
                        modality=hit.modality,
                        metadata=hit.metadata
                    )
                    boosted_hits.append(boosted_hit)
                else:
                    boosted_hits.append(hit)
            
            # Sort by score and take top results
            boosted_hits.sort(key=lambda x: x.score, reverse=True)
            hits = boosted_hits[:top_k]
        else:
            hits = retriever.search_text(current_query, k=top_k)
        
        # Boost image results for code-related queries
        code_query_terms = ['code', 'explain', 'algorithm', 'function', 'binary search', 'programming', 'implementation', 'method', 'logic']
        is_code_query = any(term in current_query.lower() for term in code_query_terms)
        
        if is_code_query:
            # Apply score boost to image results that contain OCR text
            boosted_hits = []
            for hit in hits:
                if (hit.metadata.get('modality') == 'image' or 
                    hit.metadata.get('type') == 'image_text'):
                    # Check if this image contains meaningful OCR text
                    ocr_text = hit.metadata.get('ocr_text', hit.metadata.get('text', ''))
                    if ocr_text.strip() and len(ocr_text.strip()) > 20:  # Substantial OCR content
                        # Boost score significantly for images with code content
                        original_score = hit.score
                        boosted_score = min(0.95, original_score * 1.5)  # Cap at 0.95 to maintain ordering
                        # Create new hit with boosted score
                        from rag.retrieval import RetrievalResult
                        boosted_hit = RetrievalResult(
                            score=boosted_score,
                            modality=hit.modality, 
                            metadata=hit.metadata
                        )
                        boosted_hits.append(boosted_hit)
                        continue
                boosted_hits.append(hit)
            hits = boosted_hits
    
    search_placeholder.success(f"✅ Found {len(hits)} results")
    
    # Prepare results efficiently with enhanced scoring
    results = [{"score": h.score, "metadata": h.metadata} for h in hits]
    
    # Apply additional priority weighting for specific content types
    code_query_terms = ['code', 'explain', 'algorithm', 'function', 'binary search', 'programming', 'implementation']
    is_code_query = any(term in current_query.lower() for term in code_query_terms)
    
    if is_code_query:
        for result in results:
            # Give extra weight to image_text results (OCR from images)
            if result["metadata"].get("type") == "image_text":
                ocr_content = result["metadata"].get("text", "")
                # Check if OCR content contains code-like patterns
                if any(pattern in ocr_content.lower() for pattern in ['scanf', 'printf', 'int ', 'if(', 'while(', '==', '!=']):
                    # Boost score for code images
                    result["score"] = min(0.99, result["score"] * 1.8)  # Significant boost
    
    # Sort by score (highest first) and filter by minimum threshold
    results = [r for r in results if r["score"] >= min_score_threshold]
    results.sort(key=lambda x: x["score"], reverse=True)
    
    # Debug: Show result types and scores
    if results:
        result_types = {}
        st.markdown("**🔍 Debug: Top retrieval results**")
        
        for i, r in enumerate(results[:5]):  # Show more results for debugging
            modality = r["metadata"].get("modality", r["metadata"].get("type", "unknown"))
            result_types[modality] = result_types.get(modality, 0) + 1
            source_name = r["metadata"].get('source_name', 'Unknown')
            score = r["score"]
            
            # Show what type of content this is - expand for debugging
            all_metadata_keys = list(r["metadata"].keys())
            content_preview = ""
            if modality == "image" or r["metadata"].get("type") in ["image", "image_text"]:
                ocr_text = r["metadata"].get("ocr_text", r["metadata"].get("text", ""))
                if ocr_text.strip():
                    content_preview = f" - OCR: {ocr_text[:100]}..."  # Show more OCR text
                else:
                    content_preview = " - No OCR text"
            elif modality == "text":
                text = r["metadata"].get("text", "")[:50]
                if text.strip():
                    content_preview = f" - Text: {text}..."
            
            # Debug: Show all metadata for first 3 results
            if i < 3:
                st.caption(f"**{i+1}.** {source_name} (Score: {score:.3f})")
                st.caption(f"   Modality: {modality}, Type: {r['metadata'].get('type', 'None')}")
                st.caption(f"   Keys: {', '.join(all_metadata_keys)}")
                st.caption(f"   {content_preview}")
            else:
                st.caption(f"**{i+1}.** {source_name} (Score: {score:.3f}, Type: {modality}){content_preview}")
        
        debug_info = ", ".join([f"{k}: {v}" for k, v in result_types.items()])
        st.info(f"🔍 Result types found: {debug_info}")
    else:
        st.error("🔍 DEBUG: No results returned from search!")

    # Check if this is a pure image search query (user just wants to see images)
    pure_image_search_terms = ['screenshot', 'image', 'picture', 'photo']
    is_pure_image_search = any(term in current_query.lower() for term in pure_image_search_terms) and len(current_query.split()) <= 3
    
    # Answer the user's query using retrieved content
    if results:
        # For pure image searches, skip AI generation and just show images
        if is_pure_image_search:
            st.success(f"🎯 Found images for '{current_query}' - see results below!")
        
        # Build context efficiently - process only top results
        context_parts = []
        char_count = 0
        
        for r in results[:5]:  # Process more results for better context
            if char_count > max_context_length:
                break
                
            meta = r["metadata"]
            modality = meta.get("modality", meta.get("type", "text"))
            
            if modality == "text":
                text = meta.get("text", "")[:800]  # More space for text chunks
                if text.strip():
                    context_parts.append(text.strip())
                    char_count += len(text)
            elif modality == "audio":
                text = meta.get("text", "")[:400]  # More space for audio
                if text.strip():
                    context_parts.append(f"[Audio]: {text.strip()}")
                    char_count += len(text) + 10
            elif modality == "image":
                # For images, include OCR text or note their presence
                ocr_text = meta.get("ocr_text", meta.get("text", ""))
                if ocr_text.strip():
                    # Include full OCR text for code analysis
                    img_text = f"[Image Content]: {ocr_text.strip()[:1000]}"  # Much more space for code
                    context_parts.append(img_text)
                    char_count += len(img_text)
                else:
                    img_info = f"[Image: {meta.get('source_name', 'unknown')}]"
                    context_parts.append(img_info)
                    char_count += len(img_info)
        
        # Combine context efficiently
        combined_context = "\n".join(context_parts)  # Use single newline for compactness
        if len(combined_context) > max_context_length:
            combined_context = combined_context[:max_context_length] + "..."
        
        # Generate answer using LLM with speed optimizations (skip for pure image searches)
        if OLLAMA_AVAILABLE and combined_context.strip() and not is_pure_image_search:
            try:
                # Use quality LLM for better explanations
                llm = get_quality_llm()
                if llm is None:
                    raise Exception("LLM not available")
                
                # Detect if this is about code and use appropriate prompt
                is_code_query = any(word in current_query.lower() for word in 
                                    ['code', 'explain', 'algorithm', 'function', 'binary search', 'programming', 'implementation'])
                
                if is_code_query and '[Image Content]:' in combined_context:
                    # Enhanced prompt optimized for DeepSeek-Coder
                    answer_prompt = (
                        "You are an expert programming instructor specializing in algorithm analysis. "
                        "When explaining code, provide a comprehensive analysis that includes:\n"
                        "1. **Algorithm Overview**: What does this code accomplish?\n"
                        "2. **Step-by-Step Walkthrough**: Trace through the logic line by line\n"
                        "3. **Key Programming Concepts**: Identify important patterns and techniques\n"
                        "4. **Complexity Analysis**: Time and space complexity if applicable\n"
                        "5. **Educational Insights**: Why this approach works and potential improvements\n\n"
                        "Focus on clarity and educational value. Use examples when helpful."
                    )
                    user_message = f"Analyze and explain this code in detail:\n\n{combined_context}\n\nSpecific question: {current_query}"
                else:
                    # General prompt optimized for DeepSeek-Coder's capabilities
                    answer_prompt = (
                        "You are a knowledgeable technical assistant with expertise in programming and algorithms. "
                        "Answer the user's question based on the provided context. Be detailed, accurate, and educational. "
                        "If the content involves technical concepts, explain them clearly with practical insights. "
                        "Structure your response logically and provide actionable information when relevant."
                    )
                    user_message = f"**Question:** {current_query}\n\n**Context:** {combined_context}\n\nProvide a comprehensive and detailed answer:"
                
                messages = [
                    SystemMessage(content=answer_prompt),
                    HumanMessage(content=user_message)
                ]
                
                with st.spinner("Generating answer..."):
                    resp = llm.invoke(messages)
                    answer_text = clean_llm_response(resp.content)
                    
                    st.subheader("🤖 Answer")
                    st.markdown(answer_text)
                    
            except Exception as e:
                st.error(f"Failed to generate answer: {e}")
                # Fallback to showing context
                st.subheader("📄 Retrieved Information")
                st.markdown(combined_context[:1500] + ("..." if len(combined_context) > 1500 else ""))
        else:
            # Fallback when Ollama is not available or no context or pure image search
            if is_pure_image_search:
                # For pure image searches, don't show text context - just focus on images
                pass
            elif not OLLAMA_AVAILABLE:
                st.info("💡 Install langchain-ollama and run Ollama to get AI-powered answers.")
                if combined_context.strip():
                    st.subheader("📌 Retrieved Information")
                    st.markdown(combined_context[:1500] + ("..." if len(combined_context) > 1500 else ""))
            elif combined_context.strip():
                st.subheader("📌 Retrieved Information")
                st.markdown(combined_context[:1500] + ("..." if len(combined_context) > 1500 else ""))
            else:
                st.warning("⚠️ No relevant content found in the search results.")
        
        # Show document sources and found images
        st.markdown("---")
        sources = set()
        found_images = []
        
        # Debug: Count how many images we're processing
        total_results = len(results)
        image_results = 0
        
        for r in results:
            source_name = r["metadata"].get('source_name', 'Unknown')
            sources.add(source_name)
            
            # Collect image results - check all possible image indicators
            is_image_result = (
                r["metadata"].get("modality") == "image" or 
                r["metadata"].get("type") == "image" or 
                r["metadata"].get("type") == "image_text" or
                "image_path" in r["metadata"] or
                "source_path" in r["metadata"] and any(ext in str(r["metadata"].get("source_path", "")).lower() for ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"])
            )
            
            if is_image_result:
                image_results += 1
                img_path = r["metadata"].get("image_path") or r["metadata"].get("source_path")
                st.info(f"🔍 DEBUG: Found image result - Path: {img_path}, Modality: {r['metadata'].get('modality')}, Type: {r['metadata'].get('type')}")
                
                if img_path and Path(img_path).exists():
                    found_images.append({
                        "path": img_path,
                        "name": source_name,
                        "score": r["score"],
                        "ocr_text": r["metadata"].get("ocr_text", r["metadata"].get("text", ""))
                    })
                else:
                    st.warning(f"⚠️ DEBUG: Image path not found or doesn't exist: {img_path}")
        
        # Debug summary
        st.info(f"🔍 DEBUG: Found {image_results} image results out of {total_results} total results. {len(found_images)} images have valid paths.")
        
        if sources:
            st.caption(f"📚 **Sources:** {', '.join(sorted(sources))}")
        
        # Display found images with enhanced information for cross-modal search
        if found_images:
            # Check if this was an image search
            image_search_terms = ['screenshot', 'image', 'picture', 'photo', 'diagram', 'chart', 'graph', 'interface', 'ui', 'screen']
            was_image_search = any(term in current_query.lower() for term in image_search_terms)
            
            if was_image_search:
                st.markdown(f"## 🎯 {len(found_images)} Images Found")
                if is_pure_image_search:
                    st.caption(f"Showing images for: **{current_query}**")
                else:
                    st.caption(f"Images related to: **{current_query}**")
            else:
                st.markdown("**🖼️ Found Images:**")
                
            # Show more images for image search queries
            max_images = 6 if was_image_search else 3
            cols = st.columns(min(3, len(found_images)))
            
            for idx, img_info in enumerate(found_images[:max_images]):
                with cols[idx % 3]:
                    try:
                        img = Image.open(img_info["path"])
                        # Larger images for image search results
                        img_width = 200 if was_image_search else 150
                        
                        st.image(img, 
                                caption=f"{img_info['name']} (Relevance: {img_info['score']:.3f})", 
                                width=img_width)
                        
                        # Show OCR text if available
                        ocr_text = img_info.get("ocr_text", "")
                        if ocr_text.strip():
                            # Show preview of OCR for image searches
                            if was_image_search and len(ocr_text) > 100:
                                st.caption(f"Preview: {ocr_text[:100]}...")
                            
                            with st.expander(f"📝 Text from {img_info['name']}"):
                                st.code(ocr_text, language="text")
                        else:
                            if was_image_search:
                                st.caption("No text detected in image")
                                
                    except Exception as e:
                        st.error(f"Cannot display {img_info['name']}: {e}")
    else:
        st.warning("⚠️ No relevant documents found for your query.")
        if OLLAMA_AVAILABLE:
            # Try to provide a fast general answer without context
            try:
                # Use quality LLM for fallback responses too
                llm = get_quality_llm()
                if llm is None:
                    raise Exception("LLM not available")
                messages = [
                    HumanMessage(content=f"Briefly answer: {current_query} (No documents found in knowledge base)")
                ]
                
                with st.spinner("Generating response..."):
                    resp = llm.invoke(messages)
                    answer_text = clean_llm_response(resp.content)
                    st.subheader("🤖 Response")
                    st.markdown(answer_text)
                    st.info("💡 Try uploading relevant documents to get more specific answers.")
            except Exception as e:
                st.error(f"Failed to generate response: {e}")
    
    # Close results section
    st.markdown('</div>', unsafe_allow_html=True)
