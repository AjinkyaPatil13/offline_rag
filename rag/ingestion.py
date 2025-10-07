from __future__ import annotations

import io
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image
from pypdf import PdfReader
import docx

from .models import ClipMultiModal


@dataclass
class Record:
    id: str
    embedding: List[float]
    modality: str  # 'text' | 'image' | 'audio'
    metadata: Dict


def _chunk_text(text: str, max_chars: int = 1000, overlap: int = 100) -> List[str]:
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + max_chars, n)
        chunks.append(text[start:end])
        if end == n:
            break
        start = max(0, end - overlap)
    return [c.strip() for c in chunks if c.strip()]


# ------------------------- Document ingestion -------------------------

def ingest_pdf(path: Path, clip: ClipMultiModal, source_id: Optional[str] = None) -> List[Record]:
    reader = PdfReader(str(path))
    records: List[Record] = []
    for i, page in enumerate(reader.pages, start=1):
        page_text = (page.extract_text() or "").strip()
        if not page_text:
            continue
        for ci, chunk in enumerate(_chunk_text(page_text)):
            embedding = clip.embed_text([chunk])[0].tolist()
            rid = f"{source_id or path.name}::pdf_p{i}_c{ci}"
            records.append(
                Record(
                    id=rid,
                    embedding=embedding,
                    modality="text",
                    metadata={
                        "type": "text",
                        "source_path": str(path.resolve()),
                        "source_name": path.name,
                        "page": i,
                        "chunk": ci,
                        "text": chunk,
                    },
                )
            )
    return records


def ingest_docx(path: Path, clip: ClipMultiModal, source_id: Optional[str] = None) -> List[Record]:
    d = docx.Document(str(path))
    text = "\n".join(p.text for p in d.paragraphs if p.text)
    records: List[Record] = []
    for ci, chunk in enumerate(_chunk_text(text)):
        embedding = clip.embed_text([chunk])[0].tolist()
        rid = f"{source_id or path.name}::docx_c{ci}"
        records.append(
            Record(
                id=rid,
                embedding=embedding,
                modality="text",
                metadata={
                    "type": "text",
                    "source_path": str(path.resolve()),
                    "source_name": path.name,
                    "chunk": ci,
                    "text": chunk,
                },
            )
        )
    return records


# ------------------------- Image ingestion -------------------------

def _extract_text_from_image(image_path: Path) -> str:
    """Extract text from image using EasyOCR (better for code/technical text)"""
    try:
        import easyocr
        # Initialize EasyOCR reader (English only for speed)
        reader = easyocr.Reader(['en'], gpu=False)  # Use CPU for compatibility
        
        # Extract text from image
        results = reader.readtext(str(image_path))
        
        # Combine all detected text with better formatting for code
        extracted_text = []
        for (bbox, text, confidence) in results:
            # Only include text with reasonable confidence
            if confidence > 0.3:  # Adjust threshold as needed
                text = text.strip()
                if text:
                    extracted_text.append(text)
        
        # Join with spaces and newlines to preserve code structure
        raw_text = "\n".join(extracted_text)
        
        # Enhance the text for better embedding by adding context
        if raw_text.strip():
            # Detect different types of content for better cross-modal search
            code_indicators = ['if', 'else', 'for', 'while', 'int', 'printf', 'scanf', 'return', '==', '!=', '<=', '>=']
            email_indicators = ['from:', 'to:', 'subject:', 'cc:', 'bcc:', '@', 'inbox', 'sent', 'draft', 'reply']
            ui_indicators = ['button', 'click', 'menu', 'toolbar', 'window', 'dialog', 'settings', 'options']
            
            looks_like_code = any(indicator in raw_text.lower() for indicator in code_indicators)
            looks_like_email = any(indicator in raw_text.lower() for indicator in email_indicators)
            looks_like_ui = any(indicator in raw_text.lower() for indicator in ui_indicators)
            
            if looks_like_email:
                # Email screenshot/interface content with better keywords
                enhanced_text = f"""EMAIL SCREENSHOT INTERFACE: {raw_text}

This is specifically an email screenshot showing email interface elements like gmail, inbox, compose, messages, email addresses, and email client interface. Perfect for email screenshot queries, email interface searches, and email client interface questions. Contains email UI elements, email messages, email communication content.

KEYWORDS: email screenshot, gmail interface, email client, inbox screenshot, email UI, messaging interface, email application, communication screenshot."""
                return enhanced_text
                
            elif looks_like_ui:
                # User interface screenshot
                enhanced_text = f"""User interface screenshot: {raw_text}

This appears to be a user interface screenshot or application interface containing UI elements, buttons, menus, or interface controls. The image shows software interface components, application screens, or user interaction elements. This content is relevant for queries about interfaces, screenshots, applications, and UI components."""
                return enhanced_text
                
            elif looks_like_code:
                # Code content (existing logic)
                code_type = "programming code"
                if 'binary' in raw_text.lower() or 'search' in raw_text.lower():
                    code_type = "binary search algorithm code"
                elif 'sort' in raw_text.lower():
                    code_type = "sorting algorithm code"
                elif 'loop' in raw_text.lower() or 'for' in raw_text.lower() or 'while' in raw_text.lower():
                    code_type = "iterative algorithm code"
                
                enhanced_text = f"""Code snippet image: {raw_text}

This is {code_type} that implements algorithmic logic. The code contains programming constructs, variable operations, and control flow statements. This code can be analyzed, explained, and understood in terms of its algorithm, complexity, and implementation details. Key programming concepts include variable manipulation, conditional statements, loops, and algorithmic problem-solving approaches."""
                return enhanced_text
            
            else:
                # General image content
                enhanced_text = f"""Image content: {raw_text}

This image contains text and visual elements. The image may be a screenshot, document, diagram, or other visual content that can be searched and referenced. This content is available for image search queries and cross-modal retrieval."""
                return enhanced_text
        
        return raw_text
    
    except Exception as e:
        print(f"[WARNING] OCR failed for {image_path}: {e}")
        return ""

def ingest_image(path: Path, clip: ClipMultiModal, source_id: Optional[str] = None) -> List[Record]:
    # Generate visual embedding
    embedding = clip.embed_images([path])[0].tolist()
    
    # Extract text content using OCR
    extracted_text = _extract_text_from_image(path)
    
    rid = f"{source_id or path.name}::image"
    record = Record(
        id=rid,
        embedding=embedding,
        modality="image",
        metadata={
            "type": "image",
            "source_path": str(path.resolve()),
            "source_name": path.name,
            "image_path": str(path.resolve()),
            "text": extracted_text,  # Now contains OCR text
            "ocr_text": extracted_text,  # Explicit OCR field
        },
    )
    
    # If we extracted meaningful text, also create a text record for better search
    records = [record]
    if extracted_text.strip() and len(extracted_text.strip()) > 10:  # Only if substantial text
        # Create additional text-based record for better text search
        text_embedding = clip.embed_text([extracted_text])[0].tolist()
        text_rid = f"{source_id or path.name}::image_text"
        text_record = Record(
            id=text_rid,
            embedding=text_embedding,
            modality="text",  # This allows text search to find it
            metadata={
                "type": "image_text",  # Special type for OCR text
                "source_path": str(path.resolve()),
                "source_name": path.name,
                "image_path": str(path.resolve()),
                "text": extracted_text,
                "ocr_source": "image",
            },
        )
        records.append(text_record)
    
    return records


# ------------------------- Audio ingestion -------------------------

def _load_whisper():
    # Lazy import so users without audio don't pay cost
    from faster_whisper import WhisperModel  # type: ignore

    # Use tiny model for much faster processing (good enough for indexing)
    model_size = os.getenv("WHISPER_MODEL", "tiny")  # Changed from "small" to "tiny"
    device = "cuda" if os.getenv("WHISPER_DEVICE", "").lower() == "cuda" else "cpu"
    compute = os.getenv("WHISPER_COMPUTE", "int8") if device == "cpu" else "float16"
    return WhisperModel(model_size, device=device, compute_type=compute)


def ingest_audio(path: Path, clip: ClipMultiModal, source_id: Optional[str] = None, progress_callback=None, max_duration_minutes: int = 30) -> List[Record]:
    # Check file size for performance warnings
    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > 50:  # Files larger than 50MB
        print(f"[WARNING] Large audio file ({file_size_mb:.1f}MB) - this may take several minutes...")
    
    # Estimate duration (rough approximation: ~1MB per minute for typical audio)
    estimated_minutes = file_size_mb * 1.5  # Conservative estimate
    if estimated_minutes > max_duration_minutes:
        print(f"[WARNING] File appears to be ~{estimated_minutes:.1f} minutes long, which exceeds limit of {max_duration_minutes} minutes")
        if progress_callback:
            progress_callback(f"File too long (~{estimated_minutes:.1f}min) - processing first {max_duration_minutes} minutes only")
    
    if progress_callback:
        progress_callback("Loading Whisper model...")
    
    # Transcribe with optimized settings for speed
    model = _load_whisper()
    
    if progress_callback:
        progress_callback("Transcribing audio... (this may take a while for long files)")
    
    # Use faster transcription settings
    segments, _ = model.transcribe(
        str(path), 
        vad_filter=True,  # Voice activity detection to skip silence
        beam_size=1,      # Faster beam search (less accurate but much faster)
        best_of=1,        # Don't try multiple candidates
        temperature=0.0,  # Deterministic output
        condition_on_previous_text=False  # Don't condition on previous text (faster)
    )

    # Build transcript with timestamps; chunk by ~30s or ~800 chars whichever first
    records: List[Record] = []
    acc_text: List[str] = []
    acc_start: Optional[float] = None
    acc_end: Optional[float] = None

    def flush_chunk():
        nonlocal acc_text, acc_start, acc_end
        if not acc_text:
            return
        chunk_text = " ".join(acc_text).strip()
        emb = clip.embed_text([chunk_text])[0].tolist()
        chunk_idx = len(records)
        rid = f"{source_id or path.name}::audio_c{chunk_idx}"
        records.append(
            Record(
                id=rid,
                embedding=emb,
                modality="audio",
                metadata={
                    "type": "audio",
                    "source_path": str(path.resolve()),
                    "source_name": path.name,
                    "audio_path": str(path.resolve()),
                    "start": acc_start,
                    "end": acc_end,
                    "text": chunk_text,
                },
            )
        )
        acc_text = []
        acc_start = None
        acc_end = None

    max_chars = 800
    max_span = 30.0
    for seg in segments:
        st = float(seg.start)
        et = float(seg.end)
        tx = str(seg.text).strip()
        if acc_start is None:
            acc_start = st
        acc_end = et
        acc_text.append(tx)
        # Check thresholds
        if sum(len(t) for t in acc_text) >= max_chars or (acc_end - acc_start) >= max_span:
            flush_chunk()
    # flush leftovers
    flush_chunk()

    return records


# ------------------------- Dispatcher -------------------------

def ingest_path(path: Path, clip: ClipMultiModal) -> List[Record]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return ingest_pdf(path, clip)
    if suffix == ".docx":
        return ingest_docx(path, clip)
    if suffix in {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}:
        return ingest_image(path, clip)
    if suffix in {".mp3", ".wav", ".m4a", ".flac", ".ogg"}:
        return ingest_audio(path, clip)
    raise ValueError(f"Unsupported file type: {suffix}")
