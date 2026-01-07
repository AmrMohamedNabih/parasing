"""
RAG-Optimized Data Models
Lightweight models for RAG ingestion pipeline
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from enum import Enum


class Language(Enum):
    """Detected language"""
    ARABIC = "AR"
    ENGLISH = "EN"
    MIXED = "MIXED"
    UNKNOWN = "UNKNOWN"


class TextDirection(Enum):
    """Text direction"""
    RTL = "rtl"
    LTR = "ltr"
    MIXED = "mixed"


@dataclass
class LanguageDetectionResult:
    """
    Result of language detection (Stage 1.5).
    
    This detection happens BEFORE OCR to enable:
    - Dynamic OCR engine selection
    - Performance optimization
    - Better accuracy
    """
    language: str           # "AR", "EN", "MIXED", "UNKNOWN"
    text_direction: str     # "RTL", "LTR", "MIXED"
    rtl_ratio: float        # 0.0 - 1.0
    arabic_char_count: int
    english_char_count: int
    total_chars: int
    confidence: float       # Detection confidence (0.0 - 1.0)
    
    def to_dict(self) -> Dict:
        return {
            'language': self.language,
            'text_direction': self.text_direction,
            'rtl_ratio': round(self.rtl_ratio, 3),
            'confidence': round(self.confidence, 2)
        }


@dataclass
class PageChunk:
    """
    Lightweight, RAG-ready page chunk.
    
    Temporary: One chunk per page (no semantic splitting yet).
    
    Future: Will be replaced with semantic chunking based on:
    - Sentence boundaries
    - Paragraph detection
    - Section headers
    - Maximum token limits (512/1024 tokens)
    
    Why page-level is acceptable temporarily:
    1. Reduces ingestion complexity (no sentence splitting logic)
    2. Faster processing (no NLP overhead)
    3. Simpler error handling (page = atomic unit)
    4. Easier to debug and validate
    5. Prepares data structure for future semantic chunking
    
    Trade-offs:
    - Larger chunks → less precise retrieval
    - May include irrelevant content (headers, footers)
    - Acceptable for initial RAG system deployment
    """
    
    page_number: int
    text: str                    # Full cleaned text
    language: str                # "AR", "EN", "MIXED"
    text_direction: str          # "RTL", "LTR", "MIXED"
    average_confidence: float    # 0.0 - 1.0
    word_count: int
    char_count: int
    
    # Optional metadata (minimal)
    extraction_method: str = "hybrid"  # "direct", "ocr", "hybrid"
    ocr_engine_used: Optional[str] = None  # "tesseract", "easyocr", None
    
    # Error handling
    status: str = "success"      # "success", "failed", "partial"
    error: Optional[str] = None
    
    # Performance metrics (optional)
    processing_time: Optional[float] = None
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict (NDJSON-ready)."""
        return {
            'page_number': self.page_number,
            'text': self.text,
            'language': self.language,
            'text_direction': self.text_direction,
            'average_confidence': round(self.average_confidence, 2),
            'word_count': self.word_count,
            'char_count': self.char_count,
            'extraction_method': self.extraction_method,
            'ocr_engine_used': self.ocr_engine_used,
            'status': self.status,
            'error': self.error,
            'processing_time': round(self.processing_time, 2) if self.processing_time else None
        }


@dataclass
class DocumentChunks:
    """
    Collection of page chunks for entire document.
    Lightweight wrapper for RAG output.
    """
    
    filename: str
    total_pages: int
    chunks: List[PageChunk] = field(default_factory=list)
    
    # Document-level metadata (minimal)
    total_processing_time: float = 0.0
    languages_detected: List[str] = field(default_factory=list)
    
    @property
    def successful_pages(self) -> int:
        return sum(1 for c in self.chunks if c.status == 'success')
    
    @property
    def failed_pages(self) -> int:
        return sum(1 for c in self.chunks if c.status == 'failed')
    
    @property
    def total_words(self) -> int:
        return sum(c.word_count for c in self.chunks)
    
    @property
    def average_confidence(self) -> float:
        if not self.chunks:
            return 0.0
        confidences = [c.average_confidence for c in self.chunks if c.status == 'success']
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def to_dict(self) -> Dict:
        """Convert to JSON-serializable dict."""
        return {
            'document': {
                'filename': self.filename,
                'total_pages': self.total_pages,
                'successful_pages': self.successful_pages,
                'failed_pages': self.failed_pages,
                'total_words': self.total_words,
                'average_confidence': round(self.average_confidence, 2),
                'total_processing_time': round(self.total_processing_time, 2),
                'languages_detected': self.languages_detected
            },
            'chunks': [chunk.to_dict() for chunk in self.chunks]
        }
    
    def to_ndjson_lines(self) -> List[str]:
        """
        Convert to NDJSON format (one JSON object per line).
        
        NDJSON Benefits:
        1. Streaming-friendly (process line-by-line)
        2. Append-only (no need to load entire file)
        3. Easy to parallelize (each line is independent)
        4. Standard format for RAG ingestion pipelines
        """
        import json
        return [json.dumps(chunk.to_dict(), ensure_ascii=False) for chunk in self.chunks]


@dataclass
class ProcessingStats:
    """
    Performance statistics for monitoring and optimization.
    """
    
    total_pages: int = 0
    pages_with_direct_extraction: int = 0
    pages_with_block_ocr: int = 0
    pages_with_full_page_ocr: int = 0
    pages_with_grid_ocr: int = 0
    
    # OCR engine usage
    pages_using_tesseract: int = 0
    pages_using_easyocr: int = 0
    
    # Language distribution
    arabic_pages: int = 0
    english_pages: int = 0
    mixed_pages: int = 0
    
    # Performance metrics
    total_processing_time: float = 0.0
    average_time_per_page: float = 0.0
    
    # Resource usage
    peak_memory_mb: float = 0.0
    worker_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            'total_pages': self.total_pages,
            'extraction_methods': {
                'direct': self.pages_with_direct_extraction,
                'block_ocr': self.pages_with_block_ocr,
                'full_page_ocr': self.pages_with_full_page_ocr,
                'grid_ocr': self.pages_with_grid_ocr
            },
            'ocr_engines': {
                'tesseract': self.pages_using_tesseract,
                'easyocr': self.pages_using_easyocr
            },
            'languages': {
                'arabic': self.arabic_pages,
                'english': self.english_pages,
                'mixed': self.mixed_pages
            },
            'performance': {
                'total_time': round(self.total_processing_time, 2),
                'avg_time_per_page': round(self.average_time_per_page, 2),
                'peak_memory_mb': round(self.peak_memory_mb, 2),
                'worker_count': self.worker_count
            }
        }
