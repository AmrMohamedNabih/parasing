"""
Output Utilities for RAG-Optimized Extraction
Handles NDJSON and JSON output formats
"""

import json
import os
from typing import List
from rag_models import PageChunk, DocumentChunks, ProcessingStats


def save_ndjson(chunks: List[PageChunk], output_path: str):
    """
    Save page chunks as NDJSON (newline-delimited JSON).
    
    NDJSON Benefits:
    1. Streaming-friendly (process line-by-line)
    2. Append-only (no need to load entire file)
    3. Easy to parallelize (each line is independent)
    4. Standard format for RAG ingestion pipelines
    
    Args:
        chunks: List of PageChunk objects
        output_path: Path to save NDJSON file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            json_line = json.dumps(chunk.to_dict(), ensure_ascii=False)
            f.write(json_line + '\n')
    
    print(f"✓ Saved NDJSON to: {output_path}")


def save_json(doc_chunks: DocumentChunks, output_path: str, pretty: bool = True):
    """
    Save document chunks as regular JSON.
    
    Args:
        doc_chunks: DocumentChunks object
        output_path: Path to save JSON file
        pretty: Whether to use indentation (default: True)
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        if pretty:
            json.dump(doc_chunks.to_dict(), f, ensure_ascii=False, indent=2)
        else:
            json.dump(doc_chunks.to_dict(), f, ensure_ascii=False)
    
    print(f"✓ Saved JSON to: {output_path}")


def save_plain_text(chunks: List[PageChunk], output_path: str):
    """
    Save extracted text as plain text file (for debugging).
    
    Args:
        chunks: List of PageChunk objects
        output_path: Path to save text file
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(f"{'='*60}\n")
            f.write(f"Page {chunk.page_number}\n")
            f.write(f"Language: {chunk.language} | Direction: {chunk.text_direction}\n")
            f.write(f"Confidence: {chunk.average_confidence:.2f} | Words: {chunk.word_count}\n")
            f.write(f"{'='*60}\n")
            f.write(chunk.text)
            f.write(f"\n\n")
    
    print(f"✓ Saved plain text to: {output_path}")


def print_summary(doc_chunks: DocumentChunks):
    """
    Print extraction summary to console.
    
    Args:
        doc_chunks: DocumentChunks object
    """
    print(f"\n{'='*60}")
    print(f"EXTRACTION SUMMARY")
    print(f"{'='*60}")
    print(f"File: {doc_chunks.filename}")
    print(f"Total pages: {doc_chunks.total_pages}")
    print(f"Successful: {doc_chunks.successful_pages}")
    print(f"Failed: {doc_chunks.failed_pages}")
    print(f"Total words: {doc_chunks.total_words:,}")
    print(f"Average confidence: {doc_chunks.average_confidence:.2f}")
    print(f"Languages detected: {', '.join(doc_chunks.languages_detected)}")
    print(f"Total time: {doc_chunks.total_processing_time:.2f}s")
    print(f"Throughput: {doc_chunks.total_pages / doc_chunks.total_processing_time * 60:.1f} pages/min")
    
    # Language distribution
    lang_dist = {}
    for chunk in doc_chunks.chunks:
        lang_dist[chunk.language] = lang_dist.get(chunk.language, 0) + 1
    
    print(f"\nLanguage Distribution:")
    for lang, count in sorted(lang_dist.items()):
        print(f"  {lang}: {count} pages ({count/doc_chunks.total_pages*100:.1f}%)")
    
    # Extraction methods
    method_dist = {}
    ocr_engines = {}
    for chunk in doc_chunks.chunks:
        method_dist[chunk.extraction_method] = method_dist.get(chunk.extraction_method, 0) + 1
        if chunk.ocr_engine_used:
            ocr_engines[chunk.ocr_engine_used] = ocr_engines.get(chunk.ocr_engine_used, 0) + 1
    
    print(f"\nExtraction Methods:")
    for method, count in sorted(method_dist.items()):
        print(f"  {method}: {count} pages ({count/doc_chunks.total_pages*100:.1f}%)")
    
    if ocr_engines:
        print(f"\nOCR Engines Used:")
        for engine, count in sorted(ocr_engines.items()):
            print(f"  {engine}: {count} pages")
    
    print(f"{'='*60}\n")


def load_ndjson(input_path: str) -> List[PageChunk]:
    """
    Load page chunks from NDJSON file.
    
    Args:
        input_path: Path to NDJSON file
        
    Returns:
        List of PageChunk objects
    """
    chunks = []
    
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                chunk = PageChunk(**data)
                chunks.append(chunk)
    
    return chunks


def load_json(input_path: str) -> DocumentChunks:
    """
    Load document chunks from JSON file.
    
    Args:
        input_path: Path to JSON file
        
    Returns:
        DocumentChunks object
    """
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Reconstruct DocumentChunks
    doc_data = data['document']
    chunks_data = data['chunks']
    
    chunks = [PageChunk(**chunk_data) for chunk_data in chunks_data]
    
    doc_chunks = DocumentChunks(
        filename=doc_data['filename'],
        total_pages=doc_data['total_pages'],
        chunks=chunks,
        total_processing_time=doc_data.get('total_processing_time', 0.0),
        languages_detected=doc_data.get('languages_detected', [])
    )
    
    return doc_chunks


def create_stats_report(doc_chunks: DocumentChunks) -> ProcessingStats:
    """
    Create processing statistics from document chunks.
    
    Args:
        doc_chunks: DocumentChunks object
        
    Returns:
        ProcessingStats object
    """
    stats = ProcessingStats()
    
    stats.total_pages = doc_chunks.total_pages
    stats.total_processing_time = doc_chunks.total_processing_time
    stats.average_time_per_page = doc_chunks.total_processing_time / doc_chunks.total_pages if doc_chunks.total_pages > 0 else 0
    
    for chunk in doc_chunks.chunks:
        # Extraction methods
        if chunk.extraction_method == 'direct':
            stats.pages_with_direct_extraction += 1
        elif chunk.extraction_method == 'ocr':
            stats.pages_with_full_page_ocr += 1
        elif chunk.extraction_method == 'hybrid':
            stats.pages_with_block_ocr += 1
        
        # OCR engines
        if chunk.ocr_engine_used == 'tesseract':
            stats.pages_using_tesseract += 1
        elif chunk.ocr_engine_used == 'easyocr':
            stats.pages_using_easyocr += 1
        
        # Languages
        if chunk.language == 'AR':
            stats.arabic_pages += 1
        elif chunk.language == 'EN':
            stats.english_pages += 1
        elif chunk.language == 'MIXED':
            stats.mixed_pages += 1
    
    return stats


def save_stats_report(stats: ProcessingStats, output_path: str):
    """
    Save processing statistics as JSON.
    
    Args:
        stats: ProcessingStats object
        output_path: Path to save stats JSON
    """
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats.to_dict(), f, ensure_ascii=False, indent=2)
    
    print(f"✓ Saved stats report to: {output_path}")
