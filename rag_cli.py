#!/usr/bin/env python3
"""
RAG-Optimized PDF Extractor - Command Line Interface

Usage:
    python rag_cli.py input.pdf -o output.ndjson
    python rag_cli.py input.pdf --format json --workers 4
    python rag_cli.py input.pdf --mode thorough --stats
"""

import argparse
import os
import sys
from rag_extractor import RAGOptimizedExtractor
from rag_output import (
    save_ndjson, save_json, save_plain_text,
    print_summary, create_stats_report, save_stats_report
)


def main():
    parser = argparse.ArgumentParser(
        description='RAG-Optimized PDF Extractor - High-performance extraction for RAG ingestion',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract to NDJSON (default, streaming-friendly)
  python rag_cli.py document.pdf -o output.ndjson
  
  # Extract to JSON with 4 workers
  python rag_cli.py document.pdf -o output.json --format json --workers 4
  
  # Thorough mode with statistics
  python rag_cli.py document.pdf -o output.ndjson --mode thorough --stats
  
  # Fast mode, English only
  python rag_cli.py document.pdf -o output.ndjson --mode fast --lang eng
  
  # Auto-detect workers, save all formats
  python rag_cli.py document.pdf -o output --save-all

Performance:
  - Fast mode: 1-2s per page (minimal OCR)
  - Balanced mode: 2-3s per page (selective OCR) [DEFAULT]
  - Thorough mode: 3-5s per page (comprehensive OCR)
  
  Expected throughput: 30-60 pages/min on 4-core CPU
        """
    )
    
    # Required arguments
    parser.add_argument('pdf_path', help='Path to PDF file')
    parser.add_argument('-o', '--output', required=True, help='Output file path (without extension for --save-all)')
    
    # Output format
    parser.add_argument('--format', choices=['ndjson', 'json', 'text'], default='ndjson',
                       help='Output format (default: ndjson)')
    parser.add_argument('--save-all', action='store_true',
                       help='Save in all formats (NDJSON, JSON, and plain text)')
    
    # Extraction options
    parser.add_argument('--mode', choices=['fast', 'balanced', 'thorough'], default='balanced',
                       help='Extraction mode (default: balanced)')
    parser.add_argument('--lang', default='ara+eng',
                       help='OCR language(s): ara, eng, ara+eng (default: ara+eng)')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for OCR (default: 300)')
    parser.add_argument('--enable-grid-ocr', action='store_true',
                       help='Enable Grid OCR (disabled by default, adds 15-20%% overhead)')
    
    # Multiprocessing options
    parser.add_argument('--workers', type=int, default=None,
                       help='Number of parallel workers (default: auto-detect based on CPU and memory)')
    parser.add_argument('--no-parallel', action='store_true',
                       help='Disable multiprocessing (sequential processing)')
    
    # Output options
    parser.add_argument('--stats', action='store_true',
                       help='Save processing statistics')
    parser.add_argument('--quiet', action='store_true',
                       help='Suppress progress output')
    parser.add_argument('--no-summary', action='store_true',
                       help='Don\'t print summary')
    
    args = parser.parse_args()
    
    # Validate input file
    if not os.path.exists(args.pdf_path):
        print(f"Error: PDF file not found: {args.pdf_path}")
        sys.exit(1)
    
    # Create extractor
    extractor = RAGOptimizedExtractor(
        lang=args.lang,
        mode=args.mode,
        dpi=args.dpi,
        enable_grid_ocr=args.enable_grid_ocr
    )
    
    # Extract document
    try:
        if args.no_parallel:
            # Sequential processing (for debugging)
            print("Warning: Sequential processing enabled (slow)")
            # TODO: Implement sequential fallback
            doc_chunks = extractor.extract_document(
                args.pdf_path,
                max_workers=1,
                verbose=not args.quiet
            )
        else:
            # Parallel processing (default)
            doc_chunks = extractor.extract_document(
                args.pdf_path,
                max_workers=args.workers,
                verbose=not args.quiet
            )
        
        # Save output
        if args.save_all:
            # Save all formats
            base_path = args.output
            save_ndjson(doc_chunks.chunks, f"{base_path}.ndjson")
            save_json(doc_chunks, f"{base_path}.json")
            save_plain_text(doc_chunks.chunks, f"{base_path}.txt")
        else:
            # Save single format
            if args.format == 'ndjson':
                save_ndjson(doc_chunks.chunks, args.output)
            elif args.format == 'json':
                save_json(doc_chunks, args.output)
            elif args.format == 'text':
                save_plain_text(doc_chunks.chunks, args.output)
        
        # Save statistics
        if args.stats:
            stats = create_stats_report(doc_chunks)
            stats_path = args.output.rsplit('.', 1)[0] + '_stats.json'
            save_stats_report(stats, stats_path)
        
        # Print summary
        if not args.no_summary:
            print_summary(doc_chunks)
        
        print("✅ Extraction completed successfully!")
        
    except Exception as e:
        print(f"❌ Error during extraction: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
