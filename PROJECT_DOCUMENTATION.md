# PDF Parsing System - Project Documentation

## Overview

An **Intelligent PDF Content Extraction System** using EasyOCR with sophisticated extraction pipelines for high-accuracy text and image extraction from PDF files. Supports Arabic, English, and mixed-language content with RTL text handling.

---

## Key Features

- ✅ **RAG-Optimized Pipeline**: 8-stage extraction for maximum accuracy
- ✅ **Parallel Processing**: 3-4x faster with concurrent page workers
- ✅ **EasyOCR Integration**: High-quality OCR with Arabic & English support
- ✅ **OCR Worker Pool**: Pre-loaded models for 7x faster extraction
- ✅ **Multi-language Support**: Arabic, English, RTL text detection
- ✅ **Column Detection**: Multi-column layout handling
- ✅ **Image Extraction**: OCR on embedded images
- ✅ **Web GUI**: Real-time progress tracking
- ✅ **Multiple Output Formats**: NDJSON, JSON, with statistics


---

## Project Structure

```
/Users/amrnabih/Documents/Gp/Parsing/
├── app.py                      # Flask web application
├── ocr_worker_pool.py          # EasyOCR worker pool management
├── rag_extractor.py            # RAG-optimized 8-stage extraction
├── intelligent_extractor.py    # 7-stage intelligent extraction  
├── language_detection.py       # Language & direction detection
├── rag_models.py               # Data models for RAG pipeline
├── rag_output.py               # Output formatting (NDJSON/JSON/stats)
├── rag_cli.py                  # Command-line interface
├── requirements.txt            # Python dependencies
├── templates/                  # HTML templates
│   └── index.html             # Main upload interface
├── static/                     # CSS & JavaScript assets
├── uploads/                    # Uploaded PDF files
└── output/                     # Extracted output files
```

---

## Architecture

### OCR Worker Pool

The system uses a **long-lived worker pool** architecture for optimal performance:

```
Flask App Startup
       ↓
Initialize EasyOCR Worker Pool
       ↓
Load EasyOCR Model (once, ~10-15s)
       ↓
Workers Ready ✓
       ↓
Process OCR Requests (2-5s per page)
```

**Benefits**:
- ⚡ **7x faster**: No per-request model loading
- 🧠 **Memory efficient**: Single model instance
- 🔄 **Concurrent**: Multiple pages processed in parallel

### RAG Pipeline (8 Stages)

1. **Direct Text Extraction**: Fast PyMuPDF extraction
2. **Block OCR**: Selective OCR on low-quality blocks
3. **Full-Page OCR**: Fallback for poor-quality pages
4. **Image OCR**: Extract text from embedded images
5. **Language Detection**: Detect language & text direction
6. **Quality Assessment**: Confidence scoring
7. **Chunking**: Create RAG-optimized chunks
8. **Output Generation**: NDJSON/JSON with metadata

### Parallel Processing Architecture

**Two-Level Parallelism**:
- **Level 1**: ThreadPoolExecutor for concurrent page processing
- **Level 2**: OCR Worker Pool for parallel OCR tasks

```
Pages Pool (4 workers) → OCR Worker Pool (pre-loaded models)
     ↓                           ↓
Page 1, 2, 3, 4          EasyOCR Workers (shared)
```

**Benefits**:
- ⚡ **3-4x faster**: Concurrent page processing
- 🧠 **Memory efficient**: Shared OCR models
- 🎯 **Auto-scaled**: Based on CPU cores (max 4 by default)
- 🔄 **Configurable**: Custom worker counts via `max_workers` parameter


---

## Core Components

### 1. OCR Worker Pool (`ocr_worker_pool.py`)

**Purpose**: Manages long-lived EasyOCR worker processes.

**Key Classes**:
```python
class OCRWorkerPool:
    def __init__(self, num_workers=1)
    def start()                    # Load models
    def submit_ocr_task(image, ...)  # Submit task
    def shutdown()                 # Cleanup
```

**Worker Lifecycle**:
1. Load EasyOCR model (once at startup)
2. Signal ready
3. Process tasks from queue
4. Return results to main process

### 2. RAG Extractor (`rag_extractor.py`)

**Purpose**: 8-stage extraction pipeline optimized for RAG systems.

**Key Class**:
```python
class RAGOptimizedExtractor:
    def extract_document(pdf_path, verbose=True)
    def extract_page(page, page_num, ...)
    def _run_ocr(image, page_num)
```

**Extraction Modes**:
- `fast`: Direct extraction only, minimal OCR
- `balanced`: Smart OCR on low-quality blocks (default)
- `thorough`: Aggressive OCR, maximum accuracy

### 3. Language Detection (`language_detection.py`)

**Purpose**: Detect language and text direction.

**Key Functions**:
```python
def detect_language_and_direction(text) -> LanguageDetection
    # Returns: language, text_direction, rtl_ratio
```

Automatically detects:
- Arabic vs English
- RTL vs LTR text direction
- Mixed-language content

### 4. Flask App (`app.py`)

**Purpose**: Web interface for PDF extraction.

**Key Routes**:
- `GET /` - Upload interface
- `POST /upload` - Start extraction
- `GET /status/<task_id>` - Get progress
- `GET /download/<filename>` - Download results

**Initialization**:
```python
def init_ocr_worker_pool():
    ocr_pool = initialize_ocr_pool(num_workers=1)
    # Pool ready for requests
```

---

## Usage

### Web Interface

1. **Start Flask**:
   ```bash
   python3 app.py
   ```

2. **Open Browser**: `http://localhost:5001`

3. **Upload PDF** and configure:
   - Pipeline: RAG-Optimized
   - Mode: Balanced
   - Language: Arabic + English
   - DPI: 300

4. **Download Results**: NDJSON, JSON, or stats

### Command Line

```bash
python3 rag_cli.py input.pdf \
    --mode balanced \
    --lang ara+eng \
    --dpi 300 \
    --output-format ndjson
```

---

## Configuration

### Extraction Modes

| Mode | Speed | Accuracy | Use Case |
|------|-------|----------|----------|
| **fast** | ⚡⚡⚡ | ⭐⭐ | Clean, high-quality PDFs |
| **balanced** | ⚡⚡ | ⭐⭐⭐ | General use (default) |
| **thorough** | ⚡ | ⭐⭐⭐⭐ | Poor-quality scans |

### Language Settings

- `en`: English only
- `ar`: Arabic only  
- `ara+eng`: Arabic + English (recommended)

### DPI Settings

- `150`: Fast, lower quality
- `300`: Balanced (default)
- `600`: High quality, slower

---

## Output Formats

### NDJSON (Newline-Delimited JSON)
```json
{"page_number": 1, "text": "...", "confidence": 0.95, ...}
{"page_number": 2, "text": "...", "confidence": 0.92, ...}
```

### JSON (Array)
```json
[
  {"page_number": 1, "text": "...", ...},
  {"page_number": 2, "text": "...", ...}
]
```

### Statistics
```json
{
  "total_pages": 12,
  "avg_confidence": 0.94,
  "extraction_time": "45.2s",
  "ocr_usage": {"easyocr": 8, "direct": 4}
}
```

---

## Dependencies

### Core
- `Flask>=3.0.0` - Web framework
- `PyMuPDF>=1.23.0` - PDF rendering
- `easyocr` - OCR engine
- `opencv-python>=4.8.0` - Image processing
- `numpy>=1.24.0` - Array operations
- `Pillow>=10.0.0` - Image handling

### Optional
- `pytesseract>=0.3.10` - Legacy pipeline support
- `pdfplumber>=0.10.0` - Alternative extraction
- `psutil>=5.9.0` - Memory monitoring

---

## Performance

### Typical Extraction Times (Parallel Mode)

| Pages | Mode | Time (Sequential) | Time (Parallel) | Speedup | Notes |
|-------|------|------------------|-----------------|---------|-------|
| 1-5 | Balanced | 20s | 8s | **2.5x** | Startup overhead |
| 10-20 | Balanced | 60s | 18s | **3.3x** | Optimal parallelism |
| 50+ | Balanced | 5min | 90s | **3.3x** | Worker pool efficient |

**Parallel Processing**: Uses 4 page workers by default (configurable via `max_workers`)

### Memory Usage

- **Idle**: ~500MB (EasyOCR loaded)
- **Processing (4 workers)**: ~1.5-2GB (depends on PDF size)
- **Peak**: ~3GB (large images, high DPI)

### Configuration Options

```python
# Auto-workers (default, max 4)
chunks = extractor.extract_document('doc.pdf')

# Custom worker count
chunks = extractor.extract_document('doc.pdf', max_workers=8)

# Single-threaded (sequential)
chunks = extractor.extract_document('doc.pdf', max_workers=1)
```


---

## Troubleshooting

### EasyOCR Not Working

**Symptom**: "OCR Worker Pool error"

**Solution**:
1. Check worker pool started: Look for "✅ EasyOCR ready!"
2. Restart Flask app
3. Check memory: Ensure >2GB available

### Slow Extraction

**Symptom**: Takes >5min for 10 pages

**Solutions**:
- Lower DPI to 150-200
- Use `fast` mode
- Disable grid OCR

### Arabic Text Issues

**Symptom**: Garbled Arabic text

**Solutions**:
- Ensure language set to `ara+eng`
- Check PDF has embedded fonts
- Try higher DPI (300-600)

---

## Development

### Adding New Features

1. **New Pipeline Stage**: Add to `rag_extractor.py`
2. **New Output Format**: Add to `rag_output.py`
3. **New UI Options**: Update `templates/index.html` and `static/js/app.js`

### Testing

```bash
# Test single PDF
python3 rag_cli.py testingData/sample.pdf

# Test web interface
python3 app.py
# Upload via browser
```

### Debugging

Enable verbose output:
```python
extractor = RAGOptimizedExtractor(...)
chunks = extractor.extract_document(pdf_path, verbose=True)
```

---

## License & Credits

**Author**: [Your Name]
**Version**: 2.0 (EasyOCR-Only)
**Last Updated**: January 2026

### Libraries Used
- EasyOCR - Jaided AI
- PyMuPDF - Artifex Software
- Flask - Pallets Projects
- OpenCV - Intel/Itseez

---

## Changelog

### v2.0 (January 2026)
- ✅ Simplified to EasyOCR-only
- ✅ Removed PaddleOCR dependency
- ✅ Streamlined worker pool  
- ✅ Updated UI
- ✅ Consolidated documentation

### v1.0 (December 2025)
- Initial RAG-optimized pipeline
- Multi-engine OCR support
- Web interface

---

## Contact & Support

For issues or questions:
1. Check this documentation
2. Review troubleshooting section
3. Check Git commit history for recent changes

**Repository Structure**: All code in `/Users/amrnabih/Documents/Gp/Parsing/`
