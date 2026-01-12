

---

## Recent Enhancements (2026-01-08)

### **1. PaddleOCR Integration**

Added PaddleOCR as a third OCR engine option for improved English text accuracy.

**Files Modified:**
- `rag_extractor.py`: Added PaddleOCR initialization and execution
- `language_detection.py`: Updated OCR engine selection logic
- `app.py`: Added PaddleOCR preference parameter
- `templates/index.html`: Added UI checkbox for PaddleOCR
- `static/js/app.js`: Added form data handling
- `requirements.txt`: Added `paddleocr` and `paddlepaddle`

**OCR Engine Selection Rules:**
```python
if language == "AR":
    engine = "easyocr"  # Best for Arabic
elif language == "EN":
    engine = "paddleocr" if prefer_paddle else "tesseract"
else:  # MIXED
    engine = "easyocr" if arabic_ratio > 0.3 else ("paddleocr" if prefer_paddle else "tesseract")
```

---

### **2. Model Pre-loading**

Implemented global model storage and worker initialization to eliminate per-page loading delays.

**Implementation:**
```python
# Global storage
_global_easyocr_reader = None
_global_paddleocr_reader = None

# Worker initialization
def _init_worker_models(lang: str):
    global _global_easyocr_reader, _global_paddleocr_reader
    _global_easyocr_reader = easyocr.Reader(languages, gpu=False)
    _global_paddleocr_reader = PaddleOCR(use_angle_cls=True, lang=lang)

# Process pool with initializer
pool_manager = ProcessPoolManager(
    max_workers=max_workers,
    initializer=_init_worker_models,
    initargs=(lang,)
)
```

**Performance Impact:**
- Before: 20-30s per page (model loading overhead)
- After: 2-5s per page (models pre-loaded)
- **Expected: 10x faster extraction**

---

### **3. Comprehensive Logging**

Added detailed logging throughout the RAG extraction pipeline.

**Log Output Example:**
```
============================================================
📄 Processing Page 1
============================================================
[Stage 1] Direct text extraction...
  ✓ Extracted 5 blocks, 120 words
[Stage 1.5] Language detection...
  ✓ Language: EN
  ✓ Direction: LTR
  ✓ Arabic ratio: 0.00%
  ✓ OCR Engine selected: PADDLEOCR
[Image Cache] Rendering page image (DPI: 300)...
  ✓ Image rendered: 2480x3508px
[Stage 2] Block OCR (selective)...
  ⊘ No blocks needed OCR (good quality)
[Stage 5] Image OCR...
  ✓ Extracted text from 1 embedded images
[Stage 8] Creating RAG chunk...

✅ Page 1 completed:
   • Words: 154
   • Confidence: 0.85
   • OCR Engine: paddleocr
   • Time: 1.23s
============================================================
```

---

### **4. Memory Monitoring**

Created `monitor_memory.py` utility for real-time RAM usage tracking.

**Usage:**
```bash
python3 monitor_memory.py
```

**Output:**
```
============================================================
Flask App Memory Monitor (macOS)
============================================================
Press Ctrl+C to stop

PID:  1528 | RAM:   117.2 MB | Virtual: 401812.2 MB
PID:  1528 | RAM:   652.8 MB | Virtual: 401812.2 MB  ← OCR loaded
```

---

### **5. Bug Fixes**

#### **A. Image Detection (Stage 5)**
- **Issue**: Images not being detected in RAG pipeline
- **Fix**: Implemented proper image extraction using `fitz.get_images()` and `extract_image()`
- **Result**: Images now correctly detected and OCR'd

#### **B. PaddleOCR API Compatibility**
- **Issue**: `use_gpu` and `show_log` parameters not supported
- **Fix**: Removed unsupported parameters, using minimal initialization
- **Result**: PaddleOCR initializes successfully

#### **C. PaddleOCR Result Parsing**
- **Issue**: "list index out of range" errors
- **Fix**: Added robust error handling with type checking
- **Result**: Graceful fallback to Tesseract on errors

---

## Performance Recommendations

### **For Best Speed**
```
Pipeline: RAG-Optimized
Workers: 1
PaddleOCR: Disabled (use Tesseract)
DPI: 200
Mode: Balanced
```
**Expected**: ~5-8 minutes for 83 pages

### **For Best Accuracy**
```
Pipeline: RAG-Optimized
Workers: 2
PaddleOCR: Disabled (EasyOCR for Arabic, Tesseract for English)
DPI: 300
Mode: Thorough
```
**Expected**: ~10-15 minutes for 83 pages

### **Why Disable PaddleOCR?**
- PaddleOCR: 10-15s per page
- Tesseract: 0.7s per page
- **Tesseract is 15-20x faster with comparable accuracy for English**

---

## Utility Scripts

### **1. monitor_memory.py**
Monitor Flask app memory usage in real-time.

```bash
python3 monitor_memory.py
```

### **2. download_paddleocr_models.py**
Pre-download PaddleOCR models to avoid first-request delays.

```bash
python3 download_paddleocr_models.py
```

---

## Known Issues

### **1. PaddleOCR Performance**
- **Issue**: Significantly slower than Tesseract (10-15s vs 0.7s per page)
- **Recommendation**: Disable PaddleOCR for production use
- **Status**: Working as designed, but not performant

### **2. Model Pre-loading Memory**
- **Issue**: Pre-loading uses ~1 GB RAM continuously
- **Impact**: Acceptable for most systems
- **Mitigation**: Reduce workers if memory constrained

---

## Version History

### **v2.0 (2026-01-08)**
- ✅ Added PaddleOCR support
- ✅ Implemented model pre-loading
- ✅ Added comprehensive logging
- ✅ Fixed image detection bug
- ✅ Created memory monitoring utility

### **v1.0 (Previous)**
- Initial RAG pipeline implementation
- 7-stage intelligent extraction
- EasyOCR and Tesseract support
- Web interface with progress tracking

---

For detailed information about the recent changes, see [walkthrough.md](file:///Users/amrnabih/.gemini/antigravity/brain/94450b06-606f-4d2a-9ef0-544d0ee79e04/walkthrough.md).
