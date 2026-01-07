# PDF Parsing System - Complete Documentation

## Table of Contents
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [Core Files Explained](#core-files-explained)
4. [Key Algorithms](#key-algorithms)
5. [Important Libraries](#important-libraries)
6. [Extraction Pipelines](#extraction-pipelines)
7. [Web Interface](#web-interface)
8. [Data Flow](#data-flow)

---

## Project Overview

This is an **Intelligent PDF Content Extraction System** that uses a sophisticated 7-stage pipeline to extract text and images from PDF files with high accuracy. The system supports both Arabic and English languages, handles RTL (Right-to-Left) text, and provides a web-based GUI for easy interaction.

### Key Features
- **7-Stage Intelligent Extraction Pipeline** for optimal accuracy
- **Dual OCR Engine Support**: Tesseract and EasyOCR
- **Multi-language Support**: Arabic, English, and mixed content
- **RTL Text Detection** and handling
- **Column Detection** for multi-column layouts
- **Image Extraction** with OCR on embedded images
- **Web-based GUI** with real-time progress tracking
- **Three Extraction Modes**: Fast, Balanced, Thorough

---

## Project Structure

```
/Users/amrnabih/Documents/Gp/Parsing/
├── app.py                          # Flask web application (main entry point)
├── intelligent_extractor.py        # 7-stage intelligent extraction pipeline
├── pdf_extractor.py                # Legacy hybrid extractor (direct + OCR)
├── structured_extractor.py         # Structured extractor with column detection
├── pipeline_models.py              # Data models for pipeline stages
├── app_viewer_routes.py            # Additional Flask routes (viewer)
├── requirements.txt                # Python dependencies
├── templates/                      # HTML templates
│   ├── index.html                  # Main upload interface
│   └── viewer.html                 # JSON structure viewer
├── static/                         # Static assets
│   ├── css/
│   │   ├── style.css              # Main stylesheet
│   │   └── stats.css              # Statistics styling
│   └── js/
│       ├── app.js                 # Frontend JavaScript logic
│       └── app_viewer_helper.js   # Viewer helper functions
├── uploads/                        # Uploaded PDF files
├── output/                         # Extracted JSON and text files
└── testingData/                    # Test PDF files
```

---

## Core Files Explained

### 1. **app.py** (Flask Web Application)
**Purpose**: Main web server that provides the GUI interface for PDF extraction.

**Key Components**:
- **Flask Routes**:
  - `/` - Main page (upload interface)
  - `/upload` - Handle file uploads and start extraction
  - `/status/<task_id>` - Get extraction progress
  - `/download/<filename>` - Download extracted files
  - `/view/<task_id>/<filename>` - View JSON structure
  - `/clear` - Clear uploaded/output files

- **Background Processing**: Uses Python threading to process PDFs asynchronously
- **Configuration**:
  - Upload folder: `uploads/`
  - Output folder: `output/`
  - Max file size: 100MB

**Key Functions**:
```python
extract_pdf_task(task_id, pdf_files, lang, preprocess, dpi, extract_images, mode, ocr_engine)
```
- Runs extraction in background thread
- Tracks progress in `extraction_status` dictionary
- Uses `IntelligentPDFExtractor` for processing

---

### 2. **intelligent_extractor.py** (7-Stage Pipeline)
**Purpose**: The core extraction engine implementing a sophisticated 7-stage pipeline.

**Architecture**: 7-Stage Intelligent Extraction Pipeline

#### **Stage 1: Layout-Aware Direct Text Extraction**
- **Always runs** (baseline stage)
- Extracts embedded text directly from PDF using PyMuPDF
- Preserves layout structure (blocks, fonts, sizes)
- Calculates confidence scores for each block
- Detects text direction (LTR/RTL/Mixed)

**Algorithm**:
```python
def stage1_direct_extraction(pdf_path, page_num):
    1. Open PDF page with PyMuPDF
    2. Extract text blocks with bounding boxes
    3. For each block:
       - Extract text content
       - Calculate confidence score (based on word count, char count)
       - Detect RTL ratio (Arabic/Hebrew characters)
       - Extract font information
    4. Return blocks with metadata
```

**Confidence Calculation**:
- Low word count (< 3 words) → Low confidence (0.2)
- High single-character ratio (> 50%) → Low confidence (0.3)
- Otherwise: `min(1.0, (word_count/50)*0.5 + (char_count/500)*0.5)`

---

#### **Stage 2: Block-Level OCR**
- **Selective** - Only runs on low-confidence blocks
- Triggers when: `confidence < 0.3` OR `word_count < 3` OR `text == ""`
- Performs OCR on specific regions instead of entire page

**Algorithm**:
```python
def stage2_block_ocr(pdf_path, page_num, direct_result):
    1. Identify blocks needing OCR (low confidence)
    2. For each low-confidence block:
       - Convert page to image at specified DPI
       - Crop image to block bounding box
       - Preprocess image (adaptive threshold, denoise)
       - Run OCR (Tesseract or EasyOCR)
       - Compare with original text
       - Keep better result
    3. Return improved blocks
```

**Image Preprocessing** (for Tesseract):
- Convert to grayscale
- Adaptive thresholding (Gaussian)
- Denoising (fastNlMeansDenoising)

---

#### **Stage 3: Full-Page OCR**
- **Safety net** - Runs when direct extraction is insufficient
- Triggers when: `total_chars < 100` OR `page_num == 0` OR `text_coverage < 20%`
- Always OCRs the first page (cover page)

**Algorithm**:
```python
def stage3_full_page_ocr(pdf_path, page_num, direct_result, page_width, page_height):
    1. Calculate text coverage percentage
    2. If coverage is low:
       - Convert entire page to image
       - Preprocess image
       - Run OCR on full page
       - Create single block with all text
    3. Return full-page block
```

---

#### **Stage 4: Adaptive Grid OCR**
- **Conditional** - Only in "thorough" mode
- Runs when text is still sparse after previous stages
- Divides page into 3x3 grid and OCRs empty regions

**Algorithm**:
```python
def stage4_grid_ocr(pdf_path, page_num, page_width, page_height, existing_blocks):
    1. Check if total_chars > 200 (skip if sufficient)
    2. Divide page into 3x3 grid
    3. For each grid cell:
       - Check if it overlaps with existing blocks
       - If no overlap (sparse region):
         * Crop image to grid cell
         * Run OCR
         * Add as new block if text found
    4. Return grid blocks
```

**Overlap Detection**: Uses IoU (Intersection over Union) with 0.3 threshold

---

#### **Stage 5: Image OCR**
- **Conditional** - Only if page has embedded images
- Extracts images and performs OCR on them
- Saves images to disk if requested

**Algorithm**:
```python
def stage5_image_ocr(pdf_path, page_num, images_dir):
    1. Get list of embedded images in page
    2. For each image:
       - Extract image bytes
       - Get image position (bounding box)
       - Save image to disk (optional)
       - Run OCR on image
       - Create ImageBlock with OCR text
       - Also create TextBlock if OCR found text
    3. Return image blocks
```

---

#### **Stage 6: Merge & De-duplication**
- **Always runs** - Critical for preventing duplicate text
- Merges outputs from all stages
- Removes duplicate blocks based on overlap and text similarity

**Algorithm**:
```python
def stage6_merge_deduplication(all_stage_results, page_num):
    1. Collect all blocks from all stages
    2. Define priority: Direct > Block OCR > Full OCR > Grid OCR > Image OCR
    3. For each block:
       - Check overlap with existing blocks (IoU > 0.7)
       - Check text similarity (> 0.8)
       - If duplicate found:
         * Keep higher priority block
         * Mark as merged
    4. Return unique blocks
```

**Text Similarity**: Uses `SequenceMatcher` (difflib) for ratio calculation

---

#### **Stage 7: Language-Aware Post-Processing**
- **Always runs** - Final cleanup for better quality
- Handles Arabic text normalization
- Fixes broken words
- Removes OCR artifacts

**Algorithm**:
```python
def stage7_post_processing(blocks):
    1. For each block:
       - If RTL (Arabic): Normalize characters, remove tatweel
       - Fix broken words (e.g., "th e" → "the")
       - Remove OCR artifacts (|, ~, `, excessive punctuation)
       - Normalize whitespace
    2. Return processed blocks
```

**Arabic Processing**:
- Remove tatweel (kashida): `\u0640`
- Normalize Alef variants: `\u0622`, `\u0623`, `\u0625` → `\u0627`

---

### 3. **pipeline_models.py** (Data Models)
**Purpose**: Defines data structures for the extraction pipeline using Python dataclasses.

**Key Models**:

#### **Enums**:
```python
class ExtractionStage(Enum):
    DIRECT = "direct"
    BLOCK_OCR = "block_ocr"
    FULL_PAGE_OCR = "full_page_ocr"
    GRID_OCR = "grid_ocr"
    IMAGE_OCR = "image_ocr"
    MERGED = "merged"
    POST_PROCESSED = "post_processed"

class TextDirection(Enum):
    LTR = "ltr"
    RTL = "rtl"
    MIXED = "mixed"
```

#### **TextBlock**:
```python
@dataclass
class TextBlock:
    block_id: str                    # Unique identifier
    bbox: List[float]                # [x0, y0, x1, y1]
    text: str                        # Extracted text
    confidence: float                # 0.0 - 1.0
    source_stage: ExtractionStage    # Which stage extracted this
    font: Optional[str]              # Font family
    size: Optional[float]            # Font size
    direction: TextDirection         # LTR/RTL/MIXED
    rtl_ratio: float                 # Ratio of RTL characters
    word_count: int                  # Number of words
    char_count: int                  # Number of characters
    column: Optional[int]            # Column number (1-indexed)
```

**Key Method**:
```python
def overlaps_with(other: TextBlock, threshold=0.7) -> bool:
    # Calculate IoU (Intersection over Union)
    # Returns True if IoU >= threshold
```

#### **Stage Results**:
- `DirectExtractionResult` - Stage 1 output
- `BlockOCRResult` - Stage 2 output
- `FullPageOCRResult` - Stage 3 output
- `GridOCRResult` - Stage 4 output
- `ImageOCRResult` - Stage 5 output
- `MergeResult` - Stage 6 output
- `PostProcessingResult` - Stage 7 output

#### **PageResult**:
```python
@dataclass
class PageResult:
    page_number: int
    width: float
    height: float
    blocks: List[TextBlock]
    images: List[ImageBlock]
    stage_results: Dict[ExtractionStage, StageResult]
    columns: int
    total_execution_time: float
```

**Properties**:
- `stage_stats` - Count blocks by source stage
- `total_confidence` - Average confidence across blocks
- `text_coverage` - Percentage of page covered by text

#### **DocumentResult**:
```python
@dataclass
class DocumentResult:
    filename: str
    total_pages: int
    pages: List[PageResult]
    metadata: Dict
    total_execution_time: float
```

**Properties**:
- `total_blocks` - Sum of blocks across all pages
- `total_images` - Sum of images across all pages
- `overall_stage_stats` - Aggregate statistics
- `avg_confidence` - Average confidence across document

**Key Method**:
```python
def to_dict() -> Dict:
    # Converts entire document to JSON-serializable dictionary
    # Used for saving to JSON files
```

---

### 4. **pdf_extractor.py** (Legacy Hybrid Extractor)
**Purpose**: Older extraction method (kept for backward compatibility).

**Approach**: Simple 2-stage hybrid
1. Try direct text extraction
2. If confidence < threshold, fall back to full-page OCR

**Key Class**:
```python
class HybridPDFExtractor:
    def __init__(lang='ara+eng', text_threshold=0.1)
    
    def extract_text_direct(pdf_path, page_num) -> (text, confidence)
    def extract_text_ocr(pdf_path, page_num, preprocess, dpi) -> text
    def extract_images_from_page(pdf_path, page_num, output_dir) -> List[Dict]
    def extract_from_pdf(pdf_path, ...) -> Dict
```

**Validation Logic**:
```python
def validate_page_text(text, page_area):
    - Check word count (< 5 words = invalid)
    - Check single-character ratio (> 50% = gibberish)
    - Calculate confidence: min(1.0, (word_count/50)*0.5 + (char_count/500)*0.5)
```

---

### 5. **structured_extractor.py** (Structured Extractor)
**Purpose**: Extracts PDF with preserved structure and column detection.

**Key Features**:
- Block-level structure preservation
- Column detection (1, 2, or 3 columns)
- RTL text detection
- Font information extraction

**Column Detection Algorithm**:
```python
def detect_columns(blocks, page_width):
    1. Get X-coordinates of block centers
    2. Divide page into thirds
    3. Count blocks in each third
    4. Determine columns:
       - Left + Right populated, Middle sparse → 2 columns
       - All three populated → 3 columns
       - Otherwise → 1 column
```

**Column Assignment**:
```python
def assign_column(bbox, page_width, num_columns):
    x_center = (bbox[0] + bbox[2]) / 2
    column_width = page_width / num_columns
    column = int(x_center / column_width) + 1
    return min(column, num_columns)
```

---

## Key Algorithms

### 1. **RTL (Right-to-Left) Detection**
**Purpose**: Detect if text is Arabic/Hebrew (RTL) or English (LTR).

**Algorithm**:
```python
def _detect_text_direction(text):
    # Arabic Unicode range: 0x0600-0x06FF
    # Hebrew Unicode range: 0x0590-0x05FF
    
    rtl_chars = count characters in range [0x0590, 0x06FF]
    total_chars = count non-whitespace characters
    
    rtl_ratio = rtl_chars / total_chars
    
    if rtl_ratio > 0.7:
        direction = RTL
    elif rtl_ratio > 0.3:
        direction = MIXED
    else:
        direction = LTR
    
    return (direction, rtl_ratio)
```

**Unicode Ranges**:
- Arabic: `\u0600` - `\u06FF`
- Hebrew: `\u0590` - `\u05FF`

---

### 2. **Bounding Box Overlap (IoU)**
**Purpose**: Calculate overlap between two bounding boxes.

**Algorithm**:
```python
def _bbox_overlap(bbox1, bbox2):
    # bbox format: [x0, y0, x1, y1]
    
    # Calculate intersection rectangle
    x0_i = max(bbox1[0], bbox2[0])
    y0_i = max(bbox1[1], bbox2[1])
    x1_i = min(bbox1[2], bbox2[2])
    y1_i = min(bbox1[3], bbox2[3])
    
    # Check if rectangles don't overlap
    if x1_i < x0_i or y1_i < y0_i:
        return 0.0
    
    # Calculate areas
    intersection = (x1_i - x0_i) * (y1_i - y0_i)
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union = area1 + area2 - intersection
    
    # IoU (Intersection over Union)
    return intersection / union if union > 0 else 0.0
```

**Usage**: De-duplication (threshold = 0.7), Grid OCR sparse region detection (threshold = 0.3)

---

### 3. **Text Similarity**
**Purpose**: Calculate similarity between two text strings for duplicate detection.

**Algorithm**:
```python
from difflib import SequenceMatcher

def _text_similarity(text1, text2):
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()
```

**Returns**: Float between 0.0 (completely different) and 1.0 (identical)

**Usage**: De-duplication (threshold = 0.8)

---

### 4. **Image Preprocessing (for Tesseract OCR)**
**Purpose**: Enhance image quality for better OCR accuracy.

**Algorithm**:
```python
def _preprocess_image(image):
    1. Convert PIL Image to OpenCV format (BGR)
    2. Convert to grayscale
    3. Apply adaptive thresholding:
       - Method: ADAPTIVE_THRESH_GAUSSIAN_C
       - Block size: 11
       - Constant: 2
    4. Apply denoising:
       - Method: fastNlMeansDenoising
       - h: 10 (filter strength)
       - templateWindowSize: 7
       - searchWindowSize: 21
    5. Convert back to PIL Image
    return processed_image
```

**OpenCV Functions**:
- `cv2.adaptiveThreshold()` - Adaptive binarization
- `cv2.fastNlMeansDenoising()` - Non-local means denoising

---

### 5. **EasyOCR vs Tesseract**
**Purpose**: Support two OCR engines for flexibility.

**EasyOCR**:
```python
def _run_easyocr(image):
    1. Convert PIL image to numpy array
    2. Run reader.readtext(image, detail=1)
    3. Filter results by confidence > 0.3
    4. Combine text from all detections
    return combined_text
```

**Tesseract**:
```python
def _run_tesseract(image):
    1. Preprocess image (adaptive threshold + denoise)
    2. Run pytesseract.image_to_string(image, lang=lang, config='--oem 3 --psm 6')
    return text
```

**Comparison**:
| Feature | EasyOCR | Tesseract |
|---------|---------|-----------|
| Accuracy | Higher (especially for Arabic) | Good |
| Speed | Slower | Faster |
| Setup | Requires model download | Requires system installation |
| GPU Support | Yes | No |
| Default | ✓ (in this project) | Fallback |

---

### 6. **Column Detection**
**Purpose**: Detect multi-column layouts (1, 2, or 3 columns).

**Algorithm**:
```python
def _detect_columns(blocks, page_width):
    1. Extract X-coordinates of block centers
    2. Divide page into thirds:
       - Left third: [0, page_width/3)
       - Middle third: [page_width/3, 2*page_width/3)
       - Right third: [2*page_width/3, page_width]
    3. Count blocks in each third
    4. Determine layout:
       - If left AND right have blocks, middle sparse → 2 columns
       - If all three have blocks → 3 columns
       - Otherwise → 1 column
    return num_columns
```

---

## Important Libraries

### 1. **PyMuPDF (fitz)** - v1.23.0+
**Purpose**: Fast PDF parsing and direct text extraction.

**Key Functions**:
```python
import fitz

doc = fitz.open(pdf_path)                    # Open PDF
page = doc[page_num]                         # Get page
text_dict = page.get_text("dict")            # Get text with structure
text = page.get_text()                       # Get plain text
image_list = page.get_images()               # Get embedded images
base_image = doc.extract_image(xref)         # Extract image bytes
rects = page.get_image_rects(xref)          # Get image position
```

**Advantages**:
- Very fast
- Preserves layout structure
- Provides font information
- Can extract images

---

### 2. **pdfplumber** - v0.10.0+
**Purpose**: Alternative PDF parser, better for tables.

**Key Functions**:
```python
import pdfplumber

with pdfplumber.open(pdf_path) as pdf:
    page = pdf.pages[page_num]
    text = page.extract_text()               # Extract text
    tables = page.extract_tables()           # Extract tables
```

**Advantages**:
- Better table extraction
- More accurate for complex layouts

---

### 3. **pytesseract** - v0.3.10+
**Purpose**: Python wrapper for Tesseract OCR engine.

**Key Functions**:
```python
import pytesseract

text = pytesseract.image_to_string(
    image,
    lang='ara+eng',                          # Language(s)
    config='--oem 3 --psm 6'                # OCR Engine Mode 3, Page Segmentation Mode 6
)
```

**Configuration**:
- `--oem 3`: LSTM neural net mode (best accuracy)
- `--psm 6`: Assume uniform block of text

**Languages**:
- `ara`: Arabic
- `eng`: English
- `ara+eng`: Both (combined)

---

### 4. **EasyOCR** - Latest
**Purpose**: Deep learning-based OCR with better accuracy.

**Key Functions**:
```python
import easyocr

reader = easyocr.Reader(['ar', 'en'], gpu=False)
results = reader.readtext(image, detail=1)

# results format: [(bbox, text, confidence), ...]
```

**Advantages**:
- Higher accuracy (especially for Arabic)
- No system dependencies
- GPU support
- Pre-trained models

**Disadvantages**:
- Slower than Tesseract
- Larger memory footprint
- Requires model download on first run

---

### 5. **pdf2image** - v1.16.3+
**Purpose**: Convert PDF pages to images for OCR.

**Key Functions**:
```python
from pdf2image import convert_from_path

images = convert_from_path(
    pdf_path,
    dpi=300,                                 # Resolution
    first_page=page_num + 1,
    last_page=page_num + 1
)
```

**Dependencies**: Requires `poppler-utils` system package

---

### 6. **OpenCV (cv2)** - v4.8.0+
**Purpose**: Image preprocessing for better OCR.

**Key Functions**:
```python
import cv2

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
denoised = cv2.fastNlMeansDenoising(image, None, 10, 7, 21)
```

**Image Processing**:
- Color space conversion
- Thresholding (binarization)
- Denoising
- Morphological operations

---

### 7. **Pillow (PIL)** - v10.0.0+
**Purpose**: Image manipulation and enhancement.

**Key Functions**:
```python
from PIL import Image, ImageEnhance

image = Image.open(image_path)
cropped = image.crop((x0, y0, x1, y1))
enhancer = ImageEnhance.Contrast(image)
enhanced = enhancer.enhance(2.0)
```

---

### 8. **Flask** - v3.0.0+
**Purpose**: Web framework for GUI.

**Key Features**:
```python
from flask import Flask, render_template, request, jsonify, send_file

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files[]')
    return jsonify({'task_id': task_id})
```

---

### 9. **NumPy** - v1.24.0+
**Purpose**: Array operations for image processing.

**Usage**:
```python
import numpy as np

img_array = np.array(pil_image)              # PIL to NumPy
pil_image = Image.fromarray(img_array)       # NumPy to PIL
```

---

## Extraction Pipelines

### **Pipeline Comparison**

| Feature | Intelligent (7-Stage) | Structured | Hybrid (Legacy) |
|---------|----------------------|------------|-----------------|
| **Stages** | 7 | 2 | 2 |
| **Accuracy** | Highest | Medium | Medium |
| **Speed** | Configurable | Fast | Fast |
| **Column Detection** | ✓ | ✓ | ✗ |
| **RTL Support** | ✓ | ✓ | ✗ |
| **Image OCR** | ✓ | ✓ | ✓ |
| **De-duplication** | ✓ | ✗ | ✗ |
| **Post-processing** | ✓ | ✗ | ✗ |
| **OCR Engines** | Tesseract + EasyOCR | Tesseract | Tesseract |
| **Modes** | Fast/Balanced/Thorough | N/A | N/A |

---

### **Mode Comparison (Intelligent Pipeline)**

| Stage | Fast Mode | Balanced Mode | Thorough Mode |
|-------|-----------|---------------|---------------|
| 1. Direct Extraction | ✓ | ✓ | ✓ |
| 2. Block OCR | ✓ | ✓ | ✓ |
| 3. Full-Page OCR | ✗ | ✓ | ✓ |
| 4. Grid OCR | ✗ | ✗ | ✓ |
| 5. Image OCR | ✓ | ✓ | ✓ |
| 6. Merge & Dedup | ✓ | ✓ | ✓ |
| 7. Post-processing | ✓ | ✓ | ✓ |

**Recommendations**:
- **Fast**: Digital PDFs with good embedded text
- **Balanced**: Mixed content (default, recommended)
- **Thorough**: Scanned documents, poor quality PDFs

---

## Web Interface

### **Frontend (templates/index.html)**

**Features**:
- Drag-and-drop file upload
- Multi-file selection
- Real-time progress tracking
- Configuration options:
  - Language selection (Arabic, English, Both)
  - Pipeline mode (Fast, Balanced, Thorough)
  - OCR engine (Tesseract, EasyOCR)
  - DPI setting
  - Image extraction toggle
- Results display with statistics
- JSON viewer
- Download extracted files

**JavaScript (static/js/app.js)**:
```javascript
// Key functions
uploadFiles()           // Handle file upload
pollStatus(taskId)      // Poll extraction status
displayResults(data)    // Show results
viewStructure(taskId, filename)  // Open JSON viewer
```

---

### **JSON Viewer (templates/viewer.html)**

**Features**:
- Hierarchical JSON structure display
- Syntax highlighting
- Collapsible sections
- Page navigation
- Block-level details:
  - Text content
  - Bounding boxes
  - Confidence scores
  - Source stage
  - Font information
  - Direction (LTR/RTL)

---

## Data Flow

### **Complete Extraction Flow**

```
1. User uploads PDF via web interface
   ↓
2. Flask receives file, saves to uploads/
   ↓
3. Creates background thread with task_id
   ↓
4. IntelligentPDFExtractor initialized with:
   - Language (ara+eng)
   - Mode (balanced)
   - DPI (300)
   - OCR engine (easyocr)
   ↓
5. For each page:
   ├─ Stage 1: Direct extraction (PyMuPDF)
   ├─ Stage 2: Block OCR (selective)
   ├─ Stage 3: Full-page OCR (conditional)
   ├─ Stage 4: Grid OCR (conditional)
   ├─ Stage 5: Image OCR (if images exist)
   ├─ Stage 6: Merge & de-duplicate
   └─ Stage 7: Post-process (Arabic normalization, cleanup)
   ↓
6. Create PageResult with all blocks
   ↓
7. Detect columns, assign blocks to columns
   ↓
8. Aggregate into DocumentResult
   ↓
9. Convert to JSON (to_dict())
   ↓
10. Save to output/filename_intelligent.json
   ↓
11. Update extraction_status with results
   ↓
12. Frontend polls /status/<task_id>
   ↓
13. Display results with statistics
   ↓
14. User can:
    - View JSON structure
    - Download JSON file
    - Process more PDFs
```

---

### **Data Model Flow**

```
PDF File
  ↓
DocumentResult
  ├─ filename
  ├─ total_pages
  ├─ metadata
  ├─ total_execution_time
  └─ pages: List[PageResult]
       ├─ page_number
       ├─ width, height
       ├─ columns
       ├─ total_execution_time
       ├─ blocks: List[TextBlock]
       │    ├─ block_id
       │    ├─ bbox [x0, y0, x1, y1]
       │    ├─ text
       │    ├─ confidence
       │    ├─ source_stage
       │    ├─ font, size
       │    ├─ direction (LTR/RTL/MIXED)
       │    ├─ rtl_ratio
       │    ├─ word_count, char_count
       │    └─ column
       ├─ images: List[ImageBlock]
       │    ├─ image_id
       │    ├─ bbox
       │    ├─ image_path
       │    ├─ ocr_text
       │    └─ confidence
       └─ stage_results: Dict[ExtractionStage, StageResult]
```

---

## Configuration

### **Environment Variables**
None required (all configuration via web interface)

### **System Requirements**
- Python 3.8+
- Tesseract OCR (system installation)
- Poppler utils (for pdf2image)

### **Installation**
```bash
# Install Python dependencies
pip install -r requirements.txt

# Install system dependencies (macOS)
brew install tesseract
brew install poppler

# Install system dependencies (Ubuntu)
sudo apt-get install tesseract-ocr
sudo apt-get install poppler-utils

# Download Tesseract language data
# Arabic: ara.traineddata
# English: eng.traineddata
```

### **Running the Application**
```bash
python3 app.py
# Server runs on http://0.0.0.0:5001
```

---

## Performance Characteristics

### **Speed Comparison** (approximate, 10-page PDF)

| Mode | Time | Accuracy |
|------|------|----------|
| Fast | ~10s | Good |
| Balanced | ~30s | Very Good |
| Thorough | ~60s | Excellent |

### **OCR Engine Comparison**

| Engine | Speed | Accuracy (Arabic) | Accuracy (English) |
|--------|-------|-------------------|-------------------|
| Tesseract | Fast | Good | Very Good |
| EasyOCR | Slow | Excellent | Excellent |

---

## Summary

This PDF extraction system represents a sophisticated, production-ready solution for extracting text and images from PDF files with high accuracy. The 7-stage intelligent pipeline ensures optimal results across various PDF types (digital, scanned, mixed), while the web interface provides an intuitive user experience.

**Key Strengths**:
1. **Intelligent Stage Selection**: Only runs necessary stages
2. **Dual OCR Support**: Tesseract (fast) and EasyOCR (accurate)
3. **RTL Language Support**: Proper Arabic/Hebrew handling
4. **De-duplication**: Prevents duplicate text from multiple stages
5. **Structured Output**: Preserves layout, columns, fonts
6. **Web GUI**: Easy to use, real-time progress tracking

**Use Cases**:
- Academic paper extraction
- Legal document processing
- Multilingual content extraction
- Scanned document digitization
- Archive digitization
