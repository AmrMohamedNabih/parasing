"""
Flask Web GUI for PDF Content Extractor
Supports both original pipeline and RAG-optimized pipeline with OCR Worker Pool
"""

import os
import json
import signal
import atexit
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pdf_extractor import HybridPDFExtractor  # Keep for backward compatibility
from structured_extractor import StructuredPDFExtractor  # Keep for backward compatibility
from intelligent_extractor import IntelligentPDFExtractor  # 7-stage pipeline
from rag_extractor import RAGOptimizedExtractor  # NEW: RAG-optimized pipeline
from rag_output import save_ndjson, save_json, create_stats_report, save_stats_report
from ocr_worker_pool import initialize_ocr_pool, get_ocr_pool, shutdown_ocr_pool
import threading
from datetime import datetime
import shutil

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['OUTPUT_FOLDER'] = 'output'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max

# Ensure folders exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['OUTPUT_FOLDER'], exist_ok=True)

# Store extraction progress
extraction_status = {}

# OCR Worker Pool state
ocr_pool = None
ocr_ready = False


def init_ocr_worker_pool():
    """Initialize OCR Worker Pool at app startup."""
    global ocr_pool, ocr_ready
    
    try:
        print("\n" + "="*60)
        print("🚀 Flask App Starting...")
        print("="*60)
        
        # Initialize OCR Worker Pool
        ocr_pool = initialize_ocr_pool(
            num_paddle_workers=1,  # 1 PaddleOCR worker
            num_easyocr_workers=1  # 1 EasyOCR worker
        )
        
        # Mark as ready
        ocr_ready = ocr_pool.is_ready
        
        if ocr_ready:
            print("✅ Flask app ready to accept requests!\n")
        else:
            print("⚠️  OCR Worker Pool failed to initialize\n")
    
    except Exception as e:
        print(f"❌ Failed to initialize OCR Worker Pool: {e}")
        ocr_ready = False


# Graceful shutdown handlers
def shutdown_handler(signum=None, frame=None):
    """Handle graceful shutdown."""
    print("\n🛑 Shutting down Flask app...")
    shutdown_ocr_pool()
    print("✅ Shutdown complete\n")


# Register shutdown handlers
atexit.register(shutdown_handler)
signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)


# ==================== LOADING GATE ====================

@app.before_request
def check_ocr_ready():
    """
    Loading gate: Block upload requests if OCR not ready.
    Allows health check and status endpoints.
    """
    # Allow these endpoints always
    allowed_paths = ['/', '/health', '/ocr-status', '/static']
    
    if any(request.path.startswith(path) for path in allowed_paths):
        return None
    
    # Block upload if OCR not ready
    if request.path == '/upload' and not ocr_ready:
        return jsonify({
            'status': 'loading',
            'message': 'OCR models are loading, please wait...',
            'ready': False
        }), 503


@app.route('/ocr-status')
def ocr_status():
    """
    OCR status endpoint for frontend polling.
    Returns readiness state of OCR Worker Pool.
    """
    global ocr_pool, ocr_ready
    
    return jsonify({
        'ready': ocr_ready,
        'paddle_loaded': ocr_pool.paddle_ready.is_set() if ocr_pool else False,
        'easyocr_loaded': ocr_pool.easyocr_ready.is_set() if ocr_pool else False,
        'message': 'OCR Worker Pool ready' if ocr_ready else 'Loading OCR models...'
    })


@app.route('/health')
def health():
    """Health check endpoint."""
    return jsonify({'status': 'healthy', 'ocr_ready': ocr_ready})


# ==================== EXISTING ROUTES ====================

def allowed_file(filename):
    """Check if file is a PDF"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'


def extract_pdf_task_rag(task_id, pdf_files, config):
    """
    Background task for RAG-optimized PDF extraction.
    
    Args:
        task_id: Unique task identifier
        pdf_files: List of PDF file paths
        config: Configuration dictionary with:
            - lang: Language (ara+eng, ara, eng)
            - mode: Extraction mode (fast, balanced, thorough)
            - dpi: DPI for OCR
            - enable_grid_ocr: Whether to enable Grid OCR
            - max_workers: Number of parallel workers (None = auto)
            - output_format: Output format (ndjson, json, both)
            - save_stats: Whether to save statistics
    """
    global extraction_status
    
    try:
        extraction_status[task_id] = {
            'status': 'processing',
            'total': len(pdf_files),
            'current': 0,
            'results': [],
            'errors': [],
            'pipeline': 'rag'
        }
        
        # Add debug prints to confirm config contents
        print(f"🧵 [Thread] Starting task {task_id}")
        print(f"📝 [Thread] Config: {config}")
        
        # Create RAG extractor
        # NOTE: Don't pass ocr_pool directly - it can't be pickled for multiprocessing
        # Instead, extractor will use get_ocr_pool() from ocr_worker_pool module
        extractor = RAGOptimizedExtractor(
            lang=config.get('lang', 'ara+eng'),
            mode=config.get('mode', 'balanced'),
            dpi=int(config.get('dpi', 300)),
            enable_grid_ocr=config.get('enable_grid_ocr', False),
            ocr_mode=config.get('ocr_mode', 'auto') # Pass ocr_mode to extractor
        )
        
        for i, pdf_path in enumerate(pdf_files, 1):
            try:
                filename = os.path.basename(pdf_path)
                extraction_status[task_id]['current'] = i
                extraction_status[task_id]['current_file'] = filename
                
                # Extract document
                doc_chunks = extractor.extract_document(
                    pdf_path,
                    max_workers=config.get('max_workers'),
                    ocr_mode=config.get('ocr_mode', 'auto'),
                    verbose=True  # Disable console output in web mode
                )
                
                # Prepare output paths
                base_filename = os.path.splitext(filename)[0]
                output_format = config.get('output_format', 'ndjson')
                
                output_files = []
                
                # Save NDJSON
                if output_format in ['ndjson', 'both']:
                    ndjson_filename = f"{base_filename}_rag.ndjson"
                    ndjson_path = os.path.join(app.config['OUTPUT_FOLDER'], ndjson_filename)
                    save_ndjson(doc_chunks.chunks, ndjson_path)
                    output_files.append(ndjson_filename)
                
                # Save JSON
                if output_format in ['json', 'both']:
                    json_filename = f"{base_filename}_rag.json"
                    json_path = os.path.join(app.config['OUTPUT_FOLDER'], json_filename)
                    save_json(doc_chunks, json_path, pretty=True)
                    output_files.append(json_filename)
                
                # Save statistics
                stats_filename = None
                if config.get('save_stats', False):
                    stats = create_stats_report(doc_chunks)
                    stats_filename = f"{base_filename}_stats.json"
                    stats_path = os.path.join(app.config['OUTPUT_FOLDER'], stats_filename)
                    save_stats_report(stats, stats_path)
                    output_files.append(stats_filename)
                
                # Add result
                extraction_status[task_id]['results'].append({
                    'filename': filename,
                    'output_files': output_files,
                    'success': True,
                    'pipeline': 'rag',
                    'total_pages': doc_chunks.total_pages,
                    'successful_pages': doc_chunks.successful_pages,
                    'failed_pages': doc_chunks.failed_pages,
                    'total_words': doc_chunks.total_words,
                    'avg_confidence': round(doc_chunks.average_confidence, 2),
                    'execution_time': round(doc_chunks.total_processing_time, 2),
                    'languages_detected': doc_chunks.languages_detected,
                    'throughput': round(doc_chunks.total_pages / doc_chunks.total_processing_time * 60, 1) if doc_chunks.total_processing_time > 0 else 0
                })
                
            except Exception as e:
                extraction_status[task_id]['errors'].append({
                    'filename': filename,
                    'error': str(e)
                })
        
        extraction_status[task_id]['status'] = 'completed'
        
    except Exception as e:
        extraction_status[task_id]['status'] = 'error'
        extraction_status[task_id]['error'] = str(e)


def extract_pdf_task(task_id, pdf_files, lang, preprocess, dpi, extract_images, mode, ocr_engine):
    """Background task for PDF extraction using 7-stage intelligent pipeline"""
    global extraction_status
    
    try:
        extraction_status[task_id] = {
            'status': 'processing',
            'total': len(pdf_files),
            'current': 0,
            'results': [],
            'errors': [],
            'pipeline': 'intelligent'
        }
        
        # Use intelligent extractor with selected mode and OCR engine
        extractor = IntelligentPDFExtractor(
            lang=lang,
            mode=mode,
            dpi=dpi,
            ocr_engine=ocr_engine
        )
        
        for i, pdf_path in enumerate(pdf_files, 1):
            try:
                filename = os.path.basename(pdf_path)
                extraction_status[task_id]['current'] = i
                extraction_status[task_id]['current_file'] = filename
                
                # Prepare output paths
                json_filename = f"{os.path.splitext(filename)[0]}_intelligent.json"
                json_path = os.path.join(app.config['OUTPUT_FOLDER'], json_filename)
                
                images_dir = None
                if extract_images:
                    images_dir = os.path.join(app.config['OUTPUT_FOLDER'], 
                                             f"{os.path.splitext(filename)[0]}_images")
                
                # Extract using intelligent pipeline
                results = extractor.extract_from_pdf(
                    pdf_path=pdf_path,
                    output_json_path=json_path,
                    extract_images=extract_images,
                    images_dir=images_dir
                )
                
                # Calculate statistics
                stage_stats = results.overall_stage_stats
                total_blocks = results.total_blocks
                total_images = results.total_images
                
                extraction_status[task_id]['results'].append({
                    'filename': filename,
                    'output_file': json_filename,
                    'json_path': json_filename,
                    'success': True,
                    'pipeline': 'intelligent',
                    'mode': mode,
                    'total_pages': results.total_pages,
                    'total_blocks': total_blocks,
                    'total_images': total_images,
                    'avg_confidence': round(results.avg_confidence, 2),
                    'execution_time': round(results.total_execution_time, 2),
                    'stage_stats': stage_stats,
                    'images_dir': os.path.basename(images_dir) if images_dir else None
                })
                
            except Exception as e:
                extraction_status[task_id]['errors'].append({
                    'filename': filename,
                    'error': str(e)
                })
        
        extraction_status[task_id]['status'] = 'completed'
        
    except Exception as e:
        extraction_status[task_id]['status'] = 'error'
        extraction_status[task_id]['error'] = str(e)


@app.route('/')
def index():
    """Render main page"""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_files():
    """Handle file upload - supports both pipelines"""
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files[]')
        
        # DEBUG: Print received form data
        print("📝 Form Data Received:", dict(request.form))
        
        # Get pipeline selection
        pipeline = request.form.get('pipeline', 'rag')  # Default to RAG
        
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        # Save uploaded files
        pdf_files = []
        for file in files:
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                pdf_files.append(filepath)
        
        if not pdf_files:
            return jsonify({'error': 'No valid PDF files uploaded'}), 400
        
        # Create task ID
        task_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        if pipeline == 'rag':
            # RAG-optimized pipeline
            config = {
                'lang': request.form.get('language', 'ara+eng'),
                'mode': request.form.get('mode', 'balanced'),
                'dpi': int(request.form.get('dpi', 300)),
                'enable_grid_ocr': request.form.get('enable_grid_ocr', 'false') == 'true',
                'max_workers': int(request.form.get('max_workers')) if request.form.get('max_workers') else None,
                'output_format': request.form.get('output_format', 'ndjson'),
                'save_stats': request.form.get('save_stats', 'false') == 'true',
                'prefer_paddle': request.form.get('prefer_paddle', 'false') == 'true'  # PaddleOCR preference
            }
            
            # Start extraction in background
            thread = threading.Thread(
                target=extract_pdf_task_rag,
                args=(task_id, pdf_files, config)
            )
            thread.start()
            
            return jsonify({
                'task_id': task_id,
                'pipeline': 'rag',
                'message': f'Processing {len(pdf_files)} PDF file(s) with RAG-optimized pipeline ({config["mode"]} mode)'
            })
        
        else:
            # Original intelligent pipeline
            lang = request.form.get('language', 'ara+eng')
            preprocess = request.form.get('preprocess', 'true') == 'true'
            dpi = int(request.form.get('dpi', 300))
            extract_images = request.form.get('extract_images', 'false') == 'true'
            mode = request.form.get('mode', 'balanced')
            ocr_engine = request.form.get('ocr_engine', 'easyocr')
            
            # Start extraction in background
            thread = threading.Thread(
                target=extract_pdf_task,
                args=(task_id, pdf_files, lang, preprocess, dpi, extract_images, mode, ocr_engine)
            )
            thread.start()
            
            return jsonify({
                'task_id': task_id,
                'pipeline': 'intelligent',
                'message': f'Processing {len(pdf_files)} PDF file(s) in {mode} mode with {ocr_engine.upper()}'
            })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/status/<task_id>')
def get_status(task_id):
    """Get extraction status"""
    if task_id not in extraction_status:
        return jsonify({'error': 'Task not found'}), 404
    
    return jsonify(extraction_status[task_id])


@app.route('/download/<filename>')
def download_file(filename):
    """Download extracted text file"""
    try:
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        return send_file(filepath, as_attachment=True)
    except Exception as e:
        return jsonify({'error': str(e)}), 404


@app.route('/clear')
def clear_files():
    """Clear uploaded and output files"""
    try:
        # Clear uploads
        for filename in os.listdir(app.config['UPLOAD_FOLDER']):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
        
        # Clear outputs
        for filename in os.listdir(app.config['OUTPUT_FOLDER']):
            filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
            if os.path.isfile(filepath):
                os.remove(filepath)
            elif os.path.isdir(filepath):
                shutil.rmtree(filepath)
        
        return jsonify({'message': 'Files cleared successfully'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/view/<task_id>/<filename>')
def view_structure(task_id, filename):
    """Render structure viewer for a JSON file"""
    return render_template('viewer.html', task_id=task_id, filename=filename)


@app.route('/api/structure/<filename>')
def get_structure(filename):
    """Get JSON structure data"""
    try:
        filepath = os.path.join(app.config['OUTPUT_FOLDER'], filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            # Check if it's NDJSON
            if filename.endswith('.ndjson'):
                # Read all lines and return as array
                lines = []
                for line in f:
                    if line.strip():
                        lines.append(json.loads(line))
                return jsonify({'chunks': lines, 'format': 'ndjson'})
            else:
                # Regular JSON
                data = json.load(f)
                return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 404


if __name__ == '__main__':
    # Initialize OCR Worker Pool before starting Flask
    # This must be in __main__ block for multiprocessing to work on macOS
    print("\n" + "="*60)
    print("🚀 Initializing OCR Worker Pool at startup...")
    print("="*60)
    init_ocr_worker_pool()
    
    # Start Flask app
    app.run(
        host='0.0.0.0',
        port=5001,
        debug=False,  # Must be False for multiprocessing
        use_reloader=False  # Must be False to avoid double initialization
    )
