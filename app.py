"""
Flask Web GUI for PDF Content Extractor
"""

import os
import json
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename
from pdf_extractor import HybridPDFExtractor  # Keep for backward compatibility
from structured_extractor import StructuredPDFExtractor  # Keep for backward compatibility
from intelligent_extractor import IntelligentPDFExtractor  # NEW: 7-stage pipeline
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


def allowed_file(filename):
    """Check if file is a PDF"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'pdf'


def extract_pdf_task(task_id, pdf_files, lang, preprocess, dpi, extract_images, mode, ocr_engine):
    """Background task for PDF extraction using 7-stage intelligent pipeline"""
    global extraction_status
    
    try:
        extraction_status[task_id] = {
            'status': 'processing',
            'total': len(pdf_files),
            'current': 0,
            'results': [],
            'errors': []
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
    """Handle file upload"""
    try:
        if 'files[]' not in request.files:
            return jsonify({'error': 'No files uploaded'}), 400
        
        files = request.files.getlist('files[]')
        lang = request.form.get('language', 'ara+eng')
        preprocess = request.form.get('preprocess', 'true') == 'true'
        dpi = int(request.form.get('dpi', 300))
        extract_images = request.form.get('extract_images', 'false') == 'true'
        mode = request.form.get('mode', 'balanced')  # Pipeline mode
        ocr_engine = request.form.get('ocr_engine', 'easyocr')  # OCR engine
        
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
        
        # Start extraction in background
        thread = threading.Thread(
            target=extract_pdf_task,
            args=(task_id, pdf_files, lang, preprocess, dpi, extract_images, mode, ocr_engine)
        )
        thread.start()
        
        return jsonify({
            'task_id': task_id,
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
            data = json.load(f)
        return jsonify(data)
    except Exception as e:
        return jsonify({'error': str(e)}), 404


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)

