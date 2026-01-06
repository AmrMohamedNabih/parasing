// PDF Extractor GUI - JavaScript

let selectedFiles = [];
let currentTaskId = null;

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const extractBtn = document.getElementById('extractBtn');
const clearBtn = document.getElementById('clearBtn');
const progressSection = document.getElementById('progressSection');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');

// Drag and Drop
uploadArea.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
});

uploadArea.addEventListener('dragleave', () => {
    uploadArea.classList.remove('drag-over');
});

uploadArea.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    const files = Array.from(e.dataTransfer.files).filter(f => f.type === 'application/pdf');
    addFiles(files);
});

uploadArea.addEventListener('click', () => {
    fileInput.click();
});

// File Input
fileInput.addEventListener('change', (e) => {
    const files = Array.from(e.target.files);
    addFiles(files);
});

// Add Files
function addFiles(files) {
    selectedFiles = [...selectedFiles, ...files];
    updateFileList();
    extractBtn.disabled = selectedFiles.length === 0;
}

// Update File List
function updateFileList() {
    if (selectedFiles.length === 0) {
        fileList.classList.add('hidden');
        return;
    }

    fileList.classList.remove('hidden');
    fileList.innerHTML = `
        <h4 style="margin-bottom: 1rem;">Selected Files (${selectedFiles.length})</h4>
        ${selectedFiles.map((file, index) => `
            <div class="file-item">
                <div class="file-info">
                    <svg class="file-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                    </svg>
                    <span>${file.name}</span>
                    <small style="color: var(--text-muted); margin-left: 1rem;">
                        ${(file.size / 1024 / 1024).toFixed(2)} MB
                    </small>
                </div>
                <button class="file-remove" onclick="removeFile(${index})">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <line x1="18" y1="6" x2="6" y2="18"></line>
                        <line x1="6" y1="6" x2="18" y2="18"></line>
                    </svg>
                </button>
            </div>
        `).join('')}
    `;
}

// Remove File
function removeFile(index) {
    selectedFiles.splice(index, 1);
    updateFileList();
    extractBtn.disabled = selectedFiles.length === 0;
}

// Extract Button
extractBtn.addEventListener('click', async () => {
    if (selectedFiles.length === 0) return;

    const formData = new FormData();
    selectedFiles.forEach(file => {
        formData.append('files[]', file);
    });

    formData.append('language', document.getElementById('language').value);
    formData.append('dpi', document.getElementById('dpi').value);
    formData.append('preprocess', document.getElementById('preprocess').checked);
    formData.append('extract_images', document.getElementById('extractImages').checked);
    formData.append('mode', document.getElementById('mode').value);
    formData.append('ocr_engine', document.getElementById('ocrEngine').value);

    try {
        extractBtn.disabled = true;
        extractBtn.innerHTML = '<span>Processing...</span>';

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (response.ok) {
            currentTaskId = data.task_id;
            showProgress();
            pollStatus();
        } else {
            alert('Error: ' + data.error);
            extractBtn.disabled = false;
            extractBtn.innerHTML = `
                <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <polyline points="20 6 9 17 4 12"></polyline>
                </svg>
                Extract Text
            `;
        }
    } catch (error) {
        alert('Error: ' + error.message);
        extractBtn.disabled = false;
    }
});

// Show Progress
function showProgress() {
    progressSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    resultsContainer.innerHTML = '';
}

// Poll Status
async function pollStatus() {
    if (!currentTaskId) return;

    try {
        const response = await fetch(`/status/${currentTaskId}`);
        const data = await response.json();

        updateProgress(data);

        if (data.status === 'processing') {
            setTimeout(pollStatus, 1000);
        } else if (data.status === 'completed') {
            showResults(data);
        } else if (data.status === 'error') {
            alert('Error: ' + data.error);
            resetUI();
        }
    } catch (error) {
        console.error('Error polling status:', error);
        setTimeout(pollStatus, 2000);
    }
}

// Update Progress
function updateProgress(data) {
    const percent = (data.current / data.total) * 100;

    document.getElementById('progressFill').style.width = percent + '%';
    document.getElementById('progressText').textContent =
        data.status === 'processing' ? 'Processing...' : 'Completed!';
    document.getElementById('progressCount').textContent =
        `${data.current}/${data.total}`;

    if (data.current_file) {
        document.getElementById('currentFile').textContent =
            `Current: ${data.current_file}`;
    }
}

// Show Results
function showResults(data) {
    progressSection.classList.add('hidden');
    resultsSection.classList.remove('hidden');

    resultsContainer.innerHTML = '';

    // Show successful extractions
    data.results.forEach(result => {
        const card = createResultCard(result, true);
        resultsContainer.appendChild(card);
    });

    // Show errors
    data.errors.forEach(error => {
        const card = createResultCard(error, false);
        resultsContainer.appendChild(card);
    });

    resetUI();
}

// Create Result Card
function createResultCard(result, success) {
    const card = document.createElement('div');
    card.className = `result-card ${success ? 'success' : 'error'}`;

    if (success) {
        // Build statistics info
        let statsHtml = '';

        // Intelligent pipeline stats
        statsHtml = `
            <div class="result-stats">
                <span class="stat-item">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path d="M14 2H6a2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                    </svg>
                    ${result.total_pages} pages
                </span>
                <span class="stat-item" style="color: var(--success);">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <rect x="3" y="3" width="18" height="18" rx="2"></rect>
                    </svg>
                    ${result.total_blocks} blocks
                </span>
                <span class="stat-item" style="color: var(--primary);">
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <polyline points="20 6 9 17 4 12"></polyline>
                    </svg>
                    ${result.avg_confidence} confidence
                </span>
                <span class="stat-item" style="color: var(--warning);">
                    ⏱️ ${result.execution_time}s
                </span>
            </div>
            ${result.stage_stats ? `
                <div class="stage-breakdown" style="margin-top: 0.75rem; padding: 0.75rem; background: var(--bg); border-radius: 6px;">
                    <div style="font-size: 0.75rem; font-weight: 600; margin-bottom: 0.5rem; color: var(--text-muted);">
                        Stage Breakdown (${result.mode} mode):
                    </div>
                    <div style="display: flex; gap: 0.75rem; flex-wrap: wrap; font-size: 0.75rem;">
                        ${result.stage_stats.direct ? `<span style="color: var(--success);">✓ Direct: ${result.stage_stats.direct}</span>` : ''}
                        ${result.stage_stats.block_ocr ? `<span style="color: var(--warning);">⚡ Block OCR: ${result.stage_stats.block_ocr}</span>` : ''}
                        ${result.stage_stats.full_page_ocr ? `<span style="color: var(--secondary);">📄 Full OCR: ${result.stage_stats.full_page_ocr}</span>` : ''}
                        ${result.stage_stats.grid_ocr ? `<span style="color: var(--primary);">🔲 Grid OCR: ${result.stage_stats.grid_ocr}</span>` : ''}
                        ${result.stage_stats.image_ocr ? `<span style="color: var(--danger);">🖼️ Image OCR: ${result.stage_stats.image_ocr}</span>` : ''}
                        ${result.stage_stats.post_processed ? `<span style="color: #10b981;">✨ Post-processed: ${result.stage_stats.post_processed}</span>` : ''}
                    </div>
                </div>
            ` : ''}
        `;

        // Build action buttons
        let actionsHtml = `
            <button class="btn btn-primary btn-small" onclick="viewStructure('${result.json_path}')">
                <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"></path>
                    <circle cx="12" cy="12" r="3"></circle>
                </svg>
                View Structure
            </button>
            <button class="btn btn-secondary btn-small" onclick="downloadFile('${result.output_file}')">
                <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                    <polyline points="7 10 12 15 17 10"></polyline>
                    <line x1="12" y1="15" x2="12" y2="3"></line>
                </svg>
                Download JSON
            </button>
        `;

        card.innerHTML = `
            <div class="result-header">
                <div class="result-filename">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <path d="M14 2H6a2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path>
                        <polyline points="14 2 14 8 20 8"></polyline>
                    </svg>
                    ${result.filename}
                </div>
                <span class="badge badge-success">Success</span>
            </div>
            ${statsHtml}
            ${result.preview ? `<div class="result-preview">${escapeHtml(result.preview)}...</div>` : ''}
            <div class="result-actions">
                ${actionsHtml}
            </div>
        `;
    } else {
        card.innerHTML = `
            <div class="result-header">
                <div class="result-filename">
                    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="8" x2="12" y2="12"></line>
                        <line x1="12" y1="16" x2="12.01" y2="16"></line>
                    </svg>
                    ${result.filename}
                </div>
                <span class="badge badge-error">Error</span>
            </div>
            <div class="result-preview" style="color: var(--danger);">
                Error: ${escapeHtml(result.error)}
            </div>
        `;
    }

    return card;
}

// Download File
function downloadFile(filename) {
    window.location.href = `/download/${filename}`;
}

// View Full Text
function viewFullText(text) {
    const modal = document.createElement('div');
    modal.style.cssText = `
        position: fixed;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background: rgba(0, 0, 0, 0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 1000;
        padding: 2rem;
    `;

    modal.innerHTML = `
        <div style="
            background: var(--bg-card);
            border-radius: 16px;
            padding: 2rem;
            max-width: 800px;
            max-height: 80vh;
            overflow-y: auto;
            border: 1px solid var(--border);
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                <h3>Full Extracted Text</h3>
                <button onclick="this.closest('div').parentElement.parentElement.remove()" 
                    style="background: none; border: none; color: var(--text); cursor: pointer; font-size: 1.5rem;">
                    ×
                </button>
            </div>
            <pre style="
                white-space: pre-wrap;
                font-family: 'Courier New', monospace;
                line-height: 1.6;
                color: var(--text-muted);
                direction: rtl;
                text-align: right;
            ">${text}</pre>
        </div>
    `;

    modal.addEventListener('click', (e) => {
        if (e.target === modal) {
            modal.remove();
        }
    });

    document.body.appendChild(modal);
}

// Clear Button
clearBtn.addEventListener('click', async () => {
    if (!confirm('Clear all files and results?')) return;

    try {
        await fetch('/clear');
        selectedFiles = [];
        updateFileList();
        resultsSection.classList.add('hidden');
        progressSection.classList.add('hidden');
        resetUI();
    } catch (error) {
        alert('Error clearing files: ' + error.message);
    }
});

// Reset UI
function resetUI() {
    extractBtn.disabled = selectedFiles.length === 0;
    extractBtn.innerHTML = `
        <svg class="btn-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
            <polyline points="20 6 9 17 4 12"></polyline>
        </svg>
        Extract Text
    `;
}

// Utility: Escape HTML
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// View Structure function
function viewStructure(jsonFilename) {
    // Open viewer in new window
    const url = `/view/current/${jsonFilename}`;
    window.open(url, '_blank', 'width=1200,height=800');
}
