// PDF Extractor GUI - JavaScript with RAG Pipeline Support

let selectedFiles = [];
let currentTaskId = null;
let currentPipeline = 'rag'; // Default to RAG

// DOM Elements
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const fileList = document.getElementById('fileList');
const extractBtn = document.getElementById('extractBtn');
const clearBtn = document.getElementById('clearBtn');
const progressSection = document.getElementById('progressSection');
const resultsSection = document.getElementById('resultsSection');
const resultsContainer = document.getElementById('resultsContainer');

// Pipeline Selection
const pipelineRadios = document.querySelectorAll('input[name="pipeline"]');
const ragSettings = document.getElementById('ragSettings');
const intelligentSettings = document.getElementById('intelligentSettings');
const disabledRagSettings = document.getElementById('disabledRagSettings');
const disabledIntelligentSettings = document.getElementById('disabledIntelligentSettings');

// Handle pipeline selection
pipelineRadios.forEach(radio => {
    radio.addEventListener('change', (e) => {
        currentPipeline = e.target.value;

        // Show/hide settings based on pipeline
        if (currentPipeline === 'rag') {
            // Show RAG settings
            ragSettings.classList.remove('hidden');
            intelligentSettings.classList.add('hidden');

            // Show disabled intelligent settings (what RAG doesn't use)
            disabledRagSettings.classList.remove('hidden');
            disabledIntelligentSettings.classList.add('hidden');
        } else {
            // Show Intelligent settings
            ragSettings.classList.add('hidden');
            intelligentSettings.classList.remove('hidden');

            // Show disabled RAG settings (what Intelligent doesn't use)
            disabledRagSettings.classList.add('hidden');
            disabledIntelligentSettings.classList.remove('hidden');
        }
    });
});

// Mode descriptions
const modeDescriptions = {
    fast: 'Fast: Minimal OCR, 1-2s/page',
    balanced: 'Balanced: Good speed/accuracy trade-off, 2-3s/page',
    thorough: 'Thorough: Maximum accuracy, 3-5s/page'
};

document.getElementById('mode').addEventListener('change', (e) => {
    document.getElementById('modeDescription').textContent = modeDescriptions[e.target.value];
});

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

    // Add pipeline selection
    formData.append('pipeline', currentPipeline);

    // Common settings
    formData.append('language', document.getElementById('language').value);
    formData.append('dpi', document.getElementById('dpi').value);
    formData.append('mode', document.getElementById('mode').value);

    if (currentPipeline === 'rag') {
        // RAG-specific settings
        formData.append('output_format', document.getElementById('outputFormat').value);
        formData.append('max_workers', document.getElementById('maxWorkers').value);
        formData.append('save_stats', document.getElementById('saveStats').checked);
        formData.append('enable_grid_ocr', document.getElementById('enableGridOcr').checked);
    } else {
        // Intelligent pipeline settings
        formData.append('preprocess', document.getElementById('preprocess').checked);
        formData.append('extract_images', document.getElementById('extractImages').checked);
        formData.append('ocr_engine', document.getElementById('ocrEngine').value);
    }

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
            showProgress(data.pipeline, data.message);
            pollStatus();
        } else {
            alert('Error: ' + data.error);
            resetUI();
        }
    } catch (error) {
        alert('Error: ' + error.message);
        resetUI();
    }
});

// Show Progress
function showProgress(pipeline, message) {
    progressSection.classList.remove('hidden');
    resultsSection.classList.add('hidden');
    resultsContainer.innerHTML = '';

    // Show pipeline info
    const pipelineInfo = document.getElementById('pipelineInfo');
    if (pipelineInfo) {
        pipelineInfo.innerHTML = `
            <div style="margin-top: 0.75rem; padding: 0.75rem; background: var(--bg); border-radius: 6px; font-size: 0.875rem;">
                <strong>${pipeline === 'rag' ? '⚡ RAG-Optimized' : '🔧 Intelligent'} Pipeline</strong>
                <p style="margin: 0.25rem 0 0 0; color: var(--text-muted);">${message}</p>
            </div>
        `;
    }
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
        const card = createResultCard(result, true, data.pipeline);
        resultsContainer.appendChild(card);
    });

    // Show errors
    data.errors.forEach(error => {
        const card = createResultCard(error, false, data.pipeline);
        resultsContainer.appendChild(card);
    });

    resetUI();
}

// Create Result Card
function createResultCard(result, success, pipeline) {
    const card = document.createElement('div');
    card.className = `result-card ${success ? 'success' : 'error'}`;

    if (success) {
        let statsHtml = '';
        let actionsHtml = '';

        if (pipeline === 'rag') {
            // RAG pipeline stats
            statsHtml = `
                <div class="result-stats">
                    <span class="stat-item">
                        📄 ${result.total_pages} pages
                    </span>
                    <span class="stat-item" style="color: var(--success);">
                        ✓ ${result.successful_pages} successful
                    </span>
                    ${result.failed_pages > 0 ? `
                        <span class="stat-item" style="color: var(--danger);">
                            ✗ ${result.failed_pages} failed
                        </span>
                    ` : ''}
                    <span class="stat-item" style="color: var(--primary);">
                        📝 ${result.total_words.toLocaleString()} words
                    </span>
                    <span class="stat-item">
                        🎯 ${result.avg_confidence} confidence
                    </span>
                    <span class="stat-item" style="color: var(--warning);">
                        ⏱️ ${result.execution_time}s
                    </span>
                    <span class="stat-item" style="color: var(--success);">
                        ⚡ ${result.throughput} pages/min
                    </span>
                </div>
                ${result.languages_detected && result.languages_detected.length > 0 ? `
                    <div style="margin-top: 0.5rem; font-size: 0.875rem; color: var(--text-muted);">
                        Languages: ${result.languages_detected.join(', ')}
                    </div>
                ` : ''}
            `;

            // RAG actions - handle multiple output files
            actionsHtml = result.output_files.map(filename => {
                const ext = filename.split('.').pop();
                const label = ext === 'ndjson' ? 'NDJSON' : ext === 'json' ? 'JSON' : 'Stats';
                const icon = ext.includes('stats') ? '📊' : '📥';

                return `
                    <button class="btn btn-secondary btn-small" onclick="downloadFile('${filename}')">
                        ${icon} Download ${label}
                    </button>
                `;
            }).join('');

        } else {
            // Intelligent pipeline stats
            statsHtml = `
                <div class="result-stats">
                    <span class="stat-item">
                        📄 ${result.total_pages} pages
                    </span>
                    <span class="stat-item" style="color: var(--success);">
                        📦 ${result.total_blocks} blocks
                    </span>
                    <span class="stat-item" style="color: var(--primary);">
                        🎯 ${result.avg_confidence} confidence
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
                        </div>
                    </div>
                ` : ''}
            `;

            actionsHtml = `
                <button class="btn btn-primary btn-small" onclick="viewStructure('${result.json_path}')">
                    👁️ View Structure
                </button>
                <button class="btn btn-secondary btn-small" onclick="downloadFile('${result.output_file}')">
                    📥 Download JSON
                </button>
            `;
        }

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

// View Structure function
function viewStructure(jsonFilename) {
    const url = `/view/current/${jsonFilename}`;
    window.open(url, '_blank', 'width=1200,height=800');
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
