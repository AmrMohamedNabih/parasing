FROM python:3.11-slim

# System packages required by Phase 1 (PyMuPDF, pdf2image, OpenCV, Tesseract)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-ara \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (layer cache)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-bake the e5-small model so containers don't download ~120MB at runtime.
# Stored in the image layer at ~/.cache/huggingface/
RUN python -c "\
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('intfloat/multilingual-e5-small'); \
print('Model pre-baked successfully.')"

# Copy application code
COPY . .
