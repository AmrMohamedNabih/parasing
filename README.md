# RAG Parsing Server

A production-grade PDF extraction API that receives binary PDF documents, processing them with a 7-stage intelligent pipeline, and stores structured text blocks in a PostgreSQL database for RAG (Retrieval-Augmented Generation) ingestion.

## 🚀 Key Features

*   **FastAPI Backend**: Modern, high-performance, and asynchronous.
*   **7-Stage Intelligent Pipeline**: Combines direct extraction, selective OCR, and structural analysis.
*   **User & Subject Isolation**: Organizes extractions by `user_id` and `subject_id`.
*   **PostgreSQL Persistence**: Stores extraction results in structured tables (`documents`, `pages`, `text_blocks`).
*   **Asynchronous Processing**: Background tasks handle heavy extraction work without blocking.
*   **Kafka-Ready**: Designed for easy integration with event streaming.
*   **Alembic Migrations**: Robust database schema management.

## 🏗 Directory Structure

```text
parsing/
├── app/
│   ├── api/v1/routes/   # Health and Documents endpoints
│   ├── core/            # Configuration and Dependencies
│   ├── db/              # SQLAlchemy Models and Session setup
│   ├── schemas/         # Pydantic request/response models
│   ├── services/        # Storage and Parsing orchestration
│   └── workers/         # Background task worker
├── alembic/             # Database migrations
├── testingData/         # Sample PDFs for testing
├── intelligent_extractor.py # Core pipeline logic
├── pipeline_models.py       # Data models for the pipeline
├── docker-compose.yml   # Infrastructure (PostgreSQL + pgAdmin)
└── requirements.txt     # Python dependencies
```

## 🛠 Setup & Installation

### 1. Requirements

*   Python 3.10+
*   Docker & Docker Compose
*   Tesseract OCR (for the extraction engine)
*   Poppler (for PDF processing)

### 2. Environment Setup

```bash
# Clone the repository
git clone <repo-url>
cd parsing

# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Setup .env
cp .env.example .env
```

### 3. Start Infrastructure

```bash
docker-compose up -d
```

### 4. Run Migrations

```bash
alembic upgrade head
```

### 5. Start the Server

```bash
uvicorn app.main:app --reload --port 8000
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Server health check |
| `POST` | `/api/v1/documents` | Upload a PDF (Multipart/Form-Data) |
| `GET` | `/api/v1/documents/{id}` | Get document status and stats |
| `GET` | `/api/v1/documents/{id}/pages` | Get all extracted pages and text blocks |
| `GET` | `/api/v1/documents?user_id=...&subject_id=...` | List documents by user/subject |

## 🧬 Pipeline Options

When uploading via `POST /api/v1/documents`:
*   `mode`: `fast` | `balanced` | `thorough` (default: `balanced`)
*   `ocr_engine`: `easyocr` | `tesseract` (default: `easyocr`)

---

## 💡 RAG Integration Notes

This server is designed to be the entry point for a RAG pipeline. The `text_blocks` table contains `user_id` and `subject_id` denormalized for fast filtering. The `bbox` field (JSONB) allows for spatial queries if your downstream vectorization needs layout context.
