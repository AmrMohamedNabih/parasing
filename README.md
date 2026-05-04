# RAG Parsing Server

A production-grade PDF extraction API that receives binary PDF documents, processing them with a 7-stage intelligent pipeline, and stores structured text blocks in a PostgreSQL database for RAG (Retrieval-Augmented Generation) ingestion.

## 🚀 Key Features

*   **FastAPI Backend**: Modern, high-performance, and asynchronous.
*   **Intelligent Pipeline**: Combines 7-stage direct extraction, selective OCR, structural analysis, followed by semantic chunking (NLTK sentence-aware) and neural merging (MiniLM).
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
*   NLTK & tiktoken (for semantic chunking and tokenization)

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
.\venv\Scripts\Activate.ps1
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

---

## Phase 2 — RAG Query Service

Phase 2 adds semantic search and Gemini-powered answer generation on top of Phase 1's extracted text blocks.

### Architecture

```
Phase 1 (port 8000)        Phase 2 (port 8001)
──────────────────         ──────────────────────────────────────
PDF Upload API     →  PostgreSQL  ←  Embedding Worker (asyncpg poll)
                        text_blocks          ↓
                                         Qdrant (vectors)
                                             ↓
                                    RAG Service (FastAPI)
                                    ├── POST /questions  → Kafka
                                    ├── GET  /questions/{id}
                                    └── GET  /stream/answer (SSE)
```

### Phase 2 Setup

#### 1. Add Phase 2 env vars to `.env`

```bash
QDRANT_URL=http://localhost:6333
GEMINI_API_KEY=your-key-here
GEMINI_MODEL=gemini-2.5-flash
KAFKA_QUESTION_TOPIC=question-requests
KAFKA_RESPONSE_TOPIC=question-responses
KAFKA_ERROR_TOPIC=question-errors
TOP_K_RESULTS=8
SIMILARITY_THRESHOLD=0.55
```

#### 2. Start Qdrant

```bash
docker-compose up -d qdrant
```

#### 3. Apply the new migration (adds `embedded_at` column)

```bash
alembic upgrade head
```

#### 4. Start the Embedding Worker

```bash
python embedding_worker.py
# Polls every 30 s; logs "Embedded N blocks for document X"
```

#### 5. Start the RAG Service

```bash
uvicorn rag_service.main:app --reload --port 8001
```

Swagger UI: http://localhost:8001/docs

---

### Phase 2 API Reference

| Method | Endpoint | Description |
|:---|:---|:---|
| `POST` | `/questions` | Enqueue a question (publishes to Kafka) |
| `GET` | `/questions/{id}?user_id=` | Poll question status (chunk count, is_final) |
| `GET` | `/stream/answer?question_id=&user_id=` | SSE stream of answer chunks |
| `GET` | `/health` | Liveness probe |

---

### curl Examples

#### Enqueue a question (HTTP streaming)

```bash
curl -X POST http://localhost:8001/questions \
  -H "Content-Type: application/json" \
  -d '{
    "question_id": "11111111-1111-1111-1111-111111111111",
    "user_id":     "YOUR-USER-UUID",
    "subject_id":  "YOUR-SUBJECT-UUID",
    "question":    "What are the main topics covered in chapter 3?",
    "language":    "auto",
    "stream_via":  "http"
  }'
# → {"question_id": "11111111-...", "status": "queued"}
```

#### Stream the answer (SSE)

```bash
curl -N "http://localhost:8001/stream/answer?\
question_id=11111111-1111-1111-1111-111111111111&\
user_id=YOUR-USER-UUID"
# → event: chunk
# → data: {"chunk": "Chapter 3 covers...", "chunk_index": 0, "is_final": false}
# → ...
# → data: {"chunk": "", "chunk_index": 12, "is_final": true, "sources": [...]}
```

#### Poll status (for SSE reconnect recovery)

```bash
curl "http://localhost:8001/questions/11111111-1111-1111-1111-111111111111?\
user_id=YOUR-USER-UUID"
# → {"status": "processing", "chunk_count": 5, "is_final": false}
```

#### Arabic question example

```bash
curl -X POST http://localhost:8001/questions \
  -H "Content-Type: application/json" \
  -d '{
    "question_id": "22222222-2222-2222-2222-222222222222",
    "user_id":     "YOUR-USER-UUID",
    "question":    "ما هي المواضيع الرئيسية في هذه الوثيقة؟",
    "language":    "ar",
    "stream_via":  "both"
  }'
```

> **Note on `SIMILARITY_THRESHOLD`:** The default is `0.55`. Lower values return more (noisier) results; higher values return fewer but more precise results. Tune based on your corpus.

---

### Kafka Message Formats

**Inbound** (`question-requests`):
```json
{
  "question_id": "uuid",
  "user_id":     "uuid",
  "subject_id":  "uuid (optional)",
  "document_id": "uuid (optional)",
  "question":    "string",
  "language":    "auto | ar | en",
  "top_k":       8,
  "stream_via":  "http | kafka | both"
}
```

**Outbound** (`question-responses`):
```json
{
  "question_id": "uuid",
  "chunk_index": 0,
  "chunk":       "text fragment",
  "is_final":    false,
  "sources":     []
}
```

**Final chunk** (`is_final: true`) includes sources:
```json
{
  "sources": [
    {
      "text_block_id": "uuid",
      "document_id":   "uuid",
      "page_number":   3,
      "text_snippet":  "first 120 chars of the passage...",
      "score":         0.87
    }
  ]
}
```
