# Re-export all models so Alembic can discover them via `from app.db.models import *`
from app.db.models.document import Document, DocumentPage, DocumentStatus, TextBlock
from app.db.models.subject import Subject
from app.db.models.user import User

__all__ = [
    "User",
    "Subject",
    "Document",
    "DocumentPage",
    "DocumentStatus",
    "TextBlock",
]
