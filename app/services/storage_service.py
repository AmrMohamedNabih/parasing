import uuid
from pathlib import Path

import aiofiles

from app.core.config import settings


class StorageService:
    """
    Handles local filesystem storage of uploaded PDFs.

    Storage layout:
        {PDF_STORAGE_PATH}/{user_id}/{subject_id}/{document_id}.pdf

    This class is intentionally minimal and swappable — replace the
    implementation with an S3/MinIO backend without touching any other code.
    """

    def __init__(self) -> None:
        self.base_path = Path(settings.PDF_STORAGE_PATH)

    def _build_path(
        self,
        user_id: uuid.UUID,
        subject_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> Path:
        """Return the absolute path for a document PDF, creating dirs if needed."""
        directory = self.base_path / str(user_id) / str(subject_id)
        directory.mkdir(parents=True, exist_ok=True)
        return directory / f"{document_id}.pdf"

    async def save(
        self,
        file_bytes: bytes,
        user_id: uuid.UUID,
        subject_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> str:
        """
        Persist PDF bytes to disk and return the absolute path as a string.
        Uses aiofiles for non-blocking I/O.
        """
        target_path = self._build_path(user_id, subject_id, document_id)
        async with aiofiles.open(target_path, "wb") as f:
            await f.write(file_bytes)
        return str(target_path)

    def resolve_path(
        self,
        user_id: uuid.UUID,
        subject_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> str:
        """Resolve the storage path without writing anything."""
        return str(self._build_path(user_id, subject_id, document_id))

    async def delete(
        self,
        user_id: uuid.UUID,
        subject_id: uuid.UUID,
        document_id: uuid.UUID,
    ) -> None:
        """Remove a stored PDF from disk (used in cleanup / re-upload scenarios)."""
        path = self._build_path(user_id, subject_id, document_id)
        if path.exists():
            path.unlink()


storage_service = StorageService()
