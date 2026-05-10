import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, ForeignKey, Integer, String, UniqueConstraint, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class Subject(Base):
    """
    A subject/topic that a user owns.
    All documents belong to a specific user+subject combination.
    Example: user_id=abc, name="Mathematics", or name="History"
    """
    __tablename__ = "subjects"
    __table_args__ = (
        # A user cannot have two subjects with the same name
        UniqueConstraint("user_id", "name", name="uq_subjects_user_id_name"),
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    user_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )

    # EDAG index lifecycle
    edag_status: Mapped[str] = mapped_column(
        String(20), nullable=False, default="none", server_default="none"
    )
    edag_built_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    edag_leaf_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Relationships
    user: Mapped["User"] = relationship("User", back_populates="subjects")  # noqa: F821
    documents: Mapped[list["Document"]] = relationship(  # noqa: F821
        "Document", back_populates="subject", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Subject id={self.id} name='{self.name}' user_id={self.user_id}>"
