"""Add embedded_at to text_blocks with partial index

Revision ID: 0002
Revises: 0001
Create Date: 2026-04-27
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Add embedded_at column — nullable, set to now() after Qdrant upsert
    op.add_column(
        "text_blocks",
        sa.Column("embedded_at", sa.DateTime(timezone=True), nullable=True),
    )

    # Partial index: only indexes rows still needing embedding.
    # Stays tiny as rows are processed (embedded rows are excluded).
    op.execute(
        """
        CREATE INDEX ix_text_blocks_unembedded
            ON text_blocks (document_id)
            WHERE embedded_at IS NULL
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_text_blocks_unembedded")
    op.drop_column("text_blocks", "embedded_at")
