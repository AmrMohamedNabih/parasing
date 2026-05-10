"""Add edag_status, edag_built_at, edag_leaf_count to subjects table

Revision ID: 0005_add_edag_status
Revises: f40b887ec5e0
Create Date: 2026-05-10
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005_add_edag_status"
down_revision: Union[str, None] = "f40b887ec5e0"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # edag_status: tracks the graph build lifecycle for a subject
    op.add_column(
        "subjects",
        sa.Column(
            "edag_status",
            sa.String(20),
            nullable=False,
            server_default="none",
        ),
    )
    # Add CHECK constraint separately (SQLAlchemy Column doesn't emit CHECK for String)
    op.execute(
        """
        ALTER TABLE subjects
        ADD CONSTRAINT ck_subjects_edag_status
        CHECK (edag_status IN ('none', 'building', 'ready', 'failed'))
        """
    )

    # edag_built_at: timestamp when the last successful build completed
    op.add_column(
        "subjects",
        sa.Column("edag_built_at", sa.DateTime(timezone=True), nullable=True),
    )

    # edag_leaf_count: number of leaves (chunks) in the built graph
    op.add_column(
        "subjects",
        sa.Column("edag_leaf_count", sa.Integer, nullable=True),
    )

    # Index for fast lookup of subjects needing a build or ready for EDAG queries
    op.create_index(
        "ix_subjects_edag_status",
        "subjects",
        ["edag_status"],
    )


def downgrade() -> None:
    op.drop_index("ix_subjects_edag_status", table_name="subjects")
    op.drop_column("subjects", "edag_leaf_count")
    op.drop_column("subjects", "edag_built_at")
    op.execute("ALTER TABLE subjects DROP CONSTRAINT IF EXISTS ck_subjects_edag_status")
    op.drop_column("subjects", "edag_status")
