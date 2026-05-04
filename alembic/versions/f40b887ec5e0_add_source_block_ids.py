"""add_source_block_ids

Revision ID: f40b887ec5e0
Revises: 427a05b3b3c9
Create Date: 2026-05-04 16:35:43.764623

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision: str = 'f40b887ec5e0'
down_revision: Union[str, None] = '427a05b3b3c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column('text_blocks', sa.Column('source_block_ids', postgresql.JSONB(astext_type=sa.Text()), nullable=True))


def downgrade() -> None:
    op.drop_column('text_blocks', 'source_block_ids')
