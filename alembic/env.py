"""
Alembic Environment Configuration
-----------------------------------
Reads DATABASE_URL from pydantic-settings so it stays in sync with the app.
Uses a synchronous psycopg2 connection (Alembic's requirement) by replacing
the asyncpg driver string with psycopg2 at runtime.
"""

import sys
from logging.config import fileConfig
from pathlib import Path

from alembic import context
from sqlalchemy import engine_from_config, pool

# ============================================================
# Make project root importable
# ============================================================
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import settings (reads .env)
from app.core.config import settings

# Import Base + all models so Alembic can detect schema changes
from app.db.base import Base
from app.db.models import User, Subject, Document, DocumentPage, TextBlock, QuestionHistory  # noqa: F401

# ============================================================
# Alembic config object
# ============================================================
config = context.config

# Override sqlalchemy.url with the value from settings
# Replace asyncpg driver with psycopg2 for synchronous Alembic migrations
sync_url = settings.DATABASE_URL.replace(
    "postgresql+asyncpg://", "postgresql+psycopg2://"
)
config.set_main_option("sqlalchemy.url", sync_url)

# Set up logging from the ini file
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Target metadata for autogenerate
target_metadata = Base.metadata


# ============================================================
# Migration runners
# ============================================================

def run_migrations_offline() -> None:
    """Emit SQL to stdout without connecting to the DB."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Connect to the DB and run migrations."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
