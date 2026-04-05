from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """
    SQLAlchemy declarative base.
    All ORM models must inherit from this class.
    """
    pass
