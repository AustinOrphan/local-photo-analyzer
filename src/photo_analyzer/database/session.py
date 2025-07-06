"""Database session management."""

from typing import AsyncGenerator, Generator
from contextlib import asynccontextmanager, contextmanager

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Session

from .engine import get_database_engine
from ..core.logger import get_logger

logger = get_logger(__name__)


@asynccontextmanager
async def get_async_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session context manager."""
    db_engine = get_database_engine()
    async with db_engine.get_session() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            raise


@contextmanager
def get_db_session() -> Generator[Session, None, None]:
    """Get sync database session context manager."""
    db_engine = get_database_engine()
    session_maker = db_engine.get_sync_session_maker()
    
    session = session_maker()
    try:
        yield session
        session.commit()
    except Exception as e:
        session.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        session.close()


# Dependency injection for FastAPI
async def get_db_dependency() -> AsyncGenerator[AsyncSession, None]:
    """Database dependency for FastAPI."""
    async with get_async_db_session() as session:
        yield session