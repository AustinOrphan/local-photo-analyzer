"""Database engine and connection management."""

import asyncio
from pathlib import Path
from typing import Optional, AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy import create_engine, inspect, event
from sqlalchemy.ext.asyncio import (
    create_async_engine, AsyncSession, AsyncEngine, async_sessionmaker
)
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool

from ..core.config import get_config
from ..core.logger import get_logger
from ..models.base import Base

logger = get_logger(__name__)


class DatabaseEngine:
    """Database engine management and initialization."""
    
    def __init__(self, config=None):
        self.config = config or get_config()
        self._engine: Optional[AsyncEngine] = None
        self._session_maker: Optional[async_sessionmaker] = None
        self._sync_engine = None
        self._sync_session_maker = None
    
    def get_database_url(self, async_driver: bool = True) -> str:
        """Get database URL based on configuration."""
        db_config = self.config.database
        
        if db_config.type == "sqlite":
            # Ensure database directory exists
            db_path = Path(db_config.path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            
            if async_driver:
                return f"sqlite+aiosqlite:///{db_path}"
            else:
                return f"sqlite:///{db_path}"
        
        elif db_config.type == "postgresql":
            driver = "asyncpg" if async_driver else "psycopg2"
            return (
                f"postgresql+{driver}://{db_config.username}:{db_config.password}"
                f"@{db_config.host}:{db_config.port}/{db_config.database}"
            )
        
        else:
            raise ValueError(f"Unsupported database type: {db_config.type}")
    
    def get_async_engine(self) -> AsyncEngine:
        """Get async database engine."""
        if self._engine is None:
            url = self.get_database_url(async_driver=True)
            
            # Engine configuration
            engine_kwargs = {
                "echo": self.config.database.echo,
                "pool_pre_ping": True,
            }
            
            if self.config.database.type == "sqlite":
                # SQLite specific configuration
                engine_kwargs.update({
                    "poolclass": StaticPool,
                    "connect_args": {
                        "check_same_thread": False,
                        "timeout": 30,
                    }
                })
            
            self._engine = create_async_engine(url, **engine_kwargs)
            
            # Create session maker
            self._session_maker = async_sessionmaker(
                bind=self._engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            logger.info(f"Created async database engine: {url}")
        
        return self._engine
    
    def get_sync_engine(self):
        """Get synchronous database engine for migrations."""
        if self._sync_engine is None:
            url = self.get_database_url(async_driver=False)
            
            engine_kwargs = {
                "echo": self.config.database.echo,
                "pool_pre_ping": True,
            }
            
            if self.config.database.type == "sqlite":
                engine_kwargs.update({
                    "poolclass": StaticPool,
                    "connect_args": {
                        "check_same_thread": False,
                        "timeout": 30,
                    }
                })
            
            self._sync_engine = create_engine(url, **engine_kwargs)
            self._sync_session_maker = sessionmaker(bind=self._sync_engine)
            
            logger.info(f"Created sync database engine: {url}")
        
        return self._sync_engine
    
    def get_session_maker(self) -> async_sessionmaker:
        """Get async session maker."""
        if self._session_maker is None:
            self.get_async_engine()  # This will create the session maker
        return self._session_maker
    
    def get_sync_session_maker(self) -> sessionmaker:
        """Get sync session maker."""
        if self._sync_session_maker is None:
            self.get_sync_engine()  # This will create the session maker
        return self._sync_session_maker
    
    @asynccontextmanager
    async def get_session(self) -> AsyncGenerator[AsyncSession, None]:
        """Get async database session context manager."""
        session_maker = self.get_session_maker()
        async with session_maker() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def create_all_tables(self) -> None:
        """Create all database tables."""
        engine = self.get_async_engine()
        
        # Import all models to ensure they're registered
        from ..models import photo, analysis, organization
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Created all database tables")
    
    async def drop_all_tables(self) -> None:
        """Drop all database tables."""
        engine = self.get_async_engine()
        
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.drop_all)
        
        logger.info("Dropped all database tables")
    
    async def table_exists(self, table_name: str) -> bool:
        """Check if a table exists."""
        engine = self.get_async_engine()
        
        async with engine.begin() as conn:
            result = await conn.run_sync(
                lambda sync_conn: inspect(sync_conn).has_table(table_name)
            )
            return result
    
    async def get_table_names(self) -> list[str]:
        """Get all table names."""
        engine = self.get_async_engine()
        
        async with engine.begin() as conn:
            result = await conn.run_sync(
                lambda sync_conn: inspect(sync_conn).get_table_names()
            )
            return result
    
    async def close(self) -> None:
        """Close database connections."""
        if self._engine:
            await self._engine.dispose()
            self._engine = None
            self._session_maker = None
            logger.info("Closed async database connections")
        
        if self._sync_engine:
            self._sync_engine.dispose()
            self._sync_engine = None
            self._sync_session_maker = None
            logger.info("Closed sync database connections")


# Global database engine instance
_db_engine: Optional[DatabaseEngine] = None


def get_database_engine() -> DatabaseEngine:
    """Get global database engine instance."""
    global _db_engine
    if _db_engine is None:
        _db_engine = DatabaseEngine()
    return _db_engine


# Enable WAL mode for SQLite (better concurrency)
@event.listens_for(create_engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set SQLite pragmas for better performance."""
    if 'sqlite' in str(dbapi_connection):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA synchronous=NORMAL")
        cursor.execute("PRAGMA cache_size=10000")
        cursor.execute("PRAGMA temp_store=MEMORY")
        cursor.close()