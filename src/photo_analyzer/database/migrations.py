"""Database migration management."""

import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

from sqlalchemy import text, MetaData, Table, Column, String, DateTime
from sqlalchemy.exc import OperationalError

from .engine import get_database_engine
from ..core.logger import get_logger

logger = get_logger(__name__)


class Migration:
    """Represents a single database migration."""
    
    def __init__(self, version: str, name: str, up_sql: str, down_sql: str = ""):
        self.version = version
        self.name = name
        self.up_sql = up_sql
        self.down_sql = down_sql
        self.applied_at: Optional[datetime] = None
    
    def __str__(self) -> str:
        return f"Migration {self.version}: {self.name}"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert migration to dictionary."""
        return {
            "version": self.version,
            "name": self.name,
            "up_sql": self.up_sql,
            "down_sql": self.down_sql,
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Migration":
        """Create migration from dictionary."""
        migration = cls(
            version=data["version"],
            name=data["name"],
            up_sql=data["up_sql"],
            down_sql=data.get("down_sql", ""),
        )
        if data.get("applied_at"):
            migration.applied_at = datetime.fromisoformat(data["applied_at"])
        return migration


class MigrationManager:
    """Manages database migrations."""
    
    def __init__(self, migrations_dir: Optional[Path] = None):
        self.db_engine = get_database_engine()
        self.migrations_dir = migrations_dir or Path(__file__).parent / "migrations"
        self.migrations_dir.mkdir(parents=True, exist_ok=True)
        self._migrations: List[Migration] = []
        self._load_migrations()
    
    def _load_migrations(self) -> None:
        """Load migrations from disk."""
        self._migrations = []
        
        # Load built-in migrations
        self._add_builtin_migrations()
        
        # Load custom migrations from files
        for migration_file in sorted(self.migrations_dir.glob("*.json")):
            try:
                with migration_file.open() as f:
                    data = json.load(f)
                    migration = Migration.from_dict(data)
                    self._migrations.append(migration)
                    logger.debug(f"Loaded migration: {migration}")
            except Exception as e:
                logger.error(f"Failed to load migration {migration_file}: {e}")
        
        # Sort by version
        self._migrations.sort(key=lambda m: m.version)
    
    def _add_builtin_migrations(self) -> None:
        """Add built-in migrations."""
        # Initial schema migration
        initial_migration = Migration(
            version="001",
            name="Initial schema",
            up_sql="""
            -- This migration creates the initial schema
            -- The actual table creation is handled by SQLAlchemy
            SELECT 1;
            """,
            down_sql="-- Initial migration cannot be rolled back"
        )
        self._migrations.append(initial_migration)
    
    async def _ensure_migration_table(self) -> None:
        """Ensure the migration tracking table exists."""
        engine = self.db_engine.get_async_engine()
        
        async with engine.begin() as conn:
            # Check if migration table exists
            table_exists = await conn.run_sync(
                lambda sync_conn: sync_conn.dialect.has_table(
                    sync_conn, "schema_migrations"
                )
            )
            
            if not table_exists:
                # Create migration table
                create_table_sql = """
                CREATE TABLE schema_migrations (
                    version VARCHAR(50) PRIMARY KEY,
                    name VARCHAR(200) NOT NULL,
                    applied_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
                )
                """
                await conn.execute(text(create_table_sql))
                logger.info("Created schema_migrations table")
    
    async def get_applied_migrations(self) -> List[str]:
        """Get list of applied migration versions."""
        await self._ensure_migration_table()
        
        engine = self.db_engine.get_async_engine()
        async with engine.begin() as conn:
            try:
                result = await conn.execute(
                    text("SELECT version FROM schema_migrations ORDER BY version")
                )
                return [row[0] for row in result.fetchall()]
            except OperationalError:
                # Table doesn't exist yet
                return []
    
    async def get_pending_migrations(self) -> List[Migration]:
        """Get list of pending migrations."""
        applied = await self.get_applied_migrations()
        return [m for m in self._migrations if m.version not in applied]
    
    async def apply_migration(self, migration: Migration) -> None:
        """Apply a single migration."""
        logger.info(f"Applying {migration}")
        
        engine = self.db_engine.get_async_engine()
        async with engine.begin() as conn:
            try:
                # Execute migration SQL
                if migration.up_sql.strip():
                    for statement in migration.up_sql.split(';'):
                        statement = statement.strip()
                        if statement:
                            await conn.execute(text(statement))
                
                # Record migration as applied
                await conn.execute(
                    text("""
                    INSERT INTO schema_migrations (version, name, applied_at)
                    VALUES (:version, :name, :applied_at)
                    """),
                    {
                        "version": migration.version,
                        "name": migration.name,
                        "applied_at": datetime.utcnow(),
                    }
                )
                
                logger.info(f"Applied {migration}")
                
            except Exception as e:
                logger.error(f"Failed to apply {migration}: {e}")
                raise
    
    async def rollback_migration(self, migration: Migration) -> None:
        """Rollback a single migration."""
        logger.info(f"Rolling back {migration}")
        
        if not migration.down_sql.strip():
            raise ValueError(f"Migration {migration.version} has no rollback SQL")
        
        engine = self.db_engine.get_async_engine()
        async with engine.begin() as conn:
            try:
                # Execute rollback SQL
                for statement in migration.down_sql.split(';'):
                    statement = statement.strip()
                    if statement:
                        await conn.execute(text(statement))
                
                # Remove migration record
                await conn.execute(
                    text("DELETE FROM schema_migrations WHERE version = :version"),
                    {"version": migration.version}
                )
                
                logger.info(f"Rolled back {migration}")
                
            except Exception as e:
                logger.error(f"Failed to rollback {migration}: {e}")
                raise
    
    async def migrate_up(self, target_version: Optional[str] = None) -> None:
        """Apply pending migrations up to target version."""
        await self._ensure_migration_table()
        
        pending = await self.get_pending_migrations()
        
        if target_version:
            pending = [m for m in pending if m.version <= target_version]
        
        if not pending:
            logger.info("No pending migrations")
            return
        
        logger.info(f"Applying {len(pending)} migrations")
        
        for migration in pending:
            await self.apply_migration(migration)
        
        logger.info("Migration complete")
    
    async def migrate_down(self, target_version: str) -> None:
        """Rollback migrations down to target version."""
        applied = await self.get_applied_migrations()
        
        # Find migrations to rollback (in reverse order)
        to_rollback = []
        for version in reversed(applied):
            if version > target_version:
                migration = next((m for m in self._migrations if m.version == version), None)
                if migration:
                    to_rollback.append(migration)
        
        if not to_rollback:
            logger.info("No migrations to rollback")
            return
        
        logger.info(f"Rolling back {len(to_rollback)} migrations")
        
        for migration in to_rollback:
            await self.rollback_migration(migration)
        
        logger.info("Rollback complete")
    
    async def create_tables(self) -> None:
        """Create all tables using SQLAlchemy."""
        await self.db_engine.create_all_tables()
    
    async def drop_tables(self) -> None:
        """Drop all tables."""
        await self.db_engine.drop_all_tables()
    
    async def reset_database(self) -> None:
        """Reset database (drop and recreate all tables)."""
        logger.warning("Resetting database - all data will be lost")
        await self.drop_tables()
        await self.create_tables()
        await self.migrate_up()
        logger.info("Database reset complete")
    
    async def status(self) -> Dict[str, Any]:
        """Get migration status."""
        applied = await self.get_applied_migrations()
        pending = await self.get_pending_migrations()
        
        return {
            "total_migrations": len(self._migrations),
            "applied_count": len(applied),
            "pending_count": len(pending),
            "applied_versions": applied,
            "pending_versions": [m.version for m in pending],
            "latest_applied": applied[-1] if applied else None,
        }
    
    def add_migration(self, version: str, name: str, up_sql: str, down_sql: str = "") -> None:
        """Add a new migration."""
        migration = Migration(version, name, up_sql, down_sql)
        
        # Save to file
        migration_file = self.migrations_dir / f"{version}_{name.replace(' ', '_').lower()}.json"
        with migration_file.open('w') as f:
            json.dump(migration.to_dict(), f, indent=2)
        
        # Add to memory
        self._migrations.append(migration)
        self._migrations.sort(key=lambda m: m.version)
        
        logger.info(f"Added migration: {migration}")


# Global migration manager instance
_migration_manager: Optional[MigrationManager] = None


def get_migration_manager() -> MigrationManager:
    """Get global migration manager instance."""
    global _migration_manager
    if _migration_manager is None:
        _migration_manager = MigrationManager()
    return _migration_manager