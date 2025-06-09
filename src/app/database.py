"""Database integration with SQLite."""
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Database manager for application."""

    def __init__(self, db_path: str = "data/app.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.connection = None

    @contextmanager
    def get_connection(self):
        """Context manager for database connections."""
        conn = None
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row  # Enable dict-like access
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            if conn:
                conn.close()

    def execute_query(self, query: str, params: Union[tuple, dict] = None) -> List[Dict]:
        """Execute SELECT query and return results."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]

    def execute_command(self, command: str, params: Union[tuple, dict] = None) -> int:
        """Execute INSERT/UPDATE/DELETE command."""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(command, params)
            else:
                cursor.execute(command)
            conn.commit()
            return cursor.rowcount

    def create_tables(self):
        """Create application database tables."""
        schema = """
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            email TEXT UNIQUE NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            level TEXT NOT NULL,
            message TEXT NOT NULL,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """

        with self.get_connection() as conn:
            conn.executescript(schema)
            logger.info("Database tables created successfully")

# Global database instance
db = DatabaseManager()
