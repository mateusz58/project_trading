"""Core application tests."""
import unittest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from app.config import ConfigManager
    from app.database import DatabaseManager
    from app.api_client import APIClient
except ImportError as e:
    print(f"Import error: {e}")
    print("Some modules may not be available yet")

class TestConfigManager(unittest.TestCase):
    """Test configuration management."""

    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager("config/config.yaml")

    def test_config_loading(self):
        """Test configuration loading."""
        self.assertIsNotNone(self.config_manager.config)
        self.assertIsInstance(self.config_manager.config.debug, bool)

    def test_config_get(self):
        """Test getting configuration values."""
        debug_value = self.config_manager.get('debug', True)
        self.assertIsInstance(debug_value, bool)

class TestDatabaseManager(unittest.TestCase):
    """Test database operations."""

    def setUp(self):
        """Set up test database."""
        self.db = DatabaseManager("data/test.db")
        self.db.create_tables()

    def test_database_connection(self):
        """Test database connection."""
        with self.db.get_connection() as conn:
            self.assertIsNotNone(conn)

if __name__ == "__main__":
    unittest.main()
