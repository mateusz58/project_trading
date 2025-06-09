"""Configuration management."""
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class AppConfig:
    """Application configuration."""
    debug: bool = False
    log_level: str = "INFO"
    database_url: str = "sqlite:///data/app.db"
    api_key: Optional[str] = None
    host: str = "localhost"
    port: int = 8000

class ConfigManager:
    """Manage application configuration."""

    def __init__(self, config_file: str = "config/config.yaml"):
        self.config_file = Path(config_file)
        self.config = AppConfig()
        self.load_config()

    def load_config(self):
        """Load configuration from file and environment."""
        # Load from YAML file
        if self.config_file.exists():
            with open(self.config_file, 'r') as f:
                yaml_config = yaml.safe_load(f)
                if yaml_config:
                    for key, value in yaml_config.items():
                        if hasattr(self.config, key):
                            setattr(self.config, key, value)

        # Override with environment variables
        self.config.debug = os.getenv('DEBUG', str(self.config.debug)).lower() == 'true'
        self.config.log_level = os.getenv('LOG_LEVEL', self.config.log_level)
        self.config.database_url = os.getenv('DATABASE_URL', self.config.database_url)
        self.config.api_key = os.getenv('API_KEY', self.config.api_key)
        self.config.host = os.getenv('HOST', self.config.host)
        self.config.port = int(os.getenv('PORT', str(self.config.port)))

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return getattr(self.config, key, default)

    def save_config(self):
        """Save current configuration to file."""
        self.config_file.parent.mkdir(exist_ok=True)
        config_dict = {
            'debug': self.config.debug,
            'log_level': self.config.log_level,
            'database_url': self.config.database_url,
            'host': self.config.host,
            'port': self.config.port
        }

        with open(self.config_file, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)

# Global config instance
config_manager = ConfigManager()
