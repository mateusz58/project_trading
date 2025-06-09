# Advanced Python Project Enhancement Script
# Professional structure with src/app/ layout
FEATURE_KEYS=(logging config database api_client testing docker monitoring async cli web_api)

set -eo pipefail

# Colors for better UX
readonly RED='\\033[0;31m'
readonly GREEN='\\033[0;32m'
readonly YELLOW='\\033[1;33m'
readonly BLUE='\\033[0;34m'
readonly PURPLE='\\033[0;35m'
readonly CYAN='\\033[0;36m'
readonly NC='\\033[0m'

# Get current directory name as project name
PROJECT_NAME=$(basename "$PWD")

# Feature management system
declare -A AVAILABLE_FEATURES
AVAILABLE_FEATURES[logging]="Advanced logging setup"
AVAILABLE_FEATURES[config]="Configuration management"
AVAILABLE_FEATURES[database]="Database integration"
AVAILABLE_FEATURES[api_client]="API client utilities"
AVAILABLE_FEATURES[testing]="Testing framework"
AVAILABLE_FEATURES[docker]="Docker support"
AVAILABLE_FEATURES[monitoring]="Monitoring & metrics"
AVAILABLE_FEATURES[async]="Async/await support"
AVAILABLE_FEATURES[cli]="CLI interface"
AVAILABLE_FEATURES[web_api]="Web API framework"

# Project configuration
declare -A PROJECT_CONFIG
PROJECT_CONFIG[name]="$PROJECT_NAME"
PROJECT_CONFIG[type]=""
PROJECT_CONFIG[features]=""
PROJECT_CONFIG[python_version]="3.11"

# Utility functions
log_info()    { echo -e "${CYAN}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $1"; }

command_exists() {
    command -v "$1" >/dev/null 2>&1
}

safe_create_file() {
    local file_path="$1"
    local content="$2"
    
    # Create directory if it doesn't exist
    mkdir -p "$(dirname "$file_path")"
    
    if [ -f "$file_path" ]; then
        log_warning "File $file_path already exists, skipping..."
        return 1
    else
        echo "$content" > "$file_path"
        log_success "Created: $file_path"
        return 0
    fi
}

safe_create_dir() {
    local dir_path="$1"
    if [ -d "$dir_path" ]; then
        log_info "Directory $dir_path already exists"
        return 0
    else
        mkdir -p "$dir_path"
        log_success "Created directory: $dir_path"
        return 0
    fi
}

show_banner() {
    echo -e "${PURPLE}"
    cat << "EOF"
╔═══════════════════════════════════════════════════════════╗
║           🐍 Advanced Python Project Generator            ║
║                Professional Structure                      ║
╚═══════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"
}

check_manjaro_dependencies() {
    log_info "Checking system dependencies..."
    
    local deps=("python" "git" "python-pip")
    local missing_deps=()
    
    for dep in "${deps[@]}"; do
        if ! pacman -Qi "$dep" >/dev/null 2>&1; then
            missing_deps+=("$dep")
        fi
    done
    
    if [ ${#missing_deps[@]} -gt 0 ]; then
        log_warning "Missing dependencies: ${missing_deps[*]}"
        read -p "Install missing dependencies? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            sudo pacman -S "${missing_deps[@]}"
        else
            log_error "Cannot proceed without required dependencies"
            exit 1
        fi
    fi
    
    log_success "All dependencies satisfied"
}

check_existing_structure() {
    log_info "Analyzing existing project structure..."
    
    echo -e "\\n${CYAN}Current project: ${PROJECT_NAME}${NC}"
    echo -e "${CYAN}Existing files and directories:${NC}"
    
    # Show current structure
    if command -v tree >/dev/null 2>&1; then
        tree -L 2 -a
    else
        find . -maxdepth 2 -type f -o -type d | head -20
    fi
    
    echo
    read -p "Continue enhancing this project? (Y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Nn]$ ]]; then
        log_info "Exiting..."
        exit 0
    fi
}

create_basic_structure() {
    log_info "Creating professional project structure..."
    
    # Create main directories using professional structure
    safe_create_dir "src/app"
    safe_create_dir "config"
    safe_create_dir "tests"
    safe_create_dir "scripts"
    safe_create_dir "docs"
    
    # Create __init__.py files to make packages
    safe_create_file "src/__init__.py" ""
    safe_create_file "src/app/__init__.py" ""
    safe_create_file "tests/__init__.py" ""
}

check_existing_features() {
    log_info "Checking existing features..."
    
    declare -A existing_features
    
    # Check for existing features using correct paths
    [ -f "src/app/logger.py" ] && existing_features[logging]=1
    [ -f "config/config.yaml" ] && existing_features[config]=1
    [ -f "src/app/database.py" ] && existing_features[database]=1
    [ -f "src/app/api_client.py" ] && existing_features[api_client]=1
    [ -f "tests/test_app.py" ] && existing_features[testing]=1
    [ -f "Dockerfile" ] && existing_features[docker]=1
    [ -f "src/app/monitoring.py" ] && existing_features[monitoring]=1
    [ -f "src/app/cli.py" ] && existing_features[cli]=1
    [ -f "src/app/web_api.py" ] && existing_features[web_api]=1
    [ -f "src/app/async_utils.py" ] && existing_features[async]=1
    
    if [ ${#existing_features[@]} -gt 0 ]; then
        echo -e "\\n${GREEN}Existing features detected:${NC}"
        for feature in "${!existing_features[@]}"; do
            if [[ -v AVAILABLE_FEATURES[$feature] ]]; then
                echo -e "  ✓ ${AVAILABLE_FEATURES[$feature]}"
            else
                echo -e "  ✓ ${feature} (Description not available)"
            fi
        done
    else
        echo -e "\\n${YELLOW}No existing features detected${NC}"
    fi
    
    echo
}

show_available_features() {
    echo -e "\\n${CYAN}Available features to add:${NC}"
    local counter=1
    
    # Create ordered array of feature keys
    local ordered_features=("logging" "config" "database" "api_client" "testing" "docker" "monitoring" "async" "cli" "web_api")
    
    for feature_key in "${ordered_features[@]}"; do
        # Check if feature already exists using correct paths
        local exists=false
        case $feature_key in
            logging) [ -f "src/app/logger.py" ] && exists=true ;;
            config) [ -f "config/config.yaml" ] && exists=true ;;
            database) [ -f "src/app/database.py" ] && exists=true ;;
            api_client) [ -f "src/app/api_client.py" ] && exists=true ;;
            testing) [ -f "tests/test_app.py" ] && exists=true ;;
            docker) [ -f "Dockerfile" ] && exists=true ;;
            monitoring) [ -f "src/app/monitoring.py" ] && exists=true ;;
            async) [ -f "src/app/async_utils.py" ] && exists=true ;;
            cli) [ -f "src/app/cli.py" ] && exists=true ;;
            web_api) [ -f "src/app/web_api.py" ] && exists=true ;;
        esac
        
        if [ "$exists" = true ]; then
            echo -e "${counter}) ${AVAILABLE_FEATURES[$feature_key]} ${GREEN}[INSTALLED]${NC}"
        else
            echo -e "${counter}) ${AVAILABLE_FEATURES[$feature_key]}"
        fi
        
        ((counter++))
    done
}

select_features_to_add() {
    show_available_features
    
    echo -e "\\n${YELLOW}Enter feature numbers to add (e.g., 1,3,5) or 'all' for all missing features:${NC}"
    read -p "Selection: " selection
    
    if [ "$selection" = "all" ]; then
        PROJECT_CONFIG[features]="all"
    else
        PROJECT_CONFIG[features]="$selection"
    fi
}

# Feature implementation functions
add_logging_feature() {
    log_info "Adding advanced logging feature..."
    
    safe_create_file "src/app/logger.py" "$(cat << 'EOF'
\"\"\"Advanced logging configuration.\"\"\"
import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional

def setup_logger(
    name: str = __name__,
    level: str = "INFO",
    log_file: Optional[str] = None,
    max_bytes: int = 10485760,  # 10MB
    backup_count: int = 5
) -> logging.Logger:
    \"\"\"Setup advanced logger with file rotation.\"\"\"
    
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # File handler with rotation
    if log_file:
        log_path = Path("logs")
        log_path.mkdir(exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_path / log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        )
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)
    
    return logger

# Default logger instance
logger = setup_logger(__name__, log_file="app.log")
EOF
)"
    
    safe_create_dir "logs"
    safe_create_file "logs/.gitkeep" ""
}

add_config_feature() {
    log_info "Adding configuration management feature..."
    
    safe_create_file "src/app/config.py" "$(cat << 'EOF'
\"\"\"Configuration management.\"\"\"
import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

@dataclass
class AppConfig:
    \"\"\"Application configuration.\"\"\"
    debug: bool = False
    log_level: str = "INFO"
    database_url: str = "sqlite:///data/app.db"
    api_key: Optional[str] = None
    host: str = "localhost"
    port: int = 8000

class ConfigManager:
    \"\"\"Manage application configuration.\"\"\"
    
    def __init__(self, config_file: str = "config/config.yaml"):
        self.config_file = Path(config_file)
        self.config = AppConfig()
        self.load_config()
    
    def load_config(self):
        \"\"\"Load configuration from file and environment.\"\"\"
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
        \"\"\"Get configuration value.\"\"\"
        return getattr(self.config, key, default)
    
    def save_config(self):
        \"\"\"Save current configuration to file.\"\"\"
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
EOF
)"
    
    safe_create_file "config/config.yaml" "$(cat << 'EOF'
# Application Configuration
debug: false
log_level: INFO
host: localhost
port: 8000

# Database
database_url: sqlite:///data/app.db

# External APIs
# api_key: your_api_key_here
EOF
)"
    
    safe_create_file ".env.example" "$(cat << 'EOF'
# Environment Variables Template
# Copy to .env and fill in your values

DEBUG=false
LOG_LEVEL=INFO
HOST=localhost
PORT=8000
DATABASE_URL=sqlite:///data/app.db
API_KEY=your_api_key_here
EOF
)"
}

add_database_feature() {
    log_info "Adding database integration feature..."
    
    safe_create_file "src/app/database.py" "$(cat << 'EOF'
\"\"\"Database integration with SQLite.\"\"\"
import sqlite3
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from contextlib import contextmanager
import logging

logger = logging.getLogger(__name__)

class DatabaseManager:
    \"\"\"Database manager for application.\"\"\"
    
    def __init__(self, db_path: str = "data/app.db"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(exist_ok=True)
        self.connection = None
    
    @contextmanager
    def get_connection(self):
        \"\"\"Context manager for database connections.\"\"\"
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
        \"\"\"Execute SELECT query and return results.\"\"\"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            return [dict(row) for row in cursor.fetchall()]
    
    def execute_command(self, command: str, params: Union[tuple, dict] = None) -> int:
        \"\"\"Execute INSERT/UPDATE/DELETE command.\"\"\"
        with self.get_connection() as conn:
            cursor = conn.cursor()
            if params:
                cursor.execute(command, params)
            else:
                cursor.execute(command)
            conn.commit()
            return cursor.rowcount
    
    def create_tables(self):
        \"\"\"Create application database tables.\"\"\"
        schema = \"\"\"
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
        \"\"\"
        
        with self.get_connection() as conn:
            conn.executescript(schema)
            logger.info("Database tables created successfully")

# Global database instance
db = DatabaseManager()
EOF
)"
    
    safe_create_dir "data"
    safe_create_file "data/.gitkeep" ""
}

add_api_client_feature() {
    log_info "Adding API client utilities feature..."
    
    safe_create_file "src/app/api_client.py" "$(cat << 'EOF'
\"\"\"Advanced API client utilities.\"\"\"
import time
from typing import Dict, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import logging

logger = logging.getLogger(__name__)

class APIClient:
    \"\"\"Robust API client with retry logic and rate limiting.\"\"\"
    
    def __init__(
        self,
        base_url: str,
        api_key: Optional[str] = None,
        timeout: int = 30,
        max_retries: int = 3,
        rate_limit: float = 1.0  # requests per second
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout
        self.rate_limit = rate_limit
        self.last_request_time = 0
        
        # Setup session with retry strategy
        self.session = requests.Session()
        
        retry_strategy = Retry(
            total=max_retries,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Set headers
        if api_key:
            self.session.headers.update({
                "Authorization": f"Bearer {api_key}",
                "User-Agent": "PythonApp/1.0"
            })
    
    def _rate_limit(self):
        \"\"\"Implement rate limiting.\"\"\"
        if self.rate_limit > 0:
            time_since_last = time.time() - self.last_request_time
            min_interval = 1.0 / self.rate_limit
            
            if time_since_last < min_interval:
                sleep_time = min_interval - time_since_last
                time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        params: Optional[Dict] = None,
        data: Optional[Dict] = None
    ) -> Dict[str, Any]:
        \"\"\"Make HTTP request with error handling.\"\"\"
        self._rate_limit()
        
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        try:
            response = self.session.request(
                method=method,
                url=url,
                params=params,
                json=data,
                timeout=self.timeout
            )
            
            response.raise_for_status()
            return response.json()
                
        except requests.exceptions.RequestException as e:
            logger.error(f"API request failed: {method} {url} - {e}")
            raise APIError(f"Request failed: {e}") from e
    
    def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        \"\"\"Make GET request.\"\"\"
        return self._make_request("GET", endpoint, params=params)
    
    def post(self, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        \"\"\"Make POST request.\"\"\"
        return self._make_request("POST", endpoint, data=data)

class APIError(Exception):
    \"\"\"Custom API exception.\"\"\"
    pass
EOF
)"
}

add_testing_feature() {
    log_info "Adding comprehensive testing framework..."
    
    safe_create_file "tests/test_app.py" "$(cat << 'EOF'
\"\"\"Core application tests.\"\"\"
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
    \"\"\"Test configuration management.\"\"\"
    
    def setUp(self):
        \"\"\"Set up test fixtures.\"\"\"
        self.config_manager = ConfigManager("config/config.yaml")
    
    def test_config_loading(self):
        \"\"\"Test configuration loading.\"\"\"
        self.assertIsNotNone(self.config_manager.config)
        self.assertIsInstance(self.config_manager.config.debug, bool)
    
    def test_config_get(self):
        \"\"\"Test getting configuration values.\"\"\"
        debug_value = self.config_manager.get('debug', True)
        self.assertIsInstance(debug_value, bool)

class TestDatabaseManager(unittest.TestCase):
    \"\"\"Test database operations.\"\"\"
    
    def setUp(self):
        \"\"\"Set up test database.\"\"\"
        self.db = DatabaseManager("data/test.db")
        self.db.create_tables()
    
    def test_database_connection(self):
        \"\"\"Test database connection.\"\"\"
        with self.db.get_connection() as conn:
            self.assertIsNotNone(conn)

if __name__ == "__main__":
    unittest.main()
EOF
)"
    
    safe_create_file "scripts/run_tests.sh" "$(cat << 'EOF'
#!/bin/bash
# Test runner script

echo "Running Application Tests..."

# Set PYTHONPATH to include src directory
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Run unit tests
echo "=== Unit Tests ==="
python -m pytest tests/ -v --tb=short

# Run with coverage if available
if command -v coverage >/dev/null 2>&1; then
    echo "=== Coverage Report ==="
    coverage run -m pytest tests/
    coverage report -m
    coverage html
    echo "HTML coverage report generated in htmlcov/"
fi

echo "Tests completed!"
EOF
)"
    
    chmod +x "scripts/run_tests.sh"
}

add_docker_feature() {
    log_info "Adding Docker support..."
    
    safe_create_file "Dockerfile" "$(cat << 'EOF'
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY config/ ./config/

# Create necessary directories
RUN mkdir -p logs data

# Set Python path
ENV PYTHONPATH=/app/src

# Run the application
CMD ["python", "-m", "app.main"]
EOF
)"
    
    safe_create_file "docker-compose.yml" "$(cat << 'EOF'
version: '3.8'

services:
  app:
    build: .
    container_name: python-app
    volumes:
      - ./data:/app/data
      - ./logs:/app/logs
      - ./config:/app/config
    environment:
      - PYTHONPATH=/app/src
    restart: unless-stopped
    ports:
      - "8000:8000"
EOF
)"
}

add_monitoring_feature() {
    log_info "Adding monitoring & metrics feature..."
    
    safe_create_file "src/app/monitoring.py" "$(cat << 'EOF'
\"\"\"Application monitoring and metrics.\"\"\"
import time
import psutil
from typing import Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    \"\"\"System metrics data class.\"\"\"
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    timestamp: float

class MonitoringManager:
    \"\"\"Application monitoring manager.\"\"\"
    
    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0
    
    def get_system_metrics(self) -> SystemMetrics:
        \"\"\"Get current system metrics.\"\"\"
        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            timestamp=time.time()
        )
    
    def get_app_metrics(self) -> Dict[str, Any]:
        \"\"\"Get application metrics.\"\"\"
        uptime = time.time() - self.start_time
        return {
            'uptime_seconds': uptime,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1)
        }
    
    def increment_request_count(self):
        \"\"\"Increment request counter.\"\"\"
        self.request_count += 1
    
    def increment_error_count(self):
        \"\"\"Increment error counter.\"\"\"
        self.error_count += 1
    
    def health_check(self) -> Dict[str, Any]:
        \"\"\"Perform health check.\"\"\"
        try:
            system_metrics = self.get_system_metrics()
            app_metrics = self.get_app_metrics()
            
            # Simple health check logic
            is_healthy = (
                system_metrics.cpu_percent < 90 and
                system_metrics.memory_percent < 90 and
                system_metrics.disk_usage < 90
            )
            
            return {
                'status': 'healthy' if is_healthy else 'unhealthy',
                'system': system_metrics.__dict__,
                'application': app_metrics,
                'timestamp': time.time()
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'timestamp': time.time()
            }

# Global monitoring instance
monitor = MonitoringManager()
EOF
)"
}

add_async_feature() {
    log_info "Adding async/await support..."
    
    safe_create_file "src/app/async_utils.py" "$(cat << 'EOF'
\"\"\"Async utilities and helpers.\"\"\"
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class AsyncAPIClient:
    \"\"\"Async API client for concurrent requests.\"\"\"
    
    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session = None
    
    async def __aenter__(self):
        \"\"\"Async context manager entry.\"\"\"
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        \"\"\"Async context manager exit.\"\"\"
        if self.session:
            await self.session.close()
    
    async def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        \"\"\"Make async GET request.\"\"\"
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        async with self.session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()
    
    async def post(self, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        \"\"\"Make async POST request.\"\"\"
        url = f"{self.base_url}/{endpoint.lstrip('/')}"
        
        async with self.session.post(url, json=data) as response:
            response.raise_for_status()
            return await response.json()

async def fetch_multiple_urls(urls: List[str]) -> List[Dict[str, Any]]:
    \"\"\"Fetch multiple URLs concurrently.\"\"\"
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            task = asyncio.create_task(fetch_url(session, url))
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

async def fetch_url(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
    \"\"\"Fetch single URL.\"\"\"
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return {"error": str(e), "url": url}

class AsyncTaskManager:
    \"\"\"Manage async background tasks.\"\"\"
    
    def __init__(self):
        self.tasks = []
    
    def add_task(self, coro):
        \"\"\"Add a background task.\"\"\"
        task = asyncio.create_task(coro)
        self.tasks.append(task)
        return task
    
    async def wait_all(self):
        \"\"\"Wait for all tasks to complete.\"\"\"
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
    
    def cancel_all(self):
        \"\"\"Cancel all running tasks.\"\"\"
        for task in self.tasks:
            if not task.done():
                task.cancel()

# Global task manager
task_manager = AsyncTaskManager()
EOF
)"
}

add_cli_feature() {
    log_info "Adding CLI interface..."
    
    safe_create_file "src/app/cli.py" "$(cat << 'EOF'
\"\"\"Command Line Interface for the application.\"\"\"
import argparse
import sys
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class CLIManager:
    \"\"\"Command Line Interface manager.\"\"\"
    
    def __init__(self):
        self.parser = argparse.ArgumentParser(
            description="Python Application CLI",
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        self.setup_arguments()
    
    def setup_arguments(self):
        \"\"\"Setup CLI arguments.\"\"\"
        # Global options
        self.parser.add_argument(
            '--verbose', '-v',
            action='store_true',
            help='Enable verbose logging'
        )
        
        self.parser.add_argument(
            '--config', '-c',
            type=str,
            default='config/config.yaml',
            help='Configuration file path'
        )
        
        # Subcommands
        subparsers = self.parser.add_subparsers(dest='command', help='Available commands')
        
        # Run command
        run_parser = subparsers.add_parser('run', help='Run the application')
        run_parser.add_argument(
            '--port', '-p',
            type=int,
            default=8000,
            help='Port to run on'
        )
        
        # Database commands
        db_parser = subparsers.add_parser('db', help='Database operations')
        db_subparsers = db_parser.add_subparsers(dest='db_command')
        
        db_subparsers.add_parser('init', help='Initialize database')
        db_subparsers.add_parser('migrate', help='Run database migrations')
        
        # Test command
        test_parser = subparsers.add_parser('test', help='Run tests')
        test_parser.add_argument(
            '--coverage',
            action='store_true',
            help='Run with coverage report'
        )
    
    def parse_args(self, args: Optional[List[str]] = None) -> argparse.Namespace:
        \"\"\"Parse command line arguments.\"\"\"
        return self.parser.parse_args(args)
    
    def handle_command(self, args: argparse.Namespace):
        \"\"\"Handle parsed command.\"\"\"
        if args.verbose:
            logging.basicConfig(level=logging.DEBUG)
        
        if args.command == 'run':
            self.run_application(args)
        elif args.command == 'db':
            self.handle_db_command(args)
        elif args.command == 'test':
            self.run_tests(args)
        else:
            self.parser.print_help()
    
    def run_application(self, args: argparse.Namespace):
        \"\"\"Run the main application.\"\"\"
        logger.info(f"Starting application on port {args.port}")
        # Import and run your main application here
        print(f"Application would start on port {args.port}")
    
    def handle_db_command(self, args: argparse.Namespace):
        \"\"\"Handle database commands.\"\"\"
        if args.db_command == 'init':
            logger.info("Initializing database...")
            # Initialize database here
            print("Database initialized")
        elif args.db_command == 'migrate':
            logger.info("Running migrations...")
            # Run migrations here
            print("Migrations completed")
    
    def run_tests(self, args: argparse.Namespace):
        \"\"\"Run tests.\"\"\"
        import subprocess
        
        cmd = ['python', '-m', 'pytest', 'tests/']
        if args.coverage:
            cmd = ['coverage', 'run', '-m', 'pytest', 'tests/']
        
        try:
            subprocess.run(cmd, check=True)
            if args.coverage:
                subprocess.run(['coverage', 'report'], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Tests failed: {e}")
            sys.exit(1)

def main():
    \"\"\"Main CLI entry point.\"\"\"
    cli = CLIManager()
    args = cli.parse_args()
    cli.handle_command(args)

if __name__ == '__main__':
    main()
EOF
)"
}

add_web_api_feature() {
    log_info "Adding Web API framework..."
    
    safe_create_file "src/app/web_api.py" "$(cat << 'EOF'
\"\"\"Web API using FastAPI.\"\"\"
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Pydantic models
class HealthResponse(BaseModel):
    status: str
    timestamp: float
    version: str = "1.0.0"

class APIResponse(BaseModel):
    success: bool
    data: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

# FastAPI app instance
app = FastAPI(
    title="Python Application API",
    description="RESTful API for Python application",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dependency injection
async def get_current_user():
    \"\"\"Get current user (placeholder for authentication).\"\"\"
    # Implement your authentication logic here
    return {"user_id": 1, "username": "demo"}

# Routes
@app.get("/", response_model=APIResponse)
async def root():
    \"\"\"Root endpoint.\"\"\"
    return APIResponse(
        success=True,
        data={"message": "Welcome to Python Application API"},
        message="API is running"
    )

@app.get("/health", response_model=HealthResponse)
async def health_check():
    \"\"\"Health check endpoint.\"\"\"
    import time
    return HealthResponse(
        status="healthy",
        timestamp=time.time()
    )

@app.get("/api/v1/status", response_model=APIResponse)
async def get_status():
    \"\"\"Get application status.\"\"\"
    try:
        # Add your status logic here
        status_data = {
            "uptime": "1h 30m",
            "requests_processed": 1234,
            "errors": 0
        }
        
        return APIResponse(
            success=True,
            data=status_data,
            message="Status retrieved successfully"
        )
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/v1/process", response_model=APIResponse)
async def process_data(
    data: Dict[str, Any],
    current_user: dict = Depends(get_current_user)
):
    \"\"\"Process data endpoint.\"\"\"
    try:
        # Add your data processing logic here
        processed_data = {
            "input": data,
            "processed_by": current_user["username"],
            "result": "Data processed successfully"
        }
        
        return APIResponse(
            success=True,
            data=processed_data,
            message="Data processed successfully"
        )
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        raise HTTPException(status_code=500, detail="Processing failed")

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return APIResponse(
        success=False,
        message="Endpoint not found"
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return APIResponse(
        success=False,
        message="Internal server error"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
EOF
)"
}

# Main execution flow
main() {
    show_banner
    
    # Check system and existing structure
    check_manjaro_dependencies
    check_existing_structure
    check_existing_features
    
    # Interactive feature selection
    echo -e "\\n${CYAN}=== Project Enhancement Menu ===${NC}"
    echo "1) Add new features"
    echo "2) Update existing features"
    echo "3) Show project status"
    echo "4) Exit"
    
    read -p "Choose option (1-4): " -n 1 -r main_choice
    echo
    
    case $main_choice in
        1)
            select_features_to_add
            implement_selected_features
            ;;
        2)
            echo -e "${YELLOW}Feature updates not implemented yet${NC}"
            ;;
        3)
            show_project_status
            ;;
        4)
            log_info "Goodbye!"
            exit 0
            ;;
        *)
            log_error "Invalid choice"
            exit 1
            ;;
    esac
}

implement_selected_features() {
    log_info "Implementing selected features..."

    create_basic_structure

    if [ "${PROJECT_CONFIG[features]}" = "all" ]; then
        for feature_key in "${FEATURE_KEYS[@]}"; do
            "add_${feature_key}_feature"
        done
    else
        IFS=',' read -ra FEATURES <<< "${PROJECT_CONFIG[features]}"
        for feature_num in "${FEATURES[@]}"; do
            feature_num=$(echo "$feature_num" | tr -d ' ')
            idx=$((feature_num - 1))
            feature_key="${FEATURE_KEYS[$idx]}"
            if [ -n "$feature_key" ]; then
                "add_${feature_key}_feature"
            else
                log_warning "Unknown feature number: $feature_num"
            fi
        done
    fi

    update_requirements
    show_completion_summary
}

show_project_status() {
    echo -e "\\n${CYAN}=== Project Status ===${NC}"
    echo -e "Project: ${GREEN}${PROJECT_NAME}${NC}"
    echo -e "Location: ${BLUE}$(pwd)${NC}"
    
    echo -e "\\n${CYAN}Directory Structure:${NC}"
    if command -v tree >/dev/null 2>&1; then
        tree -L 3 -a
    else
        find . -type d | head -10 | sed 's/^/  /'
    fi
    
    echo -e "\\n${CYAN}Python Files:${NC}"
    find . -name "*.py" -type f | head -10 | sed 's/^/  /'
}

update_requirements() {
    log_info "Updating requirements.txt..."
    
    # Create a temporary file with new requirements
    local temp_req=$(mktemp)
    
    # Add common requirements based on features
    cat > "$temp_req" << 'EOF'
# Core dependencies
requests>=2.28.0
pyyaml>=6.0

# Testing
pytest>=7.0.0
coverage>=6.0

# Database
sqlite3

# Async support
aiohttp>=3.8.0

# Web API
fastapi>=0.100.0
uvicorn>=0.23.0

# Monitoring
psutil>=5.9.0

# CLI
argparse

# Data processing
pandas>=1.5.0
numpy>=1.24.0
EOF
    
    # Only update if requirements.txt doesn't exist or is empty
    if [ ! -f "requirements.txt" ] || [ ! -s "requirements.txt" ]; then
        cp "$temp_req" "requirements.txt"
        log_success "Created requirements.txt"
    else
        log_info "requirements.txt already exists, skipping update"
    fi
    
    rm "$temp_req"
}

show_completion_summary() {
    echo -e "\\n${GREEN}=== Enhancement Complete! ===${NC}"
    echo -e "${CYAN}Professional project structure created with:${NC}"
    echo "  📁 src/app/ - Main application code"
    echo "  📁 config/ - Configuration files"
    echo "  📁 tests/ - Test files"
    echo "  📁 scripts/ - Utility scripts"
    echo "  📁 data/ - Database and data files"
    echo "  📁 logs/ - Log files"
    
    echo -e "\\n${CYAN}Next steps:${NC}"
    echo "1. Install dependencies: pip install -r requirements.txt"
    echo "2. Run tests: ./scripts/run_tests.sh"
    echo "3. Configure config/config.yaml for your needs"
    echo "4. Copy .env.example to .env and fill in your API keys"
    echo "5. Start developing in src/app/"
    
    echo -e "\\n${YELLOW}Re-run this script anytime to add more features!${NC}"
}

# Script entry point
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
'''

# Write the script to a file
with open('enhanced_project_generator.sh', 'w') as f:
    f.write(script_content)

print("Fixed bash script created: enhanced_project_generator.sh")
print("\nKey fixes made:")
print("1. Fixed feature number mapping in implement_selected_features()")
print("2. Removed all trading-specific content")
print("3. Made feature selection more robust")
print("4. Fixed string escaping issues in heredoc sections")
print("5. Ensured proper feature function calls")
print("6. Added proper error handling for unknown feature numbers")