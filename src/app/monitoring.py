"""Application monitoring and metrics."""
import time
import psutil
from typing import Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System metrics data class."""
    cpu_percent: float
    memory_percent: float
    disk_usage: float
    timestamp: float

class MonitoringManager:
    """Application monitoring manager."""

    def __init__(self):
        self.start_time = time.time()
        self.request_count = 0
        self.error_count = 0

    def get_system_metrics(self) -> SystemMetrics:
        """Get current system metrics."""
        return SystemMetrics(
            cpu_percent=psutil.cpu_percent(interval=1),
            memory_percent=psutil.virtual_memory().percent,
            disk_usage=psutil.disk_usage('/').percent,
            timestamp=time.time()
        )

    def get_app_metrics(self) -> Dict[str, Any]:
        """Get application metrics."""
        uptime = time.time() - self.start_time
        return {
            'uptime_seconds': uptime,
            'request_count': self.request_count,
            'error_count': self.error_count,
            'error_rate': self.error_count / max(self.request_count, 1)
        }

    def increment_request_count(self):
        """Increment request counter."""
        self.request_count += 1

    def increment_error_count(self):
        """Increment error counter."""
        self.error_count += 1

    def health_check(self) -> Dict[str, Any]:
        """Perform health check."""
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
