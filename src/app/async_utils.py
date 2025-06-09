"""Async utilities and helpers."""
import asyncio
import aiohttp
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)

class AsyncAPIClient:
    """Async API client for concurrent requests."""

    def __init__(self, base_url: str, timeout: int = 30):
        self.base_url = base_url.rstrip('/')
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def get(self, endpoint: str, params: Optional[Dict] = None) -> Dict[str, Any]:
        """Make async GET request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        async with self.session.get(url, params=params) as response:
            response.raise_for_status()
            return await response.json()

    async def post(self, endpoint: str, data: Optional[Dict] = None) -> Dict[str, Any]:
        """Make async POST request."""
        url = f"{self.base_url}/{endpoint.lstrip('/')}"

        async with self.session.post(url, json=data) as response:
            response.raise_for_status()
            return await response.json()

async def fetch_multiple_urls(urls: List[str]) -> List[Dict[str, Any]]:
    """Fetch multiple URLs concurrently."""
    async with aiohttp.ClientSession() as session:
        tasks = []
        for url in urls:
            task = asyncio.create_task(fetch_url(session, url))
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)
        return results

async def fetch_url(session: aiohttp.ClientSession, url: str) -> Dict[str, Any]:
    """Fetch single URL."""
    try:
        async with session.get(url) as response:
            response.raise_for_status()
            return await response.json()
    except Exception as e:
        logger.error(f"Failed to fetch {url}: {e}")
        return {"error": str(e), "url": url}

class AsyncTaskManager:
    """Manage async background tasks."""

    def __init__(self):
        self.tasks = []

    def add_task(self, coro):
        """Add a background task."""
        task = asyncio.create_task(coro)
        self.tasks.append(task)
        return task

    async def wait_all(self):
        """Wait for all tasks to complete."""
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)

    def cancel_all(self):
        """Cancel all running tasks."""
        for task in self.tasks:
            if not task.done():
                task.cancel()

# Global task manager
task_manager = AsyncTaskManager()
