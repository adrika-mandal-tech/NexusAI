"""Shared test fixtures for the NexusAI backend test suite."""

import pytest
from httpx import ASGITransport, AsyncClient

from nexusai.main import app


@pytest.fixture
async def client() -> AsyncClient:
    """Async HTTP client wired to the FastAPI app (no real server needed)."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as ac:
        yield ac
