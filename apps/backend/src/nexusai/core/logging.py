"""Structured JSON logging with request-ID injection.

Sets up logging once at import time. Every log line includes:
- timestamp (ISO 8601)
- level
- logger name
- message
- request_id (when available via contextvars)
"""

import logging
import sys
import uuid
from contextvars import ContextVar

from nexusai.core.config import get_settings

# ── Context variable for request correlation ─────────────────────
request_id_ctx: ContextVar[str] = ContextVar("request_id", default="")


class RequestIdFilter(logging.Filter):
    """Injects the current request_id into every log record."""

    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = request_id_ctx.get("")  # type: ignore[attr-defined]
        return True


def generate_request_id() -> str:
    """Generate a short correlation ID."""
    return uuid.uuid4().hex[:12]


def setup_logging() -> None:
    """Configure root logger with structured JSON-like format."""
    settings = get_settings()

    log_format = (
        '{"time":"%(asctime)s","level":"%(levelname)s",'
        '"logger":"%(name)s","request_id":"%(request_id)s",'
        '"message":"%(message)s"}'
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(log_format, datefmt="%Y-%m-%dT%H:%M:%S"))
    handler.addFilter(RequestIdFilter())

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO))

    # Quieten noisy third-party loggers and avoid logging request query params (e.g. API keys)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a named logger with the request-ID filter already attached."""
    logger = logging.getLogger(name)
    return logger
