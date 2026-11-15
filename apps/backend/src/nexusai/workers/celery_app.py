"""Celery application configuration."""

from __future__ import annotations

from celery import Celery

from nexusai.core.config import get_settings

settings = get_settings()

celery_app = Celery(
    "nexusai",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "nexusai.workers.tasks.ingestion",
        "nexusai.workers.tasks.csv_ingestion",
    ],
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=3600,
    worker_prefetch_multiplier=1,
)
