"""Structured logging and CloudWatch custom metrics (no-op unless CLOUDWATCH_ENABLED=true)."""

from __future__ import annotations

import json
import logging
import sys
from functools import lru_cache

from app.config import get_settings

logger = logging.getLogger("kb")


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        extra = getattr(record, "data", None)
        if extra:
            payload.update(extra)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    logger.handlers = [handler]
    logger.setLevel(logging.INFO)
    logger.propagate = False


def log_event(event: str, **data) -> None:
    logger.info(event, extra={"data": data})


@lru_cache
def _cloudwatch_client():
    import boto3

    return boto3.client("cloudwatch", region_name=get_settings().aws_region)


def emit_metric(name: str, value: float = 1.0, unit: str = "Count") -> None:
    """Publish a custom metric to CloudWatch. Never raises: metrics must not break requests."""
    settings = get_settings()
    if not settings.cloudwatch_enabled:
        return
    try:
        _cloudwatch_client().put_metric_data(
            Namespace=settings.cloudwatch_namespace,
            MetricData=[{"MetricName": name, "Value": value, "Unit": unit}],
        )
    except Exception:  # pragma: no cover - depends on AWS availability
        logger.warning("cloudwatch_put_failed", extra={"data": {"metric": name}})
