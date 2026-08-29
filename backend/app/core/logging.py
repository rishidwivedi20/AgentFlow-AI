"""Structured logging setup.

Human-readable lines in development, single-line JSON in production so logs
stay greppable and machine-parseable. Configured once from the application
factory; modules just call ``logging.getLogger(__name__)``.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

# Attributes present on every LogRecord. Anything outside this set was attached
# by the caller via `extra=` and belongs in the structured payload.
_RESERVED_RECORD_ATTRS = frozenset(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__.keys()
) | {
    "message",
    "asctime",
    "taskName",
    "color_message",  # uvicorn attaches ANSI-decorated duplicates of its messages
}


class JsonFormatter(logging.Formatter):
    """Render log records as one JSON object per line."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        for key, value in record.__dict__.items():
            if key not in _RESERVED_RECORD_ATTRS:
                payload[key] = value

        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)

        return json.dumps(payload, default=str)


class ConsoleFormatter(logging.Formatter):
    """Readable development output that still surfaces ``extra=`` fields."""

    def __init__(self) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)-8s %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )

    def format(self, record: logging.LogRecord) -> str:
        base = super().format(record)
        context = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _RESERVED_RECORD_ATTRS
        }
        if context:
            rendered = " ".join(f"{key}={value}" for key, value in context.items())
            base = f"{base} [{rendered}]"
        return base


def configure_logging(level: str = "INFO", *, json_output: bool = False) -> None:
    """Install a single stdout handler on the root logger.

    Idempotent: existing handlers are replaced, so repeated calls (reload,
    tests) do not produce duplicated log lines.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter() if json_output else ConsoleFormatter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(level)

    # Uvicorn installs its own handlers; let records propagate to ours instead
    # so every line shares one format.
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        uvicorn_logger = logging.getLogger(name)
        uvicorn_logger.handlers.clear()
        uvicorn_logger.propagate = True
