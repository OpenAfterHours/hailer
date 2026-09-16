"""Lightweight logging for Hailer with secret redaction.

Normal users see only warnings and errors on stderr. ``--verbose`` or
``HAILER_LOG_LEVEL=DEBUG`` enables debug output that covers provider calls,
session setup, Marimo connections, tool invocations and configuration.

Every handler installed here carries :class:`RedactingFilter`, which masks
bearer tokens, ``Authorization`` header values and the current values of any
environment variable whose name ends in ``_KEY``, ``_TOKEN``, ``_SECRET`` or
``_PASSWORD``.
"""

from __future__ import annotations

import logging
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

ROOT_LOGGER_NAME = "hailer"
SECRET_ENV_SUFFIXES: tuple[str, ...] = ("_KEY", "_TOKEN", "_SECRET", "_PASSWORD")
MIN_SECRET_LENGTH = 8
REDACTED = "<redacted>"

_AUTH_HEADER_RE = re.compile(r"(?i)(authorization\s*[:=]\s*)[^\r\n;,]+")
_BEARER_RE = re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=\-]+")
#: A bare OpenAI-style key inside quoted text (a keyring-sourced key is not in the environment).
_TOKEN_RE = re.compile(r"\bsk-[A-Za-z0-9_\-]{16,}")

_NOISY_LOGGERS = ("openai_codex", "urllib3", "httpx", "httpcore", "asyncio", "mcp")


class RedactingFilter(logging.Filter):
    """Mask secrets in log messages before they reach any handler."""

    def __init__(self, env: Mapping[str, str] | None = None) -> None:
        super().__init__()
        # Keep a reference (not a copy) so values set after setup are still masked.
        self._env = os.environ if env is None else env

    def secret_values(self) -> list[str]:
        values: list[str] = []
        for name, value in self._env.items():
            if not isinstance(value, str) or len(value) < MIN_SECRET_LENGTH:
                continue
            if name.upper().endswith(SECRET_ENV_SUFFIXES):
                values.append(value)
        # Longest first so a value that contains another is masked whole.
        return sorted(set(values), key=len, reverse=True)

    def redact(self, text: str) -> str:
        text = _AUTH_HEADER_RE.sub(lambda m: m.group(1) + REDACTED, text)
        text = _BEARER_RE.sub("Bearer " + REDACTED, text)
        text = _TOKEN_RE.sub(REDACTED, text)
        for value in self.secret_values():
            if value in text:
                text = text.replace(value, REDACTED)
        return text

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001 - malformed format args; keep the raw message
            message = str(record.msg)
        record.msg = self.redact(message)
        record.args = ()
        if record.exc_text:
            record.exc_text = self.redact(record.exc_text)
        return True


def redact(text: str, env: Mapping[str, str] | None = None) -> str:
    """Convenience wrapper: redact ``text`` using the same rules as the filter."""
    return RedactingFilter(env).redact(text)


def _resolve_level(level: str | None, verbose: bool, env: Mapping[str, str]) -> int:
    if verbose:
        return logging.DEBUG
    name = (level or env.get("HAILER_LOG_LEVEL") or "WARNING").upper()
    resolved = logging.getLevelName(name)
    if not isinstance(resolved, int):
        return logging.WARNING
    return resolved


def setup_logging(
    level: str | None,
    *,
    verbose: bool = False,
    log_file: Path | None = None,
    env: Mapping[str, str] | None = None,
    stream=None,
) -> logging.Logger:
    """Configure the ``hailer`` logger and return it. Safe to call more than once."""
    env_map: Mapping[str, str] = os.environ if env is None else env
    resolved = _resolve_level(level, verbose, env_map)

    logger = logging.getLogger(ROOT_LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        try:
            handler.close()
        except Exception:  # noqa: BLE001
            pass

    redactor = RedactingFilter(env_map if env is not None else None)
    formatter = logging.Formatter("%(levelname)s %(name)s: %(message)s")

    console = logging.StreamHandler(stream if stream is not None else sys.stderr)
    console.setLevel(resolved)
    console.setFormatter(formatter)
    console.addFilter(redactor)
    logger.addHandler(console)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(resolved)
        file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        file_handler.addFilter(redactor)
        logger.addHandler(file_handler)

    logger.setLevel(resolved)
    logger.propagate = False

    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.DEBUG if verbose else logging.WARNING)

    logger.debug("logging configured at %s", logging.getLevelName(resolved))
    return logger


def get_logger(name: str = ROOT_LOGGER_NAME) -> logging.Logger:
    """Return a child of the ``hailer`` logger (``get_logger("agent")`` -> ``hailer.agent``)."""
    if name == ROOT_LOGGER_NAME or name.startswith(ROOT_LOGGER_NAME + "."):
        return logging.getLogger(name)
    return logging.getLogger(f"{ROOT_LOGGER_NAME}.{name}")
