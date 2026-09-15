"""Hailer error types. Every error carries a user-facing ``hint`` with the fix.

The CLI prints ``message`` then ``hint`` and never a stack trace unless
``--verbose`` / ``HAILER_LOG_LEVEL=DEBUG`` is set.
"""

from __future__ import annotations


class HailerError(Exception):
    """Base class: ``str(err)`` is the short message; ``hint`` says what to do."""

    def __init__(self, message: str, hint: str = "") -> None:
        super().__init__(message)
        self.hint = hint


class ConfigError(HailerError):
    """hailer.toml missing, unparsable, or invalid."""


class NotebookNotFoundError(HailerError):
    """The configured marimo notebook file does not exist."""


class MarimoUnavailableError(HailerError):
    """No marimo server answers at the configured/discovered URL."""


class NoSessionError(HailerError):
    """Server is up but the notebook is not open in a browser (no kernel session)."""


class MarimoExecutionError(HailerError):
    """The execute request failed at the transport level (not a Python error in user code)."""


class CredentialsError(HailerError):
    """API key for the active provider is missing (env var or keyring)."""


class ProviderError(HailerError):
    """Model endpoint unreachable, rejected the request, or misconfigured."""


class AgentError(HailerError):
    """Codex runtime failed to start or a turn failed."""


class WebAccessDenied(HailerError):
    """URL host is not in ``[web].allowed_domains``."""


class MalformedParquetError(HailerError):
    """A parquet file could not be read; the message names the file."""
