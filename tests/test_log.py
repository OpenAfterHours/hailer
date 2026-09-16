"""Tests for hailer.log: level resolution and secret redaction."""

from __future__ import annotations

import io
import logging
from pathlib import Path

from hailer.log import REDACTED, RedactingFilter, get_logger, redact, setup_logging


def _record(msg: str, *args: object) -> logging.LogRecord:
    return logging.LogRecord("hailer.test", logging.INFO, __file__, 1, msg, args, None)


def test_filter_masks_env_secrets_only_when_long_enough() -> None:
    env = {
        "INTERNAL_MODEL_API_KEY": "fake-internal-key-123456",
        "SHORT_KEY": "abc",  # too short to mask
        "MARIMO_TOKEN": "tok-9876543210",
        "DB_PASSWORD": "hunter2hunter2",
        "CLIENT_SECRET": "s3cr3t-value",
        "PLAIN_VALUE": "plainvalue123",  # name has no secret suffix, must not be masked
    }
    flt = RedactingFilter(env)
    record = _record(
        "auth with %s and %s, token %s, pw %s, secret %s, and %s stays",
        "fake-internal-key-123456",
        "abc",
        "tok-9876543210",
        "hunter2hunter2",
        "s3cr3t-value",
        "plainvalue123",
    )
    assert flt.filter(record) is True
    message = record.getMessage()
    assert "fake-internal-key-123456" not in message
    assert "tok-9876543210" not in message
    assert "hunter2hunter2" not in message
    assert "s3cr3t-value" not in message
    assert "abc" in message
    assert "plainvalue123" in message
    assert message.count(REDACTED) == 4
    assert record.args == ()


def test_filter_masks_bearer_and_authorization_header() -> None:
    flt = RedactingFilter({})
    text = flt.redact("headers: Authorization: Bearer sk-abc.DEF_123 ok; x=Bearer zzz-999; other: value")
    assert "sk-abc.DEF_123" not in text
    assert "zzz-999" not in text
    assert "other: value" in text
    assert text.startswith("headers: Authorization: <redacted>")


def test_filter_masks_exception_text() -> None:
    env = {"OPENAI_API_KEY": "sk-live-abcdefghijkl"}
    flt = RedactingFilter(env)
    record = _record("failed")
    record.exc_text = "RuntimeError: rejected key sk-live-abcdefghijkl"
    flt.filter(record)
    assert "sk-live-abcdefghijkl" not in record.exc_text
    assert REDACTED in record.exc_text


def test_filter_reads_env_live() -> None:
    env: dict[str, str] = {}
    flt = RedactingFilter(env)
    assert flt.redact("value-1234567890") == "value-1234567890"
    env["LATE_KEY"] = "value-1234567890"  # set after the filter was created
    assert flt.redact("value-1234567890") == REDACTED


def test_redact_helper() -> None:
    assert redact("Bearer abcdef") == "Bearer <redacted>"
    assert redact("nothing here", {}) == "nothing here"


def test_setup_logging_levels() -> None:
    stream = io.StringIO()
    logger = setup_logging(None, env={}, stream=stream)
    assert logger.name == "hailer"
    assert logger.level == logging.WARNING
    assert logger.propagate is False

    logger = setup_logging("info", env={}, stream=stream)
    assert logger.level == logging.INFO

    logger = setup_logging(None, env={"HAILER_LOG_LEVEL": "debug"}, stream=stream)
    assert logger.level == logging.DEBUG

    logger = setup_logging("error", verbose=True, env={"HAILER_LOG_LEVEL": "critical"}, stream=stream)
    assert logger.level == logging.DEBUG  # --verbose wins

    logger = setup_logging("nonsense", env={}, stream=stream)
    assert logger.level == logging.WARNING  # unknown level falls back safely


def test_setup_logging_is_idempotent_and_redacts_console_output() -> None:
    stream = io.StringIO()
    env = {"INTERNAL_MODEL_API_KEY": "fake-internal-key-123456"}
    setup_logging("debug", env=env, stream=stream)
    logger = setup_logging("debug", env=env, stream=stream)
    assert len(logger.handlers) == 1  # no duplicate handlers after a second call
    get_logger("agent").warning("using key %s for provider", "fake-internal-key-123456")
    output = stream.getvalue()
    assert "fake-internal-key-123456" not in output
    assert REDACTED in output
    assert "hailer.agent" in output


def test_setup_logging_file_handler_redacts(tmp_path: Path) -> None:
    log_file = tmp_path / "logs" / "hailer.log"
    env = {"OPENAI_API_KEY": "sk-live-abcdefghijkl"}
    logger = setup_logging("debug", env=env, log_file=log_file, stream=io.StringIO())
    logger.info("Authorization: Bearer sk-live-abcdefghijkl")
    logger.debug("session setup done")
    for handler in logger.handlers:
        handler.flush()
    text = log_file.read_text(encoding="utf-8")
    assert "sk-live-abcdefghijkl" not in text
    assert "session setup done" in text
    setup_logging(None, env={}, stream=io.StringIO())  # release the file handler


def test_normal_level_hides_debug_and_info() -> None:
    stream = io.StringIO()
    logger = setup_logging(None, env={}, stream=stream)
    logger.debug("provider call details")
    logger.info("session setup")
    logger.warning("something to see")
    output = stream.getvalue()
    assert "provider call details" not in output
    assert "session setup" not in output
    assert "something to see" in output


def test_get_logger_names() -> None:
    assert get_logger().name == "hailer"
    assert get_logger("marimo_client").name == "hailer.marimo_client"
    assert get_logger("hailer.cli").name == "hailer.cli"


def test_redact_masks_bare_api_keys() -> None:
    assert redact("key sk-abcdefghijklmnopqrstuvwxyz123456 rejected", {}) == "key <redacted> rejected"
    assert redact("sk-short is not a key", {}) == "sk-short is not a key"
