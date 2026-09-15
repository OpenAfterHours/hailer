"""Tests for hailer.secrets using an in-memory keyring backend (never the real store)."""

from __future__ import annotations

import keyring
import pytest
from keyring.backend import KeyringBackend
from keyring.errors import PasswordDeleteError

from hailer import secrets
from hailer.errors import CredentialsError
from hailer.models import ProviderConfig


class MemoryKeyring(KeyringBackend):
    priority = 1  # type: ignore[assignment]

    def __init__(self) -> None:
        super().__init__()
        self.store: dict[tuple[str, str], str] = {}

    def get_password(self, service: str, username: str) -> str | None:
        return self.store.get((service, username))

    def set_password(self, service: str, username: str, password: str) -> None:
        self.store[(service, username)] = password

    def delete_password(self, service: str, username: str) -> None:
        if (service, username) not in self.store:
            raise PasswordDeleteError(username)
        del self.store[(service, username)]


class BrokenKeyring(KeyringBackend):
    priority = 1  # type: ignore[assignment]

    def get_password(self, service: str, username: str) -> str | None:
        raise RuntimeError("credential store locked")

    def set_password(self, service: str, username: str, password: str) -> None:
        raise RuntimeError("credential store locked")

    def delete_password(self, service: str, username: str) -> None:
        raise RuntimeError("credential store locked")


@pytest.fixture
def memory_keyring():
    previous = keyring.get_keyring()
    backend = MemoryKeyring()
    keyring.set_keyring(backend)
    try:
        yield backend
    finally:
        keyring.set_keyring(previous)


@pytest.fixture
def broken_keyring():
    previous = keyring.get_keyring()
    keyring.set_keyring(BrokenKeyring())
    try:
        yield
    finally:
        keyring.set_keyring(previous)


INTERNAL = ProviderConfig(
    id="internal",
    base_url="https://llm.example.internal/v1",
    env_key="INTERNAL_MODEL_API_KEY",
)
OPENAI = ProviderConfig(id="openai", env_key="OPENAI_API_KEY", requires_openai_auth=True)


def test_username_format():
    assert secrets.keyring_username("internal", "INTERNAL_MODEL_API_KEY") == "internal:INTERNAL_MODEL_API_KEY"


def test_env_wins_over_keyring(memory_keyring):
    memory_keyring.store[("hailer", "internal:INTERNAL_MODEL_API_KEY")] = "from-keyring"
    value, source = secrets.resolve_provider_key(INTERNAL, {"INTERNAL_MODEL_API_KEY": "from-env"})
    assert (value, source) == ("from-env", "env")


def test_keyring_used_when_env_missing(memory_keyring):
    memory_keyring.store[("hailer", "internal:INTERNAL_MODEL_API_KEY")] = "from-keyring"
    value, source = secrets.resolve_provider_key(INTERNAL, {})
    assert (value, source) == ("from-keyring", "keyring")


def test_missing_everywhere(memory_keyring):
    assert secrets.resolve_provider_key(INTERNAL, {}) == (None, "missing")


def test_empty_env_value_is_ignored(memory_keyring):
    assert secrets.resolve_provider_key(INTERNAL, {"INTERNAL_MODEL_API_KEY": ""}) == (None, "missing")


def test_provider_without_env_key():
    assert secrets.resolve_provider_key(ProviderConfig(id="x"), {"X": "1"}) == (None, "missing")


def test_openai_missing_is_reported_not_raised(memory_keyring):
    # The CLI decides whether this is fatal (a ChatGPT login may exist).
    assert secrets.resolve_provider_key(OPENAI, {}) == (None, "missing")


def test_keyring_failure_treated_as_missing(broken_keyring):
    assert secrets.resolve_provider_key(INTERNAL, {}) == (None, "missing")
    with pytest.raises(CredentialsError) as err:
        secrets.delete_provider_key(INTERNAL)
    assert "INTERNAL_MODEL_API_KEY" in err.value.hint
    with pytest.raises(CredentialsError) as err:
        secrets.store_provider_key(INTERNAL, "abc")
    assert "INTERNAL_MODEL_API_KEY" in err.value.hint
    assert "credential store" in str(err.value)


def test_store_and_delete_round_trip(memory_keyring):
    secrets.store_provider_key(INTERNAL, "  secret-123  ")
    assert memory_keyring.store[("hailer", "internal:INTERNAL_MODEL_API_KEY")] == "secret-123"
    assert secrets.resolve_provider_key(INTERNAL, {}) == ("secret-123", "keyring")
    assert secrets.delete_provider_key(INTERNAL) is True
    assert secrets.delete_provider_key(INTERNAL) is False
    assert secrets.resolve_provider_key(INTERNAL, {}) == (None, "missing")


def test_store_rejects_empty(memory_keyring):
    with pytest.raises(CredentialsError):
        secrets.store_provider_key(INTERNAL, "   ")
    with pytest.raises(CredentialsError):
        secrets.store_provider_key(ProviderConfig(id="nokey"), "abc")
