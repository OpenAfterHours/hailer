"""API-key resolution for model providers.

Order of precedence for a provider's key:

1. the environment variable named in ``ProviderConfig.env_key`` (automation, CI);
2. the OS credential store via ``keyring`` (Windows Credential Manager on Windows),
   stored by ``hailer login <provider>``;
3. nothing — the agent cannot start without a key and says how to store one.

The key value is only ever returned to the caller; it is never logged.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Mapping

from hailer.errors import CredentialsError
from hailer.models import ProviderConfig

try:  # log.py is written by another owner; fall back to plain logging if absent.
    from hailer.log import get_logger
except Exception:  # pragma: no cover - only when hailer.log is missing

    def get_logger(name: str = "hailer") -> logging.Logger:
        return logging.getLogger(name)


log = get_logger("hailer.secrets")

KEYRING_SERVICE = "hailer"

KeySource = str  # "env" | "keyring" | "missing"


def keyring_username(provider_id: str, env_key: str) -> str:
    """Credential-store username for a provider key, e.g. ``internal:INTERNAL_MODEL_API_KEY``."""
    return f"{provider_id}:{env_key}"


def _keyring_get(username: str) -> str | None:
    try:
        import keyring

        return keyring.get_password(KEYRING_SERVICE, username)
    except Exception as exc:  # NoKeyringError, backend failures, locked store ...
        log.debug("keyring lookup failed for %s: %s", username, type(exc).__name__)
        return None


def resolve_provider_key(
    provider: ProviderConfig, env: Mapping[str, str] = os.environ
) -> tuple[str | None, KeySource]:
    """Return ``(value, source)`` where source is ``"env"``, ``"keyring"`` or ``"missing"``."""
    env_key = provider.env_key
    if not env_key:
        return None, "missing"
    value = env.get(env_key)
    if value:
        return value, "env"
    stored = _keyring_get(keyring_username(provider.id, env_key))
    if stored:
        return stored, "keyring"
    return None, "missing"


def _env_fallback_hint(provider: ProviderConfig) -> str:
    return f"Set the {provider.env_key} environment variable in your terminal instead."


def store_provider_key(provider: ProviderConfig, value: str) -> None:
    """Save a key in the OS credential store.

    Raises ``CredentialsError`` (with the environment-variable fallback as the hint) when the
    provider has no ``env_key``, the value is empty, or the keyring backend is unusable.
    """
    if not provider.env_key:
        raise CredentialsError(
            f"Provider '{provider.id}' has no env_key; nothing to store.",
            hint=f"Add env_key = \"<VAR_NAME>\" to [model_providers.{provider.id}] in hailer.toml.",
        )
    if not value or not value.strip():
        raise CredentialsError("Refusing to store an empty key.", hint="Run the login command again and paste the key.")
    try:
        import keyring

        keyring.set_password(KEYRING_SERVICE, keyring_username(provider.id, provider.env_key), value.strip())
    except Exception as exc:
        raise CredentialsError(
            f"Could not store the key in the OS credential store ({type(exc).__name__}).",
            hint=_env_fallback_hint(provider),
        ) from exc


def delete_provider_key(provider: ProviderConfig) -> bool:
    """Remove a stored key. Returns ``True`` if something was deleted, ``False`` if nothing was stored.

    Raises ``CredentialsError`` when the keyring backend is unusable.
    """
    if not provider.env_key:
        return False
    username = keyring_username(provider.id, provider.env_key)
    try:
        import keyring

        if keyring.get_password(KEYRING_SERVICE, username) is None:
            return False
        keyring.delete_password(KEYRING_SERVICE, username)
        return True
    except Exception as exc:
        log.debug("keyring delete failed for %s: %s", username, type(exc).__name__)
        raise CredentialsError(
            f"Could not access the OS credential store ({type(exc).__name__}).",
            hint=_env_fallback_hint(provider),
        ) from exc
