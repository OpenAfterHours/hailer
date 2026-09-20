"""Discover portable host package settings for the pip install in a kernel build.

Only package-index, transport and CA settings cross the boundary. No host commands,
credential helpers, installation paths or arbitrary environment variables are copied.
Values can contain credentials: never include them (or parser errors) in diagnostics.
"""

from __future__ import annotations

import configparser
import os
import re
import sys
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import quote, urlsplit, urlunsplit

from hailer.errors import KernelRuntimeError

_OPTIONS = ("index-url", "extra-index-url", "trusted-host", "proxy", "cert", "no-index", "find-links", "timeout", "retries")
_INDEX_OPTIONS = ("index-url", "extra-index-url", "no-index", "find-links")


@dataclass
class PackageSettings:
    options: dict[str, str] = field(default_factory=dict, repr=False)
    cert: Path | None = field(default=None, repr=False)


def _error(message: str) -> KernelRuntimeError:
    return KernelRuntimeError(
        message,
        hint="Use --pip-config FILE to select build settings explicitly, or --no-host-config to disable discovery.",
    )


def _exists(path: Path) -> bool:
    try:
        path.stat()
        return True
    except FileNotFoundError:
        return False
    except OSError:
        raise _error("Cannot inspect a host package configuration file.") from None


def _pip_paths(env: Mapping[str, str], home: Path, prefix: Path, platform: str) -> list[Path]:
    explicit = env.get("PIP_CONFIG_FILE")
    if explicit and explicit.lower() in (os.devnull.lower(), "nul" if platform == "win32" else "/dev/null"):
        return []
    if platform == "win32":
        name = "pip.ini"
        system = [Path(env.get("PROGRAMDATA", "C:/ProgramData")) / "pip" / name]
        user = [home / "pip" / name, Path(env.get("APPDATA", str(home / "AppData/Roaming"))) / "pip" / name]
    else:
        name = "pip.conf"
        if platform == "darwin":
            system = [Path(p) / "pip" / name for p in env.get("XDG_DATA_DIRS", "/Library/Application Support").split(":") if p]
            data = Path(env.get("XDG_DATA_HOME", str(home / "Library/Application Support"))) / "pip"
            user_dir = data if _exists(data) else home / ".config/pip"
        else:
            system = [Path(p) / "pip" / name for p in env.get("XDG_CONFIG_DIRS", "/etc/xdg").split(":") if p]
            system.append(Path("/etc/pip.conf"))
            user_dir = Path(env.get("XDG_CONFIG_HOME", str(home / ".config"))) / "pip"
        user = [home / ".pip" / name, user_dir / name]
    custom = Path(explicit).expanduser() if explicit else None
    # pip skips user config when PIP_CONFIG_FILE points at an existing file.
    paths = system + ([] if custom and _exists(custom) else user)
    paths.append(Path(env.get("VIRTUAL_ENV", str(prefix))) / name)
    if custom:
        paths.append(custom)
    return paths


def _pip_settings(paths: list[Path], env: Mapping[str, str]) -> dict[str, str]:
    parser = configparser.RawConfigParser()
    try:
        for path in paths:
            if _exists(path):
                with path.open(encoding="utf-8-sig") as stream:
                    parser.read_file(stream)
        values: dict[str, str] = {}
        for section in ("global", "install"):
            if parser.has_section(section):
                values.update({k.replace("_", "-"): v for k, v in parser.items(section)})
    except (OSError, UnicodeError, configparser.Error):
        raise _error("Cannot read the host pip configuration.") from None
    for key in (*_OPTIONS, "client-cert"):
        value = env.get("PIP_" + key.upper().replace("-", "_"))
        if value:  # pip ignores empty environment values
            values[key] = value
    if values.get("client-cert"):
        raise _error("Automatic discovery cannot mount a pip client certificate; use an explicit build configuration.")
    return {key: values[key] for key in _OPTIONS if key in values}


def _toml(path: Path) -> dict:
    try:
        with path.open("rb") as stream:
            return tomllib.load(stream)
    except (OSError, ValueError):
        raise _error("Cannot read the host uv configuration.") from None


def _merge(low: dict, high: dict) -> dict:
    result = dict(low)
    for key, value in high.items():
        previous = result.get(key)
        if isinstance(value, dict) and isinstance(previous, dict):
            value = _merge(previous, value)
        elif isinstance(value, list) and isinstance(previous, list):
            value = value + previous
        result[key] = value
    return result


def _uv_settings(env: Mapping[str, str], home: Path, cwd: Path, platform: str) -> dict:
    if env.get("UV_NO_CONFIG", "").lower() in ("1", "true", "yes"):
        return {}
    if env.get("UV_CONFIG_FILE"):
        settings = _toml(Path(env["UV_CONFIG_FILE"]).expanduser())
        return {**settings, **settings.get("pip", {})}
    if platform == "win32":
        system = [Path(env.get("PROGRAMDATA", "C:/ProgramData")) / "uv/uv.toml"]
        user = Path(env.get("APPDATA", str(home / "AppData/Roaming"))) / "uv/uv.toml"
    else:
        system = [Path(p) / "uv/uv.toml" for p in env.get("XDG_CONFIG_DIRS", "/etc/xdg").split(":") if p]
        system.append(Path("/etc/uv/uv.toml"))
        user = Path(env.get("XDG_CONFIG_HOME", str(home / ".config"))) / "uv/uv.toml"
    settings: dict = {}
    for path in system:
        if _exists(path):
            settings = _toml(path)
            break
    if _exists(user):
        settings = _merge(settings, _toml(user))
    for directory in (cwd, *cwd.parents):
        uv = directory / "uv.toml"
        project = directory / "pyproject.toml"
        if _exists(uv):
            settings = _merge(settings, _toml(uv))
            break
        if _exists(project):
            table = _toml(project).get("tool", {}).get("uv")
            if table is not None:
                settings = _merge(settings, table)
                break
    pip = settings.get("pip", {})
    return {**settings, **pip}


def _uv_index(value: str, env: Mapping[str, str], name: str = "") -> str:
    if "://" not in value:
        raise _error("The discovered uv index must be an HTTP(S) URL.")
    if "=" in value.split("://", 1)[0]:
        name, value = value.split("=", 1)
    url = urlsplit(value)
    if name:
        key = "UV_INDEX_" + re.sub(r"[^A-Za-z0-9]", "_", name).upper()
        username = env.get(key + "_USERNAME")
        password = env.get(key + "_PASSWORD")
        if username is not None or password is not None:
            username = quote(username, safe="") if username is not None else (url.username or "")
            password = quote(password, safe="") if password is not None else (url.password or "")
            value = urlunsplit(url._replace(netloc=f"{username}:{password}@{url.netloc.rsplit('@', 1)[-1]}"))
    return value


def _uv_options(settings: dict, env: Mapping[str, str]) -> dict[str, str]:
    # pip does not implement uv's first-index resolution. Only a single default
    # index is portable without changing which repository supplies a package.
    extra = env.get("UV_INDEX") or env.get("UV_EXTRA_INDEX_URL") or settings.get("extra-index-url")
    if extra or settings.get("sources"):
        raise _error("uv package routing or additional indexes need an explicit pip build configuration.")
    value = env.get("UV_DEFAULT_INDEX") or env.get("UV_INDEX_URL")
    result: dict[str, str] = {}
    indexes = settings.get("index", [])
    if not isinstance(indexes, list) or any(not isinstance(i, dict) for i in indexes):
        raise _error("Invalid index table in the host uv configuration.")
    if any(not i.get("default") or i.get("explicit") for i in indexes):
        raise _error("uv package routing or additional indexes need an explicit pip build configuration.")
    if value:
        result["index-url"] = _uv_index(value, env)
    elif indexes:
        # Higher-precedence named definitions replace the same lower-level name.
        first = indexes[0]
        if any(i.get("name") != first.get("name") or not first.get("name") for i in indexes[1:]):
            raise _error("Multiple uv default indexes need an explicit pip build configuration.")
        result["index-url"] = _uv_index(first["url"], env, first.get("name", ""))
    elif settings.get("index-url"):
        result["index-url"] = _uv_index(settings["index-url"], env)
    no_index = env.get("UV_NO_INDEX")
    if no_index or settings.get("no-index"):
        result["no-index"] = no_index or "true"
    return result


def discover(
    *, environ: Mapping[str, str] | None = None, home: Path | None = None,
    cwd: Path | None = None, prefix: Path | None = None, platform: str | None = None,
) -> PackageSettings:
    """Read pip settings first; fall back to a single uv default index.

    Environment overrides files within each tool. Files follow global/user/site
    precedence for pip and system/user/nearest-project precedence for uv.
    Explicit function inputs keep discovery testable without reading real secrets.
    """
    env = os.environ if environ is None else environ
    home, cwd = home or Path.home(), cwd or Path.cwd()
    platform = platform or sys.platform
    values = _pip_settings(_pip_paths(env, home, prefix or Path(sys.prefix), platform), env)
    if not any(key in values for key in _INDEX_OPTIONS):
        try:
            values.update(_uv_options(_uv_settings(env, home, cwd, platform), env))
        except (TypeError, KeyError, ValueError, AttributeError):
            raise _error("Invalid package settings in the host uv configuration.") from None
    cert = values.pop("cert", "") or env.get("REQUESTS_CA_BUNDLE", "") or env.get("SSL_CERT_FILE", "")
    for key in ("index-url", "extra-index-url", "find-links"):
        for url in values.get(key, "").split():
            if not url.lower().startswith(("https://", "http://")):
                raise _error("Host package paths cannot be used inside Docker; configure an HTTP(S) mirror.")
    # Clear inherited extra indexes when a host index has been selected.
    if any(key in values for key in _INDEX_OPTIONS):
        values.setdefault("extra-index-url", "")
        values.setdefault("no-index", "false")
        values.setdefault("find-links", "")
    return PackageSettings(values, Path(cert).expanduser() if cert else None)
