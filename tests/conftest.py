"""Shared, opt-in isolation for tests that discover host package configuration."""

import os

import pytest

from hailer import kernel_packages


@pytest.fixture
def isolated_package_settings(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith(("PIP_", "UV_")) or name in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
            monkeypatch.delenv(name)
    discover = kernel_packages.discover
    root = tmp_path / "package-settings"
    root.mkdir()

    def isolated_discover():
        env = dict(os.environ)
        env.update(PROGRAMDATA=str(root / "system"), APPDATA=str(root / "user"), VIRTUAL_ENV=str(root / "venv"))
        return discover(environ=env, home=root, cwd=root, prefix=root, platform="win32")

    monkeypatch.setattr(kernel_packages, "discover", isolated_discover)
    return root
