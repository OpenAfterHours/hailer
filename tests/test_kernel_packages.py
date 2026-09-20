"""Package discovery uses fixture files, never the developer's credentials or network."""

import configparser
import csv
import os
import subprocess
import traceback
from pathlib import Path

import pytest

from hailer import kernel_image, kernel_packages as packages
from hailer.errors import KernelRuntimeError


@pytest.fixture
def host(tmp_path):
    env = {"PROGRAMDATA": str(tmp_path / "system"), "APPDATA": str(tmp_path / "user")}

    def write(relative, text):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def discover():
        return packages.discover(environ=env, home=tmp_path, cwd=tmp_path, prefix=tmp_path / "venv", platform="win32")

    return env, write, discover


def test_pip_merges_layers_install_section_and_environment(host):
    env, write, discover = host
    write("system/pip/pip.ini", "[global]\nindex-url=https://global/simple\nproxy=http://proxy\n")
    write("pip/pip.ini", "[global]\nindex-url=https://legacy/simple\n")
    write("user/pip/pip.ini", "[global]\nindex-url=https://user/simple\n[install]\nindex-url=https://install/simple\n")
    write("venv/pip.ini", "[global]\nindex-url=https://venv/simple\n")
    assert discover().options["index-url"] == "https://install/simple"
    env["PIP_INDEX_URL"] = "https://user:private%40token@env/simple"
    env["UV_INDEX_URL"] = "https://ignored/simple"
    settings = discover()
    assert settings.options["index-url"] == env["PIP_INDEX_URL"]
    assert settings.options["proxy"] == "http://proxy"
    assert settings.options["extra-index-url"] == ""
    assert "private" not in repr(settings)


def test_pip_config_file_skips_user_but_preserves_system_and_site(host):
    env, write, discover = host
    write("system/pip/pip.ini", "[global]\nproxy=http://proxy\n")
    write("user/pip/pip.ini", "[install]\nindex-url=https://skip/simple\n")
    write("venv/pip.ini", "[global]\ntrusted-host=internal\n")
    env["PIP_CONFIG_FILE"] = str(write("custom.ini", "[global]\nindex-url=https://custom/simple\n"))
    assert discover().options == {
        "index-url": "https://custom/simple", "proxy": "http://proxy", "trusted-host": "internal",
        "extra-index-url": "", "no-index": "false", "find-links": "",
    }
    env["PIP_CONFIG_FILE"] = "nul"
    assert discover().options == {}
    env["PIP_INDEX_URL"] = "https://env/simple"
    assert discover().options["index-url"] == "https://env/simple"


@pytest.mark.parametrize("platform", ["linux", "darwin", "win32"])
def test_platform_pip_locations_and_active_environment(tmp_path, platform):
    env = {"APPDATA": str(tmp_path / "roaming"), "PROGRAMDATA": str(tmp_path / "system"),
           "XDG_CONFIG_HOME": str(tmp_path / "config"), "XDG_CONFIG_DIRS": str(tmp_path / "global"),
           "VIRTUAL_ENV": str(tmp_path / "active")}
    paths = packages._pip_paths(env, tmp_path, tmp_path / "python", platform)
    name = "pip.ini" if platform == "win32" else "pip.conf"
    assert paths[-1] == tmp_path / "active" / name
    assert paths[-2] == {
        "win32": tmp_path / "roaming/pip/pip.ini",
        "linux": tmp_path / "config/pip/pip.conf",
        "darwin": tmp_path / ".config/pip/pip.conf",
    }[platform]


def test_uv_user_project_and_environment_precedence_with_named_credentials(host):
    env, write, discover = host
    write("user/uv/uv.toml", '[[index]]\nname="company"\nurl="https://user/simple"\ndefault=true\n')
    write("pyproject.toml", '[[tool.uv.index]]\nname="company"\nurl="https://project/simple"\ndefault=true\n')
    env.update(UV_INDEX_COMPANY_USERNAME="build@company", UV_INDEX_COMPANY_PASSWORD="private/token")
    assert discover().options["index-url"] == "https://build%40company:private%2Ftoken@project/simple"
    write("uv.toml", 'index-url="https://uv-file/simple"\n')
    # An index table has priority over the legacy index-url option.
    assert "@user/simple" in discover().options["index-url"]
    env["UV_DEFAULT_INDEX"] = "company=https://env/simple"
    assert "@env/simple" in discover().options["index-url"]


def test_uv_nearest_parent_and_explicit_file(host):
    env, write, discover = host
    write("uv.toml", 'index-url="https://project/simple"\n')
    assert discover().options["index-url"] == "https://project/simple"
    env["UV_CONFIG_FILE"] = str(write("custom.toml", '[pip]\nindex-url="https://custom/simple"\n'))
    assert discover().options["index-url"] == "https://custom/simple"
    env["UV_NO_CONFIG"] = "true"
    assert discover().options == {}


@pytest.mark.parametrize("variable", ["UV_INDEX_URL", "UV_DEFAULT_INDEX"])
def test_uv_environment_without_pip_installed(host, variable):
    env, _, discover = host
    env[variable] = "https://mirror/simple"
    assert discover().options["index-url"] == "https://mirror/simple"


@pytest.mark.parametrize("config", [
    '[[index]]\nurl="https://extra/simple"\n',
    'extra-index-url=["https://extra/simple"]\n',
    '[sources]\nmarimo={index="company"}\n',
])
def test_uv_routing_is_not_silently_changed_to_pip_resolution(host, config):
    _, write, discover = host
    write("uv.toml", config)
    with pytest.raises(KernelRuntimeError, match="explicit pip"):
        discover()


def test_pip_selection_ignores_unrelated_uv_configuration(host):
    env, write, discover = host
    env["PIP_INDEX_URL"] = "https://pip/simple"
    write("uv.toml", 'malformed="private-token')
    assert discover().options["index-url"] == "https://pip/simple"


def test_parser_errors_do_not_disclose_credentials_even_in_verbose_traceback(host):
    _, write, discover = host
    write("user/pip/pip.ini", "private-token malformed INI")
    with pytest.raises(KernelRuntimeError) as info:
        discover()
    assert "private-token" not in "".join(traceback.format_exception(info.value))


def test_portable_allowlist_and_certificate_discovery(host):
    env, write, discover = host
    cert = write("ca.pem", "test-ca")
    write("user/pip/pip.ini", f"[global]\ncert={cert}\nindex-url=https://mirror/simple\ntarget=/host/site-packages\n")
    env["MODEL_API_KEY"] = "not-a-package-credential"
    settings = discover()
    assert settings.cert == cert
    assert "cert" not in settings.options and "target" not in settings.options
    assert "not-a-package-credential" not in str(settings.options)


def test_local_wheel_paths_fail_before_build(host):
    env, _, discover = host
    env["PIP_FIND_LINKS"] = "C:/wheels"
    with pytest.raises(KernelRuntimeError, match="Host package paths"):
        discover()


def _secrets(args):
    result = {}
    for index, arg in enumerate(args):
        if arg == "--secret":
            fields = dict(item.split("=", 1) for item in next(csv.reader([args[index + 1]])))
            result[fields["id"]] = Path(fields["src"])
    return result


@pytest.mark.parametrize("failure", [False, True])
def test_build_secret_lifetime_contents_and_cleanup(isolated_package_settings, monkeypatch, failure):
    monkeypatch.setenv("PIP_INDEX_URL", "https://build:private-token@mirror/simple")
    seen = {}

    class Runner:
        def stream(self, args):
            secret = _secrets(args)["pip_config"]
            seen["secret"] = secret
            assert "private-token" not in str(args)
            config = configparser.RawConfigParser()
            config.read(secret)
            assert config["install"]["index-url"] == "https://build:private-token@mirror/simple"
            context = Path(args[-1])
            assert not secret.is_relative_to(context)
            assert {p.name for p in context.iterdir()} == {"Dockerfile", "hailer"}
            if os.name != "nt":
                assert secret.stat().st_mode & 0o777 == 0o600
            if failure:
                raise KeyboardInterrupt
            return subprocess.CompletedProcess(args, 0)

    if failure:
        with pytest.raises(KeyboardInterrupt):
            kernel_image.build("test", Runner())
    else:
        assert kernel_image.build("test", Runner()) == 0
    assert not seen["secret"].exists()


def test_explicit_file_and_opt_out_skip_discovery(monkeypatch, tmp_path):
    monkeypatch.setattr(packages, "discover", lambda: pytest.fail("discovery should not run"))
    config = tmp_path / "pip.ini"
    config.write_text("[global]\nindex-url=https://explicit/simple\n")
    with kernel_image.configured_build_options(pip_config=config) as options:
        assert _secrets(options)["pip_config"] == config
    with kernel_image.configured_build_options(no_host_config=True) as options:
        assert options == []


def test_discovered_certificate_and_explicit_override(isolated_package_settings, monkeypatch, tmp_path):
    cert = tmp_path / "host.pem"
    cert.write_text("host-ca")
    explicit = tmp_path / "explicit.pem"
    explicit.write_text("explicit-ca")
    monkeypatch.setenv("PIP_CERT", str(cert))
    with kernel_image.configured_build_options() as options:
        assert _secrets(options)["pip_cert"] == cert
    with kernel_image.configured_build_options(pip_cert=explicit) as options:
        assert _secrets(options)["pip_cert"] == explicit


def test_generated_config_overrides_base_install_settings(isolated_package_settings, monkeypatch, tmp_path):
    base_config = tmp_path / "base-pip.conf"
    base_config.write_text("[install]\nindex-url=https://base/simple\nextra-index-url=https://unwanted/simple\nno-index=true\n")
    monkeypatch.setenv("PIP_INDEX_URL", "https://host/simple")
    with kernel_image.configured_build_options() as options:
        secret = _secrets(options)["pip_config"]
        # Apply pip's file/section precedence to an existing base config and the secret.
        effective = packages._pip_settings([base_config, secret], {})
        assert effective["index-url"] == "https://host/simple"
        assert effective["extra-index-url"] == ""
        assert effective["no-index"] == "false"
