"""Tests for scripts/build_kernel_image.py: the ``docker buildx build`` command it assembles.

Nothing here needs Docker: ``subprocess.run`` and ``shutil.which`` are replaced, and the dry run
only prints. The build itself is exercised by the ``docker`` job in .github/workflows/test.yml and
the ``image`` job in release.yml.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

from hailer import kernel_image

pytestmark = pytest.mark.usefixtures("isolated_package_settings")

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "build_kernel_image.py"
_spec = importlib.util.spec_from_file_location("hailer_build_kernel_image_script", _SCRIPT)
assert _spec is not None and _spec.loader is not None
script = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = script
_spec.loader.exec_module(script)

ARGS = {"MARIMO_VERSION": "0.24.2", "KERNEL_CONTRACT": "marimo0.24.2-0123456789ab"}


def _build_arg_parts(args) -> list[str]:
    return [part for name, value in args.items() for part in ("--build-arg", f"{name}={value}")]


def test_buildx_command_has_platforms_tags_build_args_output_then_the_context():
    cmd = script.buildx_command(
        Path("ctx"), ARGS, tags=["reg/img:1", "reg/img:latest"], platforms=["linux/amd64", "linux/arm64"], output="push"
    )
    assert cmd == [
        "docker", "buildx", "build",
        "--platform", "linux/amd64,linux/arm64",
        "--tag", "reg/img:1",
        "--tag", "reg/img:latest",
        "--build-arg", "MARIMO_VERSION=0.24.2",
        "--build-arg", "KERNEL_CONTRACT=marimo0.24.2-0123456789ab",
        "--push",
        "ctx",
    ]  # fmt: skip


def test_buildx_command_without_platforms_or_output_and_with_extra_arguments():
    cmd = script.buildx_command("ctx", ARGS, tags=["t"], extra=["--progress", "plain"], docker="/usr/bin/docker")
    assert cmd == ["/usr/bin/docker", "buildx", "build", "--tag", "t", *_build_arg_parts(ARGS), "--progress", "plain", "ctx"]
    assert script.buildx_command("ctx", {}, tags=["t"], output="load")[-2:] == ["--load", "ctx"]


def test_buildx_command_rejects_a_bad_output_and_no_tags():
    with pytest.raises(ValueError):
        script.buildx_command("ctx", ARGS, tags=["t"], output="export")
    with pytest.raises(ValueError):
        script.buildx_command("ctx", ARGS, tags=[])


def test_defaults_are_both_platforms_and_the_image_this_hailer_runs():
    args = script.parse_args([])
    assert args.tags == [kernel_image.default_image()]
    assert args.platforms == ["linux/amd64", "linux/arm64"]
    assert args.output is None and args.extra == [] and not args.dry_run and args.context is None
    assert script.parse_args(["--push"]).platforms == ["linux/amd64", "linux/arm64"]


def test_load_defaults_to_this_machines_platform_unless_one_is_given():
    assert script.parse_args(["--load"]).platforms == []
    assert script.parse_args(["--load", "--platform", "linux/amd64"]).platforms == ["linux/amd64"]
    assert script.parse_args(["--platform", " linux/arm64 , linux/amd64 "]).platforms == ["linux/arm64", "linux/amd64"]


def test_tags_repeat_and_arguments_after_the_separator_go_to_buildx():
    args = script.parse_args(["--tag", "a:1", "--tag", "b:2", "--", "--progress", "plain", "--tag", "c:3"])
    assert args.tags == ["a:1", "b:2"]
    assert args.extra == ["--progress", "plain", "--tag", "c:3"]


@pytest.mark.parametrize("argv", [["--push", "--load"], ["--platform", " , "], ["--frobnicate"]])
def test_bad_arguments_exit_with_usage(argv, capsys):
    with pytest.raises(SystemExit) as info:
        script.parse_args(argv)
    assert info.value.code == 2


def test_dry_run_prints_the_command_and_runs_nothing(monkeypatch, capsys):
    monkeypatch.setattr(script.subprocess, "run", lambda *a, **k: pytest.fail("the dry run ran something"))
    assert script.main(["--dry-run", "--push", "--tag", "reg/img:1"]) == 0
    printed = capsys.readouterr().out.strip().splitlines()[-1]
    expected = script.buildx_command(
        script.CONTEXT_PLACEHOLDER, kernel_image.build_args(), tags=["reg/img:1"], platforms=script.DEFAULT_PLATFORMS, output="push"
    )
    assert printed == script.shlex.join(expected)


def test_dry_run_with_a_context_prepares_it_for_a_manual_run(tmp_path, capsys):
    ctx = tmp_path / "ctx"
    assert script.main(["--dry-run", "--load", "--context", str(ctx)]) == 0
    assert (ctx / "Dockerfile").is_file() and (ctx / "hailer" / "_forward.py").is_file()
    assert capsys.readouterr().out.strip().splitlines()[-1].endswith(script.shlex.quote(str(ctx)))
    assert script.main(["--dry-run", "--context", str(ctx)]) == 1, "a context that already holds hailer/ is refused"
    assert "already holds a hailer folder" in capsys.readouterr().err


def test_a_build_runs_buildx_on_a_prepared_temporary_context(monkeypatch):
    seen: dict = {}

    def fake_run(cmd, **kwargs):
        context = Path(cmd[-1])
        seen["cmd"] = list(cmd)
        seen["ready"] = (context / "Dockerfile").is_file() and (context / "hailer" / "periods.py").is_file()
        return subprocess.CompletedProcess(cmd, 7)

    monkeypatch.setattr(script.shutil, "which", lambda name: "/opt/docker" if name == "docker" else None)
    monkeypatch.setattr(script.subprocess, "run", fake_run)
    assert script.main(["--load", "--platform", "linux/amd64", "--tag", "t:1"]) == 7, "buildx's exit code"
    cmd = seen["cmd"]
    assert cmd[:3] == ["/opt/docker", "buildx", "build"] and seen["ready"]
    assert cmd[3:-1] == ["--platform", "linux/amd64", "--tag", "t:1", *_build_arg_parts(kernel_image.build_args()), "--load"]
    assert not Path(cmd[-1]).exists(), "the temporary context is removed afterwards"


BOTH = ("linux/amd64", "linux/arm64")
NOT_FOUND = (1, "ERROR: ghcr.io/openafterhours/hailer-kernel:x: not found")


def _index(*platforms: str) -> tuple[int, str]:
    """``imagetools inspect --format {{json .Manifest}}`` for an image index (plus an attestation)."""
    entries = [{"platform": {"os": p.split("/")[0], "architecture": p.split("/")[1]}} for p in platforms]
    entries.append({"platform": {"os": "unknown", "architecture": "unknown"}})
    return 0, json.dumps({"mediaType": "application/vnd.oci.image.index.v1+json", "manifests": entries})


def _registry(monkeypatch, answers: dict[str, tuple[int, str]]) -> list[list[str]]:
    """``subprocess.run`` as a registry answering ``imagetools inspect`` per tag, and a build that works."""
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(list(cmd))
        if cmd[1:4] == ["buildx", "imagetools", "inspect"]:
            code, text = answers.get(cmd[-1], NOT_FOUND)
            return subprocess.CompletedProcess(cmd, code, text if code == 0 else "", "" if code == 0 else text)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(script.shutil, "which", lambda name: "/opt/docker" if name == "docker" else None)
    monkeypatch.setattr(script.subprocess, "run", fake_run)
    return calls


def test_if_missing_builds_nothing_when_every_tag_is_published_for_both_platforms(monkeypatch, capsys):
    """A release that changed nothing in the image reuses the published one."""
    calls = _registry(monkeypatch, {"reg/img:t": _index(*BOTH)})
    assert script.main(["--push", "--if-missing", "--tag", "reg/img:t"]) == 0
    assert calls == [["/opt/docker", "buildx", "imagetools", "inspect", "--format", "{{json .Manifest}}", "reg/img:t"]]
    assert "already published: nothing to build" in capsys.readouterr().out


@pytest.mark.parametrize("missing", [NOT_FOUND, (1, "ERROR: reg/img:t: manifest unknown")])
def test_if_missing_builds_and_pushes_when_the_registry_says_the_tag_does_not_exist(monkeypatch, missing):
    calls = _registry(monkeypatch, {"reg/img:t": missing})
    assert script.main(["--push", "--if-missing", "--tag", "reg/img:t"]) == 0
    build = calls[-1]
    assert build[:3] == ["/opt/docker", "buildx", "build"] and "--push" in build
    assert script.parse_args(["--push"]).if_missing is False, "without --if-missing: always build"


@pytest.mark.parametrize(
    ("answers", "said"),
    [
        ({"reg/img:t": (1, "ERROR: failed to authorize: 401 Unauthorized")}, "could not ask the registry about reg/img:t"),
        ({"reg/img:t": (1, "ERROR: dial tcp: lookup ghcr.io: no such host")}, "could not ask the registry"),
        ({"reg/img:t": _index("linux/amd64")}, "reg/img:t is published without linux/arm64; refusing to overwrite it"),
        ({"reg/img:t": (0, json.dumps({"mediaType": "application/vnd.oci.image.manifest.v1+json"}))}, "published without linux/amd64, linux/arm64"),
        ({"reg/img:t": (0, "not json")}, "unexpected manifest for reg/img:t"),
        ({"reg/img:t": _index(*BOTH), "reg/img:u": NOT_FOUND}, "only some tags are published (missing: reg/img:u)"),
    ],
)
def test_if_missing_fails_closed_and_never_overwrites(monkeypatch, capsys, answers, said):
    calls = _registry(monkeypatch, answers)
    assert script.main(["--push", "--if-missing", *[part for tag in answers for part in ("--tag", tag)]]) == 1
    assert said in capsys.readouterr().err
    assert not any(call[1:3] == ["buildx", "build"] for call in calls), "nothing built or pushed"


def test_if_missing_needs_push(capsys):
    with pytest.raises(SystemExit) as info:
        script.parse_args(["--load", "--if-missing"])
    assert info.value.code == 2 and "--if-missing needs --push" in capsys.readouterr().err


def test_no_docker_is_a_clear_error(monkeypatch, capsys):
    monkeypatch.setattr(script.shutil, "which", lambda name: None)
    assert script.main(["--load"]) == 1
    assert "docker is not on PATH" in capsys.readouterr().err


def test_corporate_dry_run_passes_secrets_without_copying_or_printing_contents(tmp_path, capsys):
    config = tmp_path / "pip.ini"
    config.write_text("[global]\nindex-url = https://user:private-token@packages.example/simple\n")
    cert = tmp_path / "company.pem"
    cert.write_text("test-ca")
    ctx = tmp_path / "context"
    assert script.main([
        "--dry-run", "--load", "--context", str(ctx), "--base-image", "company/python:3.13",
        "--pip-config", str(config), "--pip-cert", str(cert), "--no-cache",
    ]) == 0
    printed = capsys.readouterr().out
    assert "BASE_IMAGE=company/python:3.13" in printed and "id=pip_config" in printed and "id=pip_cert" in printed
    assert "--no-cache" in printed
    assert "private-token" not in printed
    assert {p.name for p in ctx.iterdir()} == {"Dockerfile", "hailer"}


def test_corporate_missing_config_does_not_prepare_a_context(tmp_path, capsys):
    ctx = tmp_path / "context"
    assert script.main(["--dry-run", "--context", str(ctx), "--pip-config", str(tmp_path / "missing.ini")]) == 1
    assert "Cannot read --pip-config file" in capsys.readouterr().err
    assert not ctx.exists()


def test_discovered_settings_dry_run_and_opt_out(monkeypatch, capsys):
    monkeypatch.setenv("UV_DEFAULT_INDEX", "https://build:private-token@mirror/simple")
    assert script.main(["--dry-run", "--load"]) == 0
    output = capsys.readouterr().out
    assert "<discovered-pip-config>" in output and "private-token" not in output
    assert script.main(["--dry-run", "--no-host-config"]) == 0
    assert "--secret" not in capsys.readouterr().out


def test_buildx_discovered_secret_exists_only_during_build(monkeypatch):
    import csv

    monkeypatch.setenv("PIP_INDEX_URL", "https://build:private-token@mirror/simple")
    seen = []

    def run(cmd, **kwargs):
        fields = dict(item.split("=", 1) for item in next(csv.reader([cmd[cmd.index("--secret") + 1]])))
        secret = Path(fields["src"])
        assert "private-token" in secret.read_text()
        assert "private-token" not in str(cmd)
        seen.append(secret)
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(script.shutil, "which", lambda name: "docker")
    monkeypatch.setattr(script.subprocess, "run", run)
    assert script.main(["--load"]) == 0
    assert seen and not seen[0].exists()
