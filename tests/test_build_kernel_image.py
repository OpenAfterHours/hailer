"""Tests for scripts/build_kernel_image.py: the ``docker buildx build`` command it assembles.

Nothing here needs Docker: ``subprocess.run`` and ``shutil.which`` are replaced, and the dry run
only prints. The build itself is exercised by the ``docker`` job in .github/workflows/test.yml and
the ``image`` job in release.yml.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

from hailer import kernel_image

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "build_kernel_image.py"
_spec = importlib.util.spec_from_file_location("hailer_build_kernel_image_script", _SCRIPT)
assert _spec is not None and _spec.loader is not None
script = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = script
_spec.loader.exec_module(script)

ARGS = {"MARIMO_VERSION": "0.24.2", "HAILER_VERSION": "9.9.9"}


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
        "--build-arg", "HAILER_VERSION=9.9.9",
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


def test_no_docker_is_a_clear_error(monkeypatch, capsys):
    monkeypatch.setattr(script.shutil, "which", lambda name: None)
    assert script.main(["--load"]) == 1
    assert "docker is not on PATH" in capsys.readouterr().err
