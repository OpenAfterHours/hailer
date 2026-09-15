"""Tests for the pure parts of scripts/release.py: version parsing, bumping and rewriting.

The git / uv / pytest orchestration is exercised by hand with ``--dry-run``; these tests
pin the arithmetic and the file rewriting so a bump can never touch the wrong line.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _ROOT / "scripts" / "release.py"
_spec = importlib.util.spec_from_file_location("hailer_release_script", _SCRIPT)
assert _spec is not None and _spec.loader is not None
release = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = release  # dataclasses resolve deferred annotations via sys.modules
_spec.loader.exec_module(release)


@pytest.mark.parametrize(
    ("current", "part", "expected"),
    [
        ("0.1.0", "patch", "0.1.1"),
        ("0.1.9", "patch", "0.1.10"),
        ("0.1.4", "minor", "0.2.0"),
        ("0.9.1", "major", "1.0.0"),
        ("1.2.0rc1", "patch", "1.2.0"),  # patch finalises a pre-release
        ("1.2.0b2", "minor", "1.3.0"),
        ("1.2.0a1", "major", "2.0.0"),
    ],
)
def test_bump_version(current: str, part: str, expected: str) -> None:
    assert release.bump_version(current, part) == expected


@pytest.mark.parametrize("bad", ["1.2", "v1.2.3", "1.2.3.4", "1.2.3-rc1", "1.2.3.dev1", "", "abc"])
def test_parse_version_rejects_unsupported_forms(bad: str) -> None:
    with pytest.raises(release.ReleaseError):
        release.parse_version(bad)


def test_bump_version_rejects_unknown_part() -> None:
    with pytest.raises(release.ReleaseError):
        release.bump_version("0.1.0", "micro")


def test_version_ordering() -> None:
    order = ["0.9.9", "1.0.0a1", "1.0.0a2", "1.0.0b1", "1.0.0rc1", "1.0.0", "1.0.1", "1.1.0", "2.0.0"]
    keys = [release.parse_version(v) for v in order]
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)


def test_set_version_rewrites_only_the_pyproject_version_line() -> None:
    text = (
        '[project]\nname = "hailer"\nversion = "0.1.0"\n'
        'dependencies = [\n    "marimo==0.24.2",\n]\n'
        '[tool.other]\nversion_hint = "0.1.0"\n'
    )
    out = release.set_version(text, release.PYPROJECT_PATTERN, "0.2.0", what="pyproject.toml")
    assert 'version = "0.2.0"\n' in out
    assert '"marimo==0.24.2"' in out  # pinned dependency untouched
    assert 'version_hint = "0.1.0"' in out  # only a `version = ` line at column 0 counts
    assert out.replace('version = "0.2.0"', 'version = "0.1.0"') == text


def test_set_version_preserves_crlf_line_endings() -> None:
    text = '"""doc"""\r\n\r\n__version__ = "0.1.0"\r\n'
    out = release.set_version(text, release.INIT_PATTERN, "0.1.1", what="__init__.py")
    assert out == '"""doc"""\r\n\r\n__version__ = "0.1.1"\r\n'


def test_set_version_requires_exactly_one_match() -> None:
    with pytest.raises(release.ReleaseError, match="found 2"):
        release.set_version('version = "1"\nversion = "2"\n', release.PYPROJECT_PATTERN, "3", what="x")
    with pytest.raises(release.ReleaseError, match="found 0"):
        release.set_version("nothing here\n", release.PYPROJECT_PATTERN, "3", what="x")


def test_lock_pattern_targets_the_hailer_package_only() -> None:
    text = (
        '[[package]]\nname = "h11"\nversion = "0.16.0"\n\n'
        '[[package]]\nname = "hailer"\nversion = "0.1.0"\nsource = { editable = "." }\n\n'
        '[[package]]\nname = "hailer-extra"\nversion = "9.9.9"\n'
    )
    assert release.find_version(text, release.LOCK_PATTERN, what="uv.lock")["v"] == "0.1.0"


def test_repository_version_files_agree() -> None:
    """pyproject.toml, src/hailer/__init__.py and uv.lock must carry the same version."""
    versions = {
        "pyproject.toml": release.read_file_version(release.PYPROJECT, release.PYPROJECT_PATTERN),
        "__init__.py": release.read_file_version(release.INIT_PY, release.INIT_PATTERN),
        "uv.lock": release.read_file_version(release.UV_LOCK, release.LOCK_PATTERN),
    }
    assert len(set(versions.values())) == 1, versions
    release.parse_version(next(iter(versions.values())))


def test_parse_args_splits_pytest_arguments() -> None:
    args = release.parse_args(["minor", "--no-push", "--", "-x", "-k", "cli"])
    assert args.part == "minor"
    assert args.no_push is True
    assert args.pytest_args == ["-x", "-k", "cli"]

    args = release.parse_args([])
    assert args.part == "patch"
    assert args.pytest_args == []


def test_parse_args_validates_explicit_version() -> None:
    with pytest.raises(release.ReleaseError):
        release.parse_args(["--version", "1.2"])
