"""Cut a hailer release: bump the version, run the tests, commit, tag and push.

Usage (from the repository root)::

    uv run python -m scripts.release              # patch: 0.1.0 -> 0.1.1
    uv run python -m scripts.release minor        # 0.1.0 -> 0.2.0
    uv run python -m scripts.release major        # 0.1.0 -> 1.0.0
    uv run python -m scripts.release --version 1.2.0rc1
    uv run python -m scripts.release --dry-run    # preflight and plan only, changes nothing
    uv run python -m scripts.release --no-push    # everything except the push
    uv run python -m scripts.release -- -x -k cli # arguments after -- go to pytest

What happens:

1. Preflight. git and uv are on PATH, the checkout is on ``main``, no tracked file has
   uncommitted changes, local ``main`` matches ``origin/main``, ``uv.lock`` is current,
   the version files agree, and the release tag is free locally and on origin. Any
   failure stops the run before anything is touched.
2. Bump. The new version is written to ``pyproject.toml`` and
   ``src/hailer/__init__.py``; ``uv lock`` then refreshes ``uv.lock``.
3. Test. ``uv run --locked pytest``. If the suite fails, or the run is interrupted,
   the three version files are restored from HEAD and the script exits non-zero.
   Nothing is committed.
4. Publish. Commit ``Release vX.Y.Z``, create the annotated tag ``vX.Y.Z`` and
   ``git push --atomic origin main vX.Y.Z``. The tag push triggers
   ``.github/workflows/release.yml``, which builds the sdist and wheel, publishes
   them to PyPI through the ``pypi`` environment (trusted publishing) and creates
   the GitHub release.

The script only uses the standard library so it runs before the project is synced.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = ROOT / "pyproject.toml"
INIT_PY = ROOT / "src" / "hailer" / "__init__.py"
UV_LOCK = ROOT / "uv.lock"
VERSION_FILES = (PYPROJECT, INIT_PY, UV_LOCK)

RELEASE_BRANCH = "main"
REMOTE = "origin"
WORKFLOW = ROOT / ".github" / "workflows" / "release.yml"

# One version line each. Anchored at line start only so CRLF files also match.
PYPROJECT_PATTERN = re.compile(r'^version = "(?P<v>[^"\r\n]+)"', re.MULTILINE)
INIT_PATTERN = re.compile(r'^__version__ = "(?P<v>[^"\r\n]+)"', re.MULTILINE)
LOCK_PATTERN = re.compile(r'^name = "hailer"\r?\nversion = "(?P<v>[^"\r\n]+)"', re.MULTILINE)

VERSION_PATTERN = re.compile(
    r"^(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)(?:(?P<pre>a|b|rc)(?P<pre_n>\d+))?$"
)
_PRE_RANK = {"a": 0, "b": 1, "rc": 2, None: 3}
_FINAL = _PRE_RANK[None]


class ReleaseError(Exception):
    """A condition that stops the release; the message is shown to the user."""


# --------------------------------------------------------------------------- #
# Version arithmetic (pure; covered by tests/test_release.py)
# --------------------------------------------------------------------------- #


def parse_version(text: str) -> tuple[int, int, int, int, int]:
    """Return a sortable key for ``X.Y.Z`` or ``X.Y.Z(a|b|rc)N``.

    Pre-releases sort before their final version: 1.2.0a1 < 1.2.0b1 < 1.2.0rc1 < 1.2.0.
    """
    match = VERSION_PATTERN.match(text)
    if match is None:
        raise ReleaseError(f"unsupported version {text!r}: expected X.Y.Z or X.Y.Z(a|b|rc)N")
    return (
        int(match["major"]),
        int(match["minor"]),
        int(match["patch"]),
        _PRE_RANK[match["pre"]],
        int(match["pre_n"] or 0),
    )


def bump_version(current: str, part: str) -> str:
    """Return the next version. ``patch`` on a pre-release finalises it (1.2.0rc1 -> 1.2.0)."""
    major, minor, patch, pre_rank, _ = parse_version(current)
    if part == "major":
        return f"{major + 1}.0.0"
    if part == "minor":
        return f"{major}.{minor + 1}.0"
    if part == "patch":
        if pre_rank < _FINAL:
            return f"{major}.{minor}.{patch}"
        return f"{major}.{minor}.{patch + 1}"
    raise ReleaseError(f"unknown version part {part!r}")


def find_version(text: str, pattern: re.Pattern[str], *, what: str) -> re.Match[str]:
    matches = list(pattern.finditer(text))
    if len(matches) != 1:
        raise ReleaseError(f"expected exactly one version line in {what}, found {len(matches)}")
    return matches[0]


def set_version(text: str, pattern: re.Pattern[str], new: str, *, what: str) -> str:
    """Replace the single version value matched by ``pattern``; everything else is untouched."""
    match = find_version(text, pattern, what=what)
    return text[: match.start("v")] + new + text[match.end("v") :]


# --------------------------------------------------------------------------- #
# Files and processes
# --------------------------------------------------------------------------- #


def _read(path: Path) -> str:
    # newline="" keeps the file's own line endings so the rewrite is byte-exact.
    with path.open(encoding="utf-8", newline="") as fh:
        return fh.read()


def _write(path: Path, text: str) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        fh.write(text)


def read_file_version(path: Path, pattern: re.Pattern[str]) -> str:
    return find_version(_read(path), pattern, what=path.name)["v"]


def _relpaths(paths: tuple[Path, ...]) -> list[str]:
    return [p.relative_to(ROOT).as_posix() for p in paths]


def say(message: str) -> None:
    print(f"release: {message}", flush=True)


def warn(message: str) -> None:
    print(f"release: {message}", file=sys.stderr, flush=True)


def run(args: list[str], *, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    """Run a command in the repository root. Captured output is UTF-8 text."""
    result = subprocess.run(
        args,
        cwd=ROOT,
        check=False,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check and result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip() if capture else ""
        suffix = f":\n{detail}" if detail else ""
        raise ReleaseError(f"`{' '.join(args)}` failed with exit code {result.returncode}{suffix}")
    return result


def git(*args: str, check: bool = True, capture: bool = True) -> subprocess.CompletedProcess[str]:
    return run(["git", *args], check=check, capture=capture)


# --------------------------------------------------------------------------- #
# Steps
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Plan:
    current: str
    new: str

    @property
    def tag(self) -> str:
        return f"v{self.new}"


def preflight(part: str, explicit: str | None) -> Plan:
    """Check everything that can be checked before any file is modified."""
    for tool in ("git", "uv"):
        if shutil.which(tool) is None:
            raise ReleaseError(f"{tool} is not on PATH")

    branch = git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
    if branch != RELEASE_BRANCH:
        raise ReleaseError(f"releases are cut from {RELEASE_BRANCH!r}; you are on {branch!r}")

    dirty = git("status", "--porcelain", "--untracked-files=no").stdout.strip()
    if dirty:
        raise ReleaseError("the working tree has uncommitted changes to tracked files:\n" + dirty)

    say(f"fetching {REMOTE}/{RELEASE_BRANCH}")
    git("fetch", "--quiet", REMOTE, RELEASE_BRANCH)
    counts = git("rev-list", "--left-right", "--count", f"HEAD...{REMOTE}/{RELEASE_BRANCH}").stdout.split()
    ahead, behind = (int(n) for n in counts)
    if ahead or behind:
        raise ReleaseError(
            f"local {RELEASE_BRANCH} is {ahead} ahead and {behind} behind {REMOTE}/{RELEASE_BRANCH}; "
            "push or pull first so the release commit lands on the published history"
        )

    lock_check = run(["uv", "lock", "--check"], check=False)
    if lock_check.returncode != 0:
        raise ReleaseError("uv.lock is out of date with pyproject.toml; run `uv lock`, commit it, then release")

    current = read_file_version(PYPROJECT, PYPROJECT_PATTERN)
    init_version = read_file_version(INIT_PY, INIT_PATTERN)
    lock_version = read_file_version(UV_LOCK, LOCK_PATTERN)
    if not current == init_version == lock_version:
        raise ReleaseError(
            "version files disagree: "
            f"pyproject.toml={current} __init__.py={init_version} uv.lock={lock_version}"
        )

    new = explicit if explicit is not None else bump_version(current, part)
    if parse_version(new) <= parse_version(current):
        raise ReleaseError(f"new version {new} must be greater than the current {current}")

    plan = Plan(current=current, new=new)
    if git("tag", "--list", plan.tag).stdout.strip():
        raise ReleaseError(f"tag {plan.tag} already exists locally")
    if git("ls-remote", "--tags", REMOTE, f"refs/tags/{plan.tag}").stdout.strip():
        raise ReleaseError(f"tag {plan.tag} already exists on {REMOTE}")

    if not WORKFLOW.exists():
        warn(f"warning: {WORKFLOW.relative_to(ROOT).as_posix()} is missing; the tag push will not publish anything")
    return plan


def bump(new: str) -> None:
    for path, pattern in ((PYPROJECT, PYPROJECT_PATTERN), (INIT_PY, INIT_PATTERN)):
        _write(path, set_version(_read(path), pattern, new, what=path.name))
        say(f"wrote {new} to {path.relative_to(ROOT).as_posix()}")
    say("refreshing uv.lock")
    run(["uv", "lock"])
    lock_version = read_file_version(UV_LOCK, LOCK_PATTERN)
    if lock_version != new:
        raise ReleaseError(f"uv.lock records {lock_version} after `uv lock`, expected {new}")


def run_tests(pytest_args: list[str]) -> int:
    say("running the test suite")
    # Output streams straight to the terminal so failures are visible.
    result = run(["uv", "run", "--locked", "pytest", *pytest_args], check=False, capture=False)
    return result.returncode


def revert() -> None:
    """Restore the version files (index and working tree) from HEAD and re-sync the env."""
    say("restoring " + ", ".join(_relpaths(VERSION_FILES)))
    git("checkout", "HEAD", "--", *_relpaths(VERSION_FILES))
    run(["uv", "sync"], check=False)


def commit_and_tag(plan: Plan) -> None:
    git("add", "--", *_relpaths(VERSION_FILES))
    git("commit", "--quiet", "-m", f"Release {plan.tag}")
    git("tag", "-a", plan.tag, "-m", f"hailer {plan.new}")
    say(f"committed and tagged {plan.tag}")


def push(plan: Plan) -> None:
    say(f"pushing {RELEASE_BRANCH} and {plan.tag} to {REMOTE}")
    git("push", "--atomic", REMOTE, RELEASE_BRANCH, plan.tag, capture=False)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #


def parse_args(argv: list[str]) -> argparse.Namespace:
    pytest_args: list[str] = []
    if "--" in argv:
        split = argv.index("--")
        argv, pytest_args = argv[:split], argv[split + 1 :]

    parser = argparse.ArgumentParser(
        prog="python -m scripts.release",
        description="Bump the version, run the tests, then commit, tag and push the release.",
        epilog="Arguments after -- are passed to pytest.",
    )
    parser.add_argument(
        "part",
        nargs="?",
        choices=("major", "minor", "patch"),
        default="patch",
        help="which part to bump (default: patch)",
    )
    parser.add_argument("--version", dest="explicit", metavar="X.Y.Z", help="release exactly this version")
    parser.add_argument("--dry-run", action="store_true", help="run the preflight checks and show the plan only")
    parser.add_argument("--no-push", action="store_true", help="commit and tag locally but do not push")
    args = parser.parse_args(argv)
    args.pytest_args = pytest_args
    if args.explicit is not None:
        parse_version(args.explicit)  # fail fast on a malformed --version
    return args


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(sys.argv[1:] if argv is None else argv)
        plan = preflight(args.part, args.explicit)
    except ReleaseError as exc:
        warn(f"aborted: {exc}")
        return 1

    say(f"{plan.current} -> {plan.new} (tag {plan.tag})")
    if args.dry_run:
        say("dry run: nothing changed")
        return 0

    try:
        bump(plan.new)
        code = run_tests(args.pytest_args)
        if code != 0:
            raise ReleaseError(f"tests failed (pytest exit code {code})")
    except (ReleaseError, KeyboardInterrupt) as exc:
        reason = "interrupted" if isinstance(exc, KeyboardInterrupt) else str(exc)
        warn(f"aborted: {reason}")
        try:
            revert()
        except ReleaseError as revert_exc:
            warn(f"could not restore the version files automatically: {revert_exc}")
            warn(f"run: git checkout HEAD -- {' '.join(_relpaths(VERSION_FILES))}")
        else:
            warn("version bump reverted; nothing was committed")
        return 1

    try:
        commit_and_tag(plan)
    except ReleaseError as exc:
        warn(f"aborted: {exc}")
        warn("the version files may be modified or staged; inspect with `git status` and undo with:")
        warn(f"  git tag -d {plan.tag}; git reset --hard {REMOTE}/{RELEASE_BRANCH}")
        return 1

    if args.no_push:
        say(f"not pushing (--no-push). Publish with: git push --atomic {REMOTE} {RELEASE_BRANCH} {plan.tag}")
        return 0

    try:
        push(plan)
    except ReleaseError as exc:
        warn(f"push failed: {exc}")
        warn(f"the release commit and tag {plan.tag} exist locally. Retry with:")
        warn(f"  git push --atomic {REMOTE} {RELEASE_BRANCH} {plan.tag}")
        warn("or undo them with:")
        warn(f"  git tag -d {plan.tag}; git reset --hard {REMOTE}/{RELEASE_BRANCH}")
        return 1

    say(f"pushed. The {plan.tag} tag triggers the Release workflow, which publishes hailer {plan.new} to PyPI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
