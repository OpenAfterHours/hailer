"""The project's own files, read-only, for the agent: ``project_files``, ``project_search`` and
``project_read`` in :mod:`hailer.tools`.

``[project]`` in the config turns it on: the agent may list, search and read the code, docs and
configuration under ``[project].root`` (default: the workspace) so that it can answer questions
about the product the data comes from. Everything happens here, on this machine; no kernel ever
sees these files, and nothing here writes.

Which files (:meth:`ProjectFiles.files`):

- In a git repository, the files ``git ls-files --cached --others --exclude-standard`` names
  (tracked, or new and not ignored); elsewhere a walk of the folder, skipping virtual
  environments. Either way no part of the path may be one of :data:`SKIPPED_DIRS`.
- Never Hailer's state folder, ``.hailer/local.toml``, the notebooks folder (the kernel holds the
  live notebooks), the data folders, a file named like a credential (:func:`secret_named`), a link
  that leads out of the root, or a file over :data:`MAX_FILE_BYTES`.
- ``[project].include`` (globs) when set, else code, docs and config files by suffix or name
  (:data:`TEXT_SUFFIXES`, :data:`TEXT_NAMES`); minus ``[project].exclude``.

Everything returned is sent to the model endpoint, so a line that looks like a credential
(:func:`hailer.context.looks_like_secret`, and every line of a private-key block) is replaced by
:data:`WITHHELD`. Results are capped: :data:`MAX_LISTED` paths, :data:`MAX_MATCHES` matching lines,
:data:`MAX_READ_LINES` lines per read and the tool output budget the caller gives.
"""

from __future__ import annotations

import os
import re
import subprocess
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from .context import looks_like_secret
from .errors import HailerError
from .models import HailerConfig
from .statedir import local_config_path, state_dir

#: Folders whose files are never project files: version control, environments, caches.
SKIPPED_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", ".venv", "venv", "node_modules", "site-packages", "__pycache__", ".mypy_cache",
        ".pytest_cache", ".ruff_cache", ".tox", ".nox", ".cache", "__marimo__", ".marimo", ".ipynb_checkpoints",
        ".idea", ".vscode", ".gradle", ".terraform",
    }
)  # fmt: skip
#: Code, docs and configuration: the files read when ``[project].include`` is not set.
TEXT_SUFFIXES = frozenset(
    {
        ".py", ".pyi", ".ipynb", ".md", ".markdown", ".rst", ".txt", ".adoc", ".toml", ".yaml", ".yml", ".json",
        ".ini", ".cfg", ".conf", ".properties", ".sql", ".jinja", ".j2", ".sh", ".bash", ".ps1", ".bat", ".r",
        ".rmd", ".qmd", ".jl", ".scala", ".java", ".kt", ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cs", ".rb",
        ".php", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".vue", ".svelte", ".html", ".css", ".scss",
        ".graphql", ".proto", ".tf", ".hcl", ".dockerfile", ".xml", ".lkml", ".dax", ".m", ".vb", ".swift",
    }
)  # fmt: skip
#: Files without one of those suffixes that are read anyway.
TEXT_NAMES = frozenset(
    {"dockerfile", "makefile", "justfile", "procfile", "jenkinsfile", "readme", "license", "codeowners", "changelog"}
)
#: The largest file listed, searched or read.
MAX_FILE_BYTES = 1024 * 1024
#: At most this many files are considered (a larger project is cut there, with a note).
MAX_FILES = 50_000
MAX_LISTED = 200
MAX_MATCHES = 100
MAX_LINE_CHARS = 240
MAX_READ_LINES = 400
#: Source code, where a ``token = ...`` line is a credential only with a quoted value
#: (:func:`hailer.context.looks_like_secret` with ``code=True``).
CODE_SUFFIXES = frozenset(
    {
        ".py", ".pyi", ".js", ".mjs", ".cjs", ".ts", ".tsx", ".jsx", ".vue", ".svelte", ".java", ".kt", ".scala",
        ".go", ".rs", ".c", ".h", ".cpp", ".hpp", ".cs", ".rb", ".php", ".swift", ".r", ".jl", ".sql", ".m", ".vb",
    }
)  # fmt: skip
#: How long a search may take before it reports what it found so far.
SEARCH_BUDGET_SEC = 15.0
#: How long a listing is reused before the folder is looked at again.
REFRESH_SEC = 30.0
GIT_TIMEOUT_SEC = 20.0
WITHHELD = "[line withheld: it looks like a credential]"

_SECRET_NAME_RE = re.compile(
    r"""(?ix)^(?:
        \.env(?:\..+)?
      | .+\.(?:pem|key|p12|pfx|jks|keystore|kdbx|tfstate|tfvars|ovpn|asc|gpg)
      | .+\.tfstate\.backup
      | id_(?:rsa|dsa|ecdsa|ed25519)(?:\.pub)?
      | \.netrc | _netrc | \.pgpass | \.pypirc | \.npmrc | \.git-credentials | \.htpasswd
      | credentials(?:\..+)?
      | secrets?\.(?:toml|ya?ml|json|env|ini|cfg|txt)
      | .*service[-_]?account.*\.json
    )$"""
)
_ENV_EXAMPLES = frozenset({".env.example", ".env.sample", ".env.template", ".env.dist", ".env.defaults"})
_KEY_BLOCK_BEGIN = re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY( BLOCK)?-----")
_KEY_BLOCK_END = re.compile(r"-----END [A-Z0-9 ]*PRIVATE KEY( BLOCK)?-----")
_KEY_BODY = re.compile(r"[A-Za-z0-9+/=]{16,}")
_KEY_HEADERS = ("Proc-Type:", "DEK-Info:", "Comment:", "Version:")


def secret_named(name: str) -> bool:
    """Whether a file name looks like a credential (``.env``, keys, ``credentials``, ...)."""
    return name.lower() not in _ENV_EXAMPLES and bool(_SECRET_NAME_RE.match(name))


def glob_matcher(pattern: str) -> re.Pattern[str]:
    """A case-insensitive regex for a glob over POSIX paths relative to the root.

    ``*`` and ``?`` stay inside one folder, ``**`` spans folders. A pattern without ``/`` (or with
    only a trailing one) matches any part of the path (``*.sql``, ``fixtures``, ``docs/``); one with
    ``/`` matches from the root (``packages/etl/**/*.py``, ``./README.md``). A pattern that matches a
    folder matches everything inside it.
    """
    glob = pattern.strip().replace("\\", "/")
    anchored = "/" in glob.rstrip("/")  # "docs/" is any folder named docs; "./docs" is the root's
    while glob.startswith("./"):
        glob = glob[2:]
    glob = glob.strip("/") or "**"
    out: list[str] = []
    i = 0
    while i < len(glob):
        if glob.startswith("**/", i):
            out.append("(?:[^/]*/)*")
            i += 3
        elif glob.startswith("**", i):
            out.append(".*")
            i += 2
        elif glob[i] == "*":
            out.append("[^/]*")
            i += 1
        elif glob[i] == "?":
            out.append("[^/]")
            i += 1
        else:
            out.append(re.escape(glob[i]))
            i += 1
    anchor = "" if anchored else "(?:.*/)?"
    return re.compile(anchor + "".join(out) + r"(?:/.*)?\Z", re.IGNORECASE)


def _matches_any(path: str, patterns: Iterable[re.Pattern[str]]) -> bool:
    return any(p.match(path) for p in patterns)


def _default_text_file(name: str) -> bool:
    lower = name.lower()
    suffix = os.path.splitext(lower)[1]
    return suffix in TEXT_SUFFIXES or lower in TEXT_NAMES or os.path.splitext(lower)[0] in TEXT_NAMES or lower in _ENV_EXAMPLES


def _key_material(line: str) -> bool:
    """Whether a line can be part of a PEM key's body: base64, a header such as ``Proc-Type:``, or
    blank, also inside a string literal (quotes, commas, ``\\n``)."""
    text = line.strip().strip("'\"`,;+()").replace("\\n", "").strip()
    return not text or text.startswith(_KEY_HEADERS) or bool(_KEY_BODY.fullmatch(text))


def withheld(lines: Iterable[str], *, code: bool = False) -> Iterator[bool]:
    """For each line, whether it is withheld: it looks like a credential (``code``: the lines are
    source code), or it is part of a private-key block (from its BEGIN line to its END line, for as
    long as the lines look like key material)."""
    in_key = False
    for line in lines:
        if in_key and (_KEY_BLOCK_END.search(line) or _key_material(line)):
            in_key = not _KEY_BLOCK_END.search(line)
            yield True
            continue
        in_key = False
        if _KEY_BLOCK_BEGIN.search(line):
            in_key = not _KEY_BLOCK_END.search(line)
            yield True
        else:
            yield looks_like_secret(line, code=code)


def redact(lines: Iterable[str], *, code: bool = False) -> Iterator[str]:
    """``lines`` with every :func:`withheld` one replaced by :data:`WITHHELD`."""
    lines = list(lines)
    for line, hide in zip(lines, withheld(lines, code=code), strict=True):
        yield WITHHELD if hide else line


def _is_code(name: str) -> bool:
    return os.path.splitext(name.lower())[1] in CODE_SUFFIXES


def _clip(line: str, limit: int = MAX_LINE_CHARS) -> str:
    line = line.rstrip("\r\n")
    return line if len(line) <= limit else line[: limit - 1] + "…"


def _size(size: int) -> str:
    return f"{size} B" if size < 1024 else f"{size / 1024:.0f} KB" if size < 1024 * 1024 else f"{size / 1024 / 1024:.1f} MB"


@dataclass(frozen=True)
class _Listing:
    files: dict[str, int]  # name -> size
    by_folded: dict[str, str]  # casefolded name -> name
    method: str  # "git" or "walk"
    truncated: bool
    at: float


class ProjectFiles:
    """The project files of one configuration (see the module docstring); thread-safe."""

    def __init__(
        self,
        root: Path,
        *,
        include: Iterable[str] = (),
        exclude: Iterable[str] = (),
        excluded_folders: Iterable[Path] = (),
        git: str = "git",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.root = Path(root).resolve()
        self._include = [glob_matcher(g) for g in include]
        self._exclude = [glob_matcher(g) for g in exclude]
        self._excluded = self._relative_folders(excluded_folders)
        self._git = git
        self._clock = clock
        self._lock = threading.Lock()
        self._listing: _Listing | None = None

    @classmethod
    def for_config(cls, config: HailerConfig) -> ProjectFiles:
        """The project files ``config`` allows; ``HailerError`` when ``[project]`` is off."""
        project = config.project
        if project.root is None:
            raise HailerError(
                "Project files are not available: the config has no [project] section.",
                hint='Tell the user that adding [project] with root = "." to the Hailer config lets you read the '
                "project's code and docs (read-only; what you read is sent to the model).",
            )
        workspace = Path(config.workspace)
        excluded = [state_dir(workspace), local_config_path(workspace), config.notebooks_root, config.data_dir]
        excluded += [folder.path for folder in getattr(config, "data_folders", ()) if folder.path is not None]
        return cls(project.root, include=project.include, exclude=project.exclude, excluded_folders=excluded)

    # -- which files --------------------------------------------------------- #

    def _relative_folders(self, folders: Iterable[Path]) -> list[str]:
        """The casefolded POSIX paths, relative to the root, of the folders (or files) inside it."""
        found: list[str] = []
        for folder in folders:
            try:
                rel = Path(folder).resolve().relative_to(self.root)
            except (OSError, ValueError):
                continue  # outside the root: nothing of it is listed anyway
            text = rel.as_posix()
            found.append("" if text == "." else text.casefold())
        return found

    def _excluded_path(self, name: str) -> bool:
        folded = name.casefold()
        return any(prefix == "" or folded == prefix or folded.startswith(prefix + "/") for prefix in self._excluded)

    def _git_names(self) -> list[str] | None:
        """The names git reports for the root (tracked, or untracked and not ignored); ``None``
        when the root is not in a git work tree or git is missing."""
        cmd = [
            self._git, "-c", "core.fsmonitor=false", "-c", "core.quotepath=off", "-C", str(self.root),
            "ls-files", "-z", "--cached", "--others", "--exclude-standard",
        ]  # fmt: skip
        env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "GIT_OPTIONAL_LOCKS": "0"}
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=GIT_TIMEOUT_SEC, env=env, check=False)
        except (OSError, subprocess.SubprocessError):
            return None
        if result.returncode != 0:
            return None
        names = result.stdout.decode("utf-8", errors="replace").split("\0")
        return [name for name in names if name and "�" not in name]

    def _walk_names(self) -> Iterator[str]:
        for folder, dirs, files in os.walk(self.root, followlinks=False):
            dirs[:] = sorted(d for d in dirs if d not in SKIPPED_DIRS and not (Path(folder) / d / "pyvenv.cfg").exists())
            rel = Path(folder).relative_to(self.root)
            for name in sorted(files):
                yield (rel / name).as_posix()

    def _wanted(self, name: str) -> bool:
        parts = name.split("/")
        if any(part in SKIPPED_DIRS for part in parts[:-1]) or secret_named(parts[-1]):
            return False
        if self._excluded_path(name) or _matches_any(name, self._exclude):
            return False
        return _matches_any(name, self._include) if self._include else _default_text_file(parts[-1])

    def _scan(self) -> _Listing:
        names = self._git_names()
        method = "git"
        if names is None:
            names, method = self._walk_names(), "walk"
        files: dict[str, int] = {}
        truncated = False
        for name in names:
            if not self._wanted(name):
                continue
            path = self.root / name
            try:
                if path.is_symlink() and not path.resolve().is_relative_to(self.root):
                    continue  # a link out of the root
                stat = path.stat()
            except OSError:
                continue
            if not path.is_file() or stat.st_size > MAX_FILE_BYTES:
                continue
            if len(files) >= MAX_FILES:
                truncated = True
                break
            files[name] = stat.st_size
        ordered = dict(sorted(files.items(), key=lambda item: item[0].casefold()))
        return _Listing(ordered, {n.casefold(): n for n in ordered}, method, truncated, self._clock())

    def _current(self) -> _Listing:
        with self._lock:
            if self._listing is None or self._clock() - self._listing.at > REFRESH_SEC:
                if not self.root.is_dir():
                    raise HailerError(
                        f"The project folder {self.root} does not exist.",
                        hint="Tell the user to check [project].root in the Hailer config.",
                    )
                self._listing = self._scan()
            return self._listing

    def files(self) -> dict[str, int]:
        """Every project file the agent may read: POSIX name relative to the root -> size in bytes."""
        return self._current().files

    def _resolve_name(self, path: str) -> str | None:
        name = path.strip().replace("\\", "/")
        while name.startswith("./"):
            name = name[2:]
        root_text = self.root.as_posix().rstrip("/") + "/"
        if name.casefold().startswith(root_text.casefold()):
            name = name[len(root_text):]  # an absolute path inside the root
        listing = self._current()
        return name if name in listing.files else listing.by_folded.get(name.casefold())

    # -- what the tools return ----------------------------------------------- #

    def _selected(self, pattern: str) -> list[str]:
        names = list(self.files())
        if not pattern.strip():
            return names
        matcher = glob_matcher(pattern)
        return [name for name in names if matcher.match(name)]

    def listing(self, pattern: str = "") -> str:
        """``project_files``: the matching names (with sizes), or a summary by folder when there are
        more than :data:`MAX_LISTED`."""
        listing = self._current()
        names = self._selected(pattern)
        where = f"under {self.root}" + (f" matching {pattern.strip()!r}" if pattern.strip() else "")
        cut = " (the project has more files; only the first ones were looked at)" if listing.truncated else ""
        if not names:
            return f"No project files {where}.{cut} Try a broader pattern, or project_files() for an overview."
        if len(names) <= MAX_LISTED:
            lines = [f"{len(names)} project file(s) {where}{cut}:"]
            lines += [f"{name}  ({_size(listing.files[name])})" for name in names]
            return "\n".join(lines)
        depth, groups = 1, Counter[str]()
        while depth <= 3:
            groups = Counter("/".join(name.split("/")[:depth]) + ("/" if name.count("/") >= depth else "") for name in names)
            if len(groups) >= 8 or depth == 3:
                break
            depth += 1
        lines = [f"{len(names)} project files {where}{cut}; too many to list, so by folder (files in each):"]
        lines += [f"{group}  {count}" for group, count in sorted(groups.items(), key=lambda item: (-item[1], item[0]))[:60]]
        if len(groups) > 60:
            lines.append(f"... and {len(groups) - 60} more folders")
        lines.append("Narrow it with a pattern, e.g. project_files('packages/billing/**') or project_files('*.sql').")
        return "\n".join(lines)

    def _text(self, name: str) -> str | None:
        """A file's text, or ``None`` when it is binary or unreadable."""
        try:
            data = (self.root / name).read_bytes()
        except OSError:
            return None
        if b"\0" in data[:8192]:
            return None
        return data.decode("utf-8", errors="replace")

    def search(self, query: str, pattern: str = "", *, budget_sec: float = SEARCH_BUDGET_SEC) -> str:
        """``project_search``: up to :data:`MAX_MATCHES` lines matching ``query`` (a regular
        expression, case-insensitive; text that does not compile is searched literally)."""
        if not query.strip():
            raise HailerError("project_search needs something to search for.", hint="Pass a word, a name or a regular expression.")
        note = ""
        try:
            regex = re.compile(query, re.IGNORECASE)
        except re.error as err:
            regex = re.compile(re.escape(query), re.IGNORECASE)
            note = f" (not a valid regular expression: {err}; searched for the text as written)"
        names = self._selected(pattern)
        deadline = self._clock() + budget_sec
        found: list[str] = []
        files_with = searched = hidden = 0
        stopped = ""
        for name in names:
            if self._clock() > deadline:
                stopped = f"Stopped after {budget_sec:.0f} s; narrow the files with a pattern to search the rest."
                break
            text = self._text(name)
            searched += 1
            if text is None:
                continue
            lines = text.splitlines()
            hits: list[str] = []
            for number, (line, hide) in enumerate(zip(lines, withheld(lines, code=_is_code(name)), strict=True), start=1):
                if regex.search(line):
                    if hide:
                        hidden += 1
                    else:
                        hits.append(f"{name}:{number}: {_clip(line)}")
            if hits:
                files_with += 1
                found.extend(hits[: MAX_MATCHES - len(found)])
            if len(found) >= MAX_MATCHES:
                stopped = f"Stopped at {MAX_MATCHES} matching lines; narrow the query or the files to see the rest."
                break
        scope = f" in files matching {pattern.strip()!r}" if pattern.strip() else ""
        held = f" {hidden} matching line(s) withheld: they look like credentials." if hidden else ""
        if not found:
            tail = f" {stopped}" if stopped else ""
            return f"No matches for {query!r}{scope} in {searched} project file(s){note}.{held}{tail}"
        head = f"{len(found)} matching line(s) in {files_with} file(s){scope}{note}:"
        return "\n".join([head, *found, *([held.strip()] if held else []), *([stopped] if stopped else [])])

    def read(self, path: str, start_line: int = 1, end_line: int = 0, *, max_chars: int = 12_000) -> str:
        """``project_read``: lines ``start_line``..``end_line`` (1-based, inclusive; 0: as many as
        fit in ``max_chars`` and :data:`MAX_READ_LINES`) of one project file, numbered."""
        name = self._resolve_name(path)
        if name is None:
            raise HailerError(
                f"{path!r} is not a project file you can read.",
                hint="Use a path exactly as project_files or project_search show it (relative to the project root). "
                "Hailer's state, the data and notebooks folders, files git ignores and files that look like "
                "credentials are never readable.",
            )
        text = self._text(name)
        if text is None:
            raise HailerError(f"{name} is a binary file.", hint="Only text files can be read.")
        lines = text.splitlines()
        total = len(lines)
        start = max(1, int(start_line or 1))
        if total == 0:
            return f"{name} is empty."
        if start > total:
            raise HailerError(f"{name} has {total} lines; start_line {start} is past the end.", hint=f"Use a start_line from 1 to {total}.")
        last = min(total, int(end_line) if end_line and int(end_line) >= start else start + MAX_READ_LINES - 1)
        last = min(last, start + MAX_READ_LINES - 1)
        budget = max(1000, max_chars - 300)
        out: list[str] = []
        used = 0
        shown_last = start - 1
        width = len(str(last))
        hide = list(withheld(lines, code=_is_code(name)))  # the whole file: a key block may start above the window
        for number in range(start, last + 1):
            line = WITHHELD if hide[number - 1] else lines[number - 1]
            row = f"{number:>{width}}  {_clip(line, 1000)}"
            if out and used + len(row) + 1 > budget:
                break
            out.append(row)
            used += len(row) + 1
            shown_last = number
        header = f"{name}, lines {start}-{shown_last} of {total}:"
        more = f"\n(more: project_read({name!r}, start_line={shown_last + 1}))" if shown_last < total else ""
        return header + "\n" + "\n".join(out) + more


__all__ = [
    "MAX_FILE_BYTES",
    "MAX_LISTED",
    "MAX_MATCHES",
    "MAX_READ_LINES",
    "SKIPPED_DIRS",
    "TEXT_NAMES",
    "TEXT_SUFFIXES",
    "WITHHELD",
    "ProjectFiles",
    "glob_matcher",
    "redact",
    "secret_named",
    "withheld",
]
