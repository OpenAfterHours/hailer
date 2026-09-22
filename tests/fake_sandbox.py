"""A sandbox over a host folder (test helper): the :class:`~hailer.sandbox.Sandbox` contract for
the tests of what holds one (the tools, the chat, the CLI), without a marimo server.

Notebook files are the folder's (listed with marimo's rules: ``.py`` files holding
``import marimo`` and ``marimo.App``, folders starting with ``.`` or ``__`` skipped), the data
operations are :class:`~hailer.sandbox.MarimoSandbox`'s own (the host data folder), and
:meth:`client` returns whatever ``client_factory`` makes, so a test decides which sessions exist.
The real notebook operations over marimo's file API have tests/test_sandbox.py.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hailer.errors import NotebookExistsError, NotebookNotFoundError
from hailer.models import MarimoServer
from hailer.notebooks import check_notebook_name
from hailer.sandbox import MarimoSandbox, NotebookFile, SandboxEntry


def _is_notebook(path: Path) -> bool:
    try:
        head = path.read_bytes()[: 1024 * 1024]
    except OSError:
        return False
    return path.suffix == ".py" and b"import marimo" in head and b"marimo.App" in head


@dataclass
class FolderSandbox(MarimoSandbox):
    """``folder`` holds the notebooks; ``client_factory(notebook=..., token_in_links=...)`` makes the
    kernel's clients (default: a real client for ``server``). Counts stops and listings."""

    folder: Path | None = None
    client_factory: Callable[..., Any] | None = None
    stop_error: str = ""
    stopped: int = 0
    listings: int = 0
    writes: list[str] = field(default_factory=list)

    def client(self, *, notebook: str | None = None, token_in_links: bool = False, timeout: float = 10.0) -> Any:
        if self.client_factory is None:
            return super().client(notebook=notebook, token_in_links=token_in_links, timeout=timeout)
        return self.client_factory(notebook=notebook, token_in_links=token_in_links)

    def stop(self) -> None:
        self.stopped += 1

    def list_notebooks(self) -> list[NotebookFile]:
        self.listings += 1
        root = Path(self.folder)
        found: list[NotebookFile] = []
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith((".", "__"))]
            for filename in filenames:
                path = Path(dirpath) / filename
                if _is_notebook(path):
                    stat = path.stat()
                    found.append(NotebookFile(path.relative_to(root).as_posix(), stat.st_mtime, stat.st_size))
        return sorted(found, key=lambda info: info.name.lower())

    def list_tree(self) -> list[SandboxEntry]:
        """Every file and folder (not looking into ``.``/``__`` folders), as marimo's listing gives them."""
        root = Path(self.folder)
        found: list[SandboxEntry] = []
        for dirpath, dirnames, filenames in os.walk(root):
            for dirname in dirnames:
                found.append(SandboxEntry((Path(dirpath) / dirname).relative_to(root).as_posix(), True))
            dirnames[:] = [d for d in dirnames if not d.startswith((".", "__"))]
            for filename in filenames:
                path = Path(dirpath) / filename
                stat = path.stat()
                found.append(SandboxEntry(path.relative_to(root).as_posix(), False, stat.st_mtime, stat.st_size))
        return sorted(found, key=lambda item: item.name)

    def has_notebook(self, name: str) -> bool:
        return (Path(self.folder) / check_notebook_name(name)).is_file()

    def read_notebook(self, name: str) -> str:
        path = Path(self.folder) / check_notebook_name(name)
        if not path.is_file():
            raise NotebookNotFoundError(f"No notebook named {name} in the notebooks folder.")
        return path.read_text(encoding="utf-8")

    def write_notebook(self, name: str, source: str, *, replace: bool = True) -> NotebookFile:
        name = check_notebook_name(name)
        path = Path(self.folder) / name
        if path.exists() and not replace:
            raise NotebookExistsError(f"{name} already exists in the notebooks folder.", hint="Pick another name, or open the existing notebook instead.")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source, encoding="utf-8", newline="\n")
        self.writes.append(name)
        return NotebookFile(name, path.stat().st_mtime, path.stat().st_size)


def with_folder_files(box: MarimoSandbox, folder: Path, client_factory: Callable[..., Any] | None = None) -> MarimoSandbox:
    """``box`` (a real runtime's sandbox) with its notebook files served from ``folder`` and, with
    ``client_factory``, its clients faked: for CLI tests that run a real runtime start without a
    marimo server behind it."""
    view = FolderSandbox(server=box.server, notebooks_path=box.notebooks_path, folder=folder, client_factory=client_factory)
    box.list_notebooks = view.list_notebooks  # type: ignore[method-assign]
    box.list_tree = view.list_tree  # type: ignore[method-assign]
    box.read_notebook = view.read_notebook  # type: ignore[method-assign]
    box.has_notebook = view.has_notebook  # type: ignore[method-assign]
    box.write_notebook = view.write_notebook  # type: ignore[method-assign]
    if client_factory is not None:
        box.client = view.client  # type: ignore[method-assign]
    return box


def folder_sandbox(config: Any, server: MarimoServer | None = None, *, docker: bool = False, **kw: Any) -> FolderSandbox:
    """A :class:`FolderSandbox` on ``config``'s folders, as the local runtime (host paths) or, with
    ``docker``, as a container sees them (``/work/notebooks``, ``/work/data``)."""
    from hailer.marimo_client import notebook_file_key

    server = server or MarimoServer(url="http://127.0.0.1:2718", runtime="docker" if docker else "unsafe-local")
    return FolderSandbox(
        server=server,
        notebooks_path="/work/notebooks" if docker else notebook_file_key(config.notebooks_root),
        data_path="/work/data" if docker else str(config.data_dir),
        data_dir=config.data_dir,
        native_paths=not docker,
        folder=config.notebooks_root,
        **kw,
    )
