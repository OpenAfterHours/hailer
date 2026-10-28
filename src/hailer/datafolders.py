"""This machine's ``[data_folders]``: ``uvx hailer data add | remove`` and ``/data add`` edit them in
``.hailer/local.toml`` (:func:`hailer.statedir.local_config_path`), which is never committed.

The file may hold other settings and the user's comments, so it is edited line by line, not
rewritten: the ``name = ...`` line in its ``[data_folders]`` table is replaced, added or removed,
and the result is parsed again before it is written (atomically). Anything unexpected (a
``[data_folders.<name>]`` table, invalid TOML) is refused with the advice to edit the file by hand.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
import tomllib
from pathlib import Path
from typing import Any

from hailer.config import DATA_FOLDER_NAME_RE, _mount_problems, _on_windows, data_folder_fix, load_config
from hailer.errors import ConfigError
from hailer.models import HailerConfig
from hailer.statedir import ensure_project_gitignore, ensure_state_dir, is_project_layout, local_config_path

_HEADER_RE = re.compile(r"^\s*\[\s*data_folders\s*\]\s*(?:#.*)?$")
_ANY_HEADER_RE = re.compile(r"^\s*\[")
LOCAL_HEADER = "# This machine's Hailer settings, on top of the project's config. Never committed.\n"


def check_name(name: str) -> str:
    name = name.strip()
    if not DATA_FOLDER_NAME_RE.match(name):
        raise ConfigError(
            f"{name!r} is not a data folder name.",
            hint="Use lower-case letters, digits, - and _ (up to 40, starting with a letter or digit), e.g. finance or crm-exports.",
        )
    return name


def _key_re(name: str) -> re.Pattern[str]:
    quoted = re.escape(name)
    return re.compile(rf"""^\s*(?:{quoted}|"{quoted}"|'{quoted}')\s*=""")


def _section(lines: list[str]) -> tuple[int, int] | None:
    """(header index, end index) of the ``[data_folders]`` table, or ``None``."""
    start = next((i for i, line in enumerate(lines) if _HEADER_RE.match(line)), None)
    if start is None:
        return None
    end = next((i for i in range(start + 1, len(lines)) if _ANY_HEADER_RE.match(lines[i])), len(lines))
    return start, end


def _value(path: Path, description: str | None) -> str:
    if description:
        return f"{{ path = {json.dumps(str(path), ensure_ascii=False)}, description = {json.dumps(description, ensure_ascii=False)} }}"
    return json.dumps(str(path), ensure_ascii=False)  # a JSON string is a TOML basic string


def _with_entry(text: str, name: str, value: str) -> str:
    lines = text.splitlines()
    entry = f"{name} = {value}"
    found = _section(lines)
    if found is None:
        while lines and not lines[-1].strip():
            lines.pop()
        lines += ([""] if lines else []) + ["[data_folders]", entry]
    else:
        start, end = found
        key = _key_re(name)
        existing = next((i for i in range(start + 1, end) if key.match(lines[i])), None)
        if existing is not None:
            lines[existing] = entry
        else:
            last = max((i for i in range(start, end) if lines[i].strip()), default=start)
            lines.insert(last + 1, entry)
    return "\n".join(lines) + "\n"


def _without_entry(text: str, name: str) -> str:
    lines = text.splitlines()
    found = _section(lines)
    if found is not None:
        start, end = found
        key = _key_re(name)
        lines = [line for i, line in enumerate(lines) if not (start < i < end and key.match(line))]
    return "\n".join(lines) + ("\n" if lines else "")


def _folders_in(text: str, path: Path) -> dict[str, Any]:
    try:
        data = tomllib.loads(text)
    except tomllib.TOMLDecodeError as err:
        raise ConfigError(
            f"Could not update {path}: {err}.",
            hint=f"Edit {path} by hand: put the folder under [data_folders] as name = \"/path/to/folder\".",
        ) from err
    folders = data.get("data_folders", {})
    return folders if isinstance(folders, dict) else {}


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".local-", suffix=".toml", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(text)
        os.replace(temp, path)
    except BaseException:
        try:
            os.unlink(temp)
        except OSError:
            pass
        raise


def _prepare_folder(workspace: Path) -> None:
    """Make sure git ignores ``local.toml``: the state folder's ``*`` (standalone layout) or
    ``.hailer/.gitignore`` (project layout)."""
    if is_project_layout(workspace):
        ensure_project_gitignore(workspace)
    else:
        ensure_state_dir(workspace)


def add_data_folder(config: HailerConfig, name: str, folder: Path | str, *, description: str | None = None) -> Path:
    """Point the data folder ``name`` at ``folder`` (an existing folder; relative to the current
    folder) in ``.hailer/local.toml``; returns the folder as written. ``ConfigError`` for a bad name,
    a missing folder, a folder a kernel may not mount, or a file this cannot edit safely."""
    name = check_name(name)
    target = Path(folder).expanduser()
    target = (target if target.is_absolute() else Path.cwd() / target).resolve()
    if not target.is_dir():
        raise ConfigError(f"{target} is not a folder.", hint="Give the path of an existing folder with the data files.")
    problems = _mount_problems(config, target, f"The data folder {name!r}", _on_windows(), fix="")
    if problems:
        raise ConfigError(problems[0].rstrip(), hint="Choose a folder that holds only the data to analyse.")
    workspace = Path(config.workspace)
    path = local_config_path(workspace)
    text = path.read_text(encoding="utf-8") if path.is_file() else LOCAL_HEADER
    updated = _with_entry(text, name, _value(target, description))
    entry = _folders_in(updated, path).get(name)
    written = entry.get("path") if isinstance(entry, dict) else entry
    if written != str(target):
        raise ConfigError(f"Could not update {path} safely.", hint=f'Edit it by hand: under [data_folders], write {name} = "{target}".')
    _prepare_folder(workspace)
    _write(path, updated)
    return target


def remove_data_folder(config: HailerConfig, name: str) -> bool:
    """Remove the data folder ``name`` from ``.hailer/local.toml``; ``False`` when it is not there.
    ``ConfigError`` when the file holds it in a form this does not edit."""
    name = check_name(name)
    path = local_config_path(Path(config.workspace))
    if not path.is_file():
        return False
    text = path.read_text(encoding="utf-8")
    if name not in _folders_in(text, path):
        return False
    updated = _without_entry(text, name)
    if name in _folders_in(updated, path):
        raise ConfigError(f"Could not remove {name!r} from {path} safely.", hint=f"Edit {path} by hand: delete its entry under [data_folders].")
    _write(path, updated)
    return True


def reload(config: HailerConfig) -> HailerConfig:
    """``config`` loaded again from its files (after an edit)."""
    return load_config(workspace=config.workspace, config_path=config.config_path)


__all__ = ["LOCAL_HEADER", "add_data_folder", "check_name", "reload", "remove_data_folder"]
