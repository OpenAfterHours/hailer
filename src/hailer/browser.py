"""Open a URL in the user's browser without touching this process's stdio.

Used by the CLI and by the agent's notebook tools while a turn is running: a launcher that
inherited the terminal would write into the chat. ``open_url`` never raises and never prints.

On Windows ``os.startfile`` is tried first. ``webbrowser`` honours the ``BROWSER`` environment
variable, which Git Bash and WSL profiles sometimes set to a non-GUI command such as ``true``;
``os.startfile`` ignores it and hands the URL to the default browser. Elsewhere ``webbrowser``
picks the launcher, but generic launchers (``xdg-open``, ``open``) are spawned here with their
stdio discarded instead of inheriting ours.
"""

from __future__ import annotations

import os
import subprocess
import webbrowser
from collections.abc import Callable
from typing import Any

#: Injection points for tests; production code never reassigns them.
is_windows: bool = os.name == "nt"
startfile: Callable[[str], Any] | None = getattr(os, "startfile", None)
webbrowser_open: Callable[[str], bool] = webbrowser.open
webbrowser_get: Callable[[], Any] = webbrowser.get
popen: Callable[..., Any] = subprocess.Popen


def open_url(url: str) -> bool:
    """Launch ``url`` in the default browser. ``True`` when a launcher was started; never raises."""
    try:
        if is_windows:
            return _open_windows(url)
        return _open_posix(url)
    except Exception:  # noqa: BLE001 - a browser problem must never fail the caller
        return False


def _open_windows(url: str) -> bool:
    if startfile is not None:
        try:
            startfile(url)
            return True
        except OSError:
            pass
    return bool(webbrowser_open(url))


def _open_posix(url: str) -> bool:
    browser = webbrowser_get()  # webbrowser.Error when no browser is registered
    if isinstance(browser, webbrowser.GenericBrowser):
        # xdg-open / open and friends: run detached with stdio discarded.
        args = [arg.replace("%s", url) for arg in browser.args] or [url]
        popen(
            [browser.name, *args],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        return True
    return bool(browser.open(url))


__all__ = ["open_url"]
