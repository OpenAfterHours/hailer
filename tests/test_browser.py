"""Tests for hailer.browser.open_url: never raises, never inherits stdio, ignores BROWSER on Windows."""

from __future__ import annotations

import subprocess
import webbrowser

import pytest

from hailer import browser


@pytest.fixture
def calls(monkeypatch):
    record: dict[str, list] = {"startfile": [], "open": [], "popen": []}
    monkeypatch.setattr(browser, "startfile", lambda url: record["startfile"].append(url))
    monkeypatch.setattr(browser, "webbrowser_open", lambda url: record["open"].append(url) or True)
    monkeypatch.setattr(browser, "popen", lambda *a, **kw: record["popen"].append((a, kw)))
    return record


def test_windows_uses_startfile_and_ignores_browser_env(calls, monkeypatch):
    monkeypatch.setattr(browser, "is_windows", True)
    monkeypatch.setenv("BROWSER", "true")  # a Git Bash profile setting that makes webbrowser a no-op
    assert browser.open_url("http://127.0.0.1:2718/?file=x.py") is True
    assert calls["startfile"] == ["http://127.0.0.1:2718/?file=x.py"]
    assert calls["open"] == [], "webbrowser is only the fallback"


def test_windows_falls_back_to_webbrowser_when_startfile_fails(calls, monkeypatch):
    monkeypatch.setattr(browser, "is_windows", True)

    def broken(url):
        raise OSError("no association")

    monkeypatch.setattr(browser, "startfile", broken)
    assert browser.open_url("http://x") is True
    assert calls["open"] == ["http://x"]
    monkeypatch.setattr(browser, "startfile", None)
    assert browser.open_url("http://y") is True
    assert calls["open"][-1] == "http://y"


def test_never_raises(calls, monkeypatch):
    monkeypatch.setattr(browser, "is_windows", True)

    def broken(url):
        raise RuntimeError("boom")

    monkeypatch.setattr(browser, "startfile", broken)
    monkeypatch.setattr(browser, "webbrowser_open", broken)
    assert browser.open_url("http://x") is False
    monkeypatch.setattr(browser, "is_windows", False)
    monkeypatch.setattr(browser, "webbrowser_get", lambda: (_ for _ in ()).throw(webbrowser.Error("no browser")))
    assert browser.open_url("http://x") is False


def test_posix_generic_launcher_is_detached_with_stdio_discarded(calls, monkeypatch):
    monkeypatch.setattr(browser, "is_windows", False)
    monkeypatch.setattr(browser, "webbrowser_get", lambda: webbrowser.BackgroundBrowser("xdg-open"))
    assert browser.open_url("http://x") is True
    (args, kwargs) = calls["popen"][0]
    assert args[0] == ["xdg-open", "http://x"]
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert kwargs["stdout"] is subprocess.DEVNULL
    assert kwargs["stderr"] is subprocess.DEVNULL
    assert kwargs["start_new_session"] is True
    assert calls["startfile"] == [] and calls["open"] == []


def test_posix_dedicated_browser_uses_its_own_open(calls, monkeypatch):
    monkeypatch.setattr(browser, "is_windows", False)

    class Dedicated:
        opened: list[str] = []

        def open(self, url, new=0, autoraise=True):
            self.opened.append(url)
            return True

    monkeypatch.setattr(browser, "webbrowser_get", lambda: Dedicated())
    assert browser.open_url("http://x") is True
    assert Dedicated.opened == ["http://x"] and calls["popen"] == []
