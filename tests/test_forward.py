"""Tests for hailer._forward, the forwarder the docker kernel is reached through: real sockets on
127.0.0.1 (no Docker)."""

from __future__ import annotations

import ast
import asyncio
import socket
from pathlib import Path

import pytest

import hailer
from hailer import _forward

TIMEOUT = 15


async def _listening_port(server: asyncio.Server) -> int:
    return int(server.sockets[0].getsockname()[1])


def test_forwarder_pipes_bytes_both_ways_and_passes_the_half_close_on():
    """A request larger than one chunk goes up; the upstream sees EOF (it reads to the end) and
    its reply comes back down."""

    async def scenario() -> bytes:
        async def shout(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            data = await reader.read()  # until EOF: only arrives if the forwarder passes it on
            writer.write(data.upper())
            await writer.drain()
            writer.close()

        upstream = await asyncio.start_server(shout, "127.0.0.1", 0)
        forwarder = await _forward.start("127.0.0.1", await _listening_port(upstream), 0, bind="127.0.0.1")
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", await _listening_port(forwarder))
            writer.write(b"revenue by region " * 10_000)
            await writer.drain()
            writer.write_eof()
            reply = await asyncio.wait_for(reader.read(), TIMEOUT)
            writer.close()
            return reply
        finally:
            forwarder.close()
            upstream.close()

    assert asyncio.run(scenario()) == b"REVENUE BY REGION " * 10_000


def test_forwarder_serves_several_connections_at_once():
    async def scenario() -> list[bytes]:
        async def echo(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
            while line := await reader.readline():
                writer.write(line)
                await writer.drain()
            writer.close()

        upstream = await asyncio.start_server(echo, "127.0.0.1", 0)
        forwarder = await _forward.start("127.0.0.1", await _listening_port(upstream), 0, bind="127.0.0.1")
        port = await _listening_port(forwarder)
        try:
            first = await asyncio.open_connection("127.0.0.1", port)
            second = await asyncio.open_connection("127.0.0.1", port)
            replies = []
            for (reader, writer), text in ((second, b"websocket\n"), (first, b"health\n")):
                writer.write(text)
                await writer.drain()
                replies.append(await asyncio.wait_for(reader.readline(), TIMEOUT))
            for _reader, writer in (first, second):
                writer.close()
            return replies
        finally:
            forwarder.close()
            upstream.close()

    assert asyncio.run(scenario()) == [b"websocket\n", b"health\n"]


def test_forwarder_drops_a_client_while_the_kernel_is_not_listening():
    async def scenario() -> bytes:
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            dead_port = probe.getsockname()[1]  # nothing listens here once the socket closes
        forwarder = await _forward.start("127.0.0.1", dead_port, 0, bind="127.0.0.1")
        try:
            reader, writer = await asyncio.open_connection("127.0.0.1", await _listening_port(forwarder))
            try:
                data = await asyncio.wait_for(reader.read(), TIMEOUT)
            except ConnectionResetError:
                data = b""
            writer.close()
            return data
        finally:
            forwarder.close()

    assert asyncio.run(scenario()) == b""


def test_command_line():
    assert _forward.parse_args(["hailer-kernel-0123456789", "2718", "2718"]) == ("hailer-kernel-0123456789", 2718, 2718)
    for bad in ([], ["kernel", "2718"], ["kernel", "port", "2718"]):
        with pytest.raises(SystemExit) as exc:
            _forward.parse_args(bad)
        assert "usage: python -m hailer._forward" in str(exc.value)


def test_main_serves_what_the_command_line_names(monkeypatch):
    seen = []

    async def serve(host, port, listen, *, bind="0.0.0.0"):
        seen.append((host, port, listen, bind))

    monkeypatch.setattr(_forward, "serve", serve)
    assert _forward.main(["hailer-kernel-0123456789", "2718", "2718"]) == 0
    assert seen == [("hailer-kernel-0123456789", 2718, 2718, "0.0.0.0")], "every interface of its own container"


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module.split(".")[0])
    return names


def test_the_forwarder_needs_only_the_standard_library():
    """``python -m hailer._forward`` imports ``hailer/__init__.py`` first; neither may pull in anything
    beyond the standard library (the forwarder container must not depend on Hailer's dependencies)."""
    assert _imports(Path(_forward.__file__)) <= {"__future__", "asyncio", "sys", "collections"}
    assert _imports(Path(hailer.__file__)) == set()
