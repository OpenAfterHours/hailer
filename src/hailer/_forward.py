"""A TCP forwarder for the docker kernel runtime (standard library only).

Docker never publishes a port for a container on an ``--internal`` network, so the kernel
container cannot be offline and reachable at the same time on its own. A second container from
the same image runs this module: it is published on ``127.0.0.1`` on the host, joined to the
kernel's internal network, and pipes every connection it accepts to the kernel's marimo port
(HTTP, the browser's websocket and Hailer's streamed execute calls alike). It only ever connects
to that one address, so notebook code gains nothing by reaching it.

    python -m hailer._forward hailer-kernel-<id> 2718 2718

Arguments: the host to forward to, its port, and the port to listen on (every interface of the
forwarder's own container).
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Awaitable, Callable, Sequence

CHUNK_BYTES = 65536
USAGE = "usage: python -m hailer._forward <host> <port> <listen-port>"

Handler = Callable[[asyncio.StreamReader, asyncio.StreamWriter], Awaitable[None]]


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    """Copy ``reader`` to ``writer`` until EOF, then pass the EOF on (half-close)."""
    try:
        while data := await reader.read(CHUNK_BYTES):
            writer.write(data)
            await writer.drain()
        if writer.can_write_eof():
            writer.write_eof()
    except OSError:  # either side went away; the other direction ends by itself
        pass


def make_handler(host: str, port: int) -> Handler:
    """A connection handler that pipes each client to ``host:port`` and back."""

    async def handle(client_reader: asyncio.StreamReader, client_writer: asyncio.StreamWriter) -> None:
        try:
            upstream_reader, upstream_writer = await asyncio.open_connection(host, port)
        except OSError:  # the kernel is not listening yet (or any more): drop the client
            client_writer.close()
            return
        try:
            await asyncio.gather(pipe(client_reader, upstream_writer), pipe(upstream_reader, client_writer))
        finally:
            client_writer.close()
            upstream_writer.close()

    return handle


async def start(host: str, port: int, listen: int, *, bind: str = "0.0.0.0") -> asyncio.Server:
    """Listen on ``bind:listen`` and forward to ``host:port`` (``listen=0`` picks a free port)."""
    return await asyncio.start_server(make_handler(host, port), bind, listen)


async def serve(host: str, port: int, listen: int, *, bind: str = "0.0.0.0") -> None:
    server = await start(host, port, listen, bind=bind)
    async with server:
        await server.serve_forever()


def parse_args(argv: Sequence[str]) -> tuple[str, int, int]:
    """``(host, port, listen)`` from the command line; ``SystemExit`` with the usage otherwise."""
    if len(argv) != 3:
        raise SystemExit(USAGE)
    host, port, listen = argv
    try:
        return host, int(port), int(listen)
    except ValueError:
        raise SystemExit(USAGE) from None


def main(argv: Sequence[str] | None = None) -> int:
    host, port, listen = parse_args(sys.argv[1:] if argv is None else argv)
    try:
        asyncio.run(serve(host, port, listen))
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
