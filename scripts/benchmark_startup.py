"""Measure a fresh process's composer and real agent startup without model calls.

Run ``uv run --locked python scripts/benchmark_startup.py --samples 3``.
Each sample uses temporary conversation state, a dummy provider key, pipe input
and a recorded terminal. Socket connections are forbidden. The notebook wait
is simulated; this does not measure uvx installation, Docker or browser launch.
Use ``hailer --verbose`` for milestones from an actual session.
"""

from __future__ import annotations

import time

_PROCESS_STARTED = time.perf_counter()

import argparse
import asyncio
import io
import json
import statistics
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace


async def _sample(notebook_delay: float) -> dict[str, float]:
    def no_connections(event: str, _args: object) -> None:
        if event == "socket.connect":
            raise RuntimeError("The startup benchmark forbids network connections")

    # Install after asyncio has created its Windows self-pipe, but before any
    # dependency imports can run, including on the warmup worker.
    sys.addaudithook(no_connections)
    from hailer.cli import common
    from hailer.agent import HailerAgent, start_dependency_warmup
    from hailer.startup import StartupTimings

    timings = StartupTimings(started_at=_PROCESS_STARTED)
    start_dependency_warmup()

    from prompt_toolkit.data_structures import Size
    from prompt_toolkit.input import create_pipe_input
    from prompt_toolkit.output.vt100 import Vt100_Output
    from rich.console import Console

    from hailer.chat import ChatController
    from hailer.chat_ui import ChatUI
    from hailer.models import HailerConfig, ModelConfig, ProviderConfig

    ticks: list[float] = []

    async def heartbeat() -> None:
        while True:
            ticks.append(time.perf_counter())
            await asyncio.sleep(0.01)

    async def until(predicate) -> None:
        async with asyncio.timeout(30):
            while not predicate():
                await asyncio.sleep(0.005)

    with tempfile.TemporaryDirectory(prefix="hailer-startup-") as folder:
        workspace = Path(folder)
        config = HailerConfig(
            workspace=workspace,
            notebook=workspace / "notebooks" / "analysis.py",
            data_dir=workspace / "data",
            context_dir=workspace / "context",
            skills_dir=workspace / "skills",
            prompts_dir=workspace / "prompts",
            model=ModelConfig(name="startup-probe", provider="probe"),
            providers={"probe": ProviderConfig(
                id="probe", env_key="HAILER_STARTUP_PROBE_KEY", base_url="http://127.0.0.1:1/v1",
            )},
        )
        services = SimpleNamespace(**vars(common))
        services._make_agent = lambda config, bundle, sandbox: HailerAgent(
            config, bundle, sandbox=sandbox, env={"HAILER_STARTUP_PROBE_KEY": "dummy-offline-probe"},
        )
        transcript, screen = io.StringIO(), io.StringIO()
        console = Console(file=transcript, force_terminal=False, color_system=None)
        controller = ChatController(console, config, common.CliOptions(), services=services)
        with create_pipe_input() as keys:
            output = Vt100_Output(screen, lambda: Size(rows=24, columns=100), term="xterm-256color")
            made: list[ChatUI] = []
            draft = "First question\nwith pasted text"
            sent_at: float | None = None
            accepted_at: float | None = None

            def factory(console, on_submit, context):
                ui = ChatUI(console, on_submit, context, pt_input=keys, pt_output=output)
                made.append(ui)

                def send_on_first_render(_application) -> None:
                    nonlocal sent_at
                    if sent_at is None:
                        sent_at = time.perf_counter()
                        keys.send_text("\x1b[200~" + draft + "\x1b[201~")

                def record_paste(buffer) -> None:
                    nonlocal accepted_at
                    if accepted_at is None and buffer.text == draft:
                        accepted_at = time.perf_counter()

                ui.application.after_render += send_on_first_render
                ui.buffer.on_text_changed += record_paste
                return ui

            async def prepare_notebook() -> None:
                await asyncio.sleep(notebook_delay)

            # Begin observing before startup can block the loop. Waiting for a
            # polling task to notice input_ready would miss an immediate stall.
            pulse = asyncio.create_task(heartbeat())
            await asyncio.sleep(0)
            running = asyncio.create_task(controller.run_interactive(
                ui_factory=factory, prepare_notebook=prepare_notebook, timings=timings,
            ))
            try:
                await until(lambda: accepted_at is not None)
                assert accepted_at is not None and sent_at is not None
                draft_latency = accepted_at - sent_at
                await until(lambda: "ready_to_answer" in timings.milestones)
                await asyncio.sleep(0.02)
                made[0].exit()
                await asyncio.wait_for(running, 10)
            finally:
                if not running.done():
                    running.cancel()
                try:
                    await running
                except asyncio.CancelledError:
                    pass
                pulse.cancel()
                try:
                    await pulse
                except asyncio.CancelledError:
                    pass
    input_ready_at = timings.started_at + timings.milestones["input_ready"]
    return {
        **timings.milestones,
        "paste_latency": draft_latency,
        "max_input_loop_pause": max(
            (b - max(a, input_ready_at) for a, b in zip(ticks, ticks[1:]) if b > input_ready_at),
            default=0.0,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=3)
    parser.add_argument("--notebook-delay", type=float, default=0.75)
    parser.add_argument("--sample", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.samples < 1 or args.notebook_delay < 0:
        parser.error("samples must be positive and notebook-delay must be nonnegative")
    if args.sample:
        result = asyncio.run(_sample(args.notebook_delay))
        print(json.dumps({key: round(value, 4) for key, value in result.items()}))
        return
    results: list[dict[str, float]] = []
    for index in range(args.samples):
        process = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--sample", "--notebook-delay", str(args.notebook_delay)],
            capture_output=True, text=True, timeout=60,
        )
        if process.returncode:
            raise SystemExit(process.stderr or process.stdout or "Startup benchmark failed")
        result = json.loads(process.stdout)
        results.append(result)
        print(json.dumps({"sample": index + 1, "seconds": result}), flush=True)
    print(json.dumps({"median_seconds": {
        key: round(statistics.median(result[key] for result in results), 4)
        for key in results[0]
    }}))


if __name__ == "__main__":
    main()
