"""Exercise the real terminal composer without a model, API key or marimo.

Run ``python scripts/chat_ui_demo.py`` from the installed project environment.
Try typing or pasting a draft during a reply, resizing the terminal, selecting
transcript text, /long for scrollback, and Ctrl+C during /slow for cancellation.
The demo stops after three minutes by default; --timeout changes that limit.
"""

from __future__ import annotations

import argparse
import asyncio

from rich.console import Console

from hailer.chat_ui import ChatUI
from hailer.models import AgentEvent


async def demo(delay: float, timeout: float) -> None:
    console = Console(highlight=False, soft_wrap=True)
    state = {"notebook": "analysis.py", "model": "demo-model"}

    def context() -> str:
        return f"Notebook: {state['notebook']} | Model: {state['model']} | Context: 2 files"

    async def submit(text: str) -> None:
        command, _, argument = text.partition(" ")
        if command in {"/exit", "/quit"}:
            ui.console.print("Bye.", markup=False)
            ui.exit()
            return
        if command in {"/help", "/status", "/context"}:
            ui.console.print(context(), markup=False)
            ui.console.print(
                "/slow: cancellable turn | /long: long reply | /fail: simulated failure\n"
                "/notebook <name> | /model <name> | /clear | /exit",
                markup=False,
            )
            return
        if command in {"/notebook", "/model"}:
            if argument.strip():
                state[command[1:]] = argument.strip()
            ui.console.print(context(), markup=False)
            return
        if command == "/clear":
            ui.console.clear()
            return
        ui.on_event(AgentEvent("status", "Thinking..."))
        await asyncio.sleep(delay)
        ui.on_event(AgentEvent("tool_call", "demo_notebook_read"))
        await asyncio.sleep(30 if command == "/slow" else delay)
        if command == "/fail":
            raise RuntimeError("simulated failure; the draft remains editable")
        ui.on_event(AgentEvent("message_delta", "This event changes activity only."))
        await asyncio.sleep(delay)
        if command == "/long":
            final = "## Scrollback demo\n\n" + "\n".join(
                f"- Row {index}: Unicode \u03bb; select this text and scroll back to earlier messages."
                for index in range(1, 45)
            )
        else:
            final = (
                "**The simulated turn is complete.**\n\n"
                "- The input stayed editable during the turn.\n"
                "- Enter while busy keeps your draft.\n\n"
                "```python\nprint('Hailer composer demo')\n```"
            )
        ui.finish(final)

    ui = ChatUI(console, submit, context)
    ui.console.print("Hailer composer demo — simulated work only", style="bold", markup=False)
    ui.console.print("Type a message or /help. Ctrl+C cancels work; Ctrl+C while idle exits.\n", markup=False)
    try:
        await asyncio.wait_for(ui.run(), timeout)
    except TimeoutError:
        console.print(f"Demo stopped after {timeout:g} seconds.", markup=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--delay", type=float, default=0.75, help="seconds per simulated activity step")
    parser.add_argument("--timeout", type=float, default=180, help="maximum demo lifetime in seconds")
    args = parser.parse_args()
    if args.delay < 0 or args.timeout <= 0:
        parser.error("--delay must be nonnegative and --timeout must be positive")
    asyncio.run(demo(args.delay, args.timeout))
