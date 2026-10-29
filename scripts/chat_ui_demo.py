"""Exercise the real terminal composer without a model, API key or marimo.

Run ``uv run python scripts/chat_ui_demo.py`` in a terminal. Turns are simulated with the agent's
own events (``tool_call`` / ``tool_result`` with ids and summaries), so the step rows, the live
row, the footer, the cards and the callouts are the ones ``uvx hailer`` prints.

Try: ask anything (parallel steps, then a reply with a table), /failstep (a step that fails and a
turn that carries on), /fail (an error callout), /slow then Esc or Ctrl+C (a cancelled step),
/status, /help, /notebook list, /notebook open <name>, /context, /long (scrollback), Enter while a
turn runs (your draft is kept), resizing the terminal, /exit. ``--unsafe-local`` shows the kernel
warning in the footer; ``PYTHONIOENCODING=ascii`` shows the ASCII glyphs a cp1252 console gets.
"""

from __future__ import annotations

import argparse
import asyncio
import time

from rich.console import Console
from rich.text import Text

from hailer.chat_ui import ChatUI
from hailer.cli.common import _print_error, _reconfigure_streams
from hailer.errors import HailerError
from hailer.models import AgentEvent, TurnSummary
from hailer.transcript import FooterData, NotebookRow, detail_card, help_card, notebook_table, status_card
from hailer.transcript import done_items, session_ended, session_note, session_row

URL = "http://127.0.0.1:2718/?file=/work/notebooks/{name}&view-as=present"

TOTALS_CODE = """\
from hailer.periods import load_periods
df = load_periods(DATA_DIR, "sales")
q = df.with_columns(quarter=pl.when(pl.col("period") <= "25-03").then(pl.lit("Q1")).otherwise(pl.lit("Q2")))
by_region = (
    q.group_by("region", "quarter").agg(pl.col("revenue").sum())
    .pivot(on="quarter", index="region", values="revenue")
    .with_columns(change=pl.col("Q2") / pl.col("Q1") - 1)
    .sort("change", descending=True)
)
by_region
"""
CELL_CODE = """\
import marimo._code_mode as cm
async with cm.get_context() as ctx:
    ctx.create_cell(GROWTH_CELL, name="growth_by_region")
"""
BROKEN_CODE = "df = load_periods(DATA_DIR, 'sales')\ndf.group_by('regoin').agg(pl.col('revenue').sum())\n"

ANSWER = """\
**North grew fastest**: Q2 revenue was **39,608**, up **29.6%** on Q1. West is the only \
region that fell. Total revenue rose 6.9% to 123,189.

| Region | Q1 | Q2 | Change |
|:--|--:|--:|--:|
| North | 30,573 | 39,608 | +29.6% |
| East | 27,805 | 28,647 | +3.0% |
| South | 27,532 | 28,189 | +2.4% |
| West | 29,332 | 26,745 | -8.8% |

Discounts start in 25-04 (the `discount` column first appears there), so every region's Q2 \
figure is net of them. The table is in `analysis.py` as `growth_by_region`, with a bar chart.
"""


class Demo:
    def __init__(self, console: Console, *, delay: float, unsafe_local: bool) -> None:
        self.delay = delay
        self.notebook = "analysis.py"
        self.notebook_open: bool | None = None
        self.kernel = "unsafe-local" if unsafe_local else "docker"
        self.model = "gpt-5.5"
        self.turns, self.tokens, self.calls = 3, 14_820, 0
        self.ui = ChatUI(console, self.submit, self.footer)

    def footer(self) -> FooterData:
        return FooterData(
            notebook=self.notebook, notebook_open=self.notebook_open, kernel=self.kernel, model=self.model,
            turns=self.turns, tokens=self.tokens, notebook_url=URL.format(name=self.notebook),
        )

    async def pause(self, seconds: float) -> None:
        await asyncio.sleep(seconds * self.delay)

    async def prepare(self) -> bool:
        """The startup checklist as ``uvx hailer`` prints it: the agent and the notebook in parallel
        on the live row, each committing its row when it is ready."""
        g, out = self.ui.glyphs, self.ui.console

        async def agent() -> None:
            self.ui.begin_task("agent", "Agent", f"starting {self.model}")
            await self.pause(0.8)
            self.ui.end_task("agent")
            out.print(session_row(g, "ok", "Agent", g.join(self.model, "openai", "OPENAI_API_KEY from keyring"), 0.8 * self.delay))
            out.print(session_row(g, "resume", "Resumed", g.join("3 earlier turns", "/new starts fresh")))

        async def notebook() -> None:
            self.ui.begin_task("notebook", "Notebook", f"opening {self.notebook} in your browser")
            await self.pause(2.2)
            self.ui.end_task("notebook")
            self.notebook_open = True
            out.print(session_row(g, "ok", "Notebook", g.join(self.notebook, "open in your browser"), 2.2 * self.delay))
            out.print(session_note("Opens in app view (results only); Ctrl+. in the notebook shows the code"))

        await asyncio.gather(agent(), notebook())
        return True

    # -- simulated agent events ---------------------------------------------------- #

    def call(self, tool: str, **args: str) -> str:
        self.calls += 1
        call_id = f"call_{self.calls}"
        self.ui.on_event(AgentEvent("tool_call", tool, {"arguments": str(args)[:80], "args": args, "id": call_id}))
        return call_id

    def result(self, tool: str, call_id: str, summary: str, seconds: float, *, ok: bool = True) -> None:
        detail = {"id": call_id, "ok": ok, "summary": summary, "seconds": seconds * self.delay}
        self.ui.on_event(AgentEvent("tool_result", tool, detail))

    async def answer(self, text: str, started: float, steps: int, input_tokens: int, output_tokens: int) -> None:
        for _ in range(6):
            self.ui.on_event(AgentEvent("message_delta", "..."))
            await self.pause(0.15)
        summary = TurnSummary(
            final_response=text, thread_id="demo", duration_ms=int((time.monotonic() - started) * 1000),
            input_tokens=input_tokens, output_tokens=output_tokens, steps=steps,
        )
        self.ui.finish(summary)
        self.turns += 1
        self.tokens += input_tokens + output_tokens

    async def analysis(self) -> None:
        started = time.monotonic()
        await self.pause(0.9)
        periods = self.call("list_periods", name="sales")
        await self.pause(0.7)
        self.result("list_periods", periods, "6 months, 25-01 to 25-06; discount from 25-04", 0.7)
        await self.pause(0.3)
        # Two calls in one model reply: both run, their results arrive in any order.
        cells = self.call("notebook_cells")
        totals = self.call("marimo_execute", code=TOTALS_CODE)
        await self.pause(0.4)
        self.result("notebook_cells", cells, "6 cells", 0.4)
        await self.pause(1.1)
        self.result("marimo_execute", totals, "4 rows", 1.5)
        await self.pause(0.3)
        cell = self.call("marimo_execute", code=CELL_CODE)
        await self.pause(1.9)
        self.result("marimo_execute", cell, "growth_by_region, checks clean", 1.9)
        await self.answer(ANSWER, started, 4, 3_140, 612)

    async def failed_step(self) -> None:
        started = time.monotonic()
        await self.pause(0.6)
        broken = self.call("marimo_execute", code=BROKEN_CODE)
        await self.pause(0.8)
        self.result("marimo_execute", broken, "ColumnNotFoundError", 0.8, ok=False)
        await self.pause(0.5)
        fixed = self.call("marimo_execute", code=BROKEN_CODE.replace("regoin", "region"))
        await self.pause(0.9)
        self.result("marimo_execute", fixed, "4 rows", 0.9)
        await self.answer("The first attempt misspelt `region`; the corrected query returned **4 rows**.", started, 2, 2_210, 140)

    async def error(self) -> None:
        await self.pause(0.6)
        periods = self.call("list_periods", name="sales")
        await self.pause(0.7)
        self.result("list_periods", periods, "6 months, 25-01 to 25-06", 0.7)
        await self.pause(0.6)
        _print_error(self.ui.console, HailerError(
            "The model endpoint rejected the API key for provider 'openai'.",
            hint="Run: uvx hailer login openai to store a new key, or set OPENAI_API_KEY in this terminal.\n"
                 "The endpoint said: Incorrect API key provided.",
        ))

    async def slow(self) -> None:
        await self.pause(0.6)
        skill = self.call("load_skill", name="example-skill")
        await self.pause(0.5)
        self.result("load_skill", skill, "3 files", 0.5)
        await self.pause(0.3)
        self.call("fetch_page", url="https://docs.pola.rs/api/python/stable/reference/")
        self.call("marimo_execute", code=TOTALS_CODE)
        await asyncio.sleep(60)  # Esc or Ctrl+C: both steps end as "cancelled, effect unknown"

    # -- commands ---------------------------------------------------------------- #

    async def submit(self, text: str) -> None:
        out = self.ui.console
        command, _, argument = text.partition(" ")
        argument = argument.strip()
        if command in {"/exit", "/quit"}:
            self.ui.exit()  # the exit summary follows the composer (run)
        elif command == "/help":
            out.print(help_card(self.ui.glyphs))
        elif command == "/status":
            out.print(self.status())
        elif command == "/notebook" and argument == "list":
            out.print(self.notebooks())
        elif command == "/notebook" and argument.startswith("open "):
            name = argument.split(None, 1)[1].removesuffix(".py") + ".py"
            self.notebook, self.notebook_open = name, None
            self.ui.begin_task("switch", "Notebook", f"opening {name} in your browser")  # opened, never printed
            await self.pause(1.5)
            self.ui.end_task("switch")
            self.notebook_open = True
            g = self.ui.glyphs
            out.print(session_row(g, "ok", "Notebook", g.join(name, "active", "open in your browser"), 1.5 * self.delay))
        elif command == "/context":
            out.print(self.context())
        elif command == "/clear":
            out.clear()
        elif command == "/failstep":
            await self.failed_step()
        elif command == "/fail":
            await self.error()
        elif command == "/slow":
            await self.slow()
        elif command == "/long":
            started = time.monotonic()
            await self.pause(0.5)
            rows = "\n".join(f"- Row {index}: Unicode λ; select this text and scroll back." for index in range(1, 45))
            await self.answer("## Scrollback demo\n\n" + rows, started, 0, 900, 700)
        elif command.startswith("/"):
            out.print(f"Unknown command {command}; type /help for commands.", markup=False)
        else:
            await self.analysis()

    def status(self) -> object:
        g = self.ui.glyphs
        unsafe = self.kernel == "unsafe-local"
        kernel = "unsafe-local (runs as you: not isolated)" if unsafe else "docker (hailer-kernel:marimo0.23-3f9c), no network; data read-only"
        rows = [
            ("ok", "Model", g.join(self.model, "openai", "OPENAI_API_KEY from keyring")),
            ("warn" if unsafe else "ok", "Kernel", kernel),
            ("ok", "Marimo", "http://127.0.0.1:2718"),
            ("ok" if self.notebook_open else "off", "Notebook", g.join(self.notebook, "session s_7d41b0" if self.notebook_open else "not open in a browser")),
            (None, "Notebooks", "notebooks"),
            (None, "Thread", g.join("3f9c2a1e-5b7d-4c11", f"{self.turns} turns", "16204 in / 2616 out tokens")),
            (None, "Web access", "docs.pola.rs"),
        ]
        return status_card(g, rows)

    def notebooks(self) -> object:
        rows = [
            NotebookRow("analysis.py", "today 14:02", "6.1 KB", active=self.notebook == "analysis.py", open=True),
            NotebookRow("q2-review.py", "yesterday", "11.8 KB", active=self.notebook == "q2-review.py"),
            NotebookRow("returns.py", "12 Oct", "3.4 KB", active=self.notebook == "returns.py"),
        ]
        return notebook_table(self.ui.glyphs, rows, "3 notebooks in notebooks. /notebook open <name> switches.")

    def context(self) -> object:
        sections = [
            ("Context files", [Text.assemble(".config/hailer/context/sales.md", ("  (1204 bytes)", "dim"))]),
            ("", [Text("sent with every session", style="dim")]),
            ("Skills", [Text.assemble("example-skill", ("  How this team writes period reports", "dim"))]),
            ("Prompts", ["first-look"]),
            ("Web access", ["docs.pola.rs"]),
        ]
        return detail_card(self.ui.glyphs, sections, ["Skipped .config/hailer/context/notes.docx: not a text file"])

    async def run(self, timeout: float) -> None:
        console, g = self.ui.console, self.ui.glyphs
        started = time.monotonic()
        console.print(Text(g.join("Hailer composer demo", "simulated work only"), style="bold"))
        console.print("Type a message or /help. Esc or Ctrl+C cancels a turn; Ctrl+C while idle exits.", markup=False)
        console.print()
        kernel = (
            session_row(g, "warn", "Kernel", g.join("unsafe-local", "runs as you, not isolated"), 0.9)
            if self.kernel == "unsafe-local" else session_row(g, "ok", "Kernel", g.join("docker", "isolated", "no network", "data read-only"), 1.9)
        )
        console.print(kernel)
        self.ui.configure_startup(self.prepare)
        try:
            await asyncio.wait_for(self.ui.run(), timeout)
        except TimeoutError:
            console.print(f"Demo stopped after {timeout:g} seconds.", markup=False)
        out = self.ui._raw_console  # the composer has gone: what the CLI prints after it
        out.print(session_ended(g, turns=self.turns - 3, elapsed=time.monotonic() - started, tokens_used=max(0, self.tokens - 14_820)))
        stopped = [("ok", "Kernel stopped")] if self.kernel == "unsafe-local" else [("ok", "Notebooks copied back to notebooks/"), ("ok", "Kernel stopped")]
        out.print(done_items(g, stopped))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--delay", type=float, default=1.0, help="multiplier for the simulated durations")
    parser.add_argument("--timeout", type=float, default=600, help="maximum demo lifetime in seconds")
    parser.add_argument("--unsafe-local", action="store_true", help="show the unsafe-local kernel warning")
    args = parser.parse_args()
    if args.delay < 0 or args.timeout <= 0:
        parser.error("--delay must be nonnegative and --timeout must be positive")
    _reconfigure_streams()  # as uvx hailer does: a character the console cannot encode prints as "?"
    demo = Demo(Console(highlight=False, soft_wrap=True), delay=args.delay, unsafe_local=args.unsafe_local)
    asyncio.run(demo.run(args.timeout))
