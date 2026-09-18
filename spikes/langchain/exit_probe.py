"""Ctrl+C during a long tool call, then quit straight away: does close() hang on the tool thread?"""

import _thread
import tempfile
import threading
import time
from pathlib import Path

from fake_gateway import FakeGateway
from lc_agent import LangChainHailerAgent
from run_spike import SYSTEM, provider_for

with FakeGateway() as gw:
    agent = LangChainHailerAgent(provider_for(gw), "corp-gpt", "sk-spike", SYSTEM, Path(tempfile.mkdtemp()) / "x.sqlite", [])
    agent.start()
    agent.tool_delay["marimo_execute"] = 8.0
    threading.Timer(1.0, _thread.interrupt_main).start()
    t0 = time.monotonic()
    try:
        agent.run_turn("please run it")
    except KeyboardInterrupt:
        print(f"interrupted after {time.monotonic() - t0:.2f}s")
    t1 = time.monotonic()
    agent.close()
    print(f"close() took {time.monotonic() - t1:.2f}s (tool had ~7s left)")
