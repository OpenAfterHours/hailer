"""The startup benchmark must detect a freeze immediately after rendering."""

import json
import subprocess
import sys


def test_benchmark_detects_a_stall_before_its_first_poll():
    script = r'''
import asyncio
import json
import time
from hailer.chat import ChatController
from scripts.benchmark_startup import _sample

original = ChatController.astart
async def blocked_start(self):
    time.sleep(0.25)
    await original(self)
ChatController.astart = blocked_start
print(json.dumps(asyncio.run(_sample(0.01))))
'''
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=45)
    assert result.returncode == 0, result.stdout + result.stderr
    sample = json.loads(result.stdout)
    assert sample["input_ready"] < sample["ready_to_answer"]
    # Polling only after input_ready would miss this entire artificial freeze.
    assert sample["max_input_loop_pause"] >= 0.25, sample
