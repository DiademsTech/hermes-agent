"""Actual provider -> manager -> Runs queue, with no LLM or memory service."""
import asyncio
from types import SimpleNamespace

import pytest

from agent.memory_manager import MemoryManager
from agent.memory_provider import MemoryProvider, RecallStatus
from gateway.platforms.api_server_runs import _make_run_event_callback


class Provider(MemoryProvider):
    name = "test-memory"

    def __init__(self, text="private recalled content", error=False):
        self.text, self.error = text, error

    def is_available(self):
        return True

    def initialize(self, session_id="", **kwargs):
        pass

    def get_tool_schemas(self):
        return []

    def handle_tool_call(self, tool_name, args, **kwargs):
        return ""

    def prefetch(self, query, **kwargs):
        if self.error:
            raise RuntimeError("private upstream error")
        return self.text

    def recall_status(self):
        # Deliberately stale when prefetch returned nothing.
        return RecallStatus("Test", 4)


@pytest.mark.parametrize("text,error,phase,count", [
    ("private recalled content", False, "completed", 4),
    ("", False, "completed", None),
    ("", True, "failed", None),
])
def test_real_manager_to_runs_queue(text, error, phase, count):
    async def exercise():
        queue = asyncio.Queue()
        adapter = SimpleNamespace(
            _run_streams={"run-test": queue}, _run_statuses={},
            _set_run_status=lambda *a, **kw: None,
        )
        callback = _make_run_event_callback(
            adapter, "run-test", asyncio.get_running_loop(),
            _api_server=SimpleNamespace(redact_sensitive_text=lambda text: text),
        )
        manager = MemoryManager()
        manager.add_provider(Provider(text, error))
        result = await asyncio.to_thread(
            manager.prefetch_all, "private user query", event_callback=callback,
        )
        start = await asyncio.wait_for(queue.get(), 1)
        end = await asyncio.wait_for(queue.get(), 1)
        assert start["phase"] == "started"
        assert end["phase"] == phase
        assert end["count"] == count
        assert end["returned"] is bool(text and not error)
        assert result == ("" if error else text)
        assert "private" not in str([start, end])
    asyncio.run(exercise())


def test_broken_event_consumer_does_not_change_context():
    manager = MemoryManager()
    manager.add_provider(Provider())
    def broken(*args, **kwargs):
        raise RuntimeError("browser disconnected")
    assert manager.prefetch_all("question", event_callback=broken) == "private recalled content"


def test_no_provider_emits_no_memory_activity():
    events = []
    assert MemoryManager().prefetch_all("question", event_callback=lambda *a, **kw: events.append(kw)) == ""
    assert events == []
