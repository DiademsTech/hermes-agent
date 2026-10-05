"""The API server's part in the gateway's ``active_agents``.

Admissions, ``_run_agent`` turns and /v1/runs tasks end outside the messaging turn boundaries
that persist ``gateway_state.json``, so a count persisted while one ran stayed in the file until
the next inbound message (#122813). ``/health/detailed`` is served in-process and reports the
runner's live total instead.
"""

from __future__ import annotations

from typing import Any


def persist_active_work(adapter: Any) -> None:
    """Republish the gateway's ``active_agents`` after this adapter's work count dropped."""
    persist = getattr(getattr(adapter, "gateway_runner", None), "_persist_active_agents", None)
    if callable(persist):
        persist()


def live_active_agents(adapter: Any, persisted: Any) -> int:
    """The runner's live work total, the one the shutdown drain waits on; ``persisted`` when no
    runner is reachable (a standalone adapter)."""
    from gateway.status import parse_active_agents

    runner = adapter.gateway_runner
    if runner is None:
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
    count = getattr(runner, "_active_work_count", None)
    return parse_active_agents(count() if callable(count) else persisted)
