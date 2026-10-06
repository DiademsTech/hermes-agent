"""The ``active_agents`` that ``/health/detailed`` reports.

``gateway_state.json`` holds the count the gateway last persisted, at its work boundaries. Work
that ends with no boundary after it (an API run, a cron job, a deferred agent worker) leaves that
count in the file until the next inbound message (#122813). ``/health/detailed`` is served
in-process, so it reports the runner's live total instead, the one the shutdown drain waits on.
"""

from __future__ import annotations

from typing import Any


def live_active_agents(adapter: Any, persisted: Any) -> int:
    """The runner's live work total; ``persisted`` when no runner is reachable (a standalone
    adapter)."""
    from gateway.status import parse_active_agents

    runner = adapter.gateway_runner
    if runner is None:
        from gateway.run import _gateway_runner_ref

        runner = _gateway_runner_ref()
    count = getattr(runner, "_active_work_count", None)
    return parse_active_agents(count() if callable(count) else persisted)
