"""Republish the gateway's ``active_agents`` when cron work ends.

``gateway_state.json`` holds the count ``GatewayRunner._persist_active_agents`` writes at turn
boundaries. A cron job ends on a scheduler thread with no boundary after it, so the file kept a
finished job counted until the next inbound message (#122813).
"""

from __future__ import annotations

from contextlib import suppress


def republish_active_agents() -> None:
    """Schedule the live runner's ``_persist_active_agents`` on its loop, where every count is
    read; safe from any thread."""
    from gateway.run import _gateway_runner_ref

    runner = _gateway_runner_ref()
    if runner is None:
        return
    loop = getattr(runner, "_gateway_loop", None)
    if loop is None or loop.is_closed():
        return
    with suppress(RuntimeError):  # the loop closed between the check and the call
        loop.call_soon_threadsafe(runner._persist_active_agents)


def watch_cron_releases() -> None:
    """Republish ``active_agents`` whenever a cron job leaves the scheduler's running set."""
    from cron.scheduler_release_observers import register_job_release_callback

    register_job_release_callback(republish_active_agents)
