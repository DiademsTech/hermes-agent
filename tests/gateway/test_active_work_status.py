"""``active_agents`` follows live work, not the last write that persisted it.

Regression for #122813. ``/health/detailed`` is served inside the gateway, so it reports the
runner's live work total instead of ``gateway_state.json``. The file is only rewritten at work
boundaries, and an API run, a cron job or a deferred agent worker can end with no messaging turn
after it: a count persisted while one ran stayed in the file, and every reader saw a busy gateway
with nothing running. Each of them now rewrites the persisted count when it ends.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
import types
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

import cron.scheduler as sched
from cron import scheduler_thread
from gateway import run as gateway_run
from gateway import status
from gateway.config import Platform, PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from tests.gateway.restart_test_helpers import make_restart_runner


class _IdleTicker:
    """``SupervisedTickerThread`` stand-in: the gateway wires its cron, no tick ever runs."""

    def __init__(self, *_args, **_kwargs):
        pass

    def start(self):
        return None


@pytest.fixture(autouse=True)
def _cron_state(monkeypatch):
    monkeypatch.setattr(sched, "_job_release_callbacks", (), raising=False)
    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()
    yield
    sched._running_job_ids.clear()
    sched._running_fire_owners.clear()


@contextlib.asynccontextmanager
async def _serving_gateway(monkeypatch):
    """A runner serving on this loop with its primary API server, cron started as
    ``start_gateway`` starts it (ticker and housekeeping threads stubbed out)."""
    runner, _adapter = make_restart_runner()
    runner._gateway_loop = asyncio.get_running_loop()
    runner._external_drain_active = False
    runner._cleanup_agent_resources_off_loop = AsyncMock()
    api = APIServerAdapter(PlatformConfig(enabled=True))
    api.gateway_runner = runner
    runner.adapters = {Platform.API_SERVER: api}
    monkeypatch.setattr(gateway_run, "_gateway_runner_ref", lambda: runner)
    monkeypatch.setattr(gateway_run, "_cron_tick_profile_homes", lambda _config: [])
    monkeypatch.setattr(gateway_run, "_start_gateway_housekeeping", lambda *_a, **_kw: None)
    monkeypatch.setattr(scheduler_thread, "SupervisedTickerThread", _IdleTicker)
    cron_stop = gateway_run._start_gateway_start_cron_and_housekeeping(runner)[0]
    app = web.Application()
    app.router.add_get("/health/detailed", api._handle_health_detailed)
    app.router.add_post("/v1/runs", api._handle_runs)
    try:
        async with TestClient(TestServer(app)) as client:
            yield types.SimpleNamespace(runner=runner, api=api, client=client)
    finally:
        cron_stop.set()


async def _start_api_run(gw):
    """A /v1/runs turn held inside its agent until ``finish``."""
    release = threading.Event()

    def run_conversation(**_kwargs):
        release.wait(30)
        return {"final_response": "done"}

    gw.api._create_agent = MagicMock(return_value=MagicMock(run_conversation=run_conversation))
    response = await gw.client.post("/v1/runs", json={"input": "hello"})
    assert response.status == 202
    task = gw.api._active_run_tasks[(await response.json())["run_id"]]

    async def finish():
        release.set()
        await task

    return finish


async def _start_cron_job(_gw):
    """A cron run in flight, released on another thread like the scheduler pool's worker."""
    assert sched.try_register_running_job("job-1")

    async def finish():
        await asyncio.to_thread(sched.release_running_job, "job-1")
        # The release schedules the rewrite on the gateway loop before it returns. A worker that
        # finishes before ``to_thread`` chains its future completes the await without yielding,
        # so give the loop one turn to run what the release already scheduled.
        await asyncio.sleep(0)

    return finish


async def _start_deferred_worker(gw):
    """An executor worker that outlived its turn (e.g. a timed-out hygiene compression)."""
    worker = asyncio.get_running_loop().create_future()
    gw.runner._defer_agent_cleanup_until_future_done(worker, MagicMock(), context="test")

    async def finish():
        worker.set_result(None)
        await asyncio.gather(*gw.runner._deferred_agent_cleanup_tasks)

    return finish


_WORK = {"api_run": _start_api_run, "cron_job": _start_cron_job, "deferred_worker": _start_deferred_worker}


def _persisted_active_agents() -> int:
    assert status.flush_runtime_status(timeout=5.0)
    runtime = status.read_runtime_status()
    assert runtime is not None
    return runtime["active_agents"]


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [None, *_WORK])
async def test_health_detailed_reports_the_live_work_total(kind, monkeypatch):
    """Whatever the file holds: idle once the work that a stale count recorded is gone, busy
    while work runs that no write has recorded yet."""
    async with _serving_gateway(monkeypatch) as gw:
        status.write_runtime_status(gateway_state="running", active_agents=0 if kind else 1)
        finish = await _WORK[kind](gw) if kind else None
        try:
            with patch("gateway.run._resolve_gateway_model", return_value="test/model"):
                body = await (await gw.client.get("/health/detailed")).json()
        finally:
            if finish is not None:
                await finish()

    assert body["active_agents"] == (1 if kind else 0)
    assert body["gateway_busy"] is (kind is not None)


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", list(_WORK))
async def test_finished_work_is_not_left_in_the_persisted_count(kind, monkeypatch):
    """A count persisted while the work ran (a turn boundary, the startup write) is rewritten
    when that work ends, with no later messaging turn to do it."""
    async with _serving_gateway(monkeypatch) as gw:
        status.write_runtime_status(gateway_state="running", active_agents=0)
        finish = await _WORK[kind](gw)
        try:
            gw.runner._persist_active_agents()
            assert _persisted_active_agents() == 1
        finally:
            await finish()
        assert _persisted_active_agents() == 0
