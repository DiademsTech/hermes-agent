"""Run control reads leave the event loop (Diadems): the SessionDB has one connection and one
lock, so a run poll that read it on the loop froze every request and platform while a long
write (a large session delete) held it."""

import asyncio
import contextlib
import threading
from types import SimpleNamespace

import pytest

import gateway.platforms.api_server_authority_runs as runs


class _Conn:
    def __init__(self, log, rows):
        self.log, self.rows = log, rows

    def execute(self, sql, params=()):
        self.log.append(threading.get_ident())
        return SimpleNamespace(fetchall=lambda: list(self.rows))


def _adapter(log, rows):
    db = SimpleNamespace(_read_ctx=lambda: contextlib.nullcontext(_Conn(log, rows)))
    authority = SimpleNamespace(db=db)
    return SimpleNamespace(gateway_runner=SimpleNamespace(session_authority=authority),
                           _run_admission_aliases={})


@pytest.fixture(autouse=True)
def _plain_rows(monkeypatch):
    monkeypatch.setattr(runs, "_row", lambda row: dict(row))
    monkeypatch.setattr(runs, "admission_result", lambda db, admission_id: {"result": {"final_response": "ok"}})


def test_read_ahead_reads_in_a_worker_thread_and_serves_the_loop():
    log = []
    adapter = _adapter(log, [{"admission_id": "adm-1", "status": "queued"}])

    async def scenario():
        loop_thread = threading.get_ident()
        with await runs.read_ahead(adapter, "run-1"):
            reads_before = len(log)
            authority, row = runs.run_admission(adapter, "run-1")
            assert len(log) == reads_before  # served from the read-ahead, not the store
        assert log and all(ident != loop_thread for ident in log)
        return row

    assert asyncio.run(scenario())["admission_id"] == "adm-1"


def test_read_ahead_replays_a_store_error():
    def boom():
        raise runs.RuntimeStoreError("storage_unavailable")
    adapter = _adapter([], [])
    adapter.gateway_runner.session_authority.db._read_ctx = boom

    async def scenario():
        with await runs.read_ahead(adapter, "run-1"):
            with pytest.raises(runs.RuntimeStoreError):
                runs.run_admission(adapter, "run-1")

    asyncio.run(scenario())


def test_outside_the_block_run_admission_reads_the_store_again():
    log = []
    adapter = _adapter(log, [{"admission_id": "adm-1", "status": "queued"}])

    async def scenario():
        with await runs.read_ahead(adapter, "run-1"):
            pass
        before = len(log)
        runs.run_admission(adapter, "run-1")
        assert len(log) == before + 1

    asyncio.run(scenario())


def test_the_loop_keeps_running_while_the_store_is_slow():
    release = threading.Event()

    class _Slow(_Conn):
        def execute(self, sql, params=()):
            release.wait(5)
            return super().execute(sql, params)

    log = []
    db = SimpleNamespace(_read_ctx=lambda: contextlib.nullcontext(_Slow(log, [])))
    adapter = SimpleNamespace(gateway_runner=SimpleNamespace(session_authority=SimpleNamespace(db=db)),
                              _run_admission_aliases={})

    async def scenario():
        reader = asyncio.create_task(runs.read_ahead(adapter, "run-1"))
        ticks = 0
        for _ in range(5):
            await asyncio.sleep(0.01)
            ticks += 1
        release.set()
        await reader
        return ticks

    assert asyncio.run(scenario()) == 5
