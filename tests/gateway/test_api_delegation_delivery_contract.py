"""A detached API result must have an addressable consumer and one durable row."""
import asyncio
from types import SimpleNamespace

import pytest

from gateway.platforms.api_server import APIServerAdapter
from gateway.session_context import clear_session_vars
from gateway.wake import persist_delegation_delivery
from hermes_state import SessionDB
from tools.delegate_tool_dispatch import _resolve_async_wake_sid


@pytest.mark.asyncio
async def test_detached_dispatch_requires_a_declared_consumer(monkeypatch):
    monkeypatch.setenv("HERMES_SESSION_HISTORY_DELIVERY", "1")
    for capability in (None, "", "1"):
        kw = dict(chat_id="api-parent", session_id="api-parent")
        import inspect
        if capability is not None and "session_history_delivery" in inspect.signature(APIServerAdapter._bind_api_server_session).parameters:
            kw["session_history_delivery"] = capability
        tokens = APIServerAdapter._bind_api_server_session(**kw)
        try:
            args = ["api-parent"]
            if len(inspect.signature(_resolve_async_wake_sid).parameters) > 1:
                from gateway.session_context import session_history_delivery_supported
                args.append(session_history_delivery_supported())
            target = _resolve_async_wake_sid(*args)
            assert target == ("api-parent" if capability == "1" else None)
        finally:
            clear_session_vars(tokens)
    from evals.api_delegation_http_probe import probe
    result = await probe()
    for request in result["requests"]:
        assert request["status"] == 200
        runtime = request["runtime"]
        assert runtime["target"] == ("child" if request["explicit"] else None)
        if request["explicit"]:
            assert runtime["session_id"] == "child"
            assert request["header"] == "parent"
    assert result["unsolicited_calls"] == 0
    assert result["durable_child_rows"] == 1
    assert sum(m["content"] == "DELIVERY_RESULT" for m in result["resumed_history"]) == 1
    runs = {r["name"]: r for r in result["runs"]}
    assert all(r["status"] == 202 for r in runs.values())
    for name in ("caller_history", "response_chain"):
        assert runs[name]["runtime"]["target"] is None
        assert runs[name]["runtime"]["history"] == [{"role": "user", "content": "caller snapshot"}]
    assert runs["session"]["runtime"]["target"] == "child"
    assert runs["session"]["runtime"]["history"] == result["resumed_history"]
    assert runs["detached"]["runtime"]["target"] == "child"
    for name in ("inline", "inline_queued"):
        assert runs[name]["runtime"]["target"] is None
        assert runs[name]["runtime"]["history"] == result["resumed_history"]
    assert runs["declared_key"]["runtime"]["target"] == "declared"
    assert runs["declared_key"]["runtime"]["history"][0]["content"] == "DECLARED_HISTORY"


@pytest.mark.asyncio
async def test_delivery_replay_is_atomic_across_continuation_and_busy_turn(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    peer = SessionDB(db_path=tmp_path / "state.db")
    try:
        db.create_session("parent", source="api_server")
        db.create_session("other", source="api_server")
        adapters = [SimpleNamespace(_ensure_session_db=lambda: db), SimpleNamespace(_ensure_session_db=lambda: peer)]
        evt = {"type": "async_delegation", "delegation_id": "unique-unit"}
        async def send(adapter, event=evt):
            await persist_delegation_delivery(adapter, text="RESULT", session_id="parent", evt=event)
        await asyncio.gather(*(send(a) for a in adapters))
        assert len(db.get_messages("parent")) == 1
        db.end_session("parent", "compression")
        db.create_session("child", source="api_server", parent_session_id="parent")
        await send(adapters[0])
        assert db.get_messages("child") == []  # old event was already recorded in the lineage
        from hermes_state_errors import SessionTurnLeaseLostError
        assert db.acquire_session_turn_lease("child", "client-turn", wait_seconds=0)
        later = {**evt, "delegation_id": "later-unit"}
        try:
            with pytest.raises(SessionTurnLeaseLostError):
                await send(adapters[0], later)
            assert db.get_messages("child") == []
        finally:
            db.release_session_turn_lease("child", "client-turn")
        await send(adapters[0], later)
        assert len(db.get_messages("child")) == 1
        notice = {**later, "task_failure_notice": True, "results": [{"task_index": 0, "status": "failed"}]}
        await send(adapters[0], notice)
        await send(adapters[1], notice)
        assert len(db.get_messages("child")) == 2  # interim notice cannot consume the final's identity
        assert db.get_messages("other") == []
    finally:
        peer.close()
        db.close()


@pytest.mark.asyncio
async def test_run_status_recovers_bounded_subagents_without_sse_or_private_output():
    from gateway.config import PlatformConfig

    api = APIServerAdapter(PlatformConfig())
    api._set_run_status("run-recovery", "running")
    emit = api._make_run_event_callback("run-recovery", asyncio.get_running_loop())
    for i in range(40):
        emit("subagent.start", subagent_id=f"child-{i}", goal="g" * 15000)
    emit("subagent.complete", subagent_id="child-39", status="completed",
         summary="review finished", files_read=["/private/file"], output_tail="private-output", cost_usd=9)
    await asyncio.sleep(0)  # drain the event-loop publication callbacks
    status = api._set_run_status("run-recovery", "completed", output="parent consumed review")
    assert len(status["subagents"]) == 32
    last = status["subagents"][-1]
    assert last["event"] == "subagent.complete" and last["status"] == "completed"
    assert last["summary"] == "review finished" and last["files_read"] == 1
    assert len(last["goal"]) == 12000
    assert "private-output" not in str(status) and "/private/file" not in str(status)
    assert "cost_usd" not in last


@pytest.mark.asyncio
async def test_busy_api_delivery_refunds_attempts_until_the_turn_releases(tmp_path):
    import time
    from gateway.config import GatewayConfig, Platform, PlatformConfig
    from gateway.run import GatewayRunner
    from tools import async_delegation as delegation

    runner = GatewayRunner(GatewayConfig())
    api = APIServerAdapter(PlatformConfig())
    db = SessionDB(tmp_path / "delivery.db")
    db.create_session("busy-parent", source="api_server")
    api._ensure_session_db = lambda: db
    runner.adapters = {Platform.API_SERVER: api}
    evt = {"type": "async_delegation", "session_key": "busy-parent",
           "origin_session_id": "busy-parent", "delegation_id": "busy-unit",
           "status": "completed", "summary": "review finished", "dispatched_at": time.time()}
    delegation._persist_dispatch(evt)
    delegation._persist_completion(evt, {"status": "completed", "summary": "review finished"})
    try:
        assert db.acquire_session_turn_lease("busy-parent", "client-turn", wait_seconds=0)
        for _ in range(10):
            assert await runner._deliver_async_delegation_group([evt]) is False
            row = delegation.get_durable_delegation("busy-unit")
            assert (row["delivery_state"], row["delivery_attempts"]) == ("pending", 0)
        assert db.get_messages("busy-parent") == []
        db.release_session_turn_lease("busy-parent", "client-turn")
        assert await runner._deliver_async_delegation_group([evt]) is True
        await runner._deliver_async_delegation_group([evt])
        assert len(db.get_messages("busy-parent")) == 1
        assert delegation.get_durable_delegation("busy-unit")["delivery_state"] == "delivered"
        assert not api._active_run_tasks and not api._background_tasks
    finally:
        db.release_session_turn_lease("busy-parent", "client-turn")
        db.close()
