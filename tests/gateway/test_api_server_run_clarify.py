"""Native ``clarify`` questions on ``/v1/runs`` (capability ``run_clarify``).

Covers the opt-in admission flag, the ``clarify.request`` event and pollable
``waiting_for_clarification`` status, ``POST /v1/runs/{run_id}/clarify`` (accept,
idempotent repeat, conflicts, validation), the native timeout and stop outcomes,
and that runs which did not opt in keep their toolset and callback unchanged.
"""

import asyncio
import json
import threading
from unittest.mock import MagicMock, patch

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware, security_headers_middleware
from gateway.platforms.api_server_run_clarify import (
    PendingClarify,
    _validate_responses,
    _wait,
    with_clarify_toolset,
)

FORM = [
    {"question": "Which environment?", "choices": ["staging", "production"]},
    {"question": "Anything to tell the reviewers?"},
    {"question": "Notify whom?", "choices": ["ops", "sales", "legal"], "multi_select": True},
]


def _adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[m for m in (cors_middleware, security_headers_middleware) if m])
    app["api_server_adapter"] = adapter
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    app.router.add_get("/v1/runs/{run_id}/events", adapter._handle_run_events)
    app.router.add_post("/v1/runs/{run_id}/clarify", adapter._handle_run_clarify)
    app.router.add_post("/v1/runs/{run_id}/steer", adapter._handle_steer_run)
    app.router.add_post("/v1/runs/{run_id}/stop", adapter._handle_stop_run)
    return app


def _clarifying_agent(**tool_kwargs):
    """Agent double whose turn calls the real clarify tool through ``clarify_callback``."""
    agent = MagicMock()
    agent.clarify_callback = None
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
    stopped = threading.Event()
    agent.interrupt.side_effect = lambda message=None: stopped.set()

    def run(user_message=None, conversation_history=None, task_id=None):
        from tools.clarify_tool import clarify_tool
        kwargs = tool_kwargs or {"question": "", "questions": FORM}
        agent.tool_result = clarify_tool(callback=agent.clarify_callback, **kwargs)
        return {"final_response": agent.tool_result, "interrupted": stopped.is_set()}

    agent.run_conversation.side_effect = run
    return agent


async def _until(predicate, timeout=5.0):
    for _ in range(int(timeout / 0.02)):
        if predicate():
            return
        await asyncio.sleep(0.02)
    raise AssertionError("condition not reached")


async def _start(cli, adapter, body):
    response = await cli.post("/v1/runs", json=body)
    assert response.status == 202, await response.text()
    run_id = (await response.json())["run_id"]
    events = asyncio.create_task((await cli.get(f"/v1/runs/{run_id}/events")).text())
    return run_id, events


def _events(text: str) -> list:
    return [json.loads(line[5:]) for line in text.splitlines() if line.startswith("data:")]


@pytest.mark.asyncio
async def test_capabilities_advertise_run_clarify():
    async with TestClient(TestServer(_app(_adapter()))) as cli:
        data = await (await cli.get("/v1/capabilities")).json()
    assert data["features"]["run_clarify"] is True
    assert data["endpoints"]["run_clarify"] == {"method": "POST", "path": "/v1/runs/{run_id}/clarify"}


@pytest.mark.asyncio
async def test_clarify_flag_must_be_boolean():
    async with TestClient(TestServer(_app(_adapter()))) as cli:
        response = await cli.post("/v1/runs", json={"input": "hi", "clarify": "yes"})
    assert response.status == 400


@pytest.mark.asyncio
async def test_run_without_opt_in_keeps_toolset_and_callback():
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent") as create:
            agent = MagicMock()
            agent.clarify_callback = None
            agent.run_conversation.return_value = {"final_response": "done"}
            agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
            create.return_value = agent
            run_id = (await (await cli.post("/v1/runs", json={"input": "hi"})).json())["run_id"]
            await _until(lambda: adapter._run_statuses.get(run_id, {}).get("status") == "completed")
    assert "interactive_clarify" not in create.call_args.kwargs
    assert agent.clarify_callback is None


@pytest.mark.parametrize(("opted_in", "disabled", "expected"), [
    (False, [], False), (True, [], True), (True, ["clarify"], False)])
def test_create_agent_offers_clarify_only_to_opted_in_runs(monkeypatch, opted_in, disabled, expected):
    captured = {}

    class FakeAgent:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    monkeypatch.setattr("run_agent.AIAgent", FakeAgent)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: {"provider": "custom"})
    monkeypatch.setattr("gateway.run._resolve_gateway_model", lambda: "model")
    monkeypatch.setattr("gateway.run._load_gateway_config", lambda: {"agent": {"disabled_toolsets": disabled}})
    monkeypatch.setattr("gateway.run.GatewayRunner._load_reasoning_config", staticmethod(lambda model="": None))
    monkeypatch.setattr("gateway.run.GatewayRunner._load_fallback_model", staticmethod(lambda: None))
    monkeypatch.setattr("hermes_cli.tools_config._get_platform_tools", lambda *_: {"web", "terminal"})
    adapter = _adapter()
    monkeypatch.setattr(adapter, "_ensure_session_db", lambda: None)
    adapter._create_agent(session_id="s", **({"interactive_clarify": True} if opted_in else {}))
    assert ("clarify" in captured["enabled_toolsets"]) is expected
    assert {"web", "terminal"} <= set(captured["enabled_toolsets"])


def test_toolset_added_only_when_not_globally_disabled():
    assert with_clarify_toolset(["web", "terminal"], {}) == ["clarify", "terminal", "web"]
    assert with_clarify_toolset(["web"], {"agent": {"disabled_toolsets": ["clarify"]}}) == ["web"]
    assert with_clarify_toolset(["web"], {"agent": {"disabled_toolsets": "['clarify']"}}) == ["web"]


@pytest.mark.asyncio
async def test_question_answer_round_trip_with_status_recovery():
    adapter = _adapter()
    agent = _clarifying_agent()
    async with TestClient(TestServer(_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent) as create:
            run_id, events = await _start(cli, adapter, {"input": "deploy", "clarify": True})
            await _until(lambda: "clarification" in adapter._run_statuses.get(run_id, {}))
            assert create.call_args.kwargs["interactive_clarify"] is True

            # A client that missed the SSE event recovers the question from status.
            status = await (await cli.get(f"/v1/runs/{run_id}")).json()
            assert status["status"] == "waiting_for_clarification"
            pending = status["clarification"]
            assert pending["event"] == "clarify.request"
            assert pending["clarify_id"].startswith("clr_")
            assert pending["timeout_seconds"] == 3600 and pending["expires_at"] > pending["timestamp"]
            assert pending["questions"] == [
                {"id": "q0", "prompt": "Which environment?", "choices": ["staging", "production"],
                 "recommended": 0, "multi_select": False, "allow_other": True},
                {"id": "q1", "prompt": "Anything to tell the reviewers?", "choices": None,
                 "recommended": None, "multi_select": False, "allow_other": True},
                {"id": "q2", "prompt": "Notify whom?", "choices": ["ops", "sales", "legal"],
                 "recommended": 0, "multi_select": True, "allow_other": True},
            ]

            # Steer stays reserved for running turns, as during approvals.
            steer = await cli.post(f"/v1/runs/{run_id}/steer", json={"input": "faster"})
            assert steer.status == 409

            answer = {"clarify_id": pending["clarify_id"],
                      "responses": ["production", "  ship it  ", ["ops", "finance"]]}
            first = await cli.post(f"/v1/runs/{run_id}/clarify", json=answer)
            assert first.status == 200
            assert (await first.json())["replayed"] is False
            again = await cli.post(f"/v1/runs/{run_id}/clarify", json=answer)
            assert again.status == 200 and (await again.json())["replayed"] is True
            other = await cli.post(f"/v1/runs/{run_id}/clarify",
                                   json={**answer, "responses": ["staging", None, None]})
            assert other.status == 409
            assert (await other.json())["error"]["code"] == "clarify_already_answered"

            stream = _events(await asyncio.wait_for(events, 5))
            final = adapter._run_statuses[run_id]

    assert final["status"] == "completed"
    assert "clarification" not in final
    result = json.loads(final["output"])
    assert [row["user_response"] for row in result["responses"]] == ["production", "ship it", ["ops", "finance"]]
    assert "timed_out" not in result
    names = [event["event"] for event in stream]
    assert names.index("clarify.request") < names.index("clarify.responded") < names.index("run.completed")
    responded = next(event for event in stream if event["event"] == "clarify.responded")
    assert responded == {**responded, "clarify_id": pending["clarify_id"], "outcome": "answered"}
    assert "responses" not in responded  # answers never ride the event stream


@pytest.mark.asyncio
async def test_answer_validation_and_unknown_request():
    adapter = _adapter()
    agent = _clarifying_agent()
    async with TestClient(TestServer(_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent):
            run_id, events = await _start(cli, adapter, {"input": "deploy", "clarify": True})
            await _until(lambda: "clarification" in adapter._run_statuses.get(run_id, {}))
            clarify_id = adapter._run_statuses[run_id]["clarification"]["clarify_id"]
            url = f"/v1/runs/{run_id}/clarify"
            cases = [
                ({"responses": ["a", "b", ["c"]]}, 400, "invalid_clarify_response"),
                ({"clarify_id": clarify_id, "responses": ["a"]}, 400, "invalid_clarify_response"),
                ({"clarify_id": clarify_id, "responses": [["a"], "b", ["c"]]}, 400, "invalid_clarify_response"),
                ({"clarify_id": "clr_other", "responses": ["a", "b", ["c"]]}, 409, "clarify_not_pending"),
            ]
            for body, status, code in cases:
                response = await cli.post(url, json=body)
                assert response.status == status, body
                assert (await response.json())["error"]["code"] == code
            assert (await cli.post("/v1/runs/run_missing/clarify", json=cases[0][0])).status == 404
            ok = await cli.post(url, json={"clarify_id": clarify_id, "responses": [None, "", []]})
            assert ok.status == 200
            await asyncio.wait_for(events, 5)
    result = json.loads(adapter._run_statuses[run_id]["output"])
    assert [row["user_response"] for row in result["responses"]] == ["", "", ""]


@pytest.mark.asyncio
async def test_native_timeout_returns_no_answer_and_expires_request():
    adapter = _adapter()
    agent = _clarifying_agent()
    async with TestClient(TestServer(_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent), \
                patch("tools.clarify_gateway.get_clarify_timeout", return_value=1):
            run_id, events = await _start(cli, adapter, {"input": "deploy", "clarify": True})
            await _until(lambda: "clarification" in adapter._run_statuses.get(run_id, {}))
            clarify_id = adapter._run_statuses[run_id]["clarification"]["clarify_id"]
            stream = _events(await asyncio.wait_for(events, 5))
    final = adapter._run_statuses[run_id]
    result = json.loads(final["output"])
    assert result["timed_out"] is True
    assert result["notice"] == "[user did not respond within 1m]"
    assert [row["user_response"] for row in result["responses"]] == ["", "", ""]
    assert "clarification" not in final
    responded = next(event for event in stream if event["event"] == "clarify.responded")
    assert responded["clarify_id"] == clarify_id and responded["outcome"] == "expired"


@pytest.mark.asyncio
async def test_stop_withdraws_pending_question():
    adapter = _adapter()
    agent = _clarifying_agent()
    async with TestClient(TestServer(_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent):
            run_id, events = await _start(cli, adapter, {"input": "deploy", "clarify": True})
            await _until(lambda: "clarification" in adapter._run_statuses.get(run_id, {}))
            pending = adapter._run_clarifications[run_id]
            assert (await cli.post(f"/v1/runs/{run_id}/stop")).status == 200
            stream = _events(await asyncio.wait_for(events, 5))
            late = await cli.post(f"/v1/runs/{run_id}/clarify",
                                  json={"clarify_id": pending.clarify_id, "responses": ["a", "b", ["c"]]})
            late_error = (await late.json())["error"]["code"]
    assert pending.state == "cancelled"
    assert late.status == 409 and late_error == "clarify_expired"
    final = adapter._run_statuses[run_id]
    assert final["status"] == "cancelled"
    assert "clarification" not in final
    result = json.loads(agent.tool_result)
    assert result["timed_out"] is True
    assert result["notice"].startswith("[user did not respond: the run was stopped")
    assert next(e for e in stream if e["event"] == "clarify.responded")["outcome"] == "cancelled"


@pytest.mark.asyncio
async def test_legacy_single_question_shape_and_multi_select():
    adapter = _adapter()
    agent = _clarifying_agent(question="Pick targets", choices=["eu", "us"], multi_select=True)
    async with TestClient(TestServer(_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=agent):
            run_id, events = await _start(cli, adapter, {"input": "go", "clarify": True})
            await _until(lambda: "clarification" in adapter._run_statuses.get(run_id, {}))
            pending = adapter._run_statuses[run_id]["clarification"]
            assert pending["questions"][0]["choices"] == ["eu", "us"]  # no "(Recommended)" suffix
            response = await cli.post(f"/v1/runs/{run_id}/clarify",
                                      json={"clarify_id": pending["clarify_id"], "responses": [["us", "apac"]]})
            assert response.status == 200
            await asyncio.wait_for(events, 5)
    result = json.loads(adapter._run_statuses[run_id]["output"])
    assert result["question"] == "Pick targets"
    assert result["user_response"] == ["us", "apac"]


def test_wire_labels_map_back_to_original_choices():
    pending = PendingClarify(clarify_id="clr_x", run_id="run_x", multi=[False, True],
                             wire_choices=[["token [REDACTED]", "none"], ["a [REDACTED]", "b"]],
                             choices=[["token sk-live-123", "none"], ["a sk-1", "b"]])
    answers, error = _validate_responses(pending, ["token [REDACTED]", ["a [REDACTED]", "b", "b"]])
    assert error is None
    assert answers == ["token sk-live-123", ["a sk-1", "b"]]


def test_interrupt_ends_wait_as_cancelled():
    from tools.interrupt import set_interrupt

    pending = PendingClarify(clarify_id="clr_x", run_id="run_x", multi=[False], wire_choices=[None], choices=[None])
    set_interrupt(True)
    try:
        assert _wait(pending, 30) == "cancelled"
    finally:
        set_interrupt(False)
    assert pending.event.is_set()


def test_control_event_keeps_pending_question_visible():
    adapter = _adapter()
    adapter._set_run_status("run_x", "waiting_for_approval", clarification={"clarify_id": "clr_x"})
    adapter._run_streams["run_x"] = asyncio.Queue()
    from gateway.platforms.api_server_runs import _mark_run_event

    _mark_run_event(adapter, "run_x", "approval.responded", choice="once")
    status = adapter._run_statuses["run_x"]
    assert status["status"] == "waiting_for_clarification"
    assert status["clarification"] == {"clarify_id": "clr_x"}
    adapter._set_run_status("run_x", "completed")
    assert "clarification" not in adapter._run_statuses["run_x"]
