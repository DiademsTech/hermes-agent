"""POST /api/sessions/{id}/fork with ``before_message_id``: a new conversation from one turn back.

The fork holds the history as it stood just before the chosen user turn (the prefix ``/undo`` keeps),
with its tool calls; the original is neither ended nor changed. Pinned for the Diadems patch
``api-session-fork-before-message``."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


def _app(adapter):
    app = web.Application()
    app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
    app.router.add_get("/api/sessions/{session_id}/messages", adapter._handle_session_messages)
    app.router.add_post("/api/sessions/{session_id}/fork", adapter._handle_fork_session)
    return app


CALL = [{"id": "call_1", "type": "function", "function": {"name": "web_search", "arguments": "{\"q\": \"x\"}"}}]


def _conversation(db, session_id="source"):
    db.create_session(session_id, "api_server", system_prompt="You are Ice.")
    db.append_messages_batch(session_id, [
        {"role": "user", "content": "Cherche le prix"},
        {"role": "assistant", "content": "", "tool_calls": CALL},
        {"role": "tool", "tool_call_id": "call_1", "content": "12 €"},
        {"role": "assistant", "content": "Il coûte 12 €."},
        {"role": "user", "content": "Et en Belgique ?"},
        {"role": "assistant", "content": "14 €."},
    ])
    return {m["content"]: m["id"] for m in db.get_messages(session_id) if m["role"] == "user"}


@pytest.mark.asyncio
async def test_fork_before_a_turn_keeps_the_history_before_it_and_leaves_the_source(adapter, session_db):
    ids = _conversation(session_db)
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/api/sessions/source/fork",
                              json={"id": "child", "before_message_id": ids["Et en Belgique ?"], "title": "Ailleurs"})
        assert resp.status == 201, await resp.text()
        assert (await resp.json())["session"]["id"] == "child"

    copied = session_db.get_messages_as_conversation("child")
    assert [m["role"] for m in copied] == ["user", "assistant", "tool", "assistant"]
    assert copied[1]["tool_calls"][0]["function"]["name"] == "web_search"
    assert copied[-1]["content"] == "Il coûte 12 €."
    child = session_db.get_session("child")
    assert child["parent_session_id"] == "source"
    assert child["system_prompt"] == "You are Ice."
    assert child["title"] == "Ailleurs"
    assert session_db._is_explicit_branch_session("child")
    # The source goes on unchanged: not ended, every row still active.
    source = session_db.get_session("source")
    assert source["end_reason"] is None and source["ended_at"] is None
    assert len(session_db.get_messages_as_conversation("source")) == 6


@pytest.mark.asyncio
async def test_a_fork_point_must_be_an_active_user_turn_with_history_before_it(adapter, session_db):
    ids = _conversation(session_db)
    tool_row = next(m["id"] for m in session_db.get_messages("source") if m["role"] == "tool")
    async with TestClient(TestServer(_app(adapter))) as cli:
        for point in (tool_row, ids["Cherche le prix"], 999_999):
            resp = await cli.post("/api/sessions/source/fork", json={"id": f"c{point}", "before_message_id": point})
            assert resp.status == 409, point
            assert (await resp.json())["error"]["code"] == "fork_point_unavailable"
            assert session_db.get_session(f"c{point}") is None
        for bad in ("12", True, -1):
            resp = await cli.post("/api/sessions/source/fork", json={"id": "bad", "before_message_id": bad})
            assert resp.status == 400
    assert session_db.get_session("source")["end_reason"] is None


@pytest.mark.asyncio
async def test_keep_source_copies_everything_without_ending_it(adapter, session_db):
    _conversation(session_db)
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/api/sessions/source/fork", json={"id": "all", "keep_source": True})
        assert resp.status == 201
        caps = await (await cli.get("/v1/capabilities")).json()
    assert caps["features"]["session_fork_before_message"] is True
    assert len(session_db.get_messages_as_conversation("all")) == 6
    assert session_db.get_session("source")["end_reason"] is None


@pytest.mark.asyncio
async def test_a_plain_fork_still_ends_its_source_as_branched(adapter, session_db):
    _conversation(session_db)
    async with TestClient(TestServer(_app(adapter))) as cli:
        assert (await cli.post("/api/sessions/source/fork", json={"id": "plain"})).status == 201
    assert session_db.get_session("source")["end_reason"] == "branched"


@pytest.mark.asyncio
async def test_a_turn_before_a_compression_forks_with_its_own_segment(adapter, session_db):
    """A row shown from an ancestor segment forks with that segment's history, never the tip's summary."""
    ids = _conversation(session_db, "root")
    session_db.end_session("root", "compression")
    session_db.create_session("tip", "api_server", parent_session_id="root")
    session_db.append_messages_batch("tip", [
        {"role": "user", "content": "[CONTEXT COMPACTION] Résumé : prix 12 € puis 14 €."},
        {"role": "assistant", "content": "Bien noté."},
        {"role": "user", "content": "Et en Suisse ?"},
        {"role": "assistant", "content": "16 €."},
    ])
    assert set(session_db._resume_lineage_ids("tip")) == {"root", "tip"}
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/api/sessions/tip/fork",
                              json={"id": "back", "before_message_id": ids["Et en Belgique ?"]})
        assert resp.status == 201, await resp.text()
    copied = session_db.get_messages_as_conversation("back")
    assert copied[0]["content"] == "Cherche le prix"
    assert copied[-1]["content"] == "Il coûte 12 €."
    assert session_db.get_session("tip")["end_reason"] is None
