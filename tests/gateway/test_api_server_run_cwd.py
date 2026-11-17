"""POST /v1/runs ``cwd``: a run works in a validated sub-folder of its profile's terminal.cwd.

The folder becomes the run's logical session cwd (context files, system prompt, agent cwd) and
its terminal start folder, for THIS run only; anything outside ``terminal.cwd`` is refused.
"""

import os
import threading

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer
from unittest.mock import MagicMock, patch

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, _STATIC_FEATURE_FLAGS
from tests.waiting import async_wait_until


def _adapter() -> APIServerAdapter:
    return APIServerAdapter(PlatformConfig(enabled=True, extra={}))


def _app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app["api_server_adapter"] = adapter
    app.router.add_post("/v1/runs", adapter._handle_runs)
    app.router.add_get("/v1/runs/{run_id}", adapter._handle_get_run)
    return app


def _observing_agent(seen: list, gate: threading.Event | None = None, arrived: threading.Event | None = None):
    """A mock agent whose run records the cwd views a real turn would read, from its own thread."""
    from agent.runtime_cwd import resolve_agent_cwd, resolve_context_cwd
    from tools.terminal_scope import terminal_env

    agent = MagicMock()
    agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0

    def _run(**_kwargs):
        if arrived is not None:
            arrived.set()
        if gate is not None:
            gate.wait(timeout=10)
        context = resolve_context_cwd()
        seen.append({
            "agent": str(resolve_agent_cwd()),
            "context": str(context) if context is not None else None,
            "terminal": terminal_env("TERMINAL_CWD"),
        })
        return {"final_response": "done"}

    agent.run_conversation.side_effect = _run
    return agent


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    root = tmp_path / "workspace"
    (root / "projects" / "appel-offres").mkdir(parents=True)
    (root / "projects" / "compta").mkdir(parents=True)
    monkeypatch.setenv("TERMINAL_CWD", str(root))
    return root


def test_capability_is_advertised():
    assert _STATIC_FEATURE_FLAGS["run_session_cwd"] is True


@pytest.mark.asyncio
async def test_run_works_in_the_requested_project_folder(workspace):
    adapter, seen = _adapter(), []
    project = os.path.realpath(workspace / "projects" / "appel-offres")
    async with TestClient(TestServer(_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=_observing_agent(seen)):
            resp = await cli.post("/v1/runs", json={"input": "hello", "cwd": project})
            assert resp.status == 202
            await async_wait_until(lambda: seen, what="run executed")
    assert seen[0] == {"agent": project, "context": project, "terminal": project}


@pytest.mark.asyncio
async def test_run_without_cwd_keeps_the_profile_folder(workspace):
    adapter, seen = _adapter(), []
    async with TestClient(TestServer(_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", return_value=_observing_agent(seen)):
            resp = await cli.post("/v1/runs", json={"input": "hello"})
            assert resp.status == 202
            await async_wait_until(lambda: seen, what="run executed")
    assert seen[0]["terminal"] == str(workspace)
    assert os.path.realpath(seen[0]["agent"]) == os.path.realpath(workspace)


@pytest.mark.asyncio
async def test_concurrent_runs_keep_their_own_folders(workspace):
    adapter, seen = _adapter(), []
    gate, first, second = threading.Event(), threading.Event(), threading.Event()
    folders = [os.path.realpath(workspace / "projects" / name) for name in ("appel-offres", "compta")]
    agents = iter([_observing_agent(seen, gate, first), _observing_agent(seen, gate, second)])
    async with TestClient(TestServer(_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent", side_effect=lambda *a, **k: next(agents)):
            for index, folder in enumerate(folders):
                resp = await cli.post(
                    "/v1/runs", json={"input": "hi", "cwd": folder, "session_id": f"dia-{index}"})
                assert resp.status == 202
            await async_wait_until(lambda: first.is_set() and second.is_set(), what="both runs started")
            gate.set()
            await async_wait_until(lambda: len(seen) == 2, what="both runs finished")
    assert sorted(entry["terminal"] for entry in seen) == sorted(folders)
    assert all(entry["agent"] == entry["terminal"] == entry["context"] for entry in seen)


@pytest.mark.asyncio
@pytest.mark.parametrize("make_cwd", [
    lambda root: str(root.parent),                       # outside terminal.cwd
    lambda root: str(root / "projects" / "absent"),      # does not exist
    lambda root: "projects/appel-offres",                # relative
    lambda root: 42,                                     # not a string
])
async def test_invalid_cwd_is_refused_before_any_run(workspace, make_cwd):
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as cli:
        with patch.object(adapter, "_create_agent") as create:
            resp = await cli.post("/v1/runs", json={"input": "hello", "cwd": make_cwd(workspace)})
            body = await resp.json()
    assert resp.status == 400
    assert body["error"]["code"] == "invalid_cwd"
    create.assert_not_called()


@pytest.mark.asyncio
async def test_symlink_out_of_the_workspace_is_refused(workspace, tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    link = workspace / "projects" / "escape"
    try:
        os.symlink(outside, link, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable here")
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/v1/runs", json={"input": "hello", "cwd": str(link)})
        body = await resp.json()
    assert resp.status == 400
    assert body["error"]["code"] == "invalid_cwd"


@pytest.mark.asyncio
async def test_cwd_without_a_configured_working_directory_is_refused(tmp_path, monkeypatch):
    monkeypatch.delenv("TERMINAL_CWD", raising=False)
    adapter = _adapter()
    async with TestClient(TestServer(_app(adapter))) as cli:
        resp = await cli.post("/v1/runs", json={"input": "hello", "cwd": str(tmp_path)})
        body = await resp.json()
    assert resp.status == 409
    assert body["error"]["code"] == "cwd_unavailable"
