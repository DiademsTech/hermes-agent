"""Exercise simultaneous native API reads/writes in an isolated DELETE-mode home.

Run with the candidate image's Python, no network and no agent volume or secrets.
The HTTP server binds only to loopback; every session/message is synthetic.
"""

import asyncio
import json
import os
import tempfile
import time
from pathlib import Path


async def verify(home: Path) -> None:
    import importlib
    stamp = json.loads(Path('/opt/hermes/install-stamp.json').read_text())
    assert stamp['commit'] == os.environ['EXPECTED_SOURCE_COMMIT'], stamp['commit']
    assert stamp['baseVersion'] == '0.23.0'
    for name in ('docx', 'openpyxl', 'pptx', 'firecrawl'):
        importlib.import_module(name)
    from hermes_cli.config_defaults import DEFAULT_CONFIG
    assert DEFAULT_CONFIG['_config_version'] == 52

    from aiohttp import web
    from aiohttp.test_utils import TestClient, TestServer
    from plugins.web.firecrawl.provider import _load_firecrawl_cls
    from gateway.config import PlatformConfig
    from gateway.platforms.api_server import APIServerAdapter
    from hermes_state import SessionDB

    assert _load_firecrawl_cls()
    with SessionDB(home / "state.db") as db:
        assert db._conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        db.create_session("live", "api_server", system_prompt="synthetic prompt " * 5000)
        db.append_message("live", "user", "Synthetic conversation")
        prompt_hash = db._conn.execute("SELECT system_prompt_hash FROM sessions WHERE id='live'").fetchone()[0]
        # Seed scale without thousands of fsyncs; all schema/indices are native.
        def seed(conn):
            conn.executemany(
                "INSERT INTO sessions(id, source, started_at, system_prompt_hash, title) VALUES (?, ?, ?, ?, ?)",
                [(f"fixture-{i}", "api_server" if i % 10 == 0 else "webhook", i, prompt_hash, f"Fixture {i}") for i in range(8000)],
            )
        db._execute_write(seed)
        adapter = APIServerAdapter(PlatformConfig(enabled=True))
        adapter._session_db = db
        app = web.Application()
        app.router.add_get("/health", adapter._handle_health)
        app.router.add_get("/v1/capabilities", adapter._handle_capabilities)
        app.router.add_get("/api/sessions", adapter._handle_list_sessions)
        app.router.add_get("/api/sessions/{session_id}/messages", adapter._handle_session_messages)
        app.router.add_get("/v1/skills", adapter._handle_skills)
        samples = []

        async with TestClient(TestServer(app)) as client:
            async with client.get('/health') as response:
                assert response.status == 200 and (await response.json())['status'] == 'ok'
            async with client.get('/v1/capabilities') as response:
                assert response.status == 200
                features = (await response.json())['features']
                for name in ('run_submission', 'run_queue', 'run_queue_controls', 'run_clarify',
                             'run_controls_generation_bound', 'run_inline_delegations',
                             'run_status_subagents', 'webhook_execution_model', 'webhook_result_session',
                             'session_messages_include_compacted', 'skills_api'):
                    assert features.get(name) is True, name

            async def reader(index: int) -> None:
                for source in ("", "&source=api_server"):
                    started = time.monotonic()
                    async with client.get("/api/sessions?limit=200" + source) as response:
                        assert response.status == 200, (index, response.status)
                        rows = (await response.json())["data"]
                        assert len(rows) == 200
                        assert any(row["id"] == "live" for row in rows)
                        assert all(row["has_system_prompt"] for row in rows)
                        assert all("system_prompt" not in row for row in rows)
                    samples.append(time.monotonic() - started)
                async with client.get("/api/sessions/live/messages") as response:
                    assert response.status == 200

            async def writer() -> None:
                for index in range(20):
                    await asyncio.to_thread(db.append_message, "live", "assistant", f"Synthetic output {index}")
                    await asyncio.sleep(0.005)

            started = time.monotonic()
            await asyncio.wait_for(asyncio.gather(*(reader(i) for i in range(10)), writer()), timeout=20)
            duration = time.monotonic() - started
            async with client.get("/v1/skills") as response:
                assert response.status == 200
            assert len(db.get_messages("live")) == 21
            assert db._conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
            assert db._conn.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        print(json.dumps({"clients": 10, "index_reads": len(samples), "message_reads": 10,
                          "writes": 20, "sessions": 8001, "journal_mode": "delete",
                          "elapsed_seconds": round(duration, 3), "max_read_seconds": round(max(samples), 3),
                          "integrity": "ok", "skills": "ok", "firecrawl_import": "ok"}))


with tempfile.TemporaryDirectory(prefix="diadems-session-concurrency-") as directory:
    home = Path(directory)
    os.environ["HERMES_HOME"] = str(home)
    (home / "config.yaml").write_text("_config_version: 52\ndatabase:\n  journal_mode: delete\nsecurity:\n  allow_lazy_installs: false\n", encoding="utf-8")
    asyncio.run(verify(home))
