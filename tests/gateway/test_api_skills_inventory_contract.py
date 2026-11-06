"""The HTTP skill inventory must call the real installed inventory signature."""

import os
from pathlib import Path

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms import api_server_runs
from gateway.platforms.api_server import APIServerAdapter


@pytest.mark.asyncio
async def test_skills_lists_a_real_home_skill_without_mocking_inventory():
    skill = Path(os.environ["HERMES_HOME"]) / "skills" / "inventory-contract" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\nname: inventory-contract\ndescription: Synthetic inventory contract skill.\n"
        "---\n\nRead this synthetic skill.\n",
        encoding="utf-8",
    )
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    app = web.Application()
    app.router.add_get("/v1/skills", adapter._handle_skills)
    try:
        async with TestClient(TestServer(app)) as client:
            response = await client.get("/v1/skills")
            assert response.status == 200, await response.text()
            payload = await response.json()
            assert payload["object"] == "list"
            assert any(row["name"] == "inventory-contract" for row in payload["data"])
    finally:
        api_server_runs._close_run_state(adapter)
