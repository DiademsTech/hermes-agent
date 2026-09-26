"""Failed automation outputs stay internal; successful output and chat still deliver."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from gateway.config import Platform, PlatformConfig
from gateway.platforms.event import MessageEvent, ProcessingOutcome
from gateway.platforms.webhook import WebhookAdapter
from gateway.run import GatewayRunner
from gateway.session import SessionSource


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("platform", [Platform.WEBHOOK, Platform.TELEGRAM])
@pytest.mark.parametrize("outcome", ["success", "failed", "interrupted", "guardrail", "exception"])
async def test_failure_policy_at_outbound_boundary(tmp_path, monkeypatch, enabled, platform, outcome):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    (tmp_path / "config.yaml").write_text(
        f"display: {{suppress_automation_failure_messages: {str(enabled).lower()}}}\n",
        encoding="utf-8",
    )
    adapter = WebhookAdapter(PlatformConfig(enabled=True, extra={}))
    adapter.config.typing_indicator = False
    adapter.send = AsyncMock(return_value=SimpleNamespace(success=True, message_id="sent"))
    adapter.on_processing_complete = AsyncMock()
    gateway = object.__new__(GatewayRunner)
    gateway._delivery_adapter_for = lambda source: adapter
    gateway._should_send_voice_reply = lambda *a, **kw: False
    source = SessionSource(platform=platform, chat_id="automation", user_id="test")
    event = MessageEvent(text="Process the event", source=source)
    result = {outcome: True} if outcome in {"failed", "interrupted"} else {}
    if outcome == "guardrail":
        result = {"completed": False, "turn_exit_reason": "guardrail_halt"}
    original_result = result.copy()
    # Identical text must survive on success: the policy never pattern-matches content.
    text = "HTTP 402: balance exhausted"

    async def handler(inbound):
        if outcome == "exception":
            raise RuntimeError("provider unavailable")
        return await gateway._hmwa_deliver_turn_response(
            inbound, source, SimpleNamespace(session_id="test"), "key", 1,
            result, [], text, "", False,
        )

    adapter._message_handler = handler
    await adapter._process_message_background(event, "key")
    muted = enabled and platform == Platform.WEBHOOK and outcome != "success"
    assert adapter.send.await_count == (0 if muted else 1)
    if outcome != "exception" and not muted:
        args, kwargs = adapter.send.call_args
        assert (kwargs["content"] if "content" in kwargs else args[1]) == text
    expected = ProcessingOutcome.FAILURE if muted or outcome == "exception" else ProcessingOutcome.SUCCESS
    adapter.on_processing_complete.assert_awaited_once_with(event, expected)
    assert result == original_result

    # A later successful turn may reuse the event; a suppressed failure must not poison it.
    if muted:
        adapter.send.reset_mock()
        adapter.on_processing_complete.reset_mock()
        adapter._message_handler = AsyncMock(return_value="Recovered")
        await adapter._process_message_background(event, "key")
        assert adapter.send.await_count == 1
        adapter.on_processing_complete.assert_awaited_once_with(event, ProcessingOutcome.SUCCESS)
