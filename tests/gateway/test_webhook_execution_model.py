"""Trusted automation routing survives durable admission; payloads carry no authority."""
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import Platform
from gateway.session import SessionSource
from gateway.webhook_execution import normalize_choice


@pytest.mark.parametrize('choice', [
    {'model': 'orphan'}, {'provider': 'orphan'}, {'reasoning_effort': 'invented'},
    {'model': ['bad']}, {'model': 'bad\n', 'provider': 'p'},
])
def test_invalid_route_refused(choice):
    with pytest.raises(ValueError):
        normalize_choice(choice)


def test_route_resolves_complete_account_and_does_not_fall_back(monkeypatch):
    from gateway.run import GatewayRunner
    from gateway.run_turn_prepare import GatewayTurnPrepareMixin

    source = SessionSource(platform=Platform.WEBHOOK, chat_id='fixture',
        webhook_execution={'model': 'route-model', 'provider': 'route-provider', 'reasoning_effort': 'low'})
    restored = SessionSource.from_dict(source.to_dict())
    resolver = Mock(return_value={'model': 'account-default', 'provider': 'route-provider', 'api_key': 'route-key'})
    monkeypatch.setattr('gateway.run._resolve_runtime_agent_kwargs_for_provider', resolver)
    monkeypatch.setattr('gateway.run._resolve_runtime_agent_kwargs', Mock(side_effect=AssertionError('wrong account')))
    monkeypatch.setattr('gateway.run._resolve_gateway_model', lambda _: 'profile-model')
    runner = SimpleNamespace(_resolve_session_key_or_none=lambda *args: None)
    model, runtime = GatewayTurnPrepareMixin._resolve_session_agent_runtime(runner, source=restored)
    assert (model, runtime) == ('route-model', {'provider': 'route-provider', 'api_key': 'route-key'})
    resolver.assert_called_once_with('route-provider', target_model='route-model')
    runner._load_reasoning_config = lambda _: {'effort': 'high'}
    assert GatewayRunner._resolve_session_reasoning_config(runner, source=restored) == {'enabled': True, 'effort': 'low'}
    resolver.side_effect = ValueError('account disconnected')
    with pytest.raises(ValueError, match='disconnected'):
        GatewayTurnPrepareMixin._resolve_session_agent_runtime(runner, source=restored)


@pytest.mark.asyncio
async def test_http_admission_and_callback_use_retained_identity(tmp_path, monkeypatch):
    from gateway.platforms.base import SendResult
    from gateway.platforms.webhook_delivery import retained_destination
    from tests.gateway.test_webhook_adapter import _make_adapter, _create_app, _INSECURE_NO_AUTH
    from hermes_state_runtime import RuntimeStoreError

    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    monkeypatch.setattr('gateway.platform_registry.platform_registry.is_registered',
                        lambda name: name == 'http_callback')
    route = {'secret': _INSECURE_NO_AUTH, 'prompt': '{text}', 'model': 'configured-model',
             'provider': 'configured-provider', 'reasoning_effort': 'low',
             'deliver': 'http_callback', 'deliver_extra': {'chat_id': 'result', 'source_session_id': '{forged}'}}
    adapter = _make_adapter(routes={'fixture': route})
    adapter.handle_message = AsyncMock()
    target = SimpleNamespace(send=AsyncMock(return_value=SendResult(success=True)))
    async with TestClient(TestServer(_create_app(adapter))) as client:
        response = await client.post('/webhooks/fixture', json={'text': 'Check', 'forged': 'foreign-session',
            'model': 'attacker-model', 'provider': 'attacker-provider', 'reasoning_effort': 'high'},
            headers={'X-GitHub-Delivery': 'stable-identity'})
        assert response.status == 202, await response.text()
        event = adapter.handle_message.await_args.args[0]
        assert event.source.webhook_execution == {
            'model': 'configured-model', 'provider': 'configured-provider', 'reasoning_effort': 'low'}
        runner = adapter._message_handler.__self__
        adapter.gateway_runner = runner
        runner.adapters[Platform('http_callback')] = target
        destination = retained_destination(adapter, event.source.chat_id)
        assert destination['source_session_id'] != 'foreign-session'
        assert runner.session_authority.db.get_session(destination['source_session_id'])
        assert 'source_session_id' not in destination['deliver_extra']
        # Lose all process-local delivery data: the authorized journal is sufficient.
        adapter._delivery_info.clear()
        assert retained_destination(adapter, event.source.chat_id) == destination
        assert (await adapter.send(event.source.chat_id, 'Result')).success
        target.send.assert_awaited_once_with('result', 'Result',
            metadata={'source_session_id': destination['source_session_id']})
        # Revoking/changing the configured route still invalidates delivery.
        adapter._routes['fixture']['model'] = 'changed'
        with pytest.raises(RuntimeStoreError):
            retained_destination(adapter, event.source.chat_id)


def test_cli_preserves_then_clears_route_choice(monkeypatch):
    from hermes_cli.webhook import _cmd_subscribe
    from tests.hermes_cli.test_webhook_cli import _make_args
    existing = {'model': 'specific', 'provider': 'own-account', 'reasoning_effort': 'low', 'secret': 'local'}
    saved = {}
    monkeypatch.setattr('hermes_cli.webhook._load_subscriptions', lambda: {'orders': existing})
    monkeypatch.setattr('hermes_cli.webhook._replace_subscription', lambda name, route, expected: saved.update(route) or route)
    monkeypatch.setattr('hermes_cli.webhook._route_url', lambda *args: 'https://example.test/orders')
    _cmd_subscribe(_make_args(name='orders'))
    assert all(saved[key] == existing[key] for key in ('model', 'provider', 'reasoning_effort'))
    saved.clear()
    _cmd_subscribe(_make_args(name='orders', model='', provider='', reasoning_effort=''))
    assert not set(saved) & {'model', 'provider', 'reasoning_effort'}
