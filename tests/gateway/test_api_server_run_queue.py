"""FIFO API contracts without a model/provider or a live user session."""
import asyncio
import threading
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp.test_utils import TestClient, TestServer

from tests.gateway.test_api_server_runs import _make_adapter, _create_runs_app
from gateway.platforms.api_server_run_idempotency import RunIdempotencyStore


def app_for(adapter):
    app = _create_runs_app(adapter)
    app.router.add_get('/v1/runs', adapter._handle_list_runs)
    app.router.add_delete('/v1/runs/{run_id}/queue', adapter._handle_delete_queued_run)
    return app


async def settled(adapter, rid):
    async with asyncio.timeout(5):
        while rid in adapter._active_run_tasks:
            await asyncio.sleep(.01)


@pytest.mark.asyncio
async def test_fifo_cancel_middle_reload_history_and_replay(tmp_path):
    adapter = _make_adapter()
    adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    gate = threading.Event()
    started = threading.Event()
    history, calls = [], []

    def create(**kwargs):
        agent = MagicMock()
        agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
        def run(user_message, conversation_history, task_id):
            calls.append((user_message, list(conversation_history)))
            if user_message == 'first':
                started.set()
                assert gate.wait(5)
            history.extend([{'role': 'user', 'content': user_message},
                            {'role': 'assistant', 'content': 'answer ' + user_message}])
            return {'final_response': 'answer ' + user_message}
        agent.run_conversation.side_effect = run
        return agent

    async def load(sid):
        return list(history)

    async with TestClient(TestServer(app_for(adapter))) as client:
        with patch.object(adapter, '_create_agent', side_effect=create), patch.object(
            adapter, '_conversation_history_for_session', side_effect=load
        ):
            async def post(text):
                response = await client.post('/v1/runs', json={
                    'input': text, 'session_id': 'one', 'queue': True
                }, headers={'Idempotency-Key': text})
                assert response.status == 202, await response.text()
                return (await response.json())['run_id']
            first = await post('first')
            assert await asyncio.to_thread(started.wait, 3)
            middle = await post('middle')
            last = await post('last')
            assert await post('last') == last
            # Another device can enumerate the queue without an SSE stream.
            snapshot = await (await client.get('/v1/runs?session_id=one')).json()
            assert [item.get('queued_input') for item in snapshot['runs'] if 'queued_input' in item] == ['middle', 'last']
            assert (await client.delete(f'/v1/runs/{first}/queue')).status == 409
            assert (await client.delete(f'/v1/runs/{middle}/queue')).status == 200
            await settled(adapter, middle)
            assert [item[0] for item in calls] == ['first']
            gate.set()
            await settled(adapter, last)
            assert [item[0] for item in calls] == ['first', 'last']
            assert calls[1][1] == history[:2]
            assert adapter._run_idempotency_store.queue_inputs(adapter._run_owners[first], 'one') == []


@pytest.mark.asyncio
async def test_other_session_runs_while_queue_waits(tmp_path):
    adapter = _make_adapter()
    adapter._max_concurrent_runs = 2
    adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    gate = threading.Event()
    calls = []
    def create(**kwargs):
        agent = MagicMock()
        agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
        def run(user_message, **kwargs):
            calls.append(user_message)
            if user_message == 'first':
                assert gate.wait(5)
            return {'final_response': 'done'}
        agent.run_conversation.side_effect = run
        return agent
    async with TestClient(TestServer(app_for(adapter))) as client:
        with patch.object(adapter, '_create_agent', side_effect=create), patch.object(
            adapter, '_conversation_history_for_session', AsyncMock(return_value=[])
        ):
            async def post(text, sid):
                r = await client.post('/v1/runs', json={'input': text, 'session_id': sid, 'queue': True}, headers={'Idempotency-Key': text})
                assert r.status == 202, await r.text()
                return (await r.json())['run_id']
            first = await post('first', 'one')
            async with asyncio.timeout(3):
                while not calls:
                    await asyncio.sleep(.01)
            last = await post('last', 'one')
            other = await post('other', 'two')
            await settled(adapter, other)
            assert calls == ['first', 'other']
            gate.set()
            await settled(adapter, last)
            assert adapter._run_statuses[first]['status'] == 'completed'


@pytest.mark.asyncio
async def test_persisted_waiting_input_after_restart_is_visible_not_replayed(tmp_path):
    adapter = _make_adapter(api_key='secret')
    adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    async with TestClient(TestServer(app_for(adapter))) as client:
        # Resolve the real authenticated scope, then simulate a dead owner.
        from types import SimpleNamespace
        request = SimpleNamespace(headers={'Authorization': 'Bearer secret'})
        scope = adapter._run_idempotency_scope(request)
        adapter._run_idempotency_store.reserve(scope, 'key', 'fingerprint', 'run_pending',
            {'run_id': 'run_pending', 'status': 'queued', 'session_id': 'one', 'created_at': 1},
            owner_pid=0, queue_input='remember this')
        assert (await client.get('/v1/runs?session_id=one')).status == 401
        headers = {'Authorization': 'Bearer secret'}
        data = await (await client.get('/v1/runs?session_id=one', headers=headers)).json()
        assert data['runs'][0]['queued_input'] == 'remember this'
        assert data['runs'][0]['status'] == 'interrupted'
        assert not adapter._active_run_tasks
        assert (await client.delete('/v1/runs/run_pending/queue', headers=headers)).status == 200
        data = await (await client.get('/v1/runs?session_id=one', headers=headers)).json()
        assert data['runs'] == []
