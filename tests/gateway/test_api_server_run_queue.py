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


@pytest.mark.asyncio
async def test_slow_initialization_does_not_block_queue_http(tmp_path):
    adapter = _make_adapter()
    adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    def create(**kwargs):
        started.set()
        release.wait(5)
        finished.set()
        agent = MagicMock()
        agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
        agent.run_conversation.return_value = {'final_response': 'done'}
        return agent
    async with TestClient(TestServer(app_for(adapter))) as client:
        with patch.object(adapter, '_create_agent', side_effect=create), patch.object(
            adapter, '_conversation_history_for_session', AsyncMock(return_value=[])
        ):
            try:
                first = await (await client.post('/v1/runs', json={'input':'first','session_id':'one'})).json()
                assert await asyncio.to_thread(started.wait, 3)
                assert not finished.is_set(), 'HTTP admission waited for constructor I/O'
                queued = await client.post('/v1/runs', json={'input':'second','session_id':'one','queue':True}, headers={'Idempotency-Key':'second'})
                assert queued.status == 202
                snapshot = await (await client.get('/v1/runs?session_id=one')).json()
                assert any(item.get('queued_input') == 'second' for item in snapshot['runs'])
                rid = (await queued.json())['run_id']
                assert (await client.delete('/v1/runs/'+rid+'/queue')).status == 200
                assert not finished.is_set(), 'Queue controls blocked on constructor I/O'
            finally:
                release.set()
            await settled(adapter, first['run_id'])

@pytest.mark.asyncio
async def test_queue_edit_reorder_and_steer_are_serialized_with_execution(tmp_path):
    adapter = _make_adapter()
    adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    gate, started = threading.Event(), threading.Event()
    calls, steers = [], []
    def create(**kwargs):
        agent = MagicMock()
        agent.session_prompt_tokens = agent.session_completion_tokens = agent.session_total_tokens = 0
        def run(user_message, **kwargs):
            calls.append(user_message)
            if user_message == 'first':
                started.set()
                assert gate.wait(15)
            return {'final_response': 'done'}
        agent.run_conversation.side_effect = run
        agent.steer.side_effect = lambda text: steers.append(text) or True
        return agent
    app = app_for(adapter)
    app.router.add_patch('/v1/runs/queue', adapter._handle_reorder_queue)
    app.router.add_patch('/v1/runs/{run_id}/queue', adapter._handle_edit_queued_run)
    app.router.add_post('/v1/runs/{run_id}/queue/steer', adapter._handle_steer_queued_run)
    async with TestClient(TestServer(app)) as client:
        with patch.object(adapter, '_create_agent', side_effect=create), patch.object(adapter, '_conversation_history_for_session', AsyncMock(return_value=[])):
            async def post(text):
                r = await client.post('/v1/runs', json={'input':text,'session_id':'one','queue':True}, headers={'Idempotency-Key':text})
                assert r.status == 202
                return (await r.json())['run_id']
            async def rows():
                data = await (await client.get('/v1/runs?session_id=one')).json()
                return [row for row in data['runs'] if 'queued_input' in row]
            first = await post('first')
            assert await asyncio.to_thread(started.wait, 3)
            try:
                b, c, d = await post('b'), await post('c'), await post('d')
                snapshot = await rows()
                assert [r['run_id'] for r in snapshot] == [b,c,d]
                revision = snapshot[0]['queue_revision']
                assert (await client.patch(f'/v1/runs/{first}/queue', json={'input':'no','expected_revision':revision})).status == 409
                assert (await client.patch(f'/v1/runs/{b}/queue', json={'input':'edited b','expected_revision':revision})).status == 200
                assert (await client.patch(f'/v1/runs/{b}/queue', json={'input':'stale','expected_revision':revision})).status == 409
                assert (await client.patch('/v1/runs/queue', json={'session_id':'one','order':[d,c,b],'expected_order':[b,c,d]})).status == 200
                assert (await client.patch('/v1/runs/queue', json={'session_id':'one','order':[b,c,d],'expected_order':[b,c,d]})).status == 409
                assert [r['run_id'] for r in await rows()] == [d,c,b]
                assert [r['run_id'] for r in adapter._run_idempotency_store.queue_inputs(adapter._run_owners[b], 'one')] == [d,c,b]
                c_row = next(r for r in await rows() if r['run_id']==c)
                payload = {'expected_revision':c_row['queue_revision']}
                adapter._active_run_agents[first].steer.side_effect = lambda text: False
                assert (await client.post(f'/v1/runs/{c}/queue/steer', json=payload)).status == 409
                assert any(r['run_id']==c for r in await rows())
                adapter._active_run_agents[first].steer.side_effect = lambda text: steers.append(text) or True
                assert (await client.post(f'/v1/runs/{c}/queue/steer', json=payload)).status == 200
                assert (await client.post(f'/v1/runs/{c}/queue/steer', json=payload)).status == 409
                assert steers == ['c']
                assert [r['run_id'] for r in await rows()] == [d,b]
            finally:
                gate.set()
            await settled(adapter, b)
            assert calls == ['first','d','edited b']
            assert await rows() == []

@pytest.mark.asyncio
async def test_queue_control_auth_scope_and_ambiguous_steer(tmp_path):
    from types import SimpleNamespace
    adapter = _make_adapter(api_key='secret')
    adapter._run_idempotency_store = RunIdempotencyStore(str(tmp_path / 'runs.db'))
    gate, started = threading.Event(), threading.Event()
    def create(**kwargs):
        agent=MagicMock()
        agent.session_prompt_tokens=agent.session_completion_tokens=agent.session_total_tokens=0
        def run(**kwargs):
            started.set()
            assert gate.wait(15)
            return {'final_response':'done'}
        agent.run_conversation.side_effect=run
        agent.steer.side_effect=RuntimeError('ambiguous')
        return agent
    app=app_for(adapter)
    app.router.add_patch('/v1/runs/queue', adapter._handle_reorder_queue)
    app.router.add_patch('/v1/runs/{run_id}/queue', adapter._handle_edit_queued_run)
    app.router.add_post('/v1/runs/{run_id}/queue/steer', adapter._handle_steer_queued_run)
    headers={'Authorization':'Bearer secret'}
    async with TestClient(TestServer(app)) as client:
        with patch.object(adapter,'_create_agent',side_effect=create), patch.object(adapter,'_conversation_history_for_session',AsyncMock(return_value=[])):
            first=await client.post('/v1/runs',headers=headers,json={'input':'first','session_id':'one'})
            first_id=(await first.json())['run_id']
            assert await asyncio.to_thread(started.wait,3)
            try:
                r=await client.post('/v1/runs',headers={**headers,'Idempotency-Key':'next'},json={'input':'next','session_id':'one','queue':True})
                rid=(await r.json())['run_id']
                rows=(await (await client.get('/v1/runs?session_id=one',headers=headers)).json())['runs']
                revision=next(row['queue_revision'] for row in rows if row['run_id']==rid)
                payload={'expected_revision':revision}
                assert (await client.patch(f'/v1/runs/{rid}/queue',json={**payload,'input':'x'})).status==401
                assert (await client.post(f'/v1/runs/{rid}/queue/steer',json=payload)).status==401
                assert (await client.patch('/v1/runs/queue',headers=headers,json={'session_id':'other','order':[rid],'expected_order':[rid]})).status==409
                assert (await client.post(f'/v1/runs/{rid}/queue/steer',headers=headers,json=payload)).status==503
                await settled(adapter,rid)
                rows=(await (await client.get('/v1/runs?session_id=one',headers=headers)).json())['runs']
                retained=next(row for row in rows if row['run_id']==rid)
                assert retained['status']=='interrupted' and retained['queued_input']=='next'
                assert (await client.post(f'/v1/runs/{rid}/queue/steer',headers=headers,json=payload)).status==409
                assert adapter._active_run_agents[first_id].steer.call_count==1
            finally:
                gate.set()
            await settled(adapter,first_id)


def test_queue_order_is_independent_of_wall_clock(tmp_path):
    store=RunIdempotencyStore(str(tmp_path/'runs.db'))
    with patch('gateway.platforms.api_server_run_idempotency.time.time',return_value=123):
        for rid in ['z','a','m']:
            store.reserve('scope',rid,rid,rid,{'run_id':rid,'session_id':'s','status':'queued','created_at':123},queue_input=rid)
    assert [row['run_id'] for row in store.queue_inputs('scope','s')]==['z','a','m']
    store.reorder_queue_inputs(['m','z','a'])
    assert [row['run_id'] for row in RunIdempotencyStore(str(tmp_path/'runs.db')).queue_inputs('scope','s')]==['m','z','a']
