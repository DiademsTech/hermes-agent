"""Real HTTP queue edits/reorders target the same admissions the owner claims."""
import hashlib
from types import SimpleNamespace

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.platforms.api_server_runs import _http_routes
from gateway.session_api_turn import admit_api_turn
from hermes_state_runtime import claim_session_input, get_session_admission, settle_session_input

KEY = 'fixture-key-0123456789'
SCOPE = hashlib.sha256(('default\0' + KEY).encode()).hexdigest()
HEADERS = {'Authorization': 'Bearer ' + KEY}


def seed(api, owner, monkeypatch, count=3):
    api._api_key = KEY
    monkeypatch.setattr(owner, '_schedule', lambda ref: None)
    rows = []
    for i in range(count):
        rows.append(admit_api_turn(api, session_id='queue', request_id=f'run-{i}',
            user_message=f'text {i}', conversation_history=[], history_from_session=True, run_owner_scope=SCOPE)[2])
    app = web.Application()
    for method, path, handler in _http_routes(api):
        app.router.add_route(method, path, handler)
    return app, rows


@pytest.mark.asyncio
async def test_edit_reorder_and_claim_share_one_durable_fifo(api, owner, monkeypatch):
    app, original = seed(api, owner, monkeypatch)
    async with TestClient(TestServer(app)) as client:
        assert (await client.get('/v1/runs?session_id=queue')).status == 401
        rows = (await (await client.get('/v1/runs?session_id=queue', headers=HEADERS)).json())['runs']
        assert [r['run_id'] for r in rows] == ['run-0', 'run-1', 'run-2']
        old_revision = rows[1]['queue_revision']
        response = await client.patch('/v1/runs/run-1/queue', headers=HEADERS,
            json={'expected_revision': old_revision, 'input': 'edited'})
        assert response.status == 200, await response.text()
        assert (await client.patch('/v1/runs/run-1/queue', headers=HEADERS,
            json={'expected_revision': old_revision, 'input': 'stale edit'})).status == 409
        # The original accepted request is still idempotent after an explicit edit.
        replay = admit_api_turn(api, session_id='queue', request_id='run-1', user_message='text 1',
            conversation_history=[], history_from_session=True, run_owner_scope=SCOPE)[2]
        assert replay['admission_id'] == original[1]['admission_id'] and replay['payload']['text'] == 'edited'
        response = await client.patch('/v1/runs/queue', headers=HEADERS, json={
            'session_id': 'queue', 'expected_order': ['run-0', 'run-1', 'run-2'],
            'order': ['run-2', 'run-1', 'run-0']})
        assert response.status == 200, await response.text()
        assert claim_session_input(owner.db, epoch=owner.epoch, session_id='queue')['request_id'] == 'run-2'
        assert (await client.patch('/v1/runs/run-2/queue', headers=HEADERS,
            json={'expected_revision': rows[2]['queue_revision'], 'input': 'too late'})).status == 409
    # Sequence remains an acceptance identity, independent from the requested execution order.
    assert get_session_admission(owner.db, admission_id=original[2]['admission_id'])['seq'] == original[2]['seq']


@pytest.mark.asyncio
async def test_queue_does_not_expose_or_mutate_another_auth_scope(api, owner, monkeypatch):
    app, rows = seed(api, owner, monkeypatch, count=1)
    other = admit_api_turn(api, session_id='queue', request_id='private', user_message='private content',
        conversation_history=[], history_from_session=True, run_owner_scope='a' * 64)[2]
    async with TestClient(TestServer(app)) as client:
        listing = await (await client.get('/v1/runs?session_id=queue', headers=HEADERS)).json()
        assert [r['run_id'] for r in listing['runs']] == ['run-0']
        assert 'private content' not in str(listing)
        response = await client.patch('/v1/runs/private/queue', headers=HEADERS,
            json={'expected_revision': '0' * 64, 'input': 'stolen'})
        assert response.status in {404, 409}
        assert (await client.patch('/v1/runs/queue', headers=HEADERS, json={
            'session_id': 'queue', 'expected_order': ['run-0'], 'order': ['run-0']})).status == 409
    assert get_session_admission(owner.db, admission_id=other['admission_id'])['payload']['text'] == 'private content'


@pytest.mark.asyncio
async def test_lost_ack_lookup_and_cancel_do_not_start_or_stop_other_turns(api, owner, monkeypatch):
    app, rows = seed(api, owner, monkeypatch, count=2)
    api._run_idempotency_store.reserve(SCOPE, 'queue-key', 'fingerprint', 'run-1',
                                      {'run_id': 'run-1', 'status': 'queued', 'session_id': 'queue'})
    claimed = claim_session_input(owner.db, epoch=owner.epoch, session_id='queue')
    async with TestClient(TestServer(app)) as client:
        listing = await (await client.get('/v1/runs?session_id=queue&idempotency_key=queue-key', headers=HEADERS)).json()
        assert [row['run_id'] for row in listing['runs']] == ['run-1']
        assert (await client.delete('/v1/runs/run-0/queue', headers=HEADERS)).status == 409
        assert (await client.delete('/v1/runs/run-1/queue', headers=HEADERS)).status == 200
    assert get_session_admission(owner.db, admission_id=claimed['admission_id'])['status'] == 'started'
    assert get_session_admission(owner.db, admission_id=rows[1]['admission_id'])['outcome'] == 'cancelled'


def test_edit_revision_survives_an_aba_edit_and_reopened_store(api, owner, monkeypatch):
    from hermes_state import SessionDB
    from hermes_state_api_queue import edit_input, revision
    from hermes_state_runtime import RuntimeStoreError
    _, rows = seed(api, owner, monkeypatch, count=1)
    row = rows[0]
    original = revision(row)
    changed = edit_input(owner.db, epoch=owner.epoch, admission_id=row['admission_id'], scope=SCOPE,
                         expected_revision=original, text='changed')
    edit_input(owner.db, epoch=owner.epoch, admission_id=row['admission_id'], scope=SCOPE,
               expected_revision=revision(changed), text='text 0')
    with SessionDB(owner.db.db_path) as reopened:
        with pytest.raises(RuntimeStoreError, match='queue_conflict'):
            edit_input(reopened, epoch=owner.epoch, admission_id=row['admission_id'], scope=SCOPE,
                       expected_revision=original, text='stale')


def test_reordered_head_invalidates_an_inflight_preclaim(api, owner, monkeypatch):
    from hermes_state_api_queue import reorder_inputs
    _, rows = seed(api, owner, monkeypatch, count=2)
    reorder_inputs(owner.db, epoch=owner.epoch, session_id='queue', scope=SCOPE,
                   expected_order=['run-0', 'run-1'], order=['run-1', 'run-0'])
    assert claim_session_input(owner.db, epoch=owner.epoch, session_id='queue', head=rows[0]['admission_id']) is False
    assert claim_session_input(owner.db, epoch=owner.epoch, session_id='queue', head=rows[1]['admission_id'])['request_id'] == 'run-1'


@pytest.mark.asyncio
async def test_queued_steer_is_transferred_once_and_settles_with_the_running_claim(api, owner, monkeypatch):
    from gateway.session_queued_controls import publish_transfers, steer_transfers
    from gateway.session_contract import SessionRef
    from gateway.session_api_turn import _settled_api_result
    app, rows = seed(api, owner, monkeypatch, count=2)
    claimed = claim_session_input(owner.db, epoch=owner.epoch, session_id='queue')
    generation = claimed['generation']
    owner.executing['queue'] = generation
    received = []
    owner.adopt_agent('queue', generation, SimpleNamespace(steer=lambda text: received.append(text) or True))
    async with TestClient(TestServer(app)) as client:
        listing = await (await client.get('/v1/runs?session_id=queue', headers=HEADERS)).json()
        queued = next(row for row in listing['runs'] if row['run_id'] == 'run-1')
        for _ in range(2):
            response = await client.post('/v1/runs/run-1/queue/steer', headers=HEADERS,
                                         json={'expected_revision': queued['queue_revision']})
            assert response.status == 200, await response.text()
    assert received == ['text 1']
    transfer = get_session_admission(owner.db, admission_id=rows[1]['admission_id'])
    assert transfer['outcome'] == 'steering'
    settle_session_input(owner.db, epoch=owner.epoch, admission_id=claimed['admission_id'],
                         generation=generation, outcome='completed',
                         _transfers=steer_transfers(owner, 'queue', generation))
    publish_transfers(owner, SessionRef(owner.profile_id, 'queue'), generation)
    assert _settled_api_result(owner, rows[1]['admission_id'])[0]['completed'] is True
    assert claim_session_input(owner.db, epoch=owner.epoch, session_id='queue') is None
