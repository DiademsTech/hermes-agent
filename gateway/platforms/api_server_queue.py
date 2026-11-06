"""HTTP queue controls over the gateway's canonical admissions."""
import asyncio
import sqlite3
from functools import partial
from types import SimpleNamespace

from aiohttp import web

from gateway.platforms.api_server_authority_runs import (
    _authority, run_admission, run_projection, store_error_response,
)
from gateway.session_contract import SessionActor, SessionRef
from gateway.session_runtime_workers import tracked_write
from hermes_state_api_queue import check_row, edit_input, owns, reorder_inputs, revision
from hermes_state_runtime import RuntimeStoreError, list_session_admissions


def _owner(adapter):
    authority = _authority(adapter)
    if authority is None or getattr(authority, 'retiring', False):
        raise RuntimeStoreError('runtime_draining')
    return authority


def _project(adapter, row):
    projection = run_projection(adapter, row['request_id'])
    if projection is None:
        raise RuntimeStoreError('not_found')
    if row['status'] == 'queued':
        projection.update(queued_input=row['payload']['text'], queue_revision=revision(row))
    return projection


async def handle(adapter, request, *, action, _api_server):
    # Queue bodies contain user text. Room grants never authorize this surface.
    error = adapter._check_auth(request)
    if error is not None:
        return error
    try:
        if adapter._room_grant_token(request):
            raise RuntimeStoreError('permission_denied')
        authority = _owner(adapter)
        scope = adapter._run_idempotency_scope(request)
        if action == 'list':
            session_id = request.query.get('session_id')
            if not session_id:
                raise RuntimeStoreError('invalid_params')
            requested_session = session_id
            session_id = await asyncio.to_thread(authority.logical_owner, session_id)
            key = request.query.get('idempotency_key')

            def read():
                if key:
                    run_id = adapter._run_idempotency_store.run_for_key(scope, key)
                    found = run_admission(adapter, run_id) if run_id else None
                    rows = [found[1]] if found else []
                    if not rows and run_id:
                        saved = adapter._durable_run_status(request, run_id)
                        return [saved] if saved and saved.get('session_id') == session_id else []
                else:
                    rows = list_session_admissions(authority.db, session_id=session_id)
                return [{**_project(adapter, row), 'session_id': requested_session} for row in rows
                        if owns(row, scope) and row['target_session_id'] == session_id]
            return web.json_response({'runs': await asyncio.to_thread(read)})

        if action == 'cancel':
            found = await asyncio.to_thread(run_admission, adapter, request.match_info['run_id'])
            if found is None or not owns(found[1], scope):
                raise RuntimeStoreError('not_found')
            authority, row = found
            ref = SessionRef(authority.profile_id, row['target_session_id'])
            actor = SessionActor('api', authority.profile_id,
                frozenset({'session:read', 'session:submit', 'session:control'}), 'api-cancel:' + row['request_id'])
            await authority.cancel_queued(actor, ref, row['admission_id'])
            return web.json_response({'accepted': True, 'run_id': row['request_id'], 'status': 'cancelled'})

        body, error = await adapter._read_json_body(request)
        if error is not None:
            return error
        if not isinstance(body, dict):
            raise RuntimeStoreError('invalid_params')
        if action == 'reorder':
            if set(body) != {'session_id', 'expected_order', 'order'} or not isinstance(body['session_id'], str):
                raise RuntimeStoreError('invalid_params')
            body['session_id'] = await asyncio.to_thread(authority.logical_owner, body['session_id'])
            await tracked_write(authority, partial(reorder_inputs, authority.db, epoch=authority.epoch,
                scope=scope, **body))
            ref = SessionRef(authority.profile_id, body['session_id'])
            authority._publish_pending(ref)
            authority._schedule(ref)
            return web.json_response({'accepted': True})

        run_id = request.match_info['run_id']
        found = await asyncio.to_thread(run_admission, adapter, run_id)
        if found is None or not owns(found[1], scope):
            raise RuntimeStoreError('not_found')
        authority, row = found
        ref = SessionRef(authority.profile_id, row['target_session_id'])
        expected = body.get('expected_revision')
        if action == 'edit':
            if set(body) != {'expected_revision', 'input'}:
                raise RuntimeStoreError('invalid_params')
            changed = await tracked_write(authority, partial(edit_input, authority.db, epoch=authority.epoch,
                admission_id=row['admission_id'], scope=scope, expected_revision=expected, text=body['input']))
            authority._publish_pending(ref)
            return web.json_response({'accepted': True, 'queue_revision': revision(changed)})
        if action == 'steer':
            if set(body) != {'expected_revision'}:
                raise RuntimeStoreError('invalid_params')
            # Check again inside the transfer transaction: an edit between this read and the
            # commit must not deliver the old text. Shared native transfer fences settle observers.
            guard = partial(check_row, admission_id=row['admission_id'], scope=scope,
                            expected_revision=expected, queued=False)
            await asyncio.to_thread(lambda: authority.db._execute_write(guard))
            from gateway.session_queued_controls import steer_queued
            actor = SessionActor('api', authority.profile_id,
                frozenset({'session:read', 'session:submit', 'session:control'}), 'api-queue:' + run_id)
            receipt = await steer_queued(SimpleNamespace(authority=authority, actor=actor), ref,
                {'admission_id': row['admission_id']}, _guard=guard)
            return web.json_response({'accepted': True, 'receipt': receipt})
        raise RuntimeStoreError('invalid_params')
    except (RuntimeStoreError, OSError, sqlite3.Error) as exc:
        return store_error_response(_api_server._openai_error, exc)
