"""CAS controls for API admissions in the canonical FIFO, with no second scheduler.

The admission's seq and original payload_digest remain its immutable acceptance
identity. Explicit edits/reorders change only its executable payload while queued.
Replaying the original submission therefore still returns the same admission.
"""
import hashlib
import hmac
import json

from hermes_state_runtime import RuntimeStoreError, _admission, _epoch, _json, _row

ORDER_SQL = "COALESCE(json_extract(payload_json, '$.api_queue_v1.order'), seq)"


def order_key(row):
    if row.get('queue_order') is not None:
        return row['queue_order']
    payload = row.get('payload')
    if payload is None:
        payload = json.loads(row.get('payload_json') or '{}')
    return (payload.get('api_queue_v1') or {}).get('order', row['seq'])


def revision(row):
    return hashlib.sha256(_json(row['payload']).encode()).hexdigest()


def owns(row, scope):
    data = (row.get('payload') or {}).get('api_turn_v1') or {}
    stored = data.get('run_owner_scope')
    return (row.get('principal_id') == 'api' and isinstance(stored, str)
            and isinstance(scope, str) and hmac.compare_digest(stored, scope))


def check_row(conn, admission_id, scope, expected_revision, *, queued=True):
    raw = _admission(conn, admission_id)
    if raw is None:
        raise RuntimeStoreError('not_found')
    row = _row(raw)
    if not owns(row, scope):
        raise RuntimeStoreError('not_found')
    if queued and row['status'] != 'queued':
        raise RuntimeStoreError('admission_started')
    if not isinstance(expected_revision, str) or not hmac.compare_digest(revision(row), expected_revision):
        raise RuntimeStoreError('queue_conflict')
    return row


def edit_input(db, *, epoch, admission_id, scope, expected_revision, text):
    if not isinstance(text, str) or not text.strip() or len(text) > 250_000:
        raise RuntimeStoreError('invalid_params')

    def write(conn):
        _epoch(conn, epoch)
        row = check_row(conn, admission_id, scope, expected_revision)
        payload = row['payload']
        if not isinstance(payload.get('text'), str):
            raise RuntimeStoreError('unsupported_control')
        # Media, authenticated authors, route and original acceptance identity stay intact.
        payload['text'] = text
        meta = payload.setdefault('api_queue_v1', {})
        meta['revision'] = meta.get('revision', 0) + 1
        conn.execute('UPDATE session_admissions SET payload_json=? WHERE admission_id=?',
                     (_json(payload), admission_id))
        return _row(_admission(conn, admission_id))
    return db._execute_write(write)


def reorder_inputs(db, *, epoch, session_id, scope, expected_order, order):
    if (not isinstance(order, list) or not isinstance(expected_order, list)
            or not all(isinstance(item, str) for item in order + expected_order)
            or len(order) != len(set(order)) or len(expected_order) != len(set(expected_order))
            or set(order) != set(expected_order)):
        raise RuntimeStoreError('invalid_params')

    def write(conn):
        _epoch(conn, epoch)
        rows = [_row(row) for row in conn.execute(
            "SELECT * FROM session_admissions WHERE target_session_id=? AND status='queued' ORDER BY " + ORDER_SQL,
            (session_id,))]
        if any(not owns(row, scope) for row in rows):
            raise RuntimeStoreError('unsupported_control')
        if [row['request_id'] for row in rows] != expected_order:
            raise RuntimeStoreError('queue_conflict')
        positions = sorted(order_key(row) for row in rows)
        by_id = {row['request_id']: row for row in rows}
        for run_id, position in zip(order, positions):
            row = by_id[run_id]
            meta = row['payload'].setdefault('api_queue_v1', {})
            meta.update(order=position, revision=meta.get('revision', 0) + 1)
            conn.execute('UPDATE session_admissions SET payload_json=? WHERE admission_id=?',
                         (_json(row['payload']), row['admission_id']))
    return db._execute_write(write)
