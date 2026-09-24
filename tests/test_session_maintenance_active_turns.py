"""Retention must never hold the database while live turns need heartbeats."""

import time

import pytest

from hermes_state import SessionDB


@pytest.fixture
def stores(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with SessionDB(tmp_path / "state.db") as maintenance, SessionDB(tmp_path / "state.db") as writer:
        maintenance.create_session("old", source="cli")
        maintenance.append_message("old", "user", "Expired synthetic transcript")
        maintenance.end_session("old", "done")
        old = time.time() - 100 * 86400
        maintenance._execute_write(lambda conn: conn.execute(
            "UPDATE messages SET timestamp = ? WHERE session_id = 'old'", (old,)
        ))
        writer.create_session("live", source="api_server")
        yield maintenance, writer


def test_auto_prune_defers_for_other_connection_then_retries_when_idle(stores):
    maintenance, writer = stores
    assert writer.try_acquire_session_turn_lease("live", "writer")
    result = maintenance.maybe_auto_prune_and_vacuum(vacuum=False)
    assert result["skipped"] is True
    assert result["deferred"] == "active_turn"
    assert result["pruned"] == 0
    assert maintenance.get_session("old") is not None
    assert maintenance.get_meta("last_auto_prune") is None
    assert writer.refresh_session_turn_lease("live", "writer")
    writer.append_message("live", "user", "Conversation can still persist")
    writer.release_session_turn_lease("live", "writer")
    result = maintenance.maybe_auto_prune_and_vacuum(vacuum=False)
    assert result["pruned"] == 1
    assert maintenance.get_session("old") is None
    assert writer.get_session("live") is not None
    assert maintenance.get_meta("last_auto_prune") is not None
    assert maintenance.get_meta("last_vacuum") is None


def test_auto_prune_rechecks_activity_at_write_transaction(stores, monkeypatch):
    maintenance, writer = stores
    execute = maintenance._execute_write
    admitted = False

    def admit_before_transaction(fn, **kwargs):
        nonlocal admitted
        if not admitted:
            admitted = True
            assert writer.try_acquire_session_turn_lease("live", "new-writer")
        return execute(fn, **kwargs)

    monkeypatch.setattr(maintenance, "_execute_write", admit_before_transaction)
    result = maintenance.maybe_auto_prune_and_vacuum(vacuum=False)
    assert result["deferred"] == "active_turn"
    assert maintenance.get_session("old") is not None
    assert maintenance.get_meta("last_auto_prune") is None


def test_expired_lease_does_not_disable_retention(stores):
    maintenance, writer = stores
    assert writer.try_acquire_session_turn_lease("live", "dead-writer")
    writer._execute_write(lambda conn: conn.execute(
        "UPDATE session_turn_leases SET expires_at = ?", (time.time() - 1,)
    ))
    result = maintenance.maybe_auto_prune_and_vacuum(vacuum=False)
    assert result["pruned"] == 1
    assert maintenance.get_session("old") is None


def test_explicit_prune_keeps_its_existing_contract(stores):
    maintenance, writer = stores
    assert writer.try_acquire_session_turn_lease("live", "writer")
    assert maintenance.prune_sessions(older_than_days=90) == 1
    assert writer.refresh_session_turn_lease("live", "writer")
