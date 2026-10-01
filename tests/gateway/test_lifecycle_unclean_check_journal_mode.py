"""The unclean-exit integrity check only scans a WAL ``state.db``.

A rollback-journal store (``delete``/``truncate``/``persist``) cannot keep a torn
b-tree after a kill: SQLite journals each page's original image before it
overwrites the page, and the next opener rolls the hot journal back. Scanning it
anyway is a whole-store ``PRAGMA quick_check`` under a SHARED lock before the API
port binds — 11-13 minutes on a 9.1 GB DELETE-mode store after an s6 stage-3
SIGKILL. WAL stores keep the check: a SIGKILL mid-checkpoint is what it is for.
The unclean exit itself is still recorded and logged either way.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import gateway.lifecycle_ledger as ledger
from gateway.lifecycle_ledger import get_lifecycle_sentinel_path, record_startup, state_db_journal_mode

_DEAD_PID = 2 ** 22 + 12345  # beyond default pid_max; never alive
_SKIPPED = "skipped: rollback journal"


def _write_running_sentinel(home: Path) -> None:
    path = get_lifecycle_sentinel_path(home)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"phase": "running", "pid": _DEAD_PID, "start_time": 1000.0,
                    "started_at": "2026-10-01T22:41:07+00:00"}),
        encoding="utf-8",
    )


def _make_state_db(home: Path, *, journal_mode: str, corrupt: bool = False) -> Path:
    path = home / "state.db"
    home.mkdir(parents=True, exist_ok=True)
    with closing(sqlite3.connect(path)) as conn:
        conn.execute(f"PRAGMA journal_mode={journal_mode}")
        conn.execute("CREATE TABLE sessions (id INTEGER PRIMARY KEY, v TEXT)")
        conn.executemany("INSERT INTO sessions (v) VALUES (?)", [(f"row-{i}" * 40,) for i in range(4000)])
        conn.commit()
    if corrupt:  # a genuinely torn b-tree page
        with open(path, "r+b") as handle:
            handle.seek(4096 * 6)
            handle.write(b"\xEF" * 4096)
    return path


def _exit_diag_records(home: Path) -> list:
    log = home / "logs" / "gateway-exit-diag.log"
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()]


def _spy_on_the_check(monkeypatch) -> list:
    calls: list = []
    real = ledger.check_state_db_integrity

    def spy(**kwargs):
        calls.append(kwargs)
        return real(**kwargs)

    monkeypatch.setattr(ledger, "check_state_db_integrity", spy)
    return calls


# ── journal mode detection ──────────────────────────────────────────────────


def test_probe_reads_wal_from_the_header_and_rollback_modes_as_delete(tmp_path: Path) -> None:
    _make_state_db(tmp_path / "wal", journal_mode="WAL")
    _make_state_db(tmp_path / "delete", journal_mode="DELETE")
    _make_state_db(tmp_path / "truncate", journal_mode="TRUNCATE")

    assert state_db_journal_mode(home=tmp_path / "wal") == "wal"
    assert state_db_journal_mode(home=tmp_path / "delete") == "delete"
    # Rollback modes are per-connection: a fresh connection reports SQLite's default.
    assert state_db_journal_mode(home=tmp_path / "truncate") == "delete"


def test_probe_is_none_for_a_missing_or_unreadable_store(tmp_path: Path) -> None:
    assert state_db_journal_mode(home=tmp_path) is None
    assert not (tmp_path / "state.db").exists(), "the probe must never create the store"

    (tmp_path / "state.db").write_bytes(b"not a sqlite database " * 64)
    assert state_db_journal_mode(home=tmp_path) is None


# ── gating on the unclean-exit path ─────────────────────────────────────────


def test_rollback_journal_store_skips_the_quick_check(tmp_path: Path, monkeypatch, caplog) -> None:
    _make_state_db(tmp_path, journal_mode="DELETE")
    _write_running_sentinel(tmp_path)
    calls = _spy_on_the_check(monkeypatch)

    with caplog.at_level(logging.INFO, logger="gateway.lifecycle_ledger"):
        evidence = record_startup(home=tmp_path)

    assert evidence is not None
    assert calls == [], "quick_check scanned a rollback-journal store"
    assert evidence["state_db_journal_mode"] == "delete"
    assert evidence["state_db_integrity"] == _SKIPPED
    assert not [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert any("quick_check is skipped" in r.getMessage() for r in caplog.records)


def test_skipped_check_still_records_and_logs_the_unclean_exit(tmp_path: Path, caplog) -> None:
    _make_state_db(tmp_path, journal_mode="DELETE")
    _write_running_sentinel(tmp_path)

    with caplog.at_level(logging.WARNING, logger="gateway.lifecycle_ledger"):
        record_startup(home=tmp_path)

    (record,) = _exit_diag_records(tmp_path)
    assert record["tag"] == "gateway.previous_unclean_exit"
    assert record["prior_pid"] == _DEAD_PID
    assert record["state_db_journal_mode"] == "delete"
    assert record["state_db_integrity"] == _SKIPPED
    assert any("exited UNCLEANLY" in r.getMessage() for r in caplog.records)
    sentinel = json.loads(get_lifecycle_sentinel_path(tmp_path).read_text(encoding="utf-8"))
    assert sentinel["phase"] == "running" and sentinel["prior_unclean_exit"] is True


def test_wal_store_is_still_checked_after_an_unclean_exit(tmp_path: Path, monkeypatch, caplog) -> None:
    _make_state_db(tmp_path, journal_mode="WAL", corrupt=True)
    _write_running_sentinel(tmp_path)
    calls = _spy_on_the_check(monkeypatch)

    with caplog.at_level(logging.ERROR, logger="gateway.lifecycle_ledger"):
        evidence = record_startup(home=tmp_path)

    assert evidence is not None and len(calls) == 1
    assert evidence["state_db_journal_mode"] == "wal"
    verdict = evidence["state_db_integrity"]
    assert verdict not in ("ok", "absent", _SKIPPED) and not verdict.startswith("check-failed")
    assert any("FAILED integrity check" in r.getMessage() for r in caplog.records)


def test_unreadable_journal_mode_keeps_the_check(tmp_path: Path, monkeypatch) -> None:
    """Fail closed: a store whose mode cannot be read is still scanned."""
    _make_state_db(tmp_path, journal_mode="DELETE")
    _write_running_sentinel(tmp_path)
    calls = _spy_on_the_check(monkeypatch)
    monkeypatch.setattr(ledger, "state_db_journal_mode", lambda home=None: None)

    evidence = record_startup(home=tmp_path)

    assert evidence is not None and len(calls) == 1
    assert evidence["state_db_integrity"] == "ok"


# ── the premise: SQLite restores a killed rollback-journal writer ───────────

_KILLED_WRITER = r"""
import sqlite3, sys, time
conn = sqlite3.connect(sys.argv[1], isolation_level=None)
conn.execute("PRAGMA cache_size=1")  # spill uncommitted pages into the main file
conn.execute("BEGIN")
conn.execute("UPDATE sessions SET v = v || 'x'")
conn.execute("INSERT INTO sessions (v) SELECT v FROM sessions")
print("in-flight", flush=True)
time.sleep(60)
"""


def _rows(db: Path) -> tuple:
    with closing(sqlite3.connect(db)) as conn:
        return conn.execute("SELECT count(*), sum(length(v)) FROM sessions").fetchone()


def test_killed_rollback_journal_writer_is_restored_on_the_next_open(tmp_path: Path) -> None:
    """Kill a DELETE-mode writer whose uncommitted pages already reached the main file: the
    next open (the probe here) rolls the hot journal back and the store is intact, as before."""
    db = _make_state_db(tmp_path, journal_mode="DELETE")
    before, pristine = _rows(db), db.read_bytes()
    writer = subprocess.Popen([sys.executable, "-c", _KILLED_WRITER, str(db)],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert writer.stdout.readline().strip() == "in-flight"
        assert db.read_bytes() != pristine, "the writer never reached the main file"
    finally:
        writer.kill()
        writer.wait(timeout=30)
        writer.stdout.close()
    journal = Path(f"{db}-journal")
    assert journal.exists(), "the killed writer left no hot journal"

    assert state_db_journal_mode(home=tmp_path) == "delete"

    assert not journal.exists(), "opening the store did not roll the hot journal back"
    with closing(sqlite3.connect(db)) as conn:
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert _rows(db) == before
