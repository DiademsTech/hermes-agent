"""Whole-table message passes read in id slices (Diadems): in rollback-journal mode one reader
statement holds the store's SHARED lock for its whole duration, so a single pass over a large
``messages`` table blocked every gateway write behind it (69 s on a 13 GB store under virtiofs)."""

import sqlite3

import pytest

import tools.tool_result_storage as trs
from hermes_state import SessionDB
from tools.tool_result_storage import _build_persisted_message, message_id_slices, referenced_spillover_names


@pytest.fixture(autouse=True)
def _home(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    return tmp_path


def _table(rows):
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT)")
    conn.executemany("INSERT INTO messages(content) VALUES (?)", [("m",)] * rows)
    return conn


def test_slices_cover_every_row_once():
    conn = _table(2500)
    slices = list(message_id_slices(conn, size=1000))
    assert slices == [(0, 1000), (1000, 2000), (2000, 2500)]


def test_slices_reach_rows_inserted_during_the_pass():
    conn = _table(1500)
    seen = []
    for low, high in message_id_slices(conn, size=1000):
        seen.append((low, high))
        if len(seen) == 1:
            conn.executemany("INSERT INTO messages(content) VALUES (?)", [("late",)] * 700)
    assert seen[-1][1] == 2200
    covered = sum(high - low for low, high in seen)
    assert covered == 2200


def test_slices_stop_at_upto():
    conn = _table(3000)
    assert list(message_id_slices(conn, after_id=500, upto=1700, size=1000)) == [(500, 1500), (1500, 1700)]


def test_empty_table_has_no_slice():
    assert list(message_id_slices(_table(0))) == []


def test_referenced_names_match_a_single_pass(tmp_path, monkeypatch):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("s1", "cli")
    names = [f"call_{i:04d}.txt" for i in range(30)]
    for i, name in enumerate(names):
        pointer = _build_persisted_message(preview="p", has_more=True, original_size=150_000,
                                           file_path=str(tmp_path / "cache" / "spillover" / name))
        db.append_message("s1", "tool", pointer)
        db.append_message("s1", "assistant", f"plain reply {i}")
    db.close()
    spill_dir = tmp_path / "cache" / "spillover"
    spill_dir.mkdir(parents=True)
    import tools.tool_result_storage as storage
    monkeypatch.setattr(storage, "message_id_slices",
                        lambda conn, **kw: message_id_slices(conn, size=7, **kw))
    assert referenced_spillover_names(spill_dir) == set(names)


def test_a_writer_commits_between_slices(tmp_path):
    path = tmp_path / "rollback.db"
    setup = sqlite3.connect(path)
    setup.execute("PRAGMA journal_mode=DELETE")
    setup.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, content TEXT)")
    setup.executemany("INSERT INTO messages(content) VALUES (?)", [("m",)] * 30)
    setup.commit()
    reader = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=0)
    writer = sqlite3.connect(path, timeout=0)
    for low, high in message_id_slices(reader, size=10):
        reader.execute("SELECT content FROM messages WHERE id>? AND id<=?", (low, high)).fetchall()
        writer.execute("INSERT INTO messages(content) VALUES ('w')")
        writer.commit()  # would raise "database is locked" if the reader still held SHARED
        if high >= 60:
            break
