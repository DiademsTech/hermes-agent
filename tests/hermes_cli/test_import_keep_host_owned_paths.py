"""``HERMES_HOST_OWNED_PATHS``: a host's own paths inside the home stay out of a backup and of a transfer.

A real ``hermes backup`` of a source home and a real ``hermes import --keep-host`` into a destination
home whose host keeps read-only dependencies, rendered files and its own work folder inside it.
"""

import sqlite3
import zipfile
from argparse import Namespace
from pathlib import Path

import pytest

import hermes_cli.gateway_setup_service as service_setup
from hermes_cli.host_owned import host_owned_patterns, is_host_owned


@pytest.fixture(autouse=True)
def _no_real_gateway_service(monkeypatch):
    import hermes_cli.gateway as gateway_mod
    monkeypatch.setattr(service_setup, "ensure_gateway_service", lambda **kw: False)
    monkeypatch.setattr(gateway_mod, "_is_service_running", lambda: False)


def _write(root: Path, rel: str, text: str) -> None:
    (root / rel).parent.mkdir(parents=True, exist_ok=True)
    (root / rel).write_text(text)


def _db(path: Path, sessions: list[str]) -> None:
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE IF NOT EXISTS sessions (id TEXT PRIMARY KEY)")
        conn.executemany("INSERT INTO sessions VALUES (?)", [(s,) for s in sessions])


HOST = "lazy-packages,plugins/host-*,.host-work,skills/*/host-skill"


def test_patterns_ignore_unsafe_entries(monkeypatch):
    monkeypatch.setenv("HERMES_HOST_OWNED_PATHS", " lazy-packages , /etc, ../x, a/./b,plugins/host-* ,")
    assert host_owned_patterns() == (("lazy-packages",), ("plugins", "host-*"))
    assert is_host_owned(("lazy-packages", "bin", "x"))
    assert is_host_owned(("plugins", "host-memory", "__init__.py"))
    assert not is_host_owned(("plugins", "mine"))
    assert not is_host_owned(("lazy",))


def test_unset_changes_nothing(monkeypatch):
    monkeypatch.delenv("HERMES_HOST_OWNED_PATHS", raising=False)
    assert host_owned_patterns() == ()
    assert not is_host_owned(("lazy-packages",))


@pytest.fixture
def homes(tmp_path, monkeypatch):
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write(src, "config.yaml", "model:\n  provider: openrouter\n  default: x/y\n")
    _write(src, ".env", "OPENROUTER_API_KEY=SRC-OR\n")
    _write(src, "SOUL.md", "source persona\n")
    _write(src, "skills/mine/SKILL.md", "# mine\n")
    # The source happens to carry a path the destination's host owns: never written.
    _write(src, "plugins/host-memory/__init__.py", "SOURCE COPY\n")
    _db(src / "state.db", ["src-chat"])

    _write(dst, "config.yaml", "model:\n  provider: custom\n  default: host\n")
    _write(dst, ".env", "API_SERVER_KEY=DST-API\n")
    _write(dst, "SOUL.md", "host persona\n")
    _write(dst, "lazy-packages/bin/tool", "host dependency\n")
    _write(dst, "plugins/host-memory/__init__.py", "HOST PLUGIN\n")
    _write(dst, ".host-work/import/archive.zip", "the archive being imported\n")
    _write(dst, "skills/tools/host-skill/SKILL.md", "# host skill\n")
    _write(dst, "skills/old/SKILL.md", "# replaced by the transfer\n")
    _db(dst / "state.db", ["host-chat"])
    return src, dst


def _backup(monkeypatch, home: Path, out: Path) -> Path:
    from hermes_cli.backup import run_backup
    monkeypatch.setenv("HERMES_HOME", str(home))
    run_backup(Namespace(output=str(out)))
    return out


def test_backup_leaves_host_owned_paths_out(homes, tmp_path, monkeypatch):
    _src, dst = homes
    monkeypatch.setenv("HERMES_HOST_OWNED_PATHS", HOST)
    archive = _backup(monkeypatch, dst, tmp_path / "dst.zip")
    with zipfile.ZipFile(archive) as zf:
        names = set(zf.namelist())
    assert "SOUL.md" in names and "skills/old/SKILL.md" in names
    assert not any(n.startswith(("lazy-packages/", "plugins/host-memory/", ".host-work/")) for n in names)
    assert "skills/tools/host-skill/SKILL.md" not in names


def test_keep_host_import_neither_writes_nor_removes_host_paths(homes, tmp_path, monkeypatch):
    from hermes_cli.backup import run_import
    src, dst = homes
    monkeypatch.delenv("HERMES_HOST_OWNED_PATHS", raising=False)
    archive = _backup(monkeypatch, src, tmp_path / "src.zip")
    monkeypatch.setenv("HERMES_HOST_OWNED_PATHS", HOST)
    monkeypatch.setenv("HERMES_HOME", str(dst))
    assert run_import(Namespace(zipfile=str(archive), force=True, keep_host=True, messaging_off=True)) is None

    # The archive's agent replaced the destination's...
    assert (dst / "SOUL.md").read_text() == "source persona\n"
    assert (dst / "skills/mine/SKILL.md").exists()
    assert not (dst / "skills/old/SKILL.md").exists()
    # ...while every host-owned path is exactly as the host left it.
    assert (dst / "lazy-packages/bin/tool").read_text() == "host dependency\n"
    assert (dst / "plugins/host-memory/__init__.py").read_text() == "HOST PLUGIN\n"
    assert (dst / ".host-work/import/archive.zip").exists()
    assert (dst / "skills/tools/host-skill/SKILL.md").exists()
    # The undo point does not copy them either.
    undo = sorted((dst / "backups").glob("pre-transfer-*.zip"))[-1]
    with zipfile.ZipFile(undo) as zf:
        assert not any(n.startswith(("lazy-packages/", "plugins/host-memory/")) for n in zf.namelist())
