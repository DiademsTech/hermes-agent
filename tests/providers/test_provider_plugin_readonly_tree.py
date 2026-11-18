"""A provider plugin installed read-only (a read-only mount) still stages, and leaves nothing behind.

``copytree`` keeps the source's modes: a read-only directory then can be neither renamed into place (Linux
needs write access to a directory to move it) nor removed, so every scan failed to load the plugin and left a
full copy in the temporary directory.
"""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

from providers import plugin_reload

pytestmark = pytest.mark.skipif(
    sys.platform == "win32" or (hasattr(os, "geteuid") and os.geteuid() == 0),
    reason="POSIX permissions, enforced only for a non-root user",
)


def _read_only_plugin(root: Path) -> Path:
    plugin = root / "demo-plugin"
    (plugin / "pkg" / "bin").mkdir(parents=True)
    (plugin / "plugin.yaml").write_text("name: demo\nkind: model-provider\n")
    (plugin / "__init__.py").write_text("X = 1\n")
    (plugin / "pkg" / "bin" / "tool").write_text("#!/bin/sh\n")
    for dirpath, _dirs, files in os.walk(plugin, topdown=False):
        for name in files:
            os.chmod(os.path.join(dirpath, name), 0o444)
        os.chmod(dirpath, 0o555)
    return plugin


def test_read_only_plugin_stages_and_cleans_up(tmp_path, monkeypatch):
    plugin = _read_only_plugin(tmp_path / "src")
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    monkeypatch.setattr(plugin_reload, "_snapshot_root", snapshots)
    try:
        target = plugin_reload._snapshot(plugin, "demo_abc")
        assert (target / "pkg" / "bin" / "tool").read_text() == "#!/bin/sh\n"
        assert [p.name for p in target.parent.iterdir()] == ["demo-plugin"]  # no .copy-* left behind
        assert plugin_reload._snapshot(plugin, "demo_abc") == target  # reused, not copied again
        plugin_reload._rmtree(snapshots)
        assert not snapshots.exists()
    finally:
        for dirpath, _dirs, _files in os.walk(tmp_path):
            os.chmod(dirpath, stat.S_IRWXU)
