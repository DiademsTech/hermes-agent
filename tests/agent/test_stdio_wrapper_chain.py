"""Agent builds and thread-scoped silences must not grow sys.stdout/sys.stderr (#139687).

A gateway builds a fresh agent per turn (``_install_safe_stdio``) and the background review /
``execute_code`` RPC path enters ``thread_scoped_silence``. When the two alternate, each cycle
used to add a ``_SafeWriter`` and a ``_ThreadRoutingStream`` on top of the previous pair. After a
few hundred turns an attribute lookup that falls through to ``__getattr__`` (``line_buffering``,
``encoding``…) walked the whole chain and raised RecursionError, failing every turn until the
gateway restarted.
"""

from __future__ import annotations

import io
import sys

import agent.thread_scoped_output as thread_output
from agent.process_bootstrap import _SafeWriter, _install_safe_stdio
from agent.thread_scoped_output import _ThreadRoutingStream, thread_scoped_silence


def _chain_depth(stream) -> int:
    depth = 0
    while isinstance(stream, (_SafeWriter, _ThreadRoutingStream)):
        depth += 1
        stream = stream._inner if isinstance(stream, _SafeWriter) else stream._passthrough
    return depth


def _fresh_streams(monkeypatch) -> io.StringIO:
    # In the test body: pytest's capture rebinds sys.stdout between setup and call.
    monkeypatch.setattr(thread_output, "_installed", {})
    monkeypatch.setattr(thread_output, "_routing_states", {})
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    monkeypatch.setattr(sys, "stderr", io.StringIO())
    return stdout


def test_build_silence_cycles_keep_a_bounded_chain(monkeypatch):
    _fresh_streams(monkeypatch)
    for _ in range(50):
        _install_safe_stdio()
        with thread_scoped_silence():
            pass

    assert _chain_depth(sys.stdout) <= 2
    assert _chain_depth(sys.stderr) <= 2


def test_attribute_fallthrough_survives_a_long_running_gateway(monkeypatch):
    stdout = _fresh_streams(monkeypatch)
    # 600 cycles is a few days for a webhook-driven agent; v0.22.0 overflows after a few hundred.
    for _ in range(600):
        _install_safe_stdio()
        with thread_scoped_silence():
            pass

    # The setup_logging probe that failed in production: an attribute neither wrapper defines.
    assert getattr(sys.stderr, "line_buffering", False) is False
    print("still routed")
    assert stdout.getvalue() == "still routed\n"
