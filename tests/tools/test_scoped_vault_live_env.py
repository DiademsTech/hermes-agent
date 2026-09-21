"""New profile credentials reach real children without restarting the gateway."""

import json
import os
import sys
from contextlib import contextmanager

import pytest

from agent import secret_scope as ss
from tools import env_passthrough as ep
from tools.code_execution_tool import _scrub_child_env
from tools.environments.local import _make_run_env, _sanitize_subprocess_env

KEY = "VAULT_TEST_SERVICE_TOKEN"


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch):
    ep.clear_env_passthrough()
    monkeypatch.setattr(ep, "_config_passthrough", frozenset())
    monkeypatch.delenv(KEY, raising=False)
    ss.set_multiplex_active(True)
    ep.register_env_passthrough([KEY])
    yield
    ep.clear_env_passthrough()
    ss.set_multiplex_active(False)


@contextmanager
def profile(home, value):
    (home / ".env").write_text(
        f"{KEY}={value}\nUNDECLARED_TOKEN=hidden\nOPENAI_API_KEY=provider-secret\n"
        if value is not None else "", encoding="utf-8",
    )
    token = ss.set_secret_scope(ss.build_profile_secret_scope(home))
    try:
        yield
    finally:
        ss.reset_secret_scope(token)


@pytest.mark.parametrize("builder", [_make_run_env, _sanitize_subprocess_env, _scrub_child_env])
def test_add_rotate_delete_and_other_profile(builder, tmp_path, monkeypatch):
    """No process-global copy is needed; a scoped miss never borrows a stale key."""
    base = {"PATH": os.environ.get("PATH", "")}
    before = dict(base)
    for value in (None, "first", "rotated", None):
        with profile(tmp_path, value):
            child = builder(base)
            assert child.get(KEY) == value
            assert "UNDECLARED_TOKEN" not in child
            assert "OPENAI_API_KEY" not in child
        assert KEY not in os.environ
        assert base == before
    monkeypatch.setenv(KEY, "launch-profile-stale")
    with profile(tmp_path, None):
        assert KEY not in builder({KEY: "launch-profile-stale"})
    # An unbound invocation must not discover profile values on its own.
    ss.set_multiplex_active(False)
    monkeypatch.delenv(KEY)
    assert KEY not in builder(base)


def test_protected_credentials_and_plugin_strip_still_apply(tmp_path, monkeypatch):
    ep.register_env_passthrough(["OPENAI_API_KEY", "AUXILIARY_FAKE_API_KEY"])
    assert not ep.is_env_passthrough("OPENAI_API_KEY")
    assert not ep.is_env_passthrough("AUXILIARY_FAKE_API_KEY")
    monkeypatch.setattr("tools.environments.local._plugin_terminal_env_strip_keys", lambda: {KEY})
    with profile(tmp_path, "allowed"):
        assert KEY not in _sanitize_subprocess_env({})


def test_existing_terminal_sees_next_turn_vault_changes(tmp_path):
    from tools.environments.local import LocalEnvironment

    env = None
    try:
        with profile(tmp_path, None):
            env = LocalEnvironment(cwd=str(tmp_path))
            assert env.execute(f"printf '%s' \"${{{KEY}-missing}}\"")["output"] == "missing"
        for value in ("added", "rotated", None):
            with profile(tmp_path, value):
                result = env.execute(f"printf '%s' \"${{{KEY}-missing}}\"")
                assert result["output"] == (value or "missing")
    finally:
        if env is not None:
            env.cleanup()


def test_native_python_reset_picks_up_vault_without_gateway_restart(tmp_path):
    from tools.code_kernel import execute_in_session_kernel, shutdown_all_kernels

    def run(reset=False):
        return json.loads(execute_in_session_kernel(
            f"import os; print(os.environ.get('{KEY}', 'missing'))",
            task_id="vault-live-test", mode="strict", child_python=sys.executable,
            child_cwd=str(tmp_path), sandbox_tools=frozenset(), timeout=30,
            max_tool_calls=1, reset=reset, is_interrupted=lambda: False,
        ))

    shutdown_all_kernels()
    try:
        with profile(tmp_path, None):
            assert run()["output"].strip() == "missing"
        for value in ("added", "rotated", None):
            with profile(tmp_path, value):
                result = run(reset=True)
                assert result["status"] == "success", result
                assert result["output"].strip() == (value or "missing")
                assert result["kernel"]["state_reset"] is True
    finally:
        shutdown_all_kernels()
