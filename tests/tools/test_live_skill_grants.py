"""Exercise live skill grants through installed Hermes tools using synthetic secrets only.

Run inside the candidate image with /opt/hermes on PYTHONPATH. All tool state is
confined to a temporary home; no model, provider or client credential is used.
"""

import contextvars
import json
import os
import shlex
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


import pytest


@pytest.mark.platforms("posix")
def test_live_skill_grants_in_worker_tools():
    with tempfile.TemporaryDirectory(prefix="diadems-vault-contract-") as directory:
        home = Path(directory)
        os.environ.update(HERMES_HOME=directory, TERMINAL_CWD=directory, TERMINAL_ENV="local")
        (home / "config.yaml").write_text(
            "terminal:\n  env_passthrough: []\n  backend: local\n", encoding="utf-8"
        )
        from agent.secret_scope import load_env_file, set_multiplex_active, set_secret_scope
        from hermes_constants import set_hermes_home_override
        from tools import skills_tool, terminal_tool, code_execution_tool  # noqa: F401
        from tools.code_kernel import shutdown_all_kernels
        from tools.registry import registry
        from tools.terminal_tool_lifecycle import cleanup_all_environments
        from tools.thread_context import propagate_context_to_thread

        set_multiplex_active(True)
        contexts = [contextvars.Context(), contextvars.Context()]
        for context in contexts:
            context.run(set_hermes_home_override, home)
        skill = home / "skills" / "vault-contract" / "SKILL.md"
        skill.parent.mkdir(parents=True)
        code = (
            "import os,json; print(json.dumps({"
            "'present':bool(os.getenv('DIADEMS_CONTRACT_PASSWORD')),"
            "'rotated':os.getenv('DIADEMS_CONTRACT_PASSWORD')=='rotated-synthetic'}))"
        )

        def refresh(value):
            (home / ".env").write_text(
                f"DIADEMS_CONTRACT_PASSWORD={value}\n" if value else "", encoding="utf-8"
            )
            for context in contexts:
                context.run(set_secret_scope, load_env_file(home / ".env"), profile_home=directory)

        def call(context, name, args, owner="vault-contract-a"):
            callback = lambda: registry.dispatch(name, args, task_id=owner)
            with ThreadPoolExecutor(max_workers=1) as pool:
                result = pool.submit(context.run(propagate_context_to_thread, callback)).result(timeout=60)
            result = json.loads(result)
            assert not result.get("error"), {"tool": name, "error": result.get("error")}
            return result

        def probe(context, present, rotated=False, owner="vault-contract-a"):
            for name, args in (
                ("terminal", {"command": f"{shlex.quote(sys.executable)} -c {shlex.quote(code)}", "timeout": 20}),
                ("execute_code", {"code": code, "reset": True}),
            ):
                result = call(context, name, args, owner)
                assert result.get("exit_code") == 0, {"tool": name, "status": result.get("status")}
                observed = json.loads(result["output"].strip())
                assert observed == {"present": present, "rotated": rotated}, (name, observed)

        try:
            # Add the secret and its skill after importing/starting the tool runtime.
            refresh("initial-synthetic")
            skill.write_text(
                "---\nname: vault-contract\ndescription: Synthetic vault contract\n"
                "required_environment_variables:\n"
                "  - name: DIADEMS_CONTRACT_PASSWORD\n    optional: true\n"
                "---\nReport presence only.\n", encoding="utf-8"
            )
            assert call(contexts[0], "skill_view", {"name": "vault-contract"})["success"]
            probe(contexts[0], True)
            # Hermes v0.22 (#90004) keeps granted names process-wide: another conversation
            # of the same profile sees the name too. Values still resolve per profile
            # (verify-vault-live-env.py checks that isolation), and the Diadems vault is
            # granted to the whole agent anyway.
            probe(contexts[1], True, owner="vault-contract-b")
            refresh("rotated-synthetic")
            probe(contexts[0], True, True)
            refresh(None)
            probe(contexts[0], False)
            print(json.dumps({"terminal": True, "execute_code": True, "live_add": True,
                              "rotation": True, "removal": True, "names_shared_across_conversations": True}))
        finally:
            shutdown_all_kernels()
            cleanup_all_environments()

