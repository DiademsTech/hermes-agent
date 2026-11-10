#!/usr/bin/env python3
"""Required candidate contracts; absent files fail closed instead of being skipped."""
from pathlib import Path
import os
import subprocess
import sys

CONTRACTS = (
    "tests/tools/test_live_skill_grants.py",
    "tests/tools/test_skill_usage.py",
    'tests/gateway/test_session_api.py',
    'tests/gateway/test_api_server_compaction_projection.py',
    'tests/gateway/test_api_server.py',
    'tests/gateway/test_api_server_runs.py',
    'tests/gateway/test_api_canonical_queue.py',
    'tests/gateway/test_api_canonical_projection.py',
    'tests/gateway/test_api_skills_inventory_contract.py',
    'tests/gateway/test_session_fork_before_message.py',
    'tests/gateway/test_api_server_memory_sessions.py',
    'tests/gateway/test_api_server_multiplex_secret_scope.py',
    'tests/cron/test_automation_memory_policy.py',
    'tests/gateway/test_webhook_adapter.py',
    'tests/gateway/test_webhook_execution_model.py',
    'tests/gateway/test_webhook_profile_egress.py',
    'tests/hermes_cli/test_webhook_cli.py',
    'tests/agent/test_memory_recall_events.py',
    'tests/agent/test_memory_provider.py',
    'tests/agent/test_turn_context.py',
    'tests/agent/test_secret_scope.py',
    'tests/agent/test_cross_process_turn_lease.py',
    'tests/test_session_maintenance_active_turns.py',
    'tests/tools/test_cronjob_tools.py',
    'tests/gateway/test_automation_failure_delivery.py',
    'tests/agent/test_tool_call_guardrail_runtime.py',
    'tests/agent/test_turn_finalizer_iteration_limit_exit.py',
    'tests/cron/test_failure_delivery_policy.py',
    'tests/cron/test_cron_failure_deliver.py',
    'tests/gateway/test_warning_notifications.py',
    'tests/tools/test_env_passthrough.py',
    'tests/tools/test_skill_env_passthrough.py',
    'tests/tools/test_local_env_blocklist.py',
    'tests/tools/test_local_env_session_leak.py',
    'tests/tools/test_code_kernel.py',
    'tests/gateway/test_lifecycle_state_db_check.py',
    'tests/gateway/test_lifecycle_ledger.py',
    'tests/gateway/test_active_work_status.py',
    'tests/gateway/test_api_server_active_work_drain.py',
    'tests/gateway/test_cron_active_work_drain.py',
    'tests/cron/test_cron_multiplex_inflight_release.py',
    'tests/cron/test_inflight_stale_guard.py',
    'tests/gateway/test_api_authority_runs.py',
    'tests/gateway/test_api_delegation_delivery_contract.py',
    'tests/tools/test_delegate_gateway_inline.py',
    'tests/tools/test_delegation_backends.py',
    'tests/agent/test_stall_guards.py',
    'tests/agent/test_intent_ack_continuation.py',
    'tests/agent/test_turn_end_stall_diagnostics.py',
    'tests/plugins/test_diadems_pc_passwords_plugin.py',
    'tests/hermes_cli/test_login_backend_provenance.py',
    'tests/agent/test_vault_backend_registry.py',
)


def main():
    root = Path(__file__).resolve().parents[1]
    missing = [name for name in CONTRACTS if not (root / name).is_file()]
    if missing:
        raise SystemExit("Missing candidate contracts: " + ", ".join(missing))
    return subprocess.call([os.environ.get("HERMES_TEST_BASH", "bash"), "scripts/run_tests.sh", *CONTRACTS,
                            "--file-retries", "0", "-j", "3", "-q"], cwd=root)


if __name__ == "__main__":
    raise SystemExit(main())
