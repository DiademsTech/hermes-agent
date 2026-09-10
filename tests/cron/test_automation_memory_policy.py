from argparse import ArgumentParser
from unittest.mock import patch

import pytest

from cron.jobs import create_job, get_job, update_job
from hermes_cli.subcommands.cron import build_cron_parser
from gateway.session import SessionSource
from gateway.config import Platform


def test_cron_memory_default_and_individual_updates():
    first = create_job(prompt="Summarize sales", schedule="0 9 * * *")
    second = create_job(prompt="Track a commitment", schedule="0 10 * * *", memory_auto_retain=True)
    assert first["memory_auto_retain"] is False
    assert second["memory_auto_retain"] is True
    update_job(first["id"], {"memory_auto_retain": True})
    update_job(second["id"], {"memory_auto_retain": False})
    assert get_job(first["id"])["memory_auto_retain"] is True
    assert get_job(second["id"])["memory_auto_retain"] is False
    update_job(first["id"], {"name": "New name"})
    assert get_job(first["id"])["memory_auto_retain"] is True
    with pytest.raises(ValueError):
        update_job(first["id"], {"memory_auto_retain": "false"})


def test_cron_cli_boolean_flags_preserve_omitted_edit():
    parser = ArgumentParser()
    build_cron_parser(parser.add_subparsers(), cmd_cron=lambda _: None)
    assert parser.parse_args(["cron", "create", "1h", "hello"]).memory_auto_retain is False
    assert parser.parse_args(["cron", "create", "--memory-auto-retain", "1h", "hello"]).memory_auto_retain is True
    assert parser.parse_args(["cron", "edit", "job123", "--no-memory-auto-retain"]).memory_auto_retain is False
    assert parser.parse_args(["cron", "edit", "job123", "--name", "other"]).memory_auto_retain is None


def test_session_source_preserves_retention_without_changing_chat_default():
    source = SessionSource(platform=Platform.WEBHOOK, chat_id="orders", memory_auto_retain=False)
    assert SessionSource.from_dict(source.to_dict()).memory_auto_retain is False
    assert SessionSource.from_dict({"platform": "telegram", "chat_id": "chat"}).memory_auto_retain is True


@pytest.mark.parametrize("policy", [None, False, True])
def test_scheduler_applies_each_job_policy_without_disabling_recall(tmp_path, policy):
    from cron.scheduler import run_job

    job = {"id": "memory-policy", "name": "test", "prompt": "hello"}
    if policy is not None:
        job["memory_auto_retain"] = policy
    runtime = {"api_key": "test-key", "base_url": "https://example.invalid/v1",
               "provider": "openrouter", "api_mode": "chat_completions"}
    with (
        patch("cron.scheduler._hermes_home", tmp_path),
        patch("cron.scheduler._resolve_origin", return_value=None),
        patch("hermes_cli.env_loader.load_hermes_dotenv"),
        patch("hermes_cli.env_loader.reset_secret_source_cache"),
        patch("hermes_state.SessionDB"),
        patch("hermes_cli.runtime_provider.resolve_runtime_provider", return_value=runtime),
        patch("run_agent.AIAgent") as agent,
    ):
        agent.return_value.run_conversation.return_value = {"final_response": "ok"}
        success, _, _, error = run_job(job)
    assert success, error
    assert agent.call_args.kwargs["memory_auto_retain"] is (policy is True)
    assert agent.call_args.kwargs["skip_memory"] is False
