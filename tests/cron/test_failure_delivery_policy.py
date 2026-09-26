"""Automation failure policy follows the owning profile without rewriting jobs."""
from copy import deepcopy

from cron.scheduler_delivery import _resolve_delivery_targets
from hermes_constants import set_hermes_home_override, reset_hermes_home_override


def test_failure_lane_is_profile_scoped_and_success_is_preserved(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    homes = [tmp_path / "a", tmp_path / "b"]
    for home, enabled in zip(homes, [True, False]):
        home.mkdir()
        (home / "config.yaml").write_text(
            f"display: {{suppress_automation_failure_messages: {str(enabled).lower()}}}\n",
            encoding="utf-8",
        )
    job = {"deliver": "telegram:123", "failure_deliver": "telegram:456"}
    before = deepcopy(job)
    for home, muted in [(homes[0], True), (homes[1], False), (homes[0], True)]:
        token = set_hermes_home_override(home)
        try:
            assert _resolve_delivery_targets(job)[0]["chat_id"] == "123"
            failures = _resolve_delivery_targets(job, for_failure=True)
            if muted:
                assert failures == []
            else:
                assert failures[0]["chat_id"] == "456"
            assert _resolve_delivery_targets({**job, "failure_deliver": "local"}, for_failure=True) == []
            assert job == before
        finally:
            reset_hermes_home_override(token)
