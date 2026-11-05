"""delegation.model_choices: the parent picks a subagent's model task by task."""

from types import SimpleNamespace
from unittest.mock import patch

from tools import delegate_tool
from tools.delegate_model_choices import apply_reasoning, model_choices, schema_property, task_routes

CHOICES = {
    "model_choices": [
        {"id": "deep", "model": "anthropic/claude-opus-5-5", "reasoning_effort": "high",
         "label": "Claude Opus 5.5", "when": "Long analysis, code, documents to write."},
        {"id": "quick", "model": "mistral/ministral-8b", "label": "Ministral 8B", "when": "Short lookups."},
        {"id": "Bad Id", "model": "x"},
        {"id": "deep", "model": "duplicate"},
        {"id": "nomodel"},
    ]
}


def _parent():
    return SimpleNamespace(model="parent/model", provider="custom", base_url="http://gateway/v1",
                           api_key="k", request_overrides=None, reasoning_config={"effort": "low"})


def test_choices_keep_only_valid_unique_entries():
    choices = model_choices(CHOICES)
    assert list(choices) == ["deep", "quick"]
    assert choices["deep"]["model"] == "anthropic/claude-opus-5-5"
    assert model_choices({}) == {} and model_choices({"model_choices": "nope"}) == {}


def test_schema_offers_the_choices_only_when_configured():
    assert schema_property({}) is None
    prop = schema_property(CHOICES)
    assert prop["enum"] == ["deep", "quick"]
    assert "deep: Claude Opus 5.5 — Long analysis" in prop["description"]
    with patch.object(delegate_tool, "_load_config", return_value=CHOICES):
        params = delegate_tool._build_dynamic_schema_overrides()["parameters"]
    assert params["properties"]["tasks"]["items"]["properties"]["model"]["enum"] == ["deep", "quick"]
    with patch.object(delegate_tool, "_load_config", return_value={}):
        params = delegate_tool._build_dynamic_schema_overrides()["parameters"]
    assert "model" not in params["properties"]["tasks"]["items"]["properties"]
    # The static schema is never mutated.
    assert "model" not in delegate_tool.DELEGATE_TASK_SCHEMA["parameters"]["properties"]["tasks"]["items"]["properties"]


def test_routes_resolve_each_named_choice_and_leave_the_rest_alone():
    tasks = [{"goal": "a"}, {"goal": "b", "model": "deep"}]
    routes, err = task_routes(tasks, {**CHOICES, "provider": "openrouter", "model": "pinned/model"}, _parent())
    assert err is None and routes[0] is None
    creds, route_cfg, effort = routes[1]
    assert creds["model"] == "anthropic/claude-opus-5-5"
    # No provider on the choice: the parent's route, not the section's pinned one.
    assert creds["provider"] is None and route_cfg["fallback_providers"] == []
    assert effort == "high"


def test_an_unknown_choice_is_refused_with_the_list():
    routes, err = task_routes([{"goal": "a", "model": "huge"}], CHOICES, _parent())
    assert routes is None and "deep, quick" in err


def test_children_are_built_on_their_choice(monkeypatch):
    built = []

    def fake_build(**kwargs):
        built.append(kwargs)
        return SimpleNamespace(tool_progress_callback=None, reasoning_config={"effort": "low"})

    monkeypatch.setattr(delegate_tool, "_build_child_preserving_parent_tools", fake_build)
    tasks = [{"goal": "a"}, {"goal": "b", "model": "deep"}]
    routes, _ = task_routes(tasks, CHOICES, _parent())
    base = {"model": None, "provider": None, "base_url": None, "api_key": None, "api_mode": None}
    children, err = delegate_tool._build_children(
        tasks, [None, None], base, top_role="leaf", max_iterations=5, parent_agent=_parent(),
        routing_cfg={}, live_deleg_id=None, live_writers=[], task_routes=routes,
    )
    assert err is None
    assert built[0]["model"] is None and built[1]["model"] == "anthropic/claude-opus-5-5"
    assert built[1]["routing_cfg"]["fallback_providers"] == []
    assert children[1][2].reasoning_config != {"effort": "low"}


def test_reasoning_is_only_set_by_a_known_level():
    child = SimpleNamespace(reasoning_config={"effort": "low"})
    apply_reasoning(child, None)
    assert child.reasoning_config == {"effort": "low"}
    apply_reasoning(child, "not-a-level")
    assert child.reasoning_config == {"effort": "low"}
    apply_reasoning(child, "high")
    assert child.reasoning_config != {"effort": "low"}
