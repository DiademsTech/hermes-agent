"""Default delegation strategy (``delegation.backend: default``).

Every ``delegate_task`` task takes the route the ``delegation`` config section describes: a
``delegation.base_url`` endpoint, a ``delegation.provider`` pin, ``delegation.model`` and
``delegation.reasoning_effort``, or, when those are empty, the parent's own route. It adds no
schema parameters. Bundled ``kind: backend``, so it is always loaded; other strategies are plugins
of the same kind (``agent/delegation_provider.py``).
"""

from __future__ import annotations

from typing import Any, Mapping

from agent.delegation_provider import DelegationProvider, DelegationRouteError


class DefaultDelegationProvider(DelegationProvider):
    """Route every task through the ``delegation`` section."""

    @property
    def name(self) -> str:
        return "default"

    @property
    def display_name(self) -> str:
        return "Default"

    @property
    def summary(self) -> str:
        return "Every subagent uses delegation.model / delegation.provider, else the parent's model."

    def task_parameters(self, config: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        from tools.delegate_model_choices import schema_property
        prop = schema_property(config)
        return {"model_choice": prop} if prop else {}

    def route_task(self, task: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
        from tools.delegate_model_choices import model_choices
        selected = task.get("model_choice")
        if selected is not None:
            choice = model_choices(config).get(selected)
            if choice is None:
                raise DelegationRouteError("Unknown model_choice; choose a configured model or omit it.")
            # Own the entire route. An omitted provider uses the parent's connection,
            # never the section's pin or a paid fallback.
            route = {key: choice[key] for key in ("model", "provider", "base_url", "api_mode")}
            route["fallback_providers"] = []
            effort = choice.get("reasoning_effort")
            if effort is None:
                effort = config.get("reasoning_effort")
            if effort is not None:
                route["reasoning_effort"] = effort
            route["label"] = choice["label"]
            return route
        # The delegation section has no speed tier: an unknown key there never becomes one.
        return {k: v for k, v in config.items() if k != "service_tier"}


def register(ctx) -> None:
    ctx.register_delegation_provider(DefaultDelegationProvider())
