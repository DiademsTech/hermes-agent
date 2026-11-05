"""Per-task model choices for delegate_task.

``delegation.model`` pins one model for every child. ``delegation.model_choices``
lets the parent pick, task by task, among models the operator listed, each with
a short guidance of when to use it::

    delegation:
      model_choices:
        - id: deep
          model: anthropic/claude-opus-5-5
          provider: ""            # empty: the parent's provider and credentials
          reasoning_effort: high  # optional, else delegation.reasoning_effort / parent
          label: Claude Opus 5.5
          when: Long analysis, code, documents to write.

Each task may then carry ``model: <id>``; without it the child is built exactly
as before. A choice resolves through the same credential path as the
``delegation`` section itself (``_resolve_delegation_credentials``): with a
provider, the runtime provider system; without, the parent's route with
another model. A pinned choice never borrows the parent's fallback chain.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

logger = logging.getLogger(__name__)

_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")
MAX_CHOICES = 12
_ROUTE_KEYS = ("model", "provider", "base_url", "api_key", "api_mode")


def model_choices(cfg: Any) -> dict[str, dict[str, Any]]:
    """The valid ``delegation.model_choices`` by id, in their configured order."""
    raw = cfg.get("model_choices") if isinstance(cfg, dict) else None
    choices: dict[str, dict[str, Any]] = {}
    for entry in raw if isinstance(raw, list) else []:
        if not isinstance(entry, dict):
            continue
        choice_id = str(entry.get("id") or "").strip().lower()
        model = str(entry.get("model") or "").strip()
        if not _ID.match(choice_id) or not model or choice_id in choices:
            logger.warning("Ignoring malformed delegation.model_choices entry %r", entry.get("id"))
            continue
        choices[choice_id] = {
            "id": choice_id,
            "model": model,
            "provider": str(entry.get("provider") or "").strip(),
            "base_url": str(entry.get("base_url") or "").strip(),
            "api_mode": str(entry.get("api_mode") or "").strip(),
            "reasoning_effort": entry.get("reasoning_effort"),
            "label": " ".join(str(entry.get("label") or model).split())[:80],
            "when": " ".join(str(entry.get("when") or "").split())[:300],
        }
        if len(choices) >= MAX_CHOICES:
            break
    return choices


def schema_property(cfg: Any) -> Optional[dict[str, Any]]:
    """The ``tasks[].model`` property, or None when no choice is configured."""
    choices = model_choices(cfg)
    if not choices:
        return None
    lines = [f"- {c['id']}: {c['label']}" + (f" — {c['when']}" if c["when"] else "") for c in choices.values()]
    return {
        "type": "string",
        "enum": list(choices),
        "description": (
            "Optional model for this subagent, picked by what the task needs. Omit to keep the default "
            "subagent model.\n" + "\n".join(lines)
        ),
    }


def task_routes(task_list: list[dict[str, Any]], cfg: dict[str, Any], parent_agent):
    """Per task ``(creds, routing_cfg, reasoning_effort)`` or None, else ``(None, error)``."""
    from tools.delegate_tool_config import _resolve_delegation_credentials

    choices = model_choices(cfg)
    routes: list[Optional[tuple[dict[str, Any], dict[str, Any], Any]]] = []
    for i, task in enumerate(task_list):
        wanted = str(task.get("model") or "").strip().lower()
        if not wanted:
            routes.append(None)
            continue
        choice = choices.get(wanted)
        if choice is None:
            known = ", ".join(choices) or "none configured"
            return None, f"Task {i}: unknown model '{wanted}'. Choose one of: {known}, or omit 'model'."
        # The choice owns its whole route: unset keys must not fall back to the
        # section's pinned provider (a model of the parent's own connection).
        route_cfg = {k: v for k, v in cfg.items() if k not in _ROUTE_KEYS and k != "model_choices"}
        route_cfg.update({k: choice[k] for k in ("model", "provider", "base_url", "api_mode")})
        route_cfg["fallback_providers"] = []
        try:
            creds = _resolve_delegation_credentials(route_cfg, parent_agent)
        except ValueError as exc:
            return None, f"Task {i}: model '{wanted}' is unavailable: {exc}"
        routes.append((creds, route_cfg, choice.get("reasoning_effort")))
    return routes, None


def apply_reasoning(child, effort: Any) -> None:
    """The choice's own reasoning level, over delegation.reasoning_effort / the parent's."""
    if not effort and effort is not False:
        return
    from hermes_constants import parse_reasoning_effort

    parsed = parse_reasoning_effort(effort)
    if parsed is None:
        logger.warning("Unknown model_choices reasoning_effort '%s', keeping the child's level", effort)
        return
    child.reasoning_config = parsed
