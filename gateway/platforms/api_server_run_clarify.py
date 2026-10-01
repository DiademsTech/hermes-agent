"""Native ``clarify`` questions on ``/v1/runs`` (capability ``run_clarify``).

Only runs admitted with ``"clarify": true`` receive the ``clarify`` toolset and
this bridge: a client that never declared it can answer keeps today's toolset,
so automation that drives Runs never blocks on a question nobody sees.

The agent thread publishes ``clarify.request`` on the run's event stream, keeps
the same payload in the pollable status (``status: waiting_for_clarification``,
``clarification``) for a client that missed the destructive SSE queue, and
blocks until ``POST /v1/runs/{run_id}/clarify`` answers, the native clarify
timeout elapses, or the run is stopped/interrupted. The tool then returns its
native result shape, so transcripts, compression and summaries are unchanged.
"""

from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from contextlib import suppress
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

from gateway.platforms.api_server_room_grants import _json_error

REQUEST_EVENT = "clarify.request"
RESPONDED_EVENT = "clarify.responded"
WAITING_STATUS = "waiting_for_clarification"
_MAX_TEXT = 4000  # question prompt / batch title on the wire
_MAX_CHOICE = 500
_MAX_ANSWER = 8000
_MAX_MULTI_ITEMS = 16


@dataclass(eq=False)
class PendingClarify:
    """The single clarify request a run can hold (clarify is never parallel)."""

    clarify_id: str
    run_id: str
    multi: List[bool]
    wire_choices: List[Optional[List[str]]]
    choices: List[Optional[List[str]]]
    event: threading.Event = field(default_factory=threading.Event)
    lock: threading.Lock = field(default_factory=threading.Lock)
    state: str = "pending"  # pending | answered | expired | cancelled
    answers: Optional[list] = None
    fingerprint: str = ""


def with_clarify_toolset(enabled_toolsets: List[str], user_config: dict) -> List[str]:
    """Add ``clarify`` for an opted-in run unless ``agent.disabled_toolsets`` removes it."""
    disabled = (user_config.get("agent") or {}).get("disabled_toolsets")
    if disabled:
        from agent.skill_utils import parse_config_string_list
        if "clarify" in {name.strip() for name in parse_config_string_list(disabled)}:
            return list(enabled_toolsets)
    return sorted({*enabled_toolsets, "clarify"})


def _no_answer_notice(outcome: str, timeout: float) -> str:
    """Native no-answer prose: the ``[user did not respond`` prefix is what compression
    and summaries already treat as a non-answer, never as user text."""
    if outcome == "expired":
        return f"[user did not respond within {max(1, round(timeout / 60))}m]"
    return "[user did not respond: the run was stopped before an answer arrived]"


def _publish_settled(adapter, pending: PendingClarify, outcome: str, put: Callable[[dict], None]) -> None:
    """Drop the pending payload from status, then tell stream consumers how it ended.
    The status never regresses from ``stopping``/terminal or a pending approval."""
    status = adapter._run_statuses.get(pending.run_id)
    if status is not None and (status.get("clarification") or {}).get("clarify_id") == pending.clarify_id:
        status.pop("clarification", None)
        current = str(status.get("status") or "running")
        adapter._set_run_status(
            pending.run_id, "running" if current == WAITING_STATUS else current, last_event=RESPONDED_EVENT)
    put({"event": RESPONDED_EVENT, "run_id": pending.run_id, "timestamp": time.time(),
         "clarify_id": pending.clarify_id, "outcome": outcome})


def cancel_run_clarify(adapter, run_id: str) -> None:
    """Withdraw the run's pending question (stop, task cancellation); idempotent."""
    pending = (getattr(adapter, "_run_clarifications", None) or {}).get(run_id)
    if pending is None:
        return
    with pending.lock:
        if pending.state == "pending":
            pending.state = "cancelled"
        pending.event.set()


def _wait(pending: PendingClarify, timeout: float) -> str:
    """Block the agent thread until answered, expired (native timeout, ``<= 0`` unlimited)
    or interrupted. Polls in 1 s slices so ``/stop`` and inactivity heartbeats keep working."""
    from tools.interrupt import is_interrupted
    try:
        from tools.environments.base import touch_activity_if_due
    except Exception:  # pragma: no cover - minimal tool-only environments
        touch_activity_if_due = None
    deadline = None if timeout <= 0 else time.monotonic() + timeout
    activity = {"last_touch": time.monotonic(), "start": time.monotonic()}
    reason = "cancelled"
    while not pending.event.is_set():
        if is_interrupted():
            break
        remaining = 1.0 if deadline is None else deadline - time.monotonic()
        if remaining <= 0:
            reason = "expired"
            break
        if pending.event.wait(timeout=min(1.0, remaining)):
            break
        if touch_activity_if_due is not None:
            touch_activity_if_due(activity, "waiting for user clarify response")
    with pending.lock:
        if pending.state == "pending":
            pending.state = reason
        pending.event.set()
        return pending.state


def make_run_clarify_callback(adapter, run, loop, *, redact: Callable[..., str]):
    """``clarify_callback`` for one run; batch-capable, so clarify_tool hands it the whole form."""
    from tools.clarify_tool import strip_recommended

    def put(event: dict) -> None:
        with suppress(Exception):
            loop.call_soon_threadsafe(run.put_event, event)

    def ask(entries: list, title: str) -> tuple[str, list, float]:
        from tools.clarify_gateway import get_clarify_timeout
        timeout = float(get_clarify_timeout())
        questions = [{
            "id": qid, "prompt": redact(str(prompt), force=True)[:_MAX_TEXT],
            "choices": [redact(str(c), force=True)[:_MAX_CHOICE] for c in choices] if choices else None,
            # clarify_tool marks the first of several choices as the recommended one.
            "recommended": 0 if choices and len(choices) >= 2 else None,
            "multi_select": bool(multi and choices), "allow_other": True,
        } for qid, prompt, choices, multi in entries]
        clarify_id = f"clr_{uuid.uuid4().hex}"
        payload = {
            "event": REQUEST_EVENT, "run_id": run.run_id, "timestamp": time.time(), "clarify_id": clarify_id,
            **({"title": redact(title, force=True)[:_MAX_TEXT]} if title else {}),
            "questions": questions, "timeout_seconds": int(timeout) if timeout > 0 else None,
            "expires_at": time.time() + timeout if timeout > 0 else None}
        pending = PendingClarify(
            clarify_id=clarify_id, run_id=run.run_id, multi=[q["multi_select"] for q in questions],
            wire_choices=[q["choices"] for q in questions], choices=[list(e[2]) if e[2] else None for e in entries])
        # Registered before publication, so the fastest possible answer finds it.
        adapter._run_clarifications[run.run_id] = pending
        current = str(adapter._run_statuses.get(run.run_id, {}).get("status") or "running")
        adapter._set_run_status(
            run.run_id, current if current in {"waiting_for_approval", "stopping"} else WAITING_STATUS,
            last_event=REQUEST_EVENT, clarification=payload)
        put(dict(payload))
        outcome = _wait(pending, timeout)
        if outcome != "answered":  # an answer was already published by its HTTP handler
            _publish_settled(adapter, pending, outcome, put)
        return outcome, list(pending.answers or []), timeout

    def callback(question, choices, multi_select=False, questions=None):
        if questions:
            entries = [(e["qid"], e["question"], e.get("choices_offered") or None, bool(e.get("multi_select")))
                       for e in questions]
            outcome, answers, timeout = ask(entries, str(question or "").strip())
            if outcome != "answered":
                return {"answers": {}, "timed_out": True, "notice": _no_answer_notice(outcome, timeout)}
            return {"answers": {e[0]: a for e, a in zip(entries, answers) if a}, "timed_out": False}
        bare = [strip_recommended(c) for c in choices] if choices else None
        outcome, answers, timeout = ask([("q0", question, bare, bool(multi_select) and bool(bare))], "")
        if outcome != "answered":
            return _no_answer_notice(outcome, timeout)
        return (answers[0] if answers else None) or ""

    return callback


def _validate_responses(pending: PendingClarify, responses: Any) -> tuple[Optional[list], Optional[str]]:
    """Answers in question order; ``None``/``""``/``[]`` skips. Exact wire labels map back
    to the original choice text (the wire copy may be redacted or truncated)."""
    if not isinstance(responses, list) or len(responses) != len(pending.multi):
        return None, f"'responses' must be an array of {len(pending.multi)} answers in question order."
    answers: list = []
    for index, raw in enumerate(responses):
        wire, original = pending.wire_choices[index] or [], pending.choices[index] or []

        def restore(text: str) -> str:
            return original[wire.index(text)] if text in wire else text
        if raw is None or raw == "" or raw == []:
            answers.append(None)
        elif pending.multi[index]:
            items = [raw] if isinstance(raw, str) else raw
            if (not isinstance(items, list) or len(items) > _MAX_MULTI_ITEMS
                    or any(not isinstance(item, str) or len(item) > _MAX_ANSWER for item in items)):
                return None, f"responses[{index}] must be a list of up to {_MAX_MULTI_ITEMS} strings."
            selected: list = []
            for item in (restore(item.strip()) for item in items if item.strip()):
                if item not in selected:
                    selected.append(item)
            answers.append(selected or None)
        elif not isinstance(raw, str) or len(raw) > _MAX_ANSWER:
            return None, f"responses[{index}] must be a string of at most {_MAX_ANSWER} characters."
        else:
            answers.append(restore(raw.strip()) or None)
    return answers, None


async def handle_run_clarify(adapter, request, *, _api_server):
    """POST /v1/runs/{run_id}/clarify — answer the run's pending clarify request."""
    from aiohttp import web
    from gateway.platforms import api_server_runs as _runs

    _openai_error = _api_server._openai_error
    run_id, _, _, _, err = _runs._load_owned_run(
        adapter, request, _api_server=_api_server, permission=None, active_fallback=False)
    if err is not None:
        return err
    body, err = await adapter._read_json_body(request)
    if err:
        return err
    clarify_id = body.get("clarify_id")
    if not isinstance(clarify_id, str) or not clarify_id.strip() or len(clarify_id) > 128:
        return _json_error(_openai_error, "A clarify_id is required.", code="invalid_clarify_response", status=400)
    pending = adapter._run_clarifications.get(run_id)
    if pending is None or pending.clarify_id != clarify_id.strip():
        return _json_error(_openai_error, f"Run has no such clarify request: {run_id}",
                           code="clarify_not_pending", status=409)
    answers, problem = _validate_responses(pending, body.get("responses"))
    if problem:
        return _json_error(_openai_error, problem, code="invalid_clarify_response", status=400)
    fingerprint = hashlib.sha256(json.dumps(answers, ensure_ascii=False).encode()).hexdigest()
    with pending.lock:
        state = pending.state
        if state == "pending":
            pending.answers, pending.fingerprint, pending.state = answers, fingerprint, "answered"
    receipt = {"object": "hermes.run.clarify_response", "run_id": run_id, "clarify_id": pending.clarify_id,
               "accepted": True}
    if state == "pending":
        def put(event: dict) -> None:
            queue = adapter._run_streams.get(run_id)
            if queue is not None:
                with suppress(Exception):
                    queue.put_nowait(event)
        # Settle the status before the agent thread resumes and emits its next events.
        _publish_settled(adapter, pending, "answered", put)
        pending.event.set()
        return web.json_response({**receipt, "replayed": False})
    if state == "answered" and pending.fingerprint == fingerprint:
        return web.json_response({**receipt, "replayed": True})
    if state == "answered":
        return _json_error(_openai_error, "This clarify request was already answered differently.",
                           code="clarify_already_answered", status=409)
    return _json_error(_openai_error, "This clarify request is no longer waiting for an answer.",
                       code="clarify_expired", status=409)
