"""Session FIFO controls, serialized with admission on the API event loop.

No await is allowed between validating a waiting item and changing it. The
executor can finish concurrently, but only the event loop can claim the next
input. Text revisions and expected order prevent stale devices overwriting it.
"""
import hashlib
import time
from contextlib import suppress

from aiohttp import web


def error(message, status=409):
    return web.json_response({"error": {"message": message, "code": "queue_conflict"}}, status=status)


async def body_for(adapter, request):
    denied = adapter._check_auth(request)
    if denied:
        return None, denied
    if adapter._room_grant_token(request):
        return None, error("Session credential required", 403)
    return await adapter._read_json_body(request)


def waiting_item(adapter, request, body):
    rid = request.match_info["run_id"]
    if not adapter._request_owns_run(request, rid):
        return None, error("Run not found", 404)
    status = adapter._durable_run_status(request, rid) or {}
    text = adapter._queued_run_inputs.get(rid)
    if status.get("status") != "queued" or text is None or rid in adapter._stopping_run_ids:
        return None, error("Message already left the queue")
    if body.get("expected_revision") != hashlib.sha256(text.encode()).hexdigest():
        return None, error("Message changed on another device")
    return rid, None


async def edit_queue(adapter, request):
    body, denied = await body_for(adapter, request)
    if denied is not None:
        return denied
    text = body.get("input")
    if not isinstance(text, str) or not text.strip() or len(text) > 250_000:
        return error("Non-empty input up to 250000 characters required", 400)
    rid, denied = waiting_item(adapter, request, body)
    if denied is not None:
        return denied
    adapter._run_idempotency_store.update_queue_input(rid, text)
    adapter._queued_run_inputs[rid] = text
    return web.json_response({"run_id": rid, "queue_revision": hashlib.sha256(text.encode()).hexdigest()})


async def reorder_queue(adapter, request):
    body, denied = await body_for(adapter, request)
    if denied is not None:
        return denied
    sid, order, expected = body.get("session_id"), body.get("order"), body.get("expected_order")
    if not isinstance(sid, str) or not isinstance(order, list) or not isinstance(expected, list) or not all(isinstance(rid, str) for rid in order + expected):
        return error("Session and ordered run ids required", 400)
    current = [rid for rid in adapter._queued_run_inputs
               if adapter._request_owns_run(request, rid)
               and adapter._run_statuses.get(rid, {}).get("session_id") == sid
               and adapter._run_statuses[rid].get("status") == "queued"
               and rid not in adapter._stopping_run_ids]
    if current != expected or len(order) != len(current) or set(order) != set(current):
        return error("Queue changed on another device")
    adapter._run_idempotency_store.reorder_queue_inputs(order)
    # Preserve the relative order of every other profile/session.
    iterator = iter(order)
    keys = [next(iterator) if rid in current else rid for rid in adapter._queued_run_inputs]
    adapter._queued_run_inputs = {rid: adapter._queued_run_inputs[rid] for rid in keys}
    return web.json_response({"order": order})


async def steer_queue(adapter, request):
    body, denied = await body_for(adapter, request)
    if denied is not None:
        return denied
    rid, denied = waiting_item(adapter, request, body)
    if denied is not None:
        return denied
    lane = adapter._run_lanes[rid]
    target = next((key for key, agent in adapter._active_run_agents.items()
                   if adapter._run_lanes.get(key) == lane and adapter._request_owns_run(request, key)
                   and adapter._run_statuses.get(key, {}).get("status") == "running"
                   and callable(getattr(agent, "steer", None))), None)
    if target is None:
        return error("No running turn accepts steering; message remains queued")
    # Persist the claim before handing off: after a process crash the input
    # stays interrupted for manual recovery, never automatically delivered twice.
    adapter._set_run_status(rid, "stopping", last_event="queue.steering")
    try:
        accepted = bool(adapter._active_run_agents[target].steer(adapter._queued_run_inputs[rid]))
    except Exception:
        # Outcome is ambiguous. Retain text, but prevent FIFO execution/retry.
        adapter._queued_run_inputs.pop(rid, None)
        adapter._stopping_run_ids.add(rid)
        adapter._set_run_status(rid, "interrupted", last_event="queue.steer_uncertain")
        adapter._active_run_tasks[rid].cancel()
        return error("Steering outcome uncertain; do not automatically retry", 503)
    if not accepted:
        adapter._set_run_status(rid, "queued", last_event="queue.steer_rejected")
        return error("Run declined steering; message remains queued")
    adapter._run_idempotency_store.discard_queue_input(rid)
    adapter._queued_run_inputs.pop(rid, None)
    adapter._stopping_run_ids.add(rid)
    adapter._set_run_status(rid, "cancelled", last_event="queue.steered", steered_to=target)
    adapter._active_run_tasks[rid].cancel()
    adapter._set_run_status(target, "running", last_event="run.steered")
    stream = adapter._run_streams.get(target)
    if stream is not None:
        with suppress(Exception):
            stream.put_nowait({"event": "run.steered", "run_id": target, "timestamp": time.time(), "accepted": True})
    return web.json_response({"run_id": rid, "steered_to": target, "accepted": True})
