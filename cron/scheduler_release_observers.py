"""Observers of in-flight cron releases.

A job leaves the scheduler's running set on a scheduler thread, outside every gateway turn
boundary, so the gateway registers an observer that republishes its persisted ``active_agents``;
without it ``gateway_state.json`` kept a finished job counted until the next inbound message
(#122813).
"""

from __future__ import annotations

import logging
import threading
from typing import Callable

logger = logging.getLogger(__name__)

_callbacks: tuple[Callable[[], None], ...] = ()
_callbacks_lock = threading.Lock()


def register_job_release_callback(callback: Callable[[], None]) -> None:
    """Call ``callback()`` after each release (idempotent); its exceptions never reach the job."""
    global _callbacks
    with _callbacks_lock:
        if callback not in _callbacks:
            _callbacks = (*_callbacks, callback)


def notify_job_released() -> None:
    """Run the observers on the releasing thread. Callers must not hold the scheduler's
    ``_running_lock``: an observer may read the running set."""
    for callback in _callbacks:
        try:
            callback()
        except Exception:
            logger.debug("Cron job release callback failed", exc_info=True)
