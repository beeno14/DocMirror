# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Process-local FIFO admission queue for DocMirror parse requests."""

from __future__ import annotations

import asyncio
import logging
import os
from collections import deque
from dataclasses import dataclass

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ACTIVE_PARSES = 10
_DEFAULT_MAX_QUEUED_PARSES = 50
_DEFAULT_QUEUE_WAIT_TIMEOUT_SECONDS = 900.0


class ParseCapacityError(RuntimeError):
    """Base class for temporary parser admission failures."""


class ParseQueueFullError(ParseCapacityError):
    """Raised when the global parser waiting queue has reached its limit."""

    def __init__(self, *, active: int, queued: int, max_queued: int) -> None:
        super().__init__(f"parser queue is full ({active} active, {queued}/{max_queued} queued)")
        self.active = active
        self.queued = queued
        self.max_queued = max_queued


class ParseQueueWaitTimeoutError(ParseCapacityError):
    """Raised when one request waits too long for a parser-process slot."""

    def __init__(self, *, wait_timeout_seconds: float) -> None:
        super().__init__(f"parser queue wait exceeded {wait_timeout_seconds:g} seconds")
        self.wait_timeout_seconds = wait_timeout_seconds


class ParseManagerShuttingDownError(ParseCapacityError):
    """Raised when a request cannot run because the parse service is shutting down."""


@dataclass(frozen=True, slots=True)
class ParseAdmissionSnapshot:
    """Immutable point-in-time queue counters for diagnostics and tests."""

    active: int
    queued: int
    closing: bool


@dataclass(slots=True)
class _ParseWaiter:
    task_id: str
    future: asyncio.Future[None]
    enqueued_at: float
    admitted: bool = False


def _non_negative_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning("event=parse_config_invalid name=%s value=%r default=%s", name, raw, default)
        return default


def _positive_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(0.1, float(raw))
    except ValueError:
        logger.warning("event=parse_config_invalid name=%s value=%r default=%s", name, raw, default)
        return default


def max_active_parses() -> int:
    """Return the parse-process limit, where zero means unlimited."""
    return _non_negative_int_env("DOCMIRROR_MAX_ACTIVE_PARSES", _DEFAULT_MAX_ACTIVE_PARSES)


def max_queued_parses() -> int:
    """Return the maximum number of requests waiting for a parser slot."""
    return _non_negative_int_env("DOCMIRROR_MAX_QUEUED_PARSES", _DEFAULT_MAX_QUEUED_PARSES)


def queue_wait_timeout_seconds() -> float:
    """Return the maximum time one request may wait in the parser queue."""
    return _positive_float_env(
        "DOCMIRROR_QUEUE_WAIT_TIMEOUT_SECONDS",
        _DEFAULT_QUEUE_WAIT_TIMEOUT_SECONDS,
    )


class ParseAdmissionQueue:
    """Grant parse-process slots in strict FIFO order within one API worker."""

    def __init__(self) -> None:
        self._state_lock = asyncio.Lock()
        self._active_slots = 0
        self._waiters: deque[_ParseWaiter] = deque()
        self._closing = False

    async def startup(self) -> None:
        """Allow admissions for a new API lifespan."""
        async with self._state_lock:
            self._closing = False

    async def acquire(self, *, task_id: str) -> None:
        """Reserve one slot immediately or wait in FIFO order."""
        loop = asyncio.get_running_loop()
        wait_timeout = queue_wait_timeout_seconds()

        async with self._state_lock:
            limit = max_active_parses()
            if self._closing:
                raise ParseManagerShuttingDownError("DocMirror parse manager is shutting down")
            if limit <= 0 or (self._active_slots < limit and not self._waiters):
                self._active_slots += 1
                return

            queue_limit = max_queued_parses()
            queued = len(self._waiters)
            if queued >= queue_limit:
                logger.warning(
                    "event=parse_queue_full task_id=%s active=%s queued=%s queue_limit=%s",
                    task_id,
                    self._active_slots,
                    queued,
                    queue_limit,
                )
                raise ParseQueueFullError(
                    active=self._active_slots,
                    queued=queued,
                    max_queued=queue_limit,
                )

            waiter = _ParseWaiter(
                task_id=task_id,
                future=loop.create_future(),
                enqueued_at=loop.time(),
            )
            self._waiters.append(waiter)
            logger.info(
                "event=parse_queue_enter task_id=%s position=%s active=%s timeout_seconds=%s",
                task_id,
                len(self._waiters),
                self._active_slots,
                wait_timeout,
            )

        try:
            await asyncio.wait_for(asyncio.shield(waiter.future), timeout=wait_timeout)
        except asyncio.TimeoutError as exc:
            if await self._remove_waiter(waiter):
                logger.warning(
                    "event=parse_queue_timeout task_id=%s timeout_seconds=%s",
                    task_id,
                    wait_timeout,
                )
                raise ParseQueueWaitTimeoutError(wait_timeout_seconds=wait_timeout) from exc
            await waiter.future
        except asyncio.CancelledError:
            removed = await self._remove_waiter(waiter)
            if not removed and waiter.admitted:
                await self.release()
            raise

        logger.info(
            "event=parse_queue_admitted task_id=%s waited_seconds=%.3f",
            task_id,
            max(0.0, loop.time() - waiter.enqueued_at),
        )

    async def release(self) -> None:
        """Release one active slot and hand it directly to the FIFO head."""
        async with self._state_lock:
            self._active_slots = max(0, self._active_slots - 1)
            if self._closing:
                return

            limit = max_active_parses()
            if limit > 0 and self._active_slots >= limit:
                return

            while self._waiters:
                waiter = self._waiters.popleft()
                if waiter.future.done():
                    continue
                waiter.admitted = True
                self._active_slots += 1
                waiter.future.set_result(None)
                return

    async def shutdown(self) -> None:
        """Reject new admissions and fail every request still waiting."""
        async with self._state_lock:
            self._closing = True
            waiters = list(self._waiters)
            self._waiters.clear()
            for waiter in waiters:
                if not waiter.future.done():
                    waiter.future.set_exception(
                        ParseManagerShuttingDownError("DocMirror parse manager is shutting down")
                    )

        if waiters:
            logger.warning("event=parse_queue_shutdown queued=%s", len(waiters))

    async def snapshot(self) -> ParseAdmissionSnapshot:
        """Return active and queued counters under the queue lock."""
        async with self._state_lock:
            return ParseAdmissionSnapshot(
                active=self._active_slots,
                queued=len(self._waiters),
                closing=self._closing,
            )

    async def _remove_waiter(self, waiter: _ParseWaiter) -> bool:
        async with self._state_lock:
            try:
                self._waiters.remove(waiter)
            except ValueError:
                return False
            if not waiter.future.done():
                waiter.future.cancel()
            return True


__all__ = [
    "ParseAdmissionQueue",
    "ParseAdmissionSnapshot",
    "ParseCapacityError",
    "ParseManagerShuttingDownError",
    "ParseQueueFullError",
    "ParseQueueWaitTimeoutError",
    "max_active_parses",
    "max_queued_parses",
    "queue_wait_timeout_seconds",
]
