# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio

import pytest

from docmirror.server.parse_admission_queue import (
    ParseAdmissionQueue,
    ParseManagerShuttingDownError,
    ParseQueueFullError,
    ParseQueueWaitTimeoutError,
    max_active_parses,
    max_queued_parses,
    queue_wait_timeout_seconds,
)


async def _wait_for_queued(queue: ParseAdmissionQueue, expected: int) -> None:
    deadline = asyncio.get_running_loop().time() + 1.0
    snapshot = await queue.snapshot()
    while snapshot.queued != expected and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.005)
        snapshot = await queue.snapshot()
    assert snapshot.queued == expected


def test_parse_queue_defaults(monkeypatch):
    monkeypatch.delenv("DOCMIRROR_MAX_ACTIVE_PARSES", raising=False)
    monkeypatch.delenv("DOCMIRROR_MAX_QUEUED_PARSES", raising=False)
    monkeypatch.delenv("DOCMIRROR_QUEUE_WAIT_TIMEOUT_SECONDS", raising=False)

    assert max_active_parses() == 10
    assert max_queued_parses() == 50
    assert queue_wait_timeout_seconds() == 900.0


def test_queue_admits_waiters_in_fifo_order(monkeypatch):
    monkeypatch.setenv("DOCMIRROR_MAX_ACTIVE_PARSES", "1")
    monkeypatch.setenv("DOCMIRROR_MAX_QUEUED_PARSES", "2")
    monkeypatch.setenv("DOCMIRROR_QUEUE_WAIT_TIMEOUT_SECONDS", "5")
    queue = ParseAdmissionQueue()
    admitted: list[str] = []

    async def wait(task_id: str) -> None:
        await queue.acquire(task_id=task_id)
        admitted.append(task_id)

    async def scenario():
        await queue.acquire(task_id="task_first")
        second = asyncio.create_task(wait("task_second"))
        await _wait_for_queued(queue, 1)
        third = asyncio.create_task(wait("task_third"))
        await _wait_for_queued(queue, 2)

        await queue.release()
        await second
        assert admitted == ["task_second"]
        await queue.release()
        await third
        assert admitted == ["task_second", "task_third"]
        await queue.release()

        snapshot = await queue.snapshot()
        assert snapshot.active == 0
        assert snapshot.queued == 0

    asyncio.run(scenario())


def test_queue_rejects_only_after_waiting_limit_is_full(monkeypatch):
    monkeypatch.setenv("DOCMIRROR_MAX_ACTIVE_PARSES", "1")
    monkeypatch.setenv("DOCMIRROR_MAX_QUEUED_PARSES", "1")
    monkeypatch.setenv("DOCMIRROR_QUEUE_WAIT_TIMEOUT_SECONDS", "5")
    queue = ParseAdmissionQueue()

    async def scenario():
        await queue.acquire(task_id="task_first")
        second = asyncio.create_task(queue.acquire(task_id="task_second"))
        await _wait_for_queued(queue, 1)

        with pytest.raises(ParseQueueFullError) as caught:
            await queue.acquire(task_id="task_third")
        assert caught.value.active == 1
        assert caught.value.queued == 1
        assert caught.value.max_queued == 1

        await queue.release()
        await second
        await queue.release()

    asyncio.run(scenario())


def test_queue_removes_waiter_after_timeout(monkeypatch):
    monkeypatch.setenv("DOCMIRROR_MAX_ACTIVE_PARSES", "1")
    monkeypatch.setenv("DOCMIRROR_MAX_QUEUED_PARSES", "1")
    monkeypatch.setenv("DOCMIRROR_QUEUE_WAIT_TIMEOUT_SECONDS", "0.1")
    queue = ParseAdmissionQueue()

    async def scenario():
        await queue.acquire(task_id="task_first")
        with pytest.raises(ParseQueueWaitTimeoutError) as caught:
            await queue.acquire(task_id="task_second")
        assert caught.value.wait_timeout_seconds == 0.1
        assert (await queue.snapshot()).queued == 0
        await queue.release()

    asyncio.run(scenario())


def test_queue_removes_cancelled_waiter(monkeypatch):
    monkeypatch.setenv("DOCMIRROR_MAX_ACTIVE_PARSES", "1")
    monkeypatch.setenv("DOCMIRROR_MAX_QUEUED_PARSES", "1")
    monkeypatch.setenv("DOCMIRROR_QUEUE_WAIT_TIMEOUT_SECONDS", "5")
    queue = ParseAdmissionQueue()

    async def scenario():
        await queue.acquire(task_id="task_first")
        cancelled = asyncio.create_task(queue.acquire(task_id="task_cancelled"))
        await _wait_for_queued(queue, 1)

        cancelled.cancel()
        with pytest.raises(asyncio.CancelledError):
            await cancelled
        assert (await queue.snapshot()).queued == 0
        await queue.release()

    asyncio.run(scenario())


def test_queue_fails_waiters_during_shutdown(monkeypatch):
    monkeypatch.setenv("DOCMIRROR_MAX_ACTIVE_PARSES", "1")
    monkeypatch.setenv("DOCMIRROR_MAX_QUEUED_PARSES", "1")
    monkeypatch.setenv("DOCMIRROR_QUEUE_WAIT_TIMEOUT_SECONDS", "5")
    queue = ParseAdmissionQueue()

    async def scenario():
        await queue.acquire(task_id="task_first")
        second = asyncio.create_task(queue.acquire(task_id="task_second"))
        await _wait_for_queued(queue, 1)

        await queue.shutdown()
        with pytest.raises(ParseManagerShuttingDownError):
            await second
        with pytest.raises(ParseManagerShuttingDownError):
            await queue.acquire(task_id="task_third")

        await queue.release()
        snapshot = await queue.snapshot()
        assert snapshot.active == 0
        assert snapshot.queued == 0
        assert snapshot.closing is True

    asyncio.run(scenario())
