# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

from docmirror.sdk.integration.request import InputRef, ParseRequest
from docmirror.server.parse_admission_queue import ParseManagerShuttingDownError
from docmirror.server.parse_process_manager import (
    ParseProcessManager,
    bounded_request_payload,
)
from docmirror.server.parse_worker import parse_request_from_dict


def _request(tmp_path: Path, *, workers: int | str | None = None) -> ParseRequest:
    source = tmp_path / "input.txt"
    source.write_text("fixture", encoding="utf-8")
    return ParseRequest(
        inputs=[InputRef(file_path=str(source), file_id="001", file_name=source.name)],
        workers=workers,
    )


async def _sleeping_process(seconds: float) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-c",
        f"import time; time.sleep({seconds!r})",
    )


async def _wait_for_queue_size(manager: ParseProcessManager, expected: int) -> None:
    deadline = asyncio.get_running_loop().time() + 1.0
    snapshot = await manager._admission.snapshot()
    while snapshot.queued != expected and asyncio.get_running_loop().time() < deadline:
        await asyncio.sleep(0.005)
        snapshot = await manager._admission.snapshot()
    assert snapshot.queued == expected


def test_request_round_trip_and_worker_cap(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DOCMIRROR_WORKERS_PER_PARSE", "2")
    payload = bounded_request_payload(_request(tmp_path, workers=8))

    assert payload["workers"] == 2
    restored = parse_request_from_dict(payload)
    assert restored.workers == 2
    assert restored.inputs[0].file_name == "input.txt"


def test_manager_runs_queued_requests_in_fifo_order(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DOCMIRROR_MAX_ACTIVE_PARSES", "1")
    monkeypatch.setenv("DOCMIRROR_MAX_QUEUED_PARSES", "2")
    monkeypatch.setenv("DOCMIRROR_QUEUE_WAIT_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("DOCMIRROR_PARSE_TIMEOUT_SECONDS", "5")
    manager = ParseProcessManager()
    started: list[str] = []

    async def spawn_process(**kwargs):
        started.append(kwargs["task_id"])
        return await _sleeping_process(0.08)

    monkeypatch.setattr(manager, "_spawn_process", spawn_process)

    async def scenario():
        first = await manager.start(_request(tmp_path), output_root=tmp_path / "tasks", task_id="task_first")
        second_start = asyncio.create_task(
            manager.start(_request(tmp_path), output_root=tmp_path / "tasks", task_id="task_second")
        )
        await _wait_for_queue_size(manager, 1)
        third_start = asyncio.create_task(
            manager.start(_request(tmp_path), output_root=tmp_path / "tasks", task_id="task_third")
        )
        await _wait_for_queue_size(manager, 2)

        await first
        second = await second_start
        await second
        third = await third_start
        await third

        assert started == ["task_first", "task_second", "task_third"]

    asyncio.run(scenario())


def test_manager_rejects_queued_requests_during_shutdown(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DOCMIRROR_MAX_ACTIVE_PARSES", "1")
    monkeypatch.setenv("DOCMIRROR_MAX_QUEUED_PARSES", "1")
    monkeypatch.setenv("DOCMIRROR_QUEUE_WAIT_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("DOCMIRROR_PARSE_TIMEOUT_SECONDS", "5")
    monkeypatch.setenv("DOCMIRROR_PARSE_KILL_GRACE_SECONDS", "0.1")
    manager = ParseProcessManager()

    async def spawn_process(**_kwargs):
        return await _sleeping_process(5)

    monkeypatch.setattr(manager, "_spawn_process", spawn_process)

    async def scenario():
        await manager.start(_request(tmp_path), output_root=tmp_path / "tasks", task_id="task_first")
        second_start = asyncio.create_task(
            manager.start(_request(tmp_path), output_root=tmp_path / "tasks", task_id="task_second")
        )
        await _wait_for_queue_size(manager, 1)

        await manager.shutdown()
        with pytest.raises(ParseManagerShuttingDownError):
            await second_start
        snapshot = await manager._admission.snapshot()
        assert snapshot.queued == 0
        assert snapshot.active == 0

    asyncio.run(scenario())


def test_manager_accepts_concurrent_processes_when_capacity_is_unlimited(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DOCMIRROR_MAX_ACTIVE_PARSES", "0")
    monkeypatch.setenv("DOCMIRROR_PARSE_TIMEOUT_SECONDS", "5")
    manager = ParseProcessManager()

    async def spawn_process(**_kwargs):
        return await _sleeping_process(0.1)

    monkeypatch.setattr(manager, "_spawn_process", spawn_process)

    async def scenario():
        tasks = [
            await manager.start(
                _request(tmp_path),
                output_root=tmp_path / "tasks",
                task_id=f"task_{index}",
            )
            for index in range(3)
        ]
        assert all(not task.done() for task in tasks)
        await asyncio.gather(*tasks)

    asyncio.run(scenario())


def test_manager_hard_timeout_kills_worker_without_writing_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DOCMIRROR_MAX_ACTIVE_PARSES", "1")
    monkeypatch.setenv("DOCMIRROR_PARSE_TIMEOUT_SECONDS", "0.1")
    monkeypatch.setenv("DOCMIRROR_PARSE_KILL_GRACE_SECONDS", "0.1")
    manager = ParseProcessManager()

    async def spawn_process(**_kwargs):
        return await _sleeping_process(10)

    monkeypatch.setattr(manager, "_spawn_process", spawn_process)

    async def scenario():
        task = await manager.start(_request(tmp_path), output_root=tmp_path / "tasks", task_id="task_timeout")
        result = await task
        assert result.status == "failed"
        assert result.errors[0]["code"] == "PARSE_TIMEOUT"
        assert not (tmp_path / "tasks" / "task_timeout" / "manifest.json").exists()

    asyncio.run(scenario())


def test_running_worker_does_not_block_api_event_loop(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DOCMIRROR_MAX_ACTIVE_PARSES", "1")
    monkeypatch.setenv("DOCMIRROR_PARSE_TIMEOUT_SECONDS", "5")
    manager = ParseProcessManager()

    async def spawn_process(**_kwargs):
        return await _sleeping_process(0.2)

    monkeypatch.setattr(manager, "_spawn_process", spawn_process)

    async def scenario():
        task = await manager.start(_request(tmp_path), output_root=tmp_path / "tasks", task_id="task_running")
        ticks = 0
        for _ in range(3):
            await asyncio.sleep(0.01)
            ticks += 1
        assert ticks == 3
        assert not task.done()
        await task

    asyncio.run(scenario())
