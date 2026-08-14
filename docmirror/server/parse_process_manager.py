# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""API-side lifecycle manager for isolated DocMirror parser processes."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import sys
from collections import OrderedDict
from pathlib import Path
from typing import Any

from docmirror.sdk.integration.request import ParseRequest
from docmirror.server.task_executor import task_directory
from docmirror.server.task_result import TaskResult, task_result_from_manifest

logger = logging.getLogger(__name__)

_DEFAULT_MAX_ACTIVE_PARSES = 0
_DEFAULT_WORKERS_PER_PARSE = 2
_DEFAULT_PARSE_TIMEOUT_SECONDS = 1800.0
_DEFAULT_KILL_GRACE_SECONDS = 10.0
_MAX_RUNTIME_RESULTS = 1000


class ParseCapacityError(RuntimeError):
    """Raised when all configured parse-process slots are occupied."""


def _positive_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning("event=parse_config_invalid name=%s value=%r default=%s", name, raw, default)
        return default


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


def workers_per_parse() -> int:
    """Return the maximum internal worker budget passed to one parse process."""
    return _positive_int_env("DOCMIRROR_WORKERS_PER_PARSE", _DEFAULT_WORKERS_PER_PARSE)


def parse_timeout_seconds() -> float:
    """Return the absolute wall-clock limit for one parser process."""
    return _positive_float_env("DOCMIRROR_PARSE_TIMEOUT_SECONDS", _DEFAULT_PARSE_TIMEOUT_SECONDS)


def parse_kill_grace_seconds() -> float:
    """Return the graceful termination window before a forced kill."""
    return _positive_float_env("DOCMIRROR_PARSE_KILL_GRACE_SECONDS", _DEFAULT_KILL_GRACE_SECONDS)


def bounded_request_payload(request: ParseRequest) -> dict[str, Any]:
    """Serialize a REST request while enforcing the per-process worker cap."""
    payload = request.to_dict()
    cap = workers_per_parse()
    requested = payload.get("workers")
    if requested is None or (isinstance(requested, str) and requested.strip().lower() in {"", "auto"}):
        payload["workers"] = cap
        return payload
    try:
        payload["workers"] = min(cap, max(1, int(requested)))
    except (TypeError, ValueError):
        payload["workers"] = cap
    return payload


class ParseProcessManager:
    """Own parser-process slots without executing parser code in the API process."""

    def __init__(self) -> None:
        self._state_lock = asyncio.Lock()
        self._active_slots = 0
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._tasks: set[asyncio.Task[TaskResult]] = set()
        self._runtime_results: OrderedDict[tuple[str, str], TaskResult] = OrderedDict()
        self._closing = False

    async def startup(self) -> None:
        """Allow submissions for a new FastAPI lifespan."""
        async with self._state_lock:
            self._closing = False

    async def start(
        self,
        request: ParseRequest,
        *,
        output_root: Path,
        task_id: str,
    ) -> asyncio.Task[TaskResult]:
        """Reserve capacity and start monitoring one isolated parser process."""
        async with self._state_lock:
            limit = max_active_parses()
            if self._closing:
                raise RuntimeError("DocMirror parse manager is shutting down")
            if limit > 0 and self._active_slots >= limit:
                raise ParseCapacityError(f"all {limit} parser process slot(s) are occupied")
            self._active_slots += 1

        task = asyncio.create_task(
            self._run_reserved(request, output_root=output_root.resolve(), task_id=task_id),
            name=f"docmirror-process:{task_id}",
        )
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
        return task

    def runtime_result(self, task_id: str, *, output_root: Path) -> TaskResult | None:
        """Return a non-persistent terminal result created by the API parent."""
        return self._runtime_results.get((str(output_root.resolve()), task_id))

    async def shutdown(self) -> None:
        """Terminate active parser process groups and wait for monitors to finish."""
        async with self._state_lock:
            self._closing = True
            processes = list(self._processes.items())
            tasks = list(self._tasks)

        for task_id, process in processes:
            logger.warning("event=parse_worker_shutdown task_id=%s pid=%s", task_id, process.pid)
            await self._terminate_process_group(process, grace_seconds=parse_kill_grace_seconds())
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _run_reserved(
        self,
        request: ParseRequest,
        *,
        output_root: Path,
        task_id: str,
    ) -> TaskResult:
        process: asyncio.subprocess.Process | None = None
        request_path = task_directory(task_id, output_root=output_root) / ".parse-request.json"
        timeout_seconds = parse_timeout_seconds()
        try:
            request_path.parent.mkdir(parents=True, exist_ok=True)
            request_path.write_text(
                json.dumps(bounded_request_payload(request), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            process = await self._spawn_process(
                task_id=task_id,
                request_path=request_path,
                output_root=output_root,
            )
            async with self._state_lock:
                self._processes[task_id] = process
            logger.info(
                "event=parse_process_started task_id=%s pid=%s timeout_seconds=%s",
                task_id,
                process.pid,
                timeout_seconds,
            )
            try:
                exit_code = await asyncio.wait_for(process.wait(), timeout=timeout_seconds)
            except asyncio.TimeoutError:
                logger.error(
                    "event=parse_process_timeout task_id=%s pid=%s timeout_seconds=%s",
                    task_id,
                    process.pid,
                    timeout_seconds,
                )
                await self._terminate_process_group(process, grace_seconds=parse_kill_grace_seconds())
                return self._remember_runtime_failure(
                    task_id,
                    output_root=output_root,
                    code="PARSE_TIMEOUT",
                    message=f"parsing exceeded the {timeout_seconds:g}-second hard timeout",
                )

            if exit_code != 0:
                logger.error(
                    "event=parse_process_crashed task_id=%s pid=%s exit_code=%s",
                    task_id,
                    process.pid,
                    exit_code,
                )
                return self._remember_runtime_failure(
                    task_id,
                    output_root=output_root,
                    code="WORKER_CRASHED",
                    message=f"parser process exited with code {exit_code}",
                )

            manifest_path = task_directory(task_id, output_root=output_root) / "manifest.json"
            result = task_result_from_manifest(manifest_path)
            if result.status == "running":
                logger.error("event=parse_process_incomplete task_id=%s pid=%s", task_id, process.pid)
                return self._remember_runtime_failure(
                    task_id,
                    output_root=output_root,
                    code="WORKER_INCOMPLETE",
                    message="parser process exited without a terminal result",
                )
            logger.info("event=parse_process_finished task_id=%s pid=%s status=%s", task_id, process.pid, result.status)
            return result
        except asyncio.CancelledError:
            if process is not None:
                await self._terminate_process_group(process, grace_seconds=parse_kill_grace_seconds())
            raise
        except Exception as exc:
            logger.exception("event=parse_process_failed task_id=%s", task_id)
            return self._remember_runtime_failure(
                task_id,
                output_root=output_root,
                code="WORKER_START_FAILED" if process is None else "WORKER_CRASHED",
                message=str(exc),
            )
        finally:
            request_path.unlink(missing_ok=True)
            async with self._state_lock:
                self._processes.pop(task_id, None)
                self._active_slots = max(0, self._active_slots - 1)

    async def _spawn_process(
        self,
        *,
        task_id: str,
        request_path: Path,
        output_root: Path,
    ) -> asyncio.subprocess.Process:
        kwargs: dict[str, Any] = {}
        if os.name == "nt":
            kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        return await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "docmirror.server.parse_worker",
            "--task-id",
            task_id,
            "--request",
            str(request_path),
            "--output-root",
            str(output_root),
            **kwargs,
        )

    async def _terminate_process_group(
        self,
        process: asyncio.subprocess.Process,
        *,
        grace_seconds: float,
    ) -> None:
        if process.returncode is not None:
            return
        try:
            if os.name == "nt":
                process.terminate()
            else:
                os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return

        try:
            await asyncio.wait_for(process.wait(), timeout=grace_seconds)
            return
        except asyncio.TimeoutError:
            pass

        try:
            if os.name == "nt":
                process.kill()
            else:
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        await process.wait()

    def _remember_runtime_failure(
        self,
        task_id: str,
        *,
        output_root: Path,
        code: str,
        message: str,
    ) -> TaskResult:
        manifest_path = task_directory(task_id, output_root=output_root) / "manifest.json"
        if manifest_path.is_file():
            result = task_result_from_manifest(manifest_path)
        else:
            result = TaskResult(task_id=task_id, status="running")
        result.status = "failed"
        result.stage = "completed"
        result.errors = [{"code": code, "message": message, "recoverable": False}]
        progress = dict(result.progress)
        progress.update(
            {
                "running": 0,
                "phase": "completed",
                "phase_percent": 100.0,
                "message": "Document parsing failed",
            }
        )
        result.progress = progress

        key = (str(output_root.resolve()), task_id)
        self._runtime_results[key] = result
        self._runtime_results.move_to_end(key)
        while len(self._runtime_results) > _MAX_RUNTIME_RESULTS:
            self._runtime_results.popitem(last=False)
        return result


_manager: ParseProcessManager | None = None


def get_parse_process_manager() -> ParseProcessManager:
    """Return the process-local manager used by the single Uvicorn API worker."""
    global _manager
    if _manager is None:
        _manager = ParseProcessManager()
    return _manager


__all__ = [
    "ParseCapacityError",
    "ParseProcessManager",
    "bounded_request_payload",
    "get_parse_process_manager",
    "max_active_parses",
    "parse_kill_grace_seconds",
    "parse_timeout_seconds",
    "workers_per_parse",
]
