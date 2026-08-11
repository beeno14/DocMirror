# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Persistent execution helpers for the DocMirror REST Task API."""

from __future__ import annotations

import asyncio
import logging
import os
import re
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from docmirror.input.entry.factory import PerceiveOptions, perceive_document
from docmirror.input.entry.options import ParsePolicy, normalize_parse_policy
from docmirror.runtime.ledger import EventLedger, build_manifest_v2
from docmirror.runtime.progress_bus import ProgressBus, ProgressSignal
from docmirror.sdk.integration.request import ParseRequest
from docmirror.server.edition_outputs import write_outputs
from docmirror.server.task_result import TaskResult, task_result_from_manifest

_TERMINAL_STATUSES = {"success", "partial", "failed"}
_PAGE_MESSAGE_RE = re.compile(r"(?:page\s+)?(?P<current>\d+)\s*/\s*(?P<total>\d+)", re.IGNORECASE)
logger = logging.getLogger(__name__)


class _TaskProgressReporter:
    """Aggregate per-file ProgressBus signals into one durable task snapshot.

    Progress callbacks may arrive from page/projector worker threads. The lock
    protects both aggregation and manifest writes, while the interval limits
    fsync pressure from page-level signals. Phase boundaries and file
    completion are always published immediately.
    """

    def __init__(
        self,
        ledger: EventLedger,
        files: list[tuple[Path, str, str]],
        *,
        min_write_interval_s: float = 0.25,
    ) -> None:
        self._ledger = ledger
        self._lock = threading.Lock()
        self._min_write_interval_s = max(0.0, min_write_interval_s)
        self._last_write_at = 0.0
        self._files: dict[str, dict[str, Any]] = {
            file_id: {
                "file_id": file_id,
                "file_name": file_name,
                "pipeline_percent": 0.0,
                "phase": "queued",
                "phase_percent": 0.0,
                "message": f"Waiting to parse {file_name}",
                "detail": {},
                "status": "running",
                "updated_monotonic": 0.0,
            }
            for _path, file_name, file_id in files
        }

    def on_signal(self, file_id: str, signal: ProgressSignal) -> None:
        """Record one phase signal and publish a throttled aggregate."""
        now = time.monotonic()
        with self._lock:
            state = self._files.get(file_id)
            if state is None or state["status"] != "running":
                return
            phase_changed = state["phase"] != signal.phase
            # The final 1% represents artifact persistence and task publication,
            # which happen after the output builder emits its 100% signal.
            reported_pipeline = max(0.0, min(99.0, float(signal.overall_pct)))
            state.update(
                {
                    "pipeline_percent": max(float(state["pipeline_percent"]), reported_pipeline),
                    "phase": str(signal.phase),
                    "phase_percent": max(0.0, min(100.0, float(signal.phase_pct))),
                    "message": str(signal.message),
                    "detail": self._normalized_detail(file_id, signal),
                    "updated_monotonic": now,
                }
            )
            force = phase_changed or signal.phase_pct >= 100.0
            if force or now - self._last_write_at >= self._min_write_interval_s:
                self._publish_locked(now)

    def finish(self, file_id: str, *, failed: bool) -> None:
        """Mark one work unit finished and immediately publish its counters."""
        now = time.monotonic()
        with self._lock:
            state = self._files.get(file_id)
            if state is None or state["status"] != "running":
                return
            state.update(
                {
                    "pipeline_percent": 100.0,
                    "phase": "failed" if failed else "completed",
                    "phase_percent": 100.0,
                    "message": "Document parsing failed" if failed else "Document parsing complete",
                    "status": "failed" if failed else "success",
                    "updated_monotonic": now,
                }
            )
            self._publish_locked(now)

    def _normalized_detail(self, file_id: str, signal: ProgressSignal) -> dict[str, Any]:
        state = self._files[file_id]
        detail = dict(signal.detail or {})
        current_page = detail.get("current_page", detail.get("page", detail.get("current")))
        total_pages = detail.get("total_pages", detail.get("total"))
        if current_page is None or total_pages is None:
            match = _PAGE_MESSAGE_RE.search(str(signal.message))
            if match:
                current_page = int(match.group("current"))
                total_pages = int(match.group("total"))
        if current_page is not None:
            detail["current_page"] = current_page
        if total_pages is not None:
            detail["total_pages"] = total_pages
        detail["file_id"] = file_id
        detail["file_name"] = state["file_name"]
        return detail

    def _publish_locked(self, now: float) -> None:
        states = list(self._files.values())
        if not states:
            return
        active = [state for state in states if state["status"] == "running"]
        representative = max(active or states, key=lambda state: float(state["updated_monotonic"]))
        completed = sum(1 for state in states if state["status"] == "success")
        failed = sum(1 for state in states if state["status"] == "failed")
        finished = completed + failed
        detail = dict(representative["detail"])
        detail.update(
            {
                "completed_files": completed,
                "failed_files": failed,
                "total_files": len(states),
            }
        )
        message = str(representative["message"])
        if len(states) > 1 and representative["status"] == "running":
            message = f"{representative['file_name']}: {message}"
        progress = {
            "total_units": len(states),
            "completed_units": completed,
            "failed_units": failed,
            "running_units": len(states) - finished,
            # Legacy task progress remains file-completion based.
            "percent": round(finished / len(states) * 100.0, 2),
            "pipeline_percent": round(
                sum(float(state["pipeline_percent"]) for state in states) / len(states),
                2,
            ),
            "phase": str(representative["phase"]),
            "phase_percent": round(float(representative["phase_percent"]), 2),
            "message": message,
            "detail": detail,
            "updated_at": _utc_now(),
        }
        try:
            manifest = self._ledger.read_manifest()
            if not manifest or manifest.get("status") in _TERMINAL_STATUSES:
                return
            manifest["progress"] = progress
            self._ledger.write_manifest(manifest)
            self._last_write_at = now
        except Exception as exc:
            # Progress reporting must never fail document parsing.
            logger.debug("[TaskProgress] Manifest update failed: %s", exc)


def task_output_root() -> Path:
    """Resolve the task store at request time so test/deployment env changes apply."""
    configured = os.environ.get("DOCMIRROR_TASK_OUTPUT_DIR") or os.environ.get("DOCMIRROR_TASK_DIR") or "output/tasks"
    return Path(configured).resolve()


def task_directory(task_id: str, *, output_root: Path | None = None) -> Path:
    """Return a validated task directory beneath the configured task store."""
    if not task_id or any(part in {"", ".", ".."} for part in Path(task_id).parts) or Path(task_id).is_absolute():
        raise ValueError("invalid task_id")
    root = (output_root or task_output_root()).resolve()
    candidate = (root / task_id).resolve()
    if not candidate.is_relative_to(root):
        raise ValueError("invalid task_id")
    return candidate


def initialize_task_manifest(
    output_root: Path,
    task_id: str,
    inputs: list[dict[str, Any]],
    *,
    parse_policy: dict[str, Any] | None = None,
    max_workers: int | None = None,
    worker_budget: dict[str, int] | None = None,
) -> Path:
    """Create the durable manifest before execution starts."""
    task_dir = task_directory(task_id, output_root=output_root)
    task_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = task_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"task already exists: {task_id}")
    manifest = build_manifest_v2(
        task_id,
        status="running",
        stage="queued",
        inputs=inputs,
        parse_policy=parse_policy,
        runtime_control={
            "worker_budget": worker_budget or ({"page_workers_per_file": max_workers} if max_workers else {})
        },
        entry="rest",
    )
    manifest["progress"] = _progress(
        total=len(inputs),
        completed=0,
        failed=0,
        running=0,
        pipeline_percent=0.0,
        phase="queued",
        phase_percent=0.0,
        message="Waiting for document parsing to start",
        detail={"total_files": len(inputs)},
    )
    EventLedger(task_dir).write_manifest(manifest)
    return manifest_path


def read_task_manifest(task_id: str, *, output_root: Path | None = None) -> dict[str, Any]:
    """Load a task manifest, returning an empty dict when it does not exist."""
    task_dir = task_directory(task_id, output_root=output_root)
    return EventLedger(task_dir).read_manifest()


async def execute_parse_task(
    request: ParseRequest,
    *,
    output_root: Path,
    task_id: str,
    timeout_s: float = 1800.0,
) -> TaskResult:
    """Parse one or more files and atomically publish a terminal manifest.

    Each batch member owns an isolated artifact directory so markdown,
    evidence and visual-debug filenames cannot collide. A failure in one
    member never discards successful members.
    """
    task_dir = task_directory(task_id, output_root=output_root)
    ledger = EventLedger(task_dir)
    manifest = ledger.read_manifest()
    if manifest.get("status") in _TERMINAL_STATUSES:
        return task_result_from_manifest(task_dir / "manifest.json")

    if not request.inputs:
        raise ValueError("ParseRequest.inputs must contain at least one document")

    policy = _policy_from_request(request)
    from docmirror.configs.runtime.performance import resolve_worker_budget

    budget = resolve_worker_budget(request.workers, file_count=len(request.inputs))
    files = _materialize_inputs(request, task_dir)
    file_semaphore = asyncio.Semaphore(budget.file_workers)
    if not manifest:
        initialize_task_manifest(
            output_root,
            task_id,
            [_input_entry(file_id, file_name) for _path, file_name, file_id in files],
            parse_policy=policy.to_dict(),
            max_workers=budget.page_workers_per_file,
            worker_budget=_worker_budget_dict(budget),
        )
        manifest = ledger.read_manifest()

    manifest["stage"] = "parsing"
    manifest["progress"] = _progress(
        total=len(files),
        completed=0,
        failed=0,
        running=len(files),
        pipeline_percent=0.0,
        phase="load_document",
        phase_percent=0.0,
        message="Initializing document parser",
        detail={"total_files": len(files)},
    )
    ledger.write_manifest(manifest)

    batch = len(files) > 1
    input_entries = list(manifest.get("inputs") or [])
    progress_reporter = _TaskProgressReporter(ledger, files)

    try:
        from docmirror.ocr.vlm_gateway import _gateway

        _gateway.collect_fallbacks()
    except Exception:
        _gateway = None

    async def process_one(index: int, source_path: Path, original_name: str, file_id: str) -> dict[str, Any]:
        entry = (
            input_entries[index - 1]
            if index - 1 < len(input_entries)
            else {
                "file_id": file_id,
                "file_name": original_name,
            }
        )
        entry.update({"file_id": file_id, "file_name": original_name, "status": "running"})
        bus = ProgressBus(on_progress=lambda signal: progress_reporter.on_signal(file_id, signal))
        bus.bind_ledger(ledger)
        try:
            async with file_semaphore:
                bus.emit(
                    "load_document",
                    0.0,
                    "Initializing document parser...",
                    {"file_id": file_id, "file_name": original_name},
                )
                result = await asyncio.wait_for(
                    perceive_document(
                        source_path,
                        PerceiveOptions(
                            policy=policy,
                            max_workers=budget.page_workers_per_file,
                            on_progress=bus.emit,
                        ),
                    ),
                    timeout=timeout_s,
                )
            result_view = result.to_read_view() if callable(getattr(result, "to_read_view", None)) else result
            artifact_dir = task_dir / "files" / file_id if batch else task_dir
            _written_task_id, written = write_outputs(
                result,
                output_root,
                file_path=str(source_path),
                file_id=file_id,
                task_id=task_id,
                overwrite=True,
                artifact_dir=artifact_dir,
                include_mirror=False,
                include_manifest=True,
                on_progress=bus.emit,
            )
            child_manifest = EventLedger(artifact_dir).read_manifest()
            artifacts = dict(child_manifest.get("artifacts") or {})
            if not artifacts:
                artifacts = {name: path.name for name, path in written.items()}
            task_artifacts = {
                name: relative
                for name, relative in _parent_artifact_map(
                    task_dir=task_dir,
                    artifact_dir=artifact_dir,
                    artifacts=artifacts,
                    file_id=file_id,
                    batch=batch,
                ).items()
                if not name.endswith("mirror") and name != "mirror"
            }
            input_artifacts = {
                name: relative
                for name, relative in _input_artifact_map(
                    task_dir=task_dir,
                    artifact_dir=artifact_dir,
                    artifacts=artifacts,
                ).items()
                if name != "mirror"
            }
            public_availability = {
                name: value
                for name, value in (child_manifest.get("edition_availability") or {}).items()
                if name != "mirror"
            }
            from docmirror.evidence.quality import build_quality_summary

            quality_summary = build_quality_summary(result_view)
            document_type = str(getattr(getattr(result_view, "entities", None), "document_type", "") or "generic")
            entry.update(
                {
                    "status": "success",
                    "document_type": document_type,
                    "page_count": int(getattr(result_view, "page_count", 0) or 0),
                    "quality_summary": quality_summary,
                    "artifacts": input_artifacts,
                    "edition_availability": public_availability,
                    "errors": [],
                }
            )
            progress_reporter.finish(file_id, failed=False)
            return {
                "file_id": file_id,
                "file_name": original_name,
                "status": "success",
                "artifacts": task_artifacts,
                "edition_availability": public_availability,
                "mirror_completeness": child_manifest.get("mirror_completeness") or {},
                "quality_summary": quality_summary,
            }
        except asyncio.TimeoutError:
            error = {
                "code": "TIMEOUT",
                "message": f"parse exceeded {timeout_s:g}s timeout",
                "recoverable": True,
            }
            entry.update({"status": "failed", "artifacts": {}, "edition_availability": {}, "errors": [error]})
            progress_reporter.finish(file_id, failed=True)
            return {
                "file_id": file_id,
                "file_name": original_name,
                "status": "failed",
                "error": error,
            }
        except Exception as exc:
            error = {
                "code": "PARSER_ERROR",
                "message": str(exc),
                "recoverable": False,
            }
            entry.update({"status": "failed", "artifacts": {}, "edition_availability": {}, "errors": [error]})
            progress_reporter.finish(file_id, failed=True)
            return {
                "file_id": file_id,
                "file_name": original_name,
                "status": "failed",
                "error": error,
            }
        finally:
            _cleanup_managed_input(source_path, task_dir)

    outcomes = await asyncio.gather(
        *(process_one(index, path, name, file_id) for index, (path, name, file_id) in enumerate(files, start=1))
    )
    successes = [outcome for outcome in outcomes if outcome["status"] == "success"]
    failures = [outcome for outcome in outcomes if outcome["status"] == "failed"]
    status = "partial" if successes and failures else ("success" if successes else "failed")

    artifacts: dict[str, str] = {}
    edition_availability: dict[str, Any] = {}
    mirror_completeness: dict[str, Any] = {}
    quality_summary: dict[str, Any] = {}
    errors: list[dict[str, Any]] = []
    for outcome in outcomes:
        artifacts.update(outcome.get("artifacts") or {})
        if outcome.get("edition_availability"):
            if batch:
                edition_availability[outcome["file_id"]] = outcome["edition_availability"]
            else:
                edition_availability = outcome["edition_availability"]
        if outcome.get("mirror_completeness"):
            if batch:
                mirror_completeness[outcome["file_id"]] = outcome["mirror_completeness"]
            else:
                mirror_completeness = outcome["mirror_completeness"]
        if outcome.get("quality_summary"):
            if batch:
                quality_summary[outcome["file_id"]] = outcome["quality_summary"]
            else:
                quality_summary = outcome["quality_summary"]
        if outcome.get("error"):
            errors.append(
                {
                    "file_id": outcome["file_id"],
                    "file_name": outcome["file_name"],
                    **outcome["error"],
                }
            )

    fallbacks = _gateway.collect_fallbacks() if _gateway is not None else []
    manifest.update(
        {
            "status": status,
            "stage": "completed",
            "progress": _progress(
                total=len(files),
                completed=len(successes),
                failed=len(failures),
                running=0,
                pipeline_percent=100.0,
                phase="completed",
                phase_percent=100.0,
                message="Document parsing complete" if not failures else "Document parsing completed with errors",
                detail={
                    "completed_files": len(successes),
                    "failed_files": len(failures),
                    "total_files": len(files),
                },
            ),
            "inputs": input_entries,
            "artifacts": artifacts,
            "edition_availability": edition_availability,
            "mirror_completeness": mirror_completeness,
            "quality_summary": quality_summary,
            "fallbacks": fallbacks,
            "errors": errors,
        }
    )
    ledger.write_manifest(manifest)
    return task_result_from_manifest(task_dir / "manifest.json")


def _policy_from_request(request: ParseRequest) -> ParsePolicy:
    return normalize_parse_policy(
        pages=request.pages,
        max_pages=request.max_pages,
        mode=request.mode,
        doc_type=request.doc_type,
        doc_type_policy=request.doc_type_policy,
        ocr=request.ocr,
        ocr_correction=request.ocr_correction,
        ocr_language=request.ocr_language,
        ocr_country=request.ocr_country,
        ocr_locale=request.ocr_locale,
        ocr_correction_packs=request.ocr_correction_packs,
        page_split=request.page_split,
    )


def _materialize_inputs(request: ParseRequest, task_dir: Path) -> list[tuple[Path, str, str]]:
    files: list[tuple[Path, str, str]] = []
    managed_root = task_dir / "inputs"
    seen_file_ids: set[str] = set()
    for index, item in enumerate(request.inputs, start=1):
        file_id = item.file_id or f"{index:03d}"
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", file_id):
            raise ValueError(f"Invalid file_id: {file_id!r}")
        if file_id in seen_file_ids:
            raise ValueError(f"Duplicate file_id: {file_id!r}")
        seen_file_ids.add(file_id)
        file_name = item.file_name or "document"
        if item.file_path:
            path = Path(item.file_path).resolve()
        elif item.data is not None:
            managed_root.mkdir(parents=True, exist_ok=True)
            safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(file_name).name).strip("._") or "document.bin"
            path = managed_root / f"{file_id}_{safe_name}"
            path.write_bytes(item.data)
        else:
            raise ValueError(f"Input {file_id} has neither file_path nor data")
        if not path.is_file():
            raise FileNotFoundError(f"Input file not found: {path}")
        files.append((path, file_name, file_id))
    return files


def _cleanup_managed_input(source_path: Path, task_dir: Path) -> None:
    managed_root = (task_dir / "inputs").resolve()
    try:
        candidate = source_path.resolve()
        if candidate.is_relative_to(managed_root):
            candidate.unlink(missing_ok=True)
    except OSError:
        pass


def _input_entry(file_id: str, file_name: str) -> dict[str, Any]:
    return {"file_id": file_id, "file_name": file_name, "status": "queued"}


def _worker_budget_dict(budget: Any) -> dict[str, int]:
    return {
        "total": int(budget.total),
        "file_workers": int(budget.file_workers),
        "page_workers_per_file": int(budget.page_workers_per_file),
        "layout_workers": int(budget.layout_workers),
    }


def _parent_artifact_map(
    *,
    task_dir: Path,
    artifact_dir: Path,
    artifacts: dict[str, Any],
    file_id: str,
    batch: bool,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for name, relative in artifacts.items():
        candidate = artifact_dir / str(relative)
        try:
            parent_relative = str(candidate.resolve().relative_to(task_dir.resolve()))
        except ValueError:
            continue
        key = f"{file_id}_{name}" if batch else str(name)
        result[key] = parent_relative
    return result


def _input_artifact_map(
    *,
    task_dir: Path,
    artifact_dir: Path,
    artifacts: dict[str, Any],
) -> dict[str, str]:
    """Return role-keyed paths for one input, rooted at the parent task."""
    result: dict[str, str] = {}
    for role, relative in artifacts.items():
        candidate = artifact_dir / str(relative)
        try:
            result[str(role)] = str(candidate.resolve().relative_to(task_dir.resolve()))
        except ValueError:
            continue
    return result


def _progress(
    *,
    total: int,
    completed: int,
    failed: int,
    running: int,
    pipeline_percent: float | None = None,
    phase: str | None = None,
    phase_percent: float | None = None,
    message: str | None = None,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    finished = completed + failed
    progress: dict[str, Any] = {
        "total_units": total,
        "completed_units": completed,
        "failed_units": failed,
        "running_units": running,
        "percent": round(finished / max(total, 1) * 100.0, 2),
    }
    if pipeline_percent is not None:
        progress.update(
            {
                "pipeline_percent": round(max(0.0, min(100.0, pipeline_percent)), 2),
                "phase": phase or "",
                "phase_percent": round(max(0.0, min(100.0, phase_percent or 0.0)), 2),
                "message": message or "",
                "detail": dict(detail or {}),
                "updated_at": _utc_now(),
            }
        )
    return progress


def _utc_now() -> str:
    return datetime.now(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "execute_parse_task",
    "initialize_task_manifest",
    "read_task_manifest",
    "task_directory",
    "task_output_root",
]
