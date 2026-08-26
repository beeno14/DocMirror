# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from docmirror.models.entities.parse_result import DocumentEntities, PageContent, ParseResult, ResultStatus
from docmirror.models.sealed import seal_parse_result
from docmirror.server.api import app
from docmirror.server.parse_admission_queue import (
    ParseQueueFullError,
    ParseQueueWaitTimeoutError,
)
from docmirror.server.parse_process_manager import (
    get_parse_process_manager,
)
from docmirror.server.task_executor import execute_parse_task


@pytest.fixture(autouse=True)
def _execute_parses_inline(monkeypatch):
    """Keep contract monkeypatches in-process; process isolation has dedicated tests."""
    manager = get_parse_process_manager()

    async def start(request, *, output_root, task_id):
        return asyncio.create_task(execute_parse_task(request, output_root=output_root, task_id=task_id))

    monkeypatch.setattr(manager, "start", start)


def _mirror(document_type: str = "business_license") -> ParseResult:
    result = ParseResult(status=ResultStatus.SUCCESS)
    result.entities = DocumentEntities(
        document_type=document_type,
        domain_specific={
            "records": [
                {"record_id": "api:001", "normalized": {"value": "A"}, "raw": {"value": "原值A"}},
                {"record_id": "api:002", "normalized": {"value": "B"}, "raw": {"value": "原值B"}},
            ]
        },
    )
    return result


def test_task_api_wait_writes_artifacts(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DOCMIRROR_TASK_OUTPUT_DIR", str(tmp_path))

    async def fake_perceive(_path, _options):
        return _mirror()

    monkeypatch.setattr("docmirror.server.task_executor.perceive_document", fake_perceive)
    client = TestClient(app)
    response = client.post(
        "/v1/tasks?wait=true",
        files={"file": ("sample.pdf", b"PDF", "application/pdf")},
    )

    assert response.status_code == 200
    task = response.json()
    assert task["status"] == "success"
    task_id = task["task_id"]
    status = client.get(f"/v1/tasks/{task_id}").json()
    assert "mirror" not in status["artifacts"]
    artifacts = client.get(f"/v1/tasks/{task_id}/artifacts").json()["artifacts"]
    assert artifacts["community"] == "001_community.json"
    community_response = client.get(f"/v1/tasks/{task_id}/artifacts/community")
    assert community_response.status_code == 200
    community = json.loads(community_response.content)
    dataset = community["datasets"][0]
    assert dataset["row_count"] == len(dataset["rows"]) == 2
    assert [row["record_id"] for row in dataset["rows"]] == ["api:001", "api:002"]
    assert dataset["rows"][0]["raw"]["value"] == "原值A"


def test_task_api_routes_cjk_pdf_upload(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DOCMIRROR_TASK_OUTPUT_DIR", str(tmp_path))
    parsed_names: list[str] = []

    async def fake_perceive(path, _options):
        parsed_names.append(Path(path).name)
        result = _mirror()
        result.pages = [PageContent(page_number=1)]
        return seal_parse_result(result)

    monkeypatch.setattr("docmirror.server.task_executor.perceive_document", fake_perceive)
    client = TestClient(app)
    original_name = "赵思雯个人征信.pdf"
    response = client.post(
        "/v1/tasks?wait=true",
        files={"file": (original_name, b"PDF", "application/pdf")},
    )

    assert response.status_code == 200
    task = response.json()
    assert task["status"] == "success"
    assert parsed_names == ["001_upload.pdf"]
    assert task["inputs"][0]["document_type"] == "business_license"
    assert task["inputs"][0]["page_count"] == 1


def test_task_api_always_uses_fixed_delivery(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DOCMIRROR_TASK_OUTPUT_DIR", str(tmp_path))

    async def fake_perceive(_path, _options):
        return _mirror()

    monkeypatch.setattr("docmirror.server.task_executor.perceive_document", fake_perceive)
    client = TestClient(app)
    response = client.post(
        "/v1/tasks?wait=true",
        files={"file": ("sample.pdf", b"PDF", "application/pdf")},
    )

    assert response.status_code == 200
    task = response.json()
    assert task["status"] == "success"
    assert task["artifacts"]["community"] == "001_community.json"
    assert "mirror" not in task["artifacts"]
    assert "mirror" not in task["edition_availability"]
    assert "editions" not in task
    assert "formats" not in task
    task_dir = tmp_path / task["task_id"]
    assert not (task_dir / "001_mirror.json").exists()


def test_task_api_has_no_delivery_selection_parameters():
    schema = TestClient(app).get("/openapi.json").json()
    removed = {"formats", "editions", "geometry", "include_geometry", "include_text", "mirror_level"}
    for path in ("/v1/tasks", "/v1/tasks/batch"):
        parameters = {parameter["name"] for parameter in schema["paths"][path]["post"].get("parameters", [])}
        assert not parameters & removed


def test_batch_task_api_preserves_partial_failure(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DOCMIRROR_TASK_OUTPUT_DIR", str(tmp_path))

    async def fake_perceive(path, _options):
        if Path(path).read_text(encoding="utf-8") == "FAIL":
            raise ValueError("fixture failure")
        return _mirror("bank_statement")

    monkeypatch.setattr("docmirror.server.task_executor.perceive_document", fake_perceive)
    client = TestClient(app)
    response = client.post(
        "/v1/tasks/batch?wait=true",
        files=[
            ("files", ("ok.txt", b"OK", "text/plain")),
            ("files", ("fail.txt", b"FAIL", "text/plain")),
        ],
    )

    assert response.status_code == 200
    task = response.json()
    assert task["status"] == "partial"
    assert len(task["inputs"]) == 2
    assert len(task["errors"]) == 1
    assert any(key.startswith("001_") for key in task["artifacts"])


def test_task_api_accepts_background_work_and_persists_terminal_status(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DOCMIRROR_TASK_OUTPUT_DIR", str(tmp_path))

    async def fake_perceive(_path, _options):
        await asyncio.sleep(0.02)
        return _mirror()

    monkeypatch.setattr("docmirror.server.task_executor.perceive_document", fake_perceive)
    with TestClient(app) as client:
        response = client.post(
            "/v1/tasks",
            files={"file": ("sample.pdf", b"PDF", "application/pdf")},
        )
        assert response.status_code == 202
        task_id = response.json()["task_id"]

        deadline = time.monotonic() + 3.0
        status = client.get(f"/v1/tasks/{task_id}").json()
        while status["status"] == "running" and time.monotonic() < deadline:
            time.sleep(0.02)
            status = client.get(f"/v1/tasks/{task_id}").json()

        assert status["status"] == "success"
        assert status["progress"]["percent"] == 100.0
        assert status["progress"]["pipeline_percent"] == 100.0
        assert status["progress"]["phase"] == "completed"
        assert status["progress"]["phase_percent"] == 100.0
        assert status["progress"]["message"]
        assert status["progress"]["updated_at"].endswith("Z")
        assert "mirror" not in status["artifacts"]
        assert client.get(f"/v1/tasks/{task_id}/artifacts/not_declared").status_code == 404


def test_primary_task_route_accepts_multiple_files_and_role_downloads(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DOCMIRROR_TASK_OUTPUT_DIR", str(tmp_path))

    async def fake_perceive(_path, _options):
        return _mirror()

    monkeypatch.setattr("docmirror.server.task_executor.perceive_document", fake_perceive)
    client = TestClient(app)
    response = client.post(
        "/v1/tasks?wait=true",
        files=[
            ("files", ("first.pdf", b"ONE", "application/pdf")),
            ("files", ("second.png", b"TWO", "image/png")),
        ],
    )

    assert response.status_code == 200
    task = response.json()
    assert task["status"] == "success"
    assert [item["file_id"] for item in task["inputs"]] == ["001", "002"]
    assert all("mirror" not in item["artifacts"] for item in task["inputs"])
    artifact = client.get(f"/v1/tasks/{task['task_id']}/files/002/artifacts/community")
    assert artifact.status_code == 200
    assert json.loads(artifact.content)["schema"]["name"] == "docmirror.community"


def test_legacy_parse_route_returns_task_result_not_mirror(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DOCMIRROR_TASK_OUTPUT_DIR", str(tmp_path))

    async def fake_perceive(_path, _options):
        return _mirror()

    monkeypatch.setattr("docmirror.server.task_executor.perceive_document", fake_perceive)
    response = TestClient(app).post(
        "/v1/parse",
        files={"file": ("sample.pdf", b"PDF", "application/pdf")},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "success"
    assert "mirror" not in payload
    assert "mirror" not in payload["artifacts"]


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (
            ParseQueueFullError(active=10, queued=50, max_queued=50),
            "DOCMIRROR_QUEUE_FULL",
        ),
        (
            ParseQueueWaitTimeoutError(wait_timeout_seconds=900),
            "DOCMIRROR_QUEUE_WAIT_TIMEOUT",
        ),
    ],
)
def test_legacy_parse_route_returns_structured_queue_errors(
    tmp_path: Path,
    monkeypatch,
    error: Exception,
    expected_code: str,
):
    monkeypatch.setenv("DOCMIRROR_TASK_OUTPUT_DIR", str(tmp_path))
    manager = get_parse_process_manager()

    async def reject_start(_request, *, output_root, task_id):
        raise error

    monkeypatch.setattr(manager, "start", reject_start)
    response = TestClient(app).post(
        "/v1/parse",
        files={"file": ("sample.pdf", b"PDF", "application/pdf")},
    )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == expected_code
    assert response.json()["detail"]["retryable"] is True
    assert not any(tmp_path.iterdir())


def test_task_api_unknown_task_is_not_found(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("DOCMIRROR_TASK_OUTPUT_DIR", str(tmp_path))
    client = TestClient(app)

    assert client.get("/v1/tasks/task_missing").status_code == 404
    assert client.get("/v1/tasks/../outside").status_code == 404
