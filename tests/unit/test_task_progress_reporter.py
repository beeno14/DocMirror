from __future__ import annotations

from pathlib import Path

import docmirror.server.api  # noqa: F401 - initializes the SDK/server import cycle used in production
from docmirror.runtime.ledger import EventLedger
from docmirror.runtime.progress_bus import ProgressSignal
from docmirror.server.task_executor import _TaskProgressReporter, initialize_task_manifest
from docmirror.server.task_result import task_result_from_manifest


def test_task_progress_reporter_preserves_legacy_percent_and_exposes_pipeline_progress(tmp_path: Path):
    task_id = "task_progress"
    initialize_task_manifest(
        tmp_path,
        task_id,
        [
            {"file_id": "001", "file_name": "one.pdf", "status": "queued"},
            {"file_id": "002", "file_name": "two.pdf", "status": "queued"},
        ],
    )
    task_dir = tmp_path / task_id
    ledger = EventLedger(task_dir)
    reporter = _TaskProgressReporter(
        ledger,
        [(Path("one.pdf"), "one.pdf", "001"), (Path("two.pdf"), "two.pdf", "002")],
        min_write_interval_s=3600.0,
    )

    reporter.on_signal(
        "001",
        ProgressSignal(
            phase="page_extraction",
            phase_pct=33.33,
            overall_pct=28.33,
            message="Extracted page 1/3",
        ),
    )
    progress = ledger.read_manifest()["progress"]
    assert progress["percent"] == 0.0
    assert progress["pipeline_percent"] == 14.16
    assert progress["phase"] == "page_extraction"
    assert progress["detail"]["current_page"] == 1
    assert progress["detail"]["total_pages"] == 3

    # Same-phase updates are throttled, but the latest state is retained and a
    # phase completion is always flushed.
    reporter.on_signal(
        "001",
        ProgressSignal(
            phase="page_extraction",
            phase_pct=66.67,
            overall_pct=51.67,
            message="Extracted page 2/3",
        ),
    )
    assert ledger.read_manifest()["progress"]["detail"]["current_page"] == 1
    reporter.on_signal(
        "001",
        ProgressSignal(
            phase="page_extraction",
            phase_pct=100.0,
            overall_pct=75.0,
            message="Extracted page 3/3",
        ),
    )
    assert ledger.read_manifest()["progress"]["detail"]["current_page"] == 3

    reporter.finish("001", failed=False)
    progress = ledger.read_manifest()["progress"]
    assert progress["percent"] == 50.0
    assert progress["completed_units"] == 1
    assert progress["pipeline_percent"] == 50.0

    reporter.finish("002", failed=True)
    progress = task_result_from_manifest(task_dir / "manifest.json").public_dict()["progress"]
    assert progress["percent"] == 100.0
    assert progress["pipeline_percent"] == 100.0
    assert progress["completed_units"] == 1
    assert progress["failed_units"] == 1
    assert progress["updated_at"].endswith("Z")
