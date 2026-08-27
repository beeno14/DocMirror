from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from docmirror.output.markdown_renderer import MARKDOWN_PROFILE_MARKER
from scripts.validate.validate_community_artifacts import (
    _AUDIT_COLUMNS,
    _COMMUNITY_READING_MARKER,
    validate_community_artifacts,
)


@pytest.fixture
def community_artifact(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    """One complete enterprise record with independent CSV and audit evidence."""
    payload = {
        "schema": {
            "name": "docmirror.community",
            "version": "4.0.0",
            "edition": "community",
            "domain": "enterprise_credit_report",
            "support_level": "ga",
        },
        "document": {
            "id": "doc_test",
            "type": "enterprise_credit_report",
            "title": "企业信用报告",
            "page_count": 1,
            "language": ["zh"],
            "source_file": {"name": "test.pdf", "mime_type": "application/pdf", "sha256": ""},
            "units": {},
        },
        "sections": [],
        "datasets": [
            {
                "id": "ds_credit",
                "name": "enterprise_credit_accounts",
                "label": "信贷账户",
                "type": "credit_accounts",
                "section_id": "",
                "csv": "001_datasets/credit.csv",
                "row_count": 1,
                "grain": "account",
                "primary_key": "record_id",
                "schema_version": "1.0",
                "status": "complete",
                "columns": [
                    {
                        "key": "balance",
                        "label": "余额",
                        "type": "money",
                        "nullable": False,
                        "raw_available": False,
                        "evidence_available": True,
                    }
                ],
                "completeness": {
                    "expected_row_count": 1,
                    "emitted_row_count": 1,
                    "omitted_row_count": 0,
                    "verified": True,
                    "basis": "source_row_count",
                },
                "rows": [
                    {
                        "record_id": "credit:1",
                        "normalized": {"balance": "12.00"},
                        "source": {"page_range": [1, 1]},
                    }
                ],
            }
        ],
        "reading": {
            "version": "1.0",
            "profile": "community",
            "document_flow": [{"order": 1, "kind": "dataset", "ref_id": "ds_credit"}],
            "tables": [
                {
                    "id": "table_credit",
                    "dataset_id": "ds_credit",
                    "section_id": "",
                    "title": "信贷账户",
                    "column_keys": ["balance"],
                    "row_count": 1,
                }
            ],
        },
        "files": {
            "content_md": "001_content.md",
            "enhanced_reading_md": "001_enhanced_reading.md",
            "datasets_dir": "001_datasets",
            "dataset_audit_csv": "001_datasets/_audit_cells.csv",
        },
        "warnings": [],
    }
    (tmp_path / "001_datasets").mkdir()
    markdown = f"{MARKDOWN_PROFILE_MARKER}\n\n# 企业信用报告\n\n余额：12.00\n"
    (tmp_path / payload["files"]["content_md"]).write_text(markdown, encoding="utf-8")
    (tmp_path / payload["files"]["enhanced_reading_md"]).write_text(
        f"{_COMMUNITY_READING_MARKER}\n{markdown}", encoding="utf-8"
    )
    (tmp_path / "001_datasets/credit.csv").write_text("record_id,balance\ncredit:1,12.00\n", encoding="utf-8")
    with (tmp_path / payload["files"]["dataset_audit_csv"]).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(_AUDIT_COLUMNS))
        writer.writeheader()
        writer.writerow(
            {
                "dataset_id": "ds_credit",
                "record_id": "credit:1",
                "field_key": "balance",
                "value": "12.00",
                "raw": "12.00",
                "value_type": "money",
                "page_start": "1",
                "page_end": "1",
                "evidence_ref": "table:credit:r1:c1",
            }
        )
    return tmp_path / "001_community.json", payload


def _validate(artifact: tuple[Path, dict[str, Any]]) -> list[str]:
    path, payload = artifact
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    return validate_community_artifacts(path)


def test_normalized_only_v4_artifacts_pass(community_artifact) -> None:
    assert _validate(community_artifact) == []


def test_evidence_backed_v3_artifacts_still_pass(community_artifact) -> None:
    _, payload = community_artifact
    payload["schema"]["version"] = "3.0.0"
    row = payload["datasets"][0]["rows"][0]
    row.update(raw=dict(row["normalized"]), canonical_raw=dict(row["normalized"]))
    assert _validate(community_artifact) == []


@pytest.mark.parametrize("missing_pool", ["raw", "canonical_raw"])
def test_v3_still_requires_each_source_pool(community_artifact, missing_pool: str) -> None:
    _, payload = community_artifact
    payload["schema"]["version"] = "3.0.0"
    row = payload["datasets"][0]["rows"][0]
    row.update(raw=dict(row["normalized"]), canonical_raw=dict(row["normalized"]))
    row.pop(missing_pool)
    issues = _validate(community_artifact)
    assert f"ds_credit.rows[0]: missing ['{missing_pool}']" in issues
    assert f"ds_credit.rows[0].{missing_pool}: must be an object" in issues


@pytest.mark.parametrize("injected_pool", ["raw", "canonical_raw"])
def test_v4_schema_rejects_source_pools(community_artifact, injected_pool: str) -> None:
    _, payload = community_artifact
    payload["datasets"][0]["rows"][0][injected_pool] = {"balance": "12.00"}
    assert any(issue.startswith("schema:") for issue in _validate(community_artifact))


@pytest.mark.parametrize("required_block", ["record_id", "normalized", "source"])
def test_v4_retains_required_record_blocks(community_artifact, required_block: str) -> None:
    _, payload = community_artifact
    payload["datasets"][0]["rows"][0].pop(required_block)
    assert f"ds_credit.rows[0]: missing ['{required_block}']" in _validate(community_artifact)


def test_v4_retains_enterprise_field_audit_conservation(community_artifact) -> None:
    path, payload = community_artifact
    audit_path = path.parent / payload["files"]["dataset_audit_csv"]
    header = audit_path.read_text(encoding="utf-8").splitlines()[0]
    audit_path.write_text(header + "\n", encoding="utf-8")
    assert "ds_credit: 1 enterprise fields missing from audit CSV" in _validate(community_artifact)


def test_v4_retains_csv_record_conservation(community_artifact) -> None:
    path, _ = community_artifact
    (path.parent / "001_datasets/credit.csv").write_text("record_id,balance\nother:1,12.00\n", encoding="utf-8")
    assert "ds_credit: ordered record_id mismatch between JSON and CSV" in _validate(community_artifact)
