from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from docmirror.models.entities.parse_result import (
    CellValue,
    DocumentEntities,
    PageContent,
    ParseResult,
    ResultStatus,
    TableBlock,
    TableRow,
)
from docmirror.server.edition_outputs import write_outputs
from scripts.validate.validate_community_artifacts import payment_direction_cells, validate_community_artifacts

pytestmark = [pytest.mark.tier_contract]


def _write_bundle(tmp_path: Path, task_id: str) -> Path:
    result = ParseResult(
        status=ResultStatus.SUCCESS,
        pages=[
            PageContent(
                page_number=1,
                tables=[
                    TableBlock(
                        headers=["收/支", "金额"],
                        rows=[
                            TableRow(cells=[CellValue(text="收入"), CellValue(text="10.00")]),
                            TableRow(cells=[CellValue(text="支出"), CellValue(text="3.00")]),
                        ],
                    )
                ],
            )
        ],
        entities=DocumentEntities(
            document_type="alipay_payment",
            domain_specific={
                "records": [
                    {
                        "record_id": "txn:001",
                        "normalized": {"direction": "income", "amount": "10.00"},
                        "canonical_raw": {"direction": "收入", "amount": "10.00"},
                        "raw": {"收/支": "收入", "金额": "10.00"},
                        "source": {"page": 1},
                    },
                    {
                        "record_id": "txn:002",
                        "normalized": {"direction": "expense", "amount": "3.00"},
                        "canonical_raw": {"direction": "支出", "amount": "3.00"},
                        "raw": {"收/支": "支出", "金额": "3.00"},
                        "source": {"page": 1},
                    },
                ],
                "data_dictionary": {
                    "record_columns": {
                        "direction": {"label": "收/支", "type": "enum"},
                        "amount": {"label": "金额", "type": "money"},
                    }
                },
            },
        ),
    )
    _task_id, written = write_outputs(
        result,
        tmp_path,
        task_id=task_id,
        include_mirror=False,
        include_manifest=False,
    )
    return written["community"]


def test_validator_accepts_complete_json_csv_markdown_bundle(tmp_path: Path) -> None:
    community_path = _write_bundle(tmp_path, "complete_bundle")

    assert validate_community_artifacts(community_path) == []


def test_validator_detects_json_row_loss(tmp_path: Path) -> None:
    community_path = _write_bundle(tmp_path, "json_row_loss")
    payload = json.loads(community_path.read_text(encoding="utf-8"))
    payload["datasets"][0]["rows"].pop()
    community_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    issues = validate_community_artifacts(community_path)

    assert any("JSON count mismatch" in issue for issue in issues)
    assert any("CSV rows=2, JSON rows=1" in issue for issue in issues)


def test_validator_detects_csv_record_id_divergence(tmp_path: Path) -> None:
    community_path = _write_bundle(tmp_path, "csv_id_divergence")
    payload = json.loads(community_path.read_text(encoding="utf-8"))
    csv_path = community_path.parent / payload["datasets"][0]["csv"]
    with csv_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows.reverse()
    with csv_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    issues = validate_community_artifacts(community_path)

    assert any("ordered record_id mismatch" in issue for issue in issues)


def test_validator_detects_missing_markdown_profile(tmp_path: Path) -> None:
    community_path = _write_bundle(tmp_path, "markdown_marker_loss")
    payload = json.loads(community_path.read_text(encoding="utf-8"))
    content_path = community_path.parent / payload["files"]["content_md"]
    content_path.write_text("人工审查内容", encoding="utf-8")

    issues = validate_community_artifacts(community_path)

    assert "content: DMP profile marker missing" in issues


def test_validator_detects_non_community_enhanced_reading(tmp_path: Path) -> None:
    community_path = _write_bundle(tmp_path, "enhanced_marker_loss")
    payload = json.loads(community_path.read_text(encoding="utf-8"))
    enhanced_path = community_path.parent / payload["files"]["enhanced_reading_md"]
    enhanced_path.write_text("private rendering", encoding="utf-8")

    issues = validate_community_artifacts(community_path)

    assert "enhanced_reading: DMP profile marker missing" in issues
    assert "enhanced_reading: Community reading profile marker missing" in issues


def test_validator_detects_reading_model_dataset_divergence(tmp_path: Path) -> None:
    community_path = _write_bundle(tmp_path, "reading_model_divergence")
    payload = json.loads(community_path.read_text(encoding="utf-8"))
    payload["reading"]["tables"][0]["row_count"] = 1
    payload["reading"]["document_flow"] = [
        entry for entry in payload["reading"]["document_flow"] if entry["kind"] != "dataset"
    ]
    community_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    issues = validate_community_artifacts(community_path)

    assert any("reading.tables[0]: row_count=1, JSON rows=2" in issue for issue in issues)
    assert any("reading.document_flow: missing datasets=" in issue for issue in issues)


def test_validator_rejects_misrouted_english_enterprise_public_dataset(
    tmp_path: Path,
) -> None:
    community_path = _write_bundle(tmp_path, "enterprise_public_routing")
    payload = json.loads(community_path.read_text(encoding="utf-8"))
    dataset = payload["datasets"][0]
    dataset["name"] = "enterprise_public_license_records"
    dataset["label"] = "enterprise public license records"
    dataset["rows"][0]["normalized"]["content"] = "duplicated generic content"
    payload["reading"]["tables"][0]["section_id"] = "sec_wrong"
    payload["public_records"] = {"license_records": []}
    community_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    issues = validate_community_artifacts(community_path)

    assert any(
        "enterprise public dataset must belong to a public_records section" in issue
        for issue in issues
    )
    assert any(
        "enterprise public dataset label must be Chinese" in issue
        for issue in issues
    )
    assert any(
        "reading table section_id=sec_wrong" in issue
        for issue in issues
    )
    assert any("generic public-record keys=['content']" in issue for issue in issues)
    assert any("community: top-level blocks=" in issue for issue in issues)


def test_validator_detects_semantic_dataset_divergence(tmp_path: Path) -> None:
    community_path = _write_bundle(tmp_path, "semantic_dataset_divergence")
    payload = json.loads(community_path.read_text(encoding="utf-8"))
    semantic_path = community_path.parent / payload["files"]["semantic_json"]
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    semantic["datasets"][0]["rows"].pop()
    semantic_path.write_text(json.dumps(semantic, ensure_ascii=False), encoding="utf-8")

    issues = validate_community_artifacts(community_path)

    assert "semantic: datasets diverge from Community JSON" in issues


def test_validator_detects_completeness_contradiction(tmp_path: Path) -> None:
    community_path = _write_bundle(tmp_path, "completeness_contradiction")
    payload = json.loads(community_path.read_text(encoding="utf-8"))
    payload["datasets"][0]["completeness"].update({"emitted_row_count": 1, "omitted_row_count": 1, "verified": True})
    community_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    issues = validate_community_artifacts(community_path)

    assert any("JSON count mismatch" in issue for issue in issues)
    assert any("completeness.verified contradicts" in issue for issue in issues)


def test_validator_detects_audit_record_loss(tmp_path: Path) -> None:
    community_path = _write_bundle(tmp_path, "audit_record_loss")
    payload = json.loads(community_path.read_text(encoding="utf-8"))
    audit_path = community_path.parent / payload["files"]["dataset_audit_csv"]
    with audit_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    kept = [row for row in rows if row["record_id"] != "txn:002"]
    with audit_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(kept)

    issues = validate_community_artifacts(community_path)

    assert "ds_transactions: 1 records missing from audit CSV" in issues


def test_validator_requires_evidence_for_enterprise_audit_fields(tmp_path: Path) -> None:
    community_path = _write_bundle(tmp_path, "enterprise_evidence_loss")
    payload = json.loads(community_path.read_text(encoding="utf-8"))
    semantic_path = community_path.parent / payload["files"]["semantic_json"]
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    semantic["classification"]["document_type"] = "enterprise_credit_report"
    semantic_path.write_text(json.dumps(semantic, ensure_ascii=False), encoding="utf-8")
    audit_path = community_path.parent / payload["files"]["dataset_audit_csv"]
    with audit_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows[0]["evidence_ref"] = "[]"
    with audit_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    issues = validate_community_artifacts(community_path)

    assert any("enterprise field missing evidence_ref" in issue for issue in issues)


def test_validator_requires_every_enterprise_field_in_audit_csv(tmp_path: Path) -> None:
    community_path = _write_bundle(tmp_path, "enterprise_field_loss")
    payload = json.loads(community_path.read_text(encoding="utf-8"))
    semantic_path = community_path.parent / payload["files"]["semantic_json"]
    semantic = json.loads(semantic_path.read_text(encoding="utf-8"))
    semantic["classification"]["document_type"] = "enterprise_credit_report"
    semantic_path.write_text(json.dumps(semantic, ensure_ascii=False), encoding="utf-8")
    audit_path = community_path.parent / payload["files"]["dataset_audit_csv"]
    with audit_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    kept = [row for row in rows if not (row["record_id"] == "txn:001" and row["field_key"] == "amount")]
    with audit_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(kept)

    issues = validate_community_artifacts(community_path)

    assert "ds_transactions: 1 enterprise fields missing from audit CSV" in issues


def test_validator_accepts_reserved_reconciliation_audit_rows(tmp_path: Path) -> None:
    community_path = _write_bundle(tmp_path, "reconciliation_audit")
    payload = json.loads(community_path.read_text(encoding="utf-8"))
    audit_path = community_path.parent / payload["files"]["dataset_audit_csv"]
    with audit_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    rows.append(
        {
            **{key: "" for key in rows[0]},
            "dataset_id": "_audit_reconciliations",
            "record_id": "audit:credit_account_balance",
            "field_key": "difference",
            "value": "0.01",
            "raw": "0.01",
            "value_type": "decimal",
            "csv_escape_applied": "false",
        }
    )
    with audit_path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    assert validate_community_artifacts(community_path) == []


def test_payment_direction_cells_counts_gfm_data_rows_separately_from_headers() -> None:
    markdown = """| 收/支 | 金额 |
| --- | ---: |
| 收入 | 10.00 |
| 支出 | 3.00 |
| 不计收支 | 7.00 |
"""

    data_cells, header_cells = payment_direction_cells(markdown)

    assert data_cells == {"收入": 1, "支出": 1, "不计收支": 1}
    assert header_cells == {"收/支": 1}
