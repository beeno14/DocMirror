# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Final Candidate-B contracts for numeric monthly status cells."""

from __future__ import annotations

from types import SimpleNamespace

from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    collect_extraction_issues,
)
from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
    PersonalDetailOCRCorrectionOverlay,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.repayment_grid import (
    extract_credit_repayment_records,
    records_from_micro_grid_dict,
)


def _monthly_row(
    record_id: str,
    *,
    status_key: str = "status",
    status: str,
    overdue_amount: object,
) -> dict[str, object]:
    return {
        "record_id": record_id,
        "repayment_id": record_id,
        "grid_id": record_id.rsplit(":", 1)[0],
        "account_id": "account:1",
        "year": 2024,
        "month": 1,
        status_key: status,
        "overdue_amount": overdue_amount,
        "source_cell_refs": [
            {
                "logical_page": 2,
                "bbox": [10, 10, 20, 20],
                "geometry_scope": "cell",
                "field_name": status_key,
            },
            {
                "logical_page": 2,
                "bbox": [20, 10, 30, 20],
                "geometry_scope": "cell",
                "field_name": "overdue_amount",
            },
        ],
    }


def test_final_overlay_withholds_numeric_status_without_positive_amount() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    source = {
        "repayment_records": [
            _monthly_row("grid:missing:2024-01", status="2", overdue_amount=None),
            {
                "record_id": "grid:zero:2024-01",
                "normalized": _monthly_row(
                    "grid:zero:2024-01",
                    status_key="status_code",
                    status="3",
                    overdue_amount="0",
                ),
                "canonical_raw": {},
            },
            _monthly_row("grid:negative:2024-01", status="4", overdue_amount="-1"),
            _monthly_row(
                "grid:invalid:2024-01",
                status="7",
                overdue_amount="not-an-amount",
            ),
            _monthly_row("grid:positive:2024-01", status="5", overdue_amount="12.50"),
        ],
        # The gate is deliberately confined to personal-detail repayment
        # records; a similarly shaped neighboring dataset is untouched.
        "postpaid_payment_history": [
            {"record_id": "postpaid:1", "status": "2", "overdue_amount": None}
        ],
    }

    corrected = overlay.correct_business_candidates(
        source,
        stage="candidate_b_final_validation",
    )
    rows = corrected["repayment_records"]

    assert rows[0]["status"] == "unknown"
    assert rows[0]["canonical_raw"]["status"] == "2"
    assert rows[1]["normalized"]["status_code"] == "unknown"
    assert rows[1]["canonical_raw"]["status_code"] == "3"
    assert rows[2]["status"] == "unknown"
    assert rows[2]["overdue_amount"] is None
    assert rows[2]["canonical_raw"] == {"status": "4", "overdue_amount": "-1"}
    assert rows[3]["status"] == "unknown"
    assert rows[3]["overdue_amount"] is None
    assert rows[3]["canonical_raw"] == {
        "status": "7",
        "overdue_amount": "not-an-amount",
    }
    assert rows[4]["status"] == "5"
    assert rows[4]["overdue_amount"] == "12.50"
    assert corrected["postpaid_payment_history"][0]["status"] == "2"

    numeric_statuses = {"1", "2", "3", "4", "5", "6", "7"}
    for row in rows:
        values = row.get("normalized") if isinstance(row.get("normalized"), dict) else row
        effective_status = str(values.get("status_code") or values.get("status") or "")
        if effective_status in numeric_statuses:
            assert float(str(values["overdue_amount"])) > 0

    anomalies = overlay.audit()["cell_anomalies"]
    status_anomalies = [item for item in anomalies if item["field_name"] == "status_code"]
    assert {item["record_id"] for item in status_anomalies} == {
        "grid:missing:2024-01",
        "grid:zero:2024-01",
        "grid:negative:2024-01",
        "grid:invalid:2024-01",
    }
    assert all(item["normalized_value_withheld"] for item in status_anomalies)
    assert not any(
        item["record_id"] == "grid:positive:2024-01" for item in anomalies
    )
    assert not any(
        item["record_id"] == "grid:zero:2024-01"
        and item["field_name"] == "overdue_amount"
        for item in anomalies
    )


def test_community_projection_never_publishes_unpaired_numeric_status() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    corrected = overlay.correct_business_candidates(
        {
            "repayment_records": [
                _monthly_row("grid:bad:2024-01", status="2", overdue_amount=None),
                _monthly_row("grid:good:2024-01", status="6", overdue_amount="125"),
            ]
        },
        stage="candidate_b_final_validation",
    )
    issue_context = SimpleNamespace(
        _personal_detail_extraction_issues=[],
        ocr_correction_audit=overlay.audit,
    )
    corrected["personal_detail_extraction_issues"] = collect_extraction_issues(
        issue_context
    )

    projected = project_personal_detail_datasets(corrected)
    monthly_rows = projected["credit_account_monthly_performance"]

    assert len(monthly_rows) == 1
    assert monthly_rows[0]["monthly_performance_id"] == "grid:good:2024-01"
    assert monthly_rows[0]["status_code"] == "6"
    assert monthly_rows[0]["status_amount"] == "125"
    assert "review" not in monthly_rows[0]
    assert not any(
        row.get("status_code") in {"1", "2", "3", "4", "5", "6", "7"}
        and float(str(row.get("status_amount") or 0)) <= 0
        for row in monthly_rows
    )

    status_issues = [
        issue
        for issue in projected["extraction_issues"]
        if issue.get("field_name") == "status_code"
    ]
    assert len(status_issues) == 1
    assert status_issues[0]["issue_code"] == "candidate_b_monthly_status_grid_unresolved"
    assert status_issues[0]["target_dataset"] == "credit_account_monthly_performance"
    assert not any(
        issue.get("target_record_id") == "grid:good:2024-01"
        for issue in projected["extraction_issues"]
    )


def test_candidate_b_amount_pairing_reaches_community_as_zero_or_field_issue() -> None:
    lines = [
        {
            "content": "2024年01月-2024年02月的还款记录",
            "bbox": [280.0, 194.0, 510.0, 217.0],
            "confidence": 1.0,
        },
        {
            "content": "1 2 3 4 5 6 7 8 9 10 11 12",
            "bbox": [130.0, 222.0, 730.0, 241.0],
            "confidence": 1.0,
        },
        {"content": "C", "bbox": [145.0, 249.0, 165.0, 274.0], "confidence": 1.0},
        {"content": "C", "bbox": [195.0, 249.0, 215.0, 274.0], "confidence": 1.0},
        {"content": "2024", "bbox": [75.0, 270.0, 112.0, 288.0], "confidence": 1.0},
        # January has an explicit printed zero. February's amount cell is blank.
        {"content": "0", "bbox": [148.0, 270.0, 162.0, 288.0], "confidence": 1.0},
    ]
    extracted = extract_credit_repayment_records(
        lines,
        page=4,
        page_width=834,
        page_height=600,
        enable_candidate_b_amount_pairing=True,
    )
    rows = records_from_micro_grid_dict(
        extracted["micro_grid"],
        accept_exact_row_numeric_status=True,
    )
    for row in rows:
        row["record_id"] = row["repayment_id"]
        row["account_id"] = "account:1"

    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    corrected = overlay.correct_business_candidates(
        {"repayment_records": rows},
        stage="candidate_b_final_validation",
    )
    issue_context = SimpleNamespace(
        _personal_detail_extraction_issues=[],
        ocr_correction_audit=overlay.audit,
    )
    corrected["personal_detail_extraction_issues"] = collect_extraction_issues(
        issue_context
    )
    projected = project_personal_detail_datasets(corrected)
    by_month = {
        row["performance_month"]: row
        for row in projected["credit_account_monthly_performance"]
    }

    assert by_month["2024-01"]["status_code"] == "C"
    assert by_month["2024-01"]["status_amount"] == "0"
    assert "review" not in by_month["2024-01"]
    assert by_month["2024-02"]["status_code"] == "C"
    assert "status_amount" not in by_month["2024-02"]
    amount_issues = [
        issue
        for issue in projected["extraction_issues"]
        if issue.get("field_name") == "status_amount"
    ]
    assert len(amount_issues) == 1
    assert amount_issues[0]["target_record_id"].endswith(":2024-02")
    assert not any(
        issue.get("target_record_id", "").endswith(":2024-01")
        for issue in projected["extraction_issues"]
    )
