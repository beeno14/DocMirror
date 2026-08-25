# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Final Candidate-B contracts for numeric monthly status cells."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.community_plugin import (
    _apply_personal_detail_dataset_status,
)
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    collect_extraction_issues,
    make_issue,
)
from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
    PersonalDetailOCRCorrectionOverlay,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
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


def _exact_monthly_status_ref(
    *,
    grid_id: str,
    performance_month: str,
    column: int,
) -> dict[str, object]:
    return {
        "source": "sealed_native_physical_table_cell",
        "logical_page": 2,
        "source_page": 1,
        "table_id": "monthly-table:1",
        "row": 4,
        "column": column,
        "bbox": [10.0, 10.0, 20.0, 20.0],
        "geometry_scope": "cell",
        "field_name": "status",
        "grid_id": grid_id,
        "performance_month": performance_month,
        "evidence_ids": [f"native:monthly-table:4:{column}"],
        "binding": "source_monthly_field_cell",
        "binding_quality": "source_monthly_field_cell",
    }


def _unresolved_monthly_row(
    record_id: str,
    *,
    month: int = 1,
    source_refs: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    row = _monthly_row(record_id, status="unknown", overdue_amount=None)
    row["month"] = month
    if source_refs is not None:
        row["source_cell_refs"] = source_refs
    return row


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
    # Both detector grids describe the same account/month.  The emitted good
    # observation closes that identity, so the bad grid alias must not leave a
    # contradictory active omission/status issue.
    assert status_issues == []
    assert not any(
        issue.get("target_record_id") == "grid:good:2024-01"
        for issue in projected["extraction_issues"]
    )


def test_final_gate_reports_exact_source_bound_month_fields_and_keeps_aggregate() -> None:
    grid_id = "grid:exact"
    target = f"{grid_id}:2024-01"
    projected = project_personal_detail_datasets(
        {
            "repayment_records": [
                _unresolved_monthly_row(
                    target,
                    source_refs=[
                        _exact_monthly_status_ref(
                            grid_id=grid_id,
                            performance_month="2024-01",
                            column=1,
                        )
                    ],
                )
            ]
        }
    )

    assert projected.get("credit_account_monthly_performance", []) == []
    local = [
        issue
        for issue in projected.get("extraction_issues", [])
        if issue.get("issue_code")
        == "candidate_b_monthly_owned_grid_missing_field"
    ]
    target_ids = {issue["target_record_id"] for issue in local}
    assert len(target_ids) == 1
    assert next(iter(target_ids)).startswith("source_account_month:")
    assert {issue["field_name"] for issue in local} == {"performance_month"}
    assert sum(
        issue.get("issue_code") == "candidate_b_monthly_status_grid_unresolved"
        for issue in projected["extraction_issues"]
    ) == 1
    for issue in local:
        issue_evidence = [
            evidence
            for evidence in projected["extraction_issue_evidence"]
            if evidence.get("extraction_issue_id")
            == issue["extraction_issue_id"]
            and evidence.get("evidence_kind") == "observed"
        ]
        assert {
            evidence["evidence_path"]: evidence.get("string_value")
            for evidence in issue_evidence
        } == {
            "account_id": "account:1",
            "performance_month": "2024-01",
        }
        assert len(issue["source_refs"]) == 1
        ref = issue["source_refs"][0]
        assert ref["field_name"] == issue["field_name"]
        assert ref["source_field_name"] == "status"
        assert ref["source_page"] == 1
        assert ref["evidence_ids"] == ["native:monthly-table:4:1"]
        assert (ref["table_id"], ref["row"], ref["column"]) == (
            "monthly-table:1",
            4,
            1,
        )
        assert ref["source"] == "candidate_b_monthly_owned_grid_cell"
        assert ref["source_origin"] == "sealed_native_physical_table_cell"
        assert ref["binding"] == "source_account_month_identity"
        assert ref["binding_quality"] == "source_account_month_identity"
        assert ref["account_id"] == "account:1"


def test_final_gate_accepts_fresh_geometry_proof_and_counts_account_month_set() -> None:
    unresolved = _unresolved_monthly_row(
        "grid:fresh:2024-02",
        month=2,
        source_refs=[
            {
                "page": 8,
                "logical_page": 8,
                "source_page": 4,
                "source": "sealed_native_physical_table_cell",
                "table_id": "monthly-table:fresh",
                "coordinate_system": "pdf_points_top_left",
                "geometry_scope": "cell",
                "grid_id": "grid:fresh",
                "row": 4,
                "col": 2,
                "field_name": "status",
                "performance_month": "2024-02",
                "bbox": [120.0, 400.0, 145.0, 414.0],
                "evidence_ids": ["native:monthly-table:fresh:4:2"],
                "binding": "source_monthly_field_cell",
                "binding_quality": "source_monthly_field_cell",
                "geometry_provenance": {
                    "active_cell_geometry_exact": True,
                    "coordinate_system": "pdf_points_top_left",
                    "source": "source_table_geometry",
                },
            }
        ],
    )
    unresolved["_account_month_identity_proof"] = {
        "account_id": "account:1",
        "performance_month": "2024-02",
        "grid_id": "grid:fresh",
        "owner_basis": "canonical_account_segment",
        "account_anchor_exact": True,
        "printed_month_range_exact": True,
        "grid_geometry_exact": True,
        "unique_owner": True,
    }
    unresolved["_account_month_identity_proof_status"] = "exact"
    owner_unresolved = make_issue(
        category="ocr_structure_correction",
        issue_code="candidate_b_monthly_grid_owner_unresolved_field",
        message="A printed grid month has no unique account owner.",
        parser_stage="candidate_b_relationship_schema",
        target_dataset="repayment_records",
        target_record_id="grid:ambiguous:2024-03",
        field_name="performance_month",
        observed_value={
            "grid_id": "grid:ambiguous",
            "performance_month": "2024-03",
        },
    )

    projected = project_personal_detail_datasets(
        {
            "repayment_records": [
                _monthly_row(
                    "grid:emitted:2024-01",
                    status="N",
                    overdue_amount="0",
                ),
                unresolved,
            ],
            "personal_detail_extraction_issues": [owner_unresolved],
        }
    )

    local = [
        issue
        for issue in projected["extraction_issues"]
        if issue.get("issue_code")
        == "candidate_b_monthly_owned_grid_missing_field"
    ]
    assert {issue["field_name"] for issue in local} == {"performance_month"}
    assert len({issue["target_record_id"] for issue in local}) == 1
    assert all(
        issue["target_record_id"].startswith("source_account_month:")
        for issue in local
    )
    for issue in local:
        evidence = {
            row["evidence_path"]: row.get("string_value")
            for row in projected["extraction_issue_evidence"]
            if row["extraction_issue_id"] == issue["extraction_issue_id"]
            and row["evidence_kind"] == "observed"
        }
        assert evidence["account_id"] == "account:1"
        assert evidence["performance_month"] == "2024-02"
        assert set(evidence) == {"account_id", "performance_month"}
        assert len(issue["source_refs"]) == 1
        assert issue["source_refs"][0]["source"] == (
            "candidate_b_monthly_owned_grid_cell"
        )
    monthly_status = next(
        row
        for row in projected["dataset_status"]
        if row["dataset_name"] == "credit_account_monthly_performance"
    )
    assert monthly_status["observed_row_count"] == 1
    assert monthly_status["expected_row_count"] == 2


@pytest.mark.parametrize(
    ("owner_basis", "proof_status"),
    (
        ("page_proximity", "exact"),
        (
            "canonical_account_segment",
            "unproven_exact_anchor_range_geometry_owner",
        ),
    ),
)
def test_final_gate_rejects_forged_or_explicitly_unproven_identity(
    owner_basis: str,
    proof_status: str,
) -> None:
    row = _monthly_row(
        "grid:forged:2024-01",
        status="N",
        overdue_amount="0",
    )
    row["_account_month_identity_proof"] = {
        "account_id": "account:1",
        "performance_month": "2024-01",
        "grid_id": "grid:forged",
        "owner_basis": owner_basis,
        "account_anchor_exact": True,
        "printed_month_range_exact": True,
        "grid_geometry_exact": True,
        "unique_owner": True,
    }
    row["_account_month_identity_proof_status"] = proof_status

    projected = project_personal_detail_datasets({"repayment_records": [row]})

    assert projected.get("credit_account_monthly_performance", []) == []
    monthly_status = next(
        status
        for status in projected["dataset_status"]
        if status["dataset_name"] == "credit_account_monthly_performance"
    )
    assert monthly_status["observed_row_count"] == 0
    assert "expected_row_count" not in monthly_status


def test_monthly_expected_count_ignores_non_omission_issue_identity_mentions() -> None:
    projected = project_personal_detail_datasets(
        {
            "repayment_records": [
                _monthly_row(
                    "grid:emitted:2024-01",
                    status="N",
                    overdue_amount="0",
                )
            ],
            "personal_detail_extraction_issues": [
                make_issue(
                    category="ocr_cell_level_error",
                    issue_code="candidate_b_monthly_duplicate_conflict",
                    message="A conflicting detector replay mentioned another month.",
                    parser_stage="candidate_b_relationship_schema",
                    target_dataset="repayment_records",
                    target_record_id="grid:diagnostic:2024-02",
                    field_name="status_code",
                    observed_value={
                        "account_id": "account:1",
                        "grid_id": "grid:diagnostic",
                        "performance_month": "2024-02",
                    },
                    reason_codes=("duplicate_account_month",),
                )
            ],
        }
    )

    monthly_status = next(
        status
        for status in projected["dataset_status"]
        if status["dataset_name"] == "credit_account_monthly_performance"
    )
    assert monthly_status["observed_row_count"] == 1
    assert monthly_status["expected_row_count"] == 1


def test_final_gate_deduplicates_local_month_fields_and_drops_stale_emitted_claims() -> None:
    unresolved_grid = "grid:duplicate"
    unresolved_target = f"{unresolved_grid}:2024-01"
    exact_ref = _exact_monthly_status_ref(
        grid_id=unresolved_grid,
        performance_month="2024-01",
        column=1,
    )
    existing_local = make_issue(
        category="ocr_cell_level_error",
        issue_code="candidate_b_monthly_grid_contract_missing_field",
        message="already localized",
        parser_stage="candidate_b_relationship_schema",
        target_dataset="repayment_records",
        target_record_id=unresolved_target,
        field_name="status",
        observed_value={
            "grid_id": unresolved_grid,
            "performance_month": "2024-01",
        },
        source_refs=[exact_ref],
    )
    emitted_target = "grid:emitted:2024-01"
    stale = make_issue(
        category="ocr_cell_level_error",
        issue_code="candidate_b_monthly_grid_contract_missing_field",
        message="stale omitted field",
        parser_stage="candidate_b_relationship_schema",
        target_dataset="repayment_records",
        target_record_id=emitted_target,
        field_name="performance_month",
        observed_value={
            "grid_id": "grid:emitted",
            "performance_month": "2024-01",
        },
        source_refs=[
            _exact_monthly_status_ref(
                grid_id="grid:emitted",
                performance_month="2024-01",
                column=1,
            )
        ],
    )
    projected = project_personal_detail_datasets(
        {
            "repayment_records": [
                _unresolved_monthly_row(
                    unresolved_target,
                    source_refs=[exact_ref, dict(exact_ref)],
                ),
                _monthly_row(
                    emitted_target,
                    status="N",
                    overdue_amount="0",
                ),
            ],
            "personal_detail_extraction_issues": [existing_local, stale],
        }
    )

    local = [
        issue
        for issue in projected.get("extraction_issues", [])
        if issue.get("issue_code")
        in {
            "candidate_b_monthly_grid_contract_missing_field",
            "candidate_b_monthly_owned_grid_missing_field",
        }
    ]
    # The unresolved and emitted grids alias the same account/month.  Closure
    # is identity-based, so neither the prior grid-local claim nor a newly
    # generated account-month claim survives.
    assert local == []


@pytest.mark.parametrize(
    "missing",
    (
        "source_page",
        "evidence_ids",
        "table_id",
        "row",
        "column",
        "bbox",
        "grid_id",
        "performance_month",
        "binding",
        "binding_quality",
        "source",
    ),
)
def test_final_gate_refuses_local_claim_when_exact_provenance_is_incomplete(
    missing: str,
) -> None:
    grid_id = "grid:incomplete"
    target = f"{grid_id}:2024-01"
    source_ref = _exact_monthly_status_ref(
        grid_id=grid_id,
        performance_month="2024-01",
        column=1,
    )
    source_ref.pop(missing)
    projected = project_personal_detail_datasets(
        {
            "repayment_records": [
                _unresolved_monthly_row(target, source_refs=[source_ref])
            ]
        }
    )

    assert sum(
        issue.get("issue_code") == "candidate_b_monthly_status_grid_unresolved"
        for issue in projected["extraction_issues"]
    ) == 1
    assert not any(
        issue.get("issue_code")
        == "candidate_b_monthly_owned_grid_missing_field"
        for issue in projected["extraction_issues"]
    )


@pytest.mark.parametrize(
    ("field", "forged_value"),
    (
        ("grid_id", "grid:neighbor"),
        ("performance_month", "2024-02"),
        ("source", "unrelated_cell"),
    ),
)
def test_final_gate_refuses_local_claim_for_mismatched_cell_identity(
    field: str,
    forged_value: str,
) -> None:
    grid_id = "grid:exact-identity"
    target = f"{grid_id}:2024-01"
    source_ref = _exact_monthly_status_ref(
        grid_id=grid_id,
        performance_month="2024-01",
        column=1,
    )
    source_ref[field] = forged_value

    projected = project_personal_detail_datasets(
        {
            "repayment_records": [
                _unresolved_monthly_row(target, source_refs=[source_ref])
            ]
        }
    )

    assert not any(
        issue.get("issue_code")
        == "candidate_b_monthly_owned_grid_missing_field"
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


def test_zero_overdue_status_withholds_nonzero_amount_without_inference() -> None:
    rows = [
        _monthly_row("grid:n-conflict:2024-01", status="N", overdue_amount="10"),
        _monthly_row("grid:star-conflict:2024-01", status="*", overdue_amount="20"),
        _monthly_row("grid:clean:2024-01", status="N", overdue_amount="0"),
        _monthly_row("grid:numeric:2024-01", status="1", overdue_amount="4691"),
        _monthly_row("grid:missing:2024-01", status="N", overdue_amount=None),
    ]
    for row in rows:
        row["status_amount_semantics"] = "delinquent_amount"

    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                {
                    "record_id": "account:1",
                    "account_id": "account:1",
                    "sequence": 1,
                    "account_type": "non_revolving_loan",
                }
            ],
            "repayment_records": rows,
        }
    )
    by_id = {
        row["monthly_performance_id"]: row
        for row in projected["credit_account_monthly_performance"]
    }

    for record_id, observed in (
        ("grid:n-conflict:2024-01", "10"),
        ("grid:star-conflict:2024-01", "20"),
    ):
        row = by_id[record_id]
        assert row["status_code"] in {"N", "*"}
        assert row.get("status_amount") is None
        assert row.get("status_amount_semantics") is None
        assert row["canonical_raw"]["status_amount"] == observed

    assert by_id["grid:clean:2024-01"]["status_amount"] == "0"
    assert by_id["grid:numeric:2024-01"]["status_code"] == "1"
    assert by_id["grid:numeric:2024-01"]["status_amount"] == "4691"
    assert by_id["grid:missing:2024-01"].get("status_amount") is None

    amount_issues = [
        issue
        for issue in projected["extraction_issues"]
        if issue.get("field_name") == "status_amount"
    ]
    conflict_issues = [
        issue
        for issue in amount_issues
        if issue.get("issue_code")
        == "candidate_b_monthly_zero_status_amount_conflict"
    ]
    assert {issue["target_record_id"] for issue in conflict_issues} == {
        "grid:n-conflict:2024-01",
        "grid:star-conflict:2024-01",
    }
    assert {issue["observed_value"] for issue in conflict_issues} == {"10", "20"}
    conflict_issue_ids = {
        str(issue.get("extraction_issue_id") or "") for issue in conflict_issues
    }
    assert all(
        any(
            evidence.get("extraction_issue_id") == issue_id
            and evidence.get("evidence_kind") == "reason"
            and evidence.get("string_value") == "normalized_value_withheld"
            for evidence in projected["extraction_issue_evidence"]
        )
        for issue_id in conflict_issue_ids
    )
    assert all(
        issue.get("issue_code") != "candidate_b_monthly_status_amount_unresolved"
        for issue in amount_issues
        if issue.get("target_record_id")
        in {"grid:n-conflict:2024-01", "grid:star-conflict:2024-01"}
    )
    missing = next(
        issue
        for issue in amount_issues
        if issue.get("target_record_id") == "grid:missing:2024-01"
    )
    assert missing["issue_code"] == "candidate_b_monthly_status_amount_unresolved"
    assert not any(
        issue.get("target_record_id")
        in {"grid:clean:2024-01", "grid:numeric:2024-01"}
        for issue in amount_issues
    )


def test_monthly_dataset_status_ignores_unlocalized_structural_grid_count() -> None:
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                {
                    "record_id": "account:1",
                    "account_id": "account:1",
                    "account_type": "credit_card",
                }
            ],
            "repayment_records": [
                _monthly_row(
                    "grid:1:2024-01",
                    status="N",
                    overdue_amount="0",
                )
            ],
            "personal_detail_extraction_issues": [
                make_issue(
                    category="ocr_structure_correction",
                    issue_code="canonical_monthly_reconstruction_incomplete",
                    message="Two source-proven monthly positions were not materialized.",
                    parser_stage="canonical_monthly_grid_materialization",
                    target_dataset="repayment_records",
                    observed_value={"canonical_row_count": 1},
                    candidate_value={
                        "source_structure_row_count": 3,
                        "unreconciled_source_position_count": 2,
                        "account_month_expected_row_count": None,
                        "localization_status": "pending_unique_account_owner_reconciliation",
                    },
                    reason_codes=(
                        "source_structure_is_audit_only",
                        "raw_grid_positions_not_a_population_denominator",
                        "dataset_incomplete",
                    ),
                )
            ],
        }
    )

    monthly_status = next(
        row
        for row in projected["dataset_status"]
        if row["dataset_name"] == "credit_account_monthly_performance"
    )
    assert monthly_status["observed_row_count"] == 1
    assert monthly_status["expected_row_count"] == 1


def test_monthly_expected_count_does_not_double_count_materialized_withheld_rows() -> None:
    unresolved = _monthly_row(
        "grid:1:2024-02",
        status="unknown",
        overdue_amount=None,
    )
    unresolved["month"] = 2
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                {
                    "record_id": "account:1",
                    "account_id": "account:1",
                    "account_type": "credit_card",
                }
            ],
            "repayment_records": [
                _monthly_row(
                    "grid:1:2024-01",
                    status="N",
                    overdue_amount="0",
                ),
                unresolved,
            ],
            "personal_detail_extraction_issues": [
                make_issue(
                    category="ocr_structure_correction",
                    issue_code="canonical_monthly_reconstruction_incomplete",
                    message="One source-proven monthly position was not materialized.",
                    parser_stage="canonical_monthly_grid_materialization",
                    target_dataset="repayment_records",
                    observed_value={"canonical_row_count": 2},
                    candidate_value={
                        "source_structure_row_count": 3,
                        "unreconciled_source_position_count": 1,
                        "account_month_expected_row_count": None,
                        "localization_status": "pending_unique_account_owner_reconciliation",
                    },
                    reason_codes=(
                        "source_structure_is_audit_only",
                        "raw_grid_positions_not_a_population_denominator",
                        "dataset_incomplete",
                    ),
                )
            ],
        }
    )

    monthly_status = next(
        row
        for row in projected["dataset_status"]
        if row["dataset_name"] == "credit_account_monthly_performance"
    )
    assert len(projected["credit_account_monthly_performance"]) == 1
    assert monthly_status["observed_row_count"] == 1
    assert monthly_status["expected_row_count"] == 1
    assert monthly_status["expected_row_count"] - monthly_status["observed_row_count"] == 0


def test_monthly_dataset_status_does_not_turn_account_gap_count_into_months() -> None:
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                {
                    "record_id": "account:1",
                    "account_id": "account:1",
                    "account_type": "credit_card",
                }
            ],
            "repayment_records": [
                _monthly_row(
                    "grid:1:2024-01",
                    status="N",
                    overdue_amount="0",
                )
            ],
            "personal_detail_extraction_issues": [
                make_issue(
                    category="schema_incompleteness",
                    issue_code="monthly_population_incomplete_from_account_gap",
                    message="The canonical grid plane contains three source positions.",
                    parser_stage="candidate_b_account_monthly_population",
                    target_dataset="repayment_records",
                    observed_value={"canonical_grid_row_count": 3},
                    candidate_value={
                        "missing_account_category_sequences": {"credit_card": [2]}
                    },
                    reason_codes=("dataset_incomplete",),
                )
            ],
        }
    )

    monthly_status = next(
        row
        for row in projected["dataset_status"]
        if row["dataset_name"] == "credit_account_monthly_performance"
    )
    assert monthly_status["observed_row_count"] == 1
    assert monthly_status["expected_row_count"] == 1


def test_monthly_expected_count_excludes_owner_unresolved_grid_positions() -> None:
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                {
                    "record_id": "account:1",
                    "account_id": "account:1",
                    "account_type": "credit_card",
                }
            ],
            "repayment_records": [
                _monthly_row(
                    "grid:owned:2024-01",
                    status="N",
                    overdue_amount="0",
                )
            ],
            "personal_detail_extraction_issues": [
                make_issue(
                    category="ocr_structure_correction",
                    issue_code="candidate_b_monthly_status_grid_unresolved",
                    message="Two owned monthly positions were withheld.",
                    parser_stage="candidate_b_monthly_status_grid",
                    target_dataset="repayment_records",
                    observed_value={
                        "grid_id": "grid:owned",
                        "withheld_month_count": 2,
                    },
                    candidate_value={"emitted_month_count_for_grid": 1},
                    reason_codes=("dataset_incomplete",),
                ),
                make_issue(
                    category="ocr_structure_correction",
                    issue_code="candidate_b_monthly_grid_owner_unresolved",
                    message="Three exact grid/month positions have no account owner.",
                    parser_stage="candidate_b_monthly_linkage",
                    target_dataset="repayment_records",
                    observed_value={
                        "grid_id": "grid:unowned",
                        "observed_candidate_count": 3,
                    },
                    candidate_value={"expected_month_count": 3},
                    reason_codes=("relation_withheld",),
                ),
            ],
        }
    )

    monthly_status = next(
        row
        for row in projected["dataset_status"]
        if row["dataset_name"] == "credit_account_monthly_performance"
    )
    assert monthly_status["observed_row_count"] == 1
    assert monthly_status["expected_row_count"] == 1


def test_zero_row_monthly_dataset_is_published_from_status_and_owner_issues() -> None:
    projected = project_personal_detail_datasets(
        {
            "personal_detail_extraction_issues": [
                make_issue(
                    category="ocr_structure_correction",
                    issue_code="candidate_b_monthly_status_grid_unresolved",
                    message="Two owned monthly positions were withheld.",
                    parser_stage="candidate_b_monthly_status_grid",
                    target_dataset="repayment_records",
                    observed_value={
                        "grid_id": "grid:owned",
                        "withheld_month_count": 2,
                    },
                    reason_codes=("dataset_incomplete",),
                ),
                make_issue(
                    category="ocr_structure_correction",
                    issue_code="candidate_b_monthly_grid_owner_unresolved",
                    message="Three exact grid/month positions have no account owner.",
                    parser_stage="candidate_b_monthly_linkage",
                    target_dataset="repayment_records",
                    observed_value={
                        "grid_id": "grid:unowned",
                        "observed_candidate_count": 3,
                    },
                    candidate_value={"expected_month_count": 3},
                    reason_codes=("relation_withheld",),
                ),
            ]
        }
    )

    assert projected.get("credit_account_monthly_performance", []) == []
    monthly_status = next(
        row
        for row in projected["dataset_status"]
        if row["dataset_name"] == "credit_account_monthly_performance"
    )
    assert monthly_status["observed_row_count"] == 0
    assert "expected_row_count" not in monthly_status


def test_monthly_owner_unresolved_grid_count_is_deduplicated_and_not_conflict_summed() -> None:
    def owner_issue(count: int, *, message: str) -> dict:
        return make_issue(
            category="ocr_structure_correction",
            issue_code="candidate_b_monthly_grid_owner_unresolved",
            message=message,
            parser_stage="candidate_b_monthly_linkage",
            target_dataset="repayment_records",
            observed_value={
                "grid_id": "grid:unowned",
                "observed_candidate_count": count,
            },
            candidate_value={"expected_month_count": count},
            reason_codes=("relation_withheld",),
        )

    duplicate = project_personal_detail_datasets(
        {
            "personal_detail_extraction_issues": [
                owner_issue(3, message="First exact owner failure."),
                owner_issue(3, message="Duplicate exact owner failure."),
            ]
        }
    )
    duplicate_status = next(
        row
        for row in duplicate["dataset_status"]
        if row["dataset_name"] == "credit_account_monthly_performance"
    )
    assert "expected_row_count" not in duplicate_status

    conflicting = project_personal_detail_datasets(
        {
            "personal_detail_extraction_issues": [
                owner_issue(3, message="First conflicting owner failure."),
                owner_issue(4, message="Second conflicting owner failure."),
            ]
        }
    )
    conflicting_status = next(
        row
        for row in conflicting["dataset_status"]
        if row["dataset_name"] == "credit_account_monthly_performance"
    )
    assert conflicting_status.get("expected_row_count", 0) == 0


def test_monthly_owner_unresolved_count_is_not_added_without_observed_agreement() -> None:
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                {
                    "record_id": "account:1",
                    "account_id": "account:1",
                    "account_type": "credit_card",
                }
            ],
            "repayment_records": [
                _monthly_row(
                    "grid:owned:2024-01",
                    status="N",
                    overdue_amount="0",
                )
            ],
            "personal_detail_extraction_issues": [
                make_issue(
                    category="ocr_structure_correction",
                    issue_code="candidate_b_monthly_grid_owner_unresolved",
                    message="The candidate and observed grid populations disagree.",
                    parser_stage="candidate_b_monthly_linkage",
                    target_dataset="repayment_records",
                    observed_value={
                        "grid_id": "grid:unowned",
                        "observed_candidate_count": 2,
                    },
                    candidate_value={"expected_month_count": 3},
                    reason_codes=("relation_withheld",),
                )
            ]
        }
    )
    assert len(projected["credit_account_monthly_performance"]) == 1
    monthly_status = next(
        row
        for row in projected["dataset_status"]
        if row["dataset_name"] == "credit_account_monthly_performance"
    )
    assert monthly_status["observed_row_count"] == 1
    assert monthly_status["expected_row_count"] == 1


def test_monthly_status_grid_count_is_deduplicated_and_not_conflict_summed() -> None:
    def status_issue(count: int, *, message: str) -> dict:
        return make_issue(
            category="ocr_structure_correction",
            issue_code="candidate_b_monthly_status_grid_unresolved",
            message=message,
            parser_stage="candidate_b_monthly_status_grid",
            target_dataset="repayment_records",
            observed_value={
                "grid_id": "grid:status-unresolved",
                "withheld_month_count": count,
            },
            reason_codes=("dataset_incomplete",),
        )

    duplicate = project_personal_detail_datasets(
        {
            "personal_detail_extraction_issues": [
                status_issue(3, message="First exact status-grid failure."),
                status_issue(3, message="Duplicate exact status-grid failure."),
            ]
        }
    )
    duplicate_status = next(
        row
        for row in duplicate["dataset_status"]
        if row["dataset_name"] == "credit_account_monthly_performance"
    )
    assert "expected_row_count" not in duplicate_status

    conflicting = project_personal_detail_datasets(
        {
            "personal_detail_extraction_issues": [
                status_issue(3, message="First conflicting status-grid failure."),
                status_issue(4, message="Second conflicting status-grid failure."),
            ]
        }
    )
    conflicting_status = next(
        row
        for row in conflicting["dataset_status"]
        if row["dataset_name"] == "credit_account_monthly_performance"
    )
    assert conflicting_status.get("expected_row_count", 0) == 0


def test_count_only_empty_monthly_status_does_not_define_canonical_denominator() -> None:
    projected = project_personal_detail_datasets(
        {
            "personal_detail_dataset_status": [
                {
                    "record_id": "dataset_status:repayment_records",
                    "dataset_name": "repayment_records",
                    "applicability": "applicable",
                    "presence_status": "partial",
                    "observed_row_count": 0,
                    "expected_row_count": 176,
                }
            ]
        }
    )

    assert projected.get("credit_account_monthly_performance", []) == []
    monthly_status = next(
        row
        for row in projected["dataset_status"]
        if row["dataset_name"] == "credit_account_monthly_performance"
    )
    assert monthly_status["observed_row_count"] == 0
    assert "expected_row_count" not in monthly_status


def test_authenticated_monthly_closure_drives_status_and_community_completeness() -> None:
    def exact_row(
        grid_id: str,
        performance_month: str,
        *,
        status: str,
        overdue_amount: object,
    ) -> dict[str, object]:
        row = _monthly_row(
            f"{grid_id}:{performance_month}",
            status=status,
            overdue_amount=overdue_amount,
        )
        row["month"] = int(performance_month[-2:])
        row["_account_month_identity_proof"] = {
            "account_id": "account:1",
            "performance_month": performance_month,
            "grid_id": grid_id,
            "owner_basis": "canonical_account_segment",
            "account_anchor_exact": True,
            "printed_month_range_exact": True,
            "grid_geometry_exact": True,
            "unique_owner": True,
        }
        row["_account_month_identity_proof_status"] = "exact"
        return row

    prepared = prepare_personal_detail_source_collections(
        {
            "facts": {},
            "datasets": {
                "repayment_records": [
                    exact_row(
                        "grid:published",
                        "2024-01",
                        status="N",
                        overdue_amount="0",
                    ),
                    exact_row(
                        "grid:withheld",
                        "2024-02",
                        status="unknown",
                        overdue_amount=None,
                    ),
                ]
            },
        }
    )
    projected = project_personal_detail_datasets(prepared["datasets"])

    assert "_personal_detail_account_month_closure_proof" not in projected
    assert len(projected["credit_account_monthly_performance"]) == 1
    monthly_status = next(
        row
        for row in projected["dataset_status"]
        if row["dataset_name"] == "credit_account_monthly_performance"
    )
    assert monthly_status["observed_row_count"] == 1
    assert monthly_status["expected_row_count"] == 2

    payload = {
        "datasets": [
            {
                "name": "credit_account_monthly_performance",
                "row_count": 1,
                "rows": projected["credit_account_monthly_performance"],
                "completeness": {
                    "expected_row_count": 1,
                    "emitted_row_count": 1,
                    "omitted_row_count": 0,
                    "verified": True,
                },
            },
            {"name": "dataset_status", "rows": projected["dataset_status"]},
        ]
    }
    _apply_personal_detail_dataset_status(payload)

    completeness = payload["datasets"][0]["completeness"]
    assert completeness["expected_row_count"] == 2
    assert completeness["emitted_row_count"] == 1
    assert completeness["omitted_row_count"] == 1
    assert completeness["verified"] is False


def test_authenticated_nonzero_monthly_closure_publishes_empty_dataset() -> None:
    row = _monthly_row(
        "grid:withheld:2024-01",
        status="unknown",
        overdue_amount=None,
    )
    row["_account_month_identity_proof"] = {
        "account_id": "account:1",
        "performance_month": "2024-01",
        "grid_id": "grid:withheld",
        "owner_basis": "canonical_account_segment",
        "account_anchor_exact": True,
        "printed_month_range_exact": True,
        "grid_geometry_exact": True,
        "unique_owner": True,
    }
    row["_account_month_identity_proof_status"] = "exact"
    prepared = prepare_personal_detail_source_collections(
        {"facts": {}, "datasets": {"repayment_records": [row]}}
    )

    projected = project_personal_detail_datasets(prepared["datasets"])

    assert projected["credit_account_monthly_performance"] == []
    monthly_status = next(
        status
        for status in projected["dataset_status"]
        if status["dataset_name"] == "credit_account_monthly_performance"
    )
    assert monthly_status["observed_row_count"] == 0
    assert monthly_status["expected_row_count"] == 1


def test_forged_monthly_closure_proof_cannot_authenticate_status_count() -> None:
    projected = project_personal_detail_datasets(
        {
            "_personal_detail_account_month_closure_proof": [
                {
                    "record_id": "personal_detail_account_month_closure_proof:1",
                    "schema": "docmirror.pboc.account_month_closure_proof.v1",
                    "identity_fields": ["account_id", "performance_month"],
                    "proof_basis": "exact_source_account_month_identity_set",
                    "expected_identity_count": 176,
                    "identity_sha256": "0" * 64,
                }
            ],
            "personal_detail_dataset_status": [
                {
                    "record_id": "personal_detail_dataset_status:repayment_records",
                    "dataset_status_id": "personal_detail_dataset_status:repayment_records",
                    "dataset_name": "repayment_records",
                    "applicability": "applicable",
                    "presence_status": "partial",
                    "observed_row_count": 0,
                    "expected_row_count": 176,
                    "reason": "forged_count_only_status",
                }
            ],
        }
    )

    monthly_status = next(
        status
        for status in projected["dataset_status"]
        if status["dataset_name"] == "credit_account_monthly_performance"
    )
    assert monthly_status["observed_row_count"] == 0
    assert "expected_row_count" not in monthly_status
