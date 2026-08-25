# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""PBOC-native personal detailed credit-report v2 tests."""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from docmirror.models.entities.parse_result import DocumentEntities, PageContent, ParseResult
from docmirror.models.schemas.registry import (
    _personal_detail_invariant_errors,
    get_projection_schema,
    validate_projection_payload,
)
from docmirror.models.sealed import seal_parse_result
from docmirror.output.community_bundle import project_community_bundle
from docmirror.plugins.credit_report.community_plugin import (
    _apply_personal_detail_dataset_status,
    _compact_personal_detail_public_projection,
    _CreditReportCommunityBundle,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_extraction import (
    _extract_header_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_parser import (
    PBOCPersonalDetailNativeParser,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    _CANONICAL_MONTHLY_STATUS_CODES,
    PBOC_DATASET_ORDER,
    _canonical_field_name,
    personal_detail_data_dictionary,
    personal_detail_semantic_extensions,
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
)
from docmirror.plugins.credit_report.projection import _records


def test_monthly_status_contract_allows_field_local_null_with_required_identity() -> None:
    schema = json.loads(
        get_projection_schema("personal_credit_report_detailed").path.read_text(
            encoding="utf-8"
        )
    )
    dataset_rules = schema["$defs"]["canonicalDataset"]["allOf"]
    monthly_rule = next(
        rule
        for rule in dataset_rules
        if rule.get("if", {}).get("properties", {}).get("name", {}).get("const")
        == "credit_account_monthly_performance"
    )
    normalized_contract = monthly_rule["then"]["properties"]["rows"]["items"][
        "properties"
    ]["normalized"]

    assert "status_code" in normalized_contract["required"]
    assert None in normalized_contract["properties"]["status_code"]["enum"]


def _record(record_id: str, **values: Any) -> dict[str, Any]:
    return {"record_id": record_id, **values}


def _responsibility_invariant_payload(*rows: dict[str, Any]) -> dict[str, Any]:
    return {
        "datasets": [
            {"name": name, "row_count": 0, "rows": []}
            for name in ("report_metadata", "report_query", "subject_profile")
        ]
        + [
            {
                "name": "repayment_responsibilities",
                "row_count": len(rows),
                "rows": [
                    {"record_id": f"responsibility:{index}", "normalized": row}
                    for index, row in enumerate(rows, start=1)
                ],
            }
        ]
    }


def test_repayment_responsibility_source_label_controls_status_alternative() -> None:
    payload = _responsibility_invariant_payload(
        {
            "related_party_category": "organization",
            "source_status_value": "0",
            "overdue_months": 0,
            "repayment_status_code": None,
        },
        {
            "related_party_category": "organization",
            "source_status_value": "N",
            "overdue_months": None,
            "repayment_status_code": "N",
        },
        {
            "related_party_category": "person",
            "source_status_value": None,
            "overdue_months": None,
            "repayment_status_code": None,
        },
    )

    assert _personal_detail_invariant_errors(payload) == ()


def test_repayment_responsibility_status_alternatives_reject_loss_or_collision() -> None:
    lost_source_value = _responsibility_invariant_payload(
        {
            "related_party_category": "organization",
            "source_status_value": "0",
            "overdue_months": None,
            "repayment_status_code": None,
        }
    )
    conflicting_fields = _responsibility_invariant_payload(
        {
            "related_party_category": "person",
            "source_status_value": "N",
            "overdue_months": 0,
            "repayment_status_code": "N",
        }
    )

    assert _personal_detail_invariant_errors(lost_source_value) == (
        "repayment_responsibilities: source status requires separated overdue months or status",
    )
    assert _personal_detail_invariant_errors(conflicting_fields) == (
        "repayment_responsibilities: overdue months and repayment status are mutually exclusive",
    )


def test_monthly_repayment_business_enum_matches_non_null_json_values() -> None:
    expected = set(_CANONICAL_MONTHLY_STATUS_CODES) | {"unknown"}
    dictionary_codes = set(
        personal_detail_data_dictionary()["enums"]["repayment_status_code"]
    )
    schema_path = (
        Path(__file__).parents[2]
        / "docmirror"
        / "configs"
        / "schemas"
        / "personal_credit_report_detailed.schema.json"
    )
    json_schema = json.loads(schema_path.read_text(encoding="utf-8"))
    monthly_branch = next(
        branch
        for branch in json_schema["$defs"]["canonicalDataset"]["allOf"]
        if branch.get("if", {})
        .get("properties", {})
        .get("name", {})
        .get("const")
        == "credit_account_monthly_performance"
    )
    schema_codes = set(
        monthly_branch["then"]["properties"]["rows"]["items"]["properties"]
        ["normalized"]["properties"]["status_code"]["enum"]
    )

    assert dictionary_codes == expected
    assert schema_codes - {None} == expected
    assert None in schema_codes


def test_v2_business_rows_quarantine_internal_ocr_metadata() -> None:
    source = {
        "repayment_records": [
            {
                "record_id": "monthly:review",
                "normalized": {
                    "repayment_id": "monthly:review",
                    "account_id": "account:1",
                    "year": 2025,
                    "month": 1,
                    "status": "2",
                    "extraction_status": "review",
                    "audit": {"reason": "status_geometry_reused_across_years"},
                    "raw_status": "2",
                    "recognition_source": "canonical_row_sequence",
                    "status_bbox": [1, 2, 3, 4],
                    "source_refs_by_field": {"status": [{"logical_page": 2}]},
                    "page": 2,
                    "source_page": 1,
                },
                "raw": {
                    "status": "2",
                    "audit": {"reason": "status_geometry_reused_across_years"},
                    "status_bbox": [1, 2, 3, 4],
                    "source_refs_by_field": {"status": [{"logical_page": 2}]},
                },
                "canonical_raw": {
                    "status": "2",
                    "recognition_source": "canonical_row_sequence",
                    "status_bbox": [1, 2, 3, 4],
                    "source_refs_by_field": {"status": [{"logical_page": 2}]},
                },
                "source_refs_by_field": {"status": [{"logical_page": 2}]},
            }
        ]
    }

    row = project_personal_detail_datasets(source)["credit_account_monthly_performance"][0]

    assert row["normalized"]["status_code"] == "2"
    assert row["normalized"]["extraction_status"] == "review"
    assert row["review"] == {
        "status": "requires_review",
        "extraction_status": "review",
        "raw_status": "2",
        "recognition_source": "canonical_row_sequence",
        "diagnostics": {"reason": "status_geometry_reused_across_years"},
    }
    for snapshot in ("normalized", "raw", "canonical_raw"):
        assert not (
            set(row[snapshot])
            & {
                "audit",
                "recognition_source",
                "status_bbox",
                "source_refs_by_field",
            }
        )
    assert "source_refs_by_field" not in row


def test_v2_monthly_gate_keeps_exact_status_issues_and_aggregates_grid() -> None:
    projected = project_personal_detail_datasets(
        {
            "repayment_records": [
                _record(
                    "grid:1:2024-01",
                    repayment_id="grid:1:2024-01",
                    grid_id="grid:1",
                    account_id="account:1",
                    year=2024,
                    month=1,
                    status="unknown",
                ),
                _record(
                    "grid:1:2024-02",
                    repayment_id="grid:1:2024-02",
                    grid_id="grid:1",
                    account_id="account:1",
                    year=2024,
                    month=2,
                    status="?",
                ),
                _record(
                    "grid:1:2024-01:duplicate-detector-candidate",
                    repayment_id="grid:1:2024-01:duplicate-detector-candidate",
                    grid_id="grid:1",
                    account_id="account:1",
                    year=2024,
                    month=1,
                    status="unknown",
                ),
                _record(
                    "grid:1:2024-03",
                    repayment_id="grid:1:2024-03",
                    grid_id="grid:1",
                    account_id="account:1",
                    year=2024,
                    month=3,
                    status="N",
                ),
            ],
            "personal_detail_extraction_issues": [
                _record(
                    "issue:jan",
                    extraction_issue_id="issue:jan",
                    category="ocr_cell_level_error",
                    issue_code="pboc_cell_contract_unresolved",
                    severity="warning",
                    status="requires_review",
                    target_dataset="repayment_records",
                    target_record_id="grid:1:2024-01",
                    field_name="status",
                    observed_value="unknown",
                ),
                _record(
                    "issue:feb",
                    extraction_issue_id="issue:feb",
                    category="ocr_cell_level_error",
                    issue_code="pboc_cell_contract_unresolved",
                    severity="warning",
                    status="requires_review",
                    target_dataset="repayment_records",
                    target_record_id="grid:1:2024-02",
                    field_name="status",
                    observed_value="?",
                ),
                _record(
                    "issue:jan:canonical-alias-duplicate",
                    extraction_issue_id="issue:jan:canonical-alias-duplicate",
                    category="ocr_cell_level_error",
                    issue_code="pboc_cell_contract_unresolved",
                    severity="warning",
                    status="requires_review",
                    target_dataset="repayment_records",
                    target_record_id="grid:1:2024-01",
                    field_name="status_code",
                    observed_value="unknown",
                ),
            ],
        }
    )

    rows = projected["credit_account_monthly_performance"]
    assert [row["status_code"] for row in rows] == ["N"]
    issues = [
        row
        for row in projected["extraction_issues"]
        if row.get("issue_code") == "candidate_b_monthly_status_grid_unresolved"
    ]
    assert len(issues) == 1
    assert issues[0]["target_dataset"] == "credit_account_monthly_performance"
    assert issues[0]["field_name"] == "status_code"
    assert "target_record_id" not in issues[0]
    local_status_issues = [
        row
        for row in projected["extraction_issues"]
        if row.get("issue_code") == "pboc_cell_contract_unresolved"
    ]
    assert len(local_status_issues) == 2
    assert {
        (
            row.get("target_record_id"),
            row.get("field_name"),
            row.get("status"),
        )
        for row in local_status_issues
    } == {
        ("grid:1:2024-01", "status_code", "requires_review"),
        ("grid:1:2024-02", "status_code", "requires_review"),
    }
    assert {
        (
            row.get("business_record_id"),
            row.get("field_name"),
            row.get("reason"),
        )
        for row in projected["field_observations"]
        if row.get("reason") == "pboc_cell_contract_unresolved"
    } == {
        (
            "grid:1:2024-01",
            "status_code",
            "pboc_cell_contract_unresolved",
        ),
        (
            "grid:1:2024-02",
            "status_code",
            "pboc_cell_contract_unresolved",
        ),
    }
    observed = {
        row["evidence_path"]: row.get("integer_value") or row.get("string_value")
        for row in projected["extraction_issue_evidence"]
        if row["extraction_issue_id"] == issues[0]["extraction_issue_id"]
        and row["evidence_kind"] == "observed"
    }
    assert observed["grid_id"] == "grid:1"
    assert observed["withheld_candidate_count"] == 3
    assert observed["withheld_month_count"] == 2
    assert observed["withheld_months[0]"] == "2024-01"
    assert observed["withheld_months[1]"] == "2024-02"
    assert observed["first_withheld_month"] == "2024-01"
    assert observed["last_withheld_month"] == "2024-02"
    status_row = next(
        row
        for row in projected["dataset_status"]
        if row["dataset_name"] == "credit_account_monthly_performance"
    )
    assert status_row["presence_status"] == "partial"
    assert status_row["observed_row_count"] == 1
    assert status_row["expected_row_count"] == 1


def test_v2_monthly_gate_prunes_stale_exact_status_issue_after_resolution() -> None:
    projected = project_personal_detail_datasets(
        {
            "repayment_records": [
                {
                    # Real Candidate-B rows carry no outer ``record_id`` and
                    # place the relationship ``grid_id`` before their row ID.
                    "repayment_id": "monthly:resolved",
                    "grid_id": "grid:resolved",
                    "account_id": "account:1",
                    "year": 2024,
                    "month": 1,
                    "status": "N",
                    "overdue_amount": 0,
                },
                {
                    "repayment_id": "monthly:orphan-resolved",
                    "grid_id": "grid:orphan-resolved",
                    "year": 2024,
                    "month": 2,
                    "status": "N",
                    "overdue_amount": 0,
                },
            ],
            "personal_detail_extraction_issues": [
                _record(
                    "issue:stale-status",
                    extraction_issue_id="issue:stale-status",
                    category="ocr_cell_level_error",
                    issue_code="pboc_cell_contract_unresolved",
                    severity="warning",
                    status="requires_review",
                    target_dataset="repayment_records",
                    target_record_id="monthly:resolved",
                    field_name="status",
                    observed_value="unknown",
                ),
                _record(
                    "issue:stale-orphan-status",
                    extraction_issue_id="issue:stale-orphan-status",
                    category="ocr_cell_level_error",
                    issue_code="pboc_cell_contract_unresolved",
                    severity="warning",
                    status="requires_review",
                    target_dataset="repayment_records",
                    target_record_id="monthly:orphan-resolved",
                    field_name="status",
                    observed_value="unknown",
                ),
            ],
        }
    )

    [row] = projected["credit_account_monthly_performance"]
    assert row["status_code"] == "N"
    assert row["monthly_performance_id"] == "monthly:resolved"
    assert not any(
        issue.get("issue_code") == "pboc_cell_contract_unresolved"
        and issue.get("target_record_id")
        in {"monthly:resolved", "monthly:orphan-resolved"}
        for issue in projected.get("extraction_issues", [])
    )
    assert not any(
        observation.get("reason") == "pboc_cell_contract_unresolved"
        and observation.get("business_record_id")
        in {"monthly:resolved", "monthly:orphan-resolved"}
        for observation in projected.get("field_observations", [])
    )
    aggregate = next(
        issue
        for issue in projected["extraction_issues"]
        if issue.get("issue_code")
        == "candidate_b_monthly_status_grid_unresolved"
    )
    assert "target_record_id" not in aggregate


def test_v2_monthly_gate_keeps_exact_status_issue_for_noncanonical_orphan() -> None:
    projected = project_personal_detail_datasets(
        {
            "repayment_records": [
                {
                    "repayment_id": "monthly:orphan-unknown",
                    "grid_id": "grid:orphan-unknown",
                    "year": 2024,
                    "month": 1,
                    "status": "unknown",
                    "overdue_amount": None,
                }
            ],
            "personal_detail_extraction_issues": [
                _record(
                    "issue:orphan-unknown-status",
                    extraction_issue_id="issue:orphan-unknown-status",
                    category="ocr_cell_level_error",
                    issue_code="pboc_cell_contract_unresolved",
                    severity="warning",
                    status="requires_review",
                    target_dataset="repayment_records",
                    target_record_id="monthly:orphan-unknown",
                    field_name="status",
                    observed_value="unknown",
                )
            ],
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
    local_issue = next(
        issue
        for issue in projected["extraction_issues"]
        if issue.get("issue_code") == "pboc_cell_contract_unresolved"
    )
    assert local_issue["target_record_id"] == "monthly:orphan-unknown"
    assert local_issue["field_name"] == "status_code"
    assert local_issue["status"] == "requires_review"
    assert any(
        observation.get("business_record_id") == "monthly:orphan-unknown"
        and observation.get("field_name") == "status_code"
        and observation.get("reason") == "pboc_cell_contract_unresolved"
        for observation in projected["field_observations"]
    )
    assert sum(
        issue.get("issue_code") == "candidate_b_monthly_status_grid_unresolved"
        for issue in projected["extraction_issues"]
    ) == 1


def test_v2_successful_monthly_row_drops_diagnostics_without_review() -> None:
    projected = project_personal_detail_datasets(
        {
            "repayment_records": [
                _record(
                    "monthly:accepted",
                    repayment_id="monthly:accepted",
                    account_id="account:1",
                    year=2025,
                    month=2,
                    status="N",
                    overdue_amount="0",
                    audit={
                        "alignment_status": "exact",
                        "source_ref": {"logical_page": 2, "bbox": [1, 2, 3, 4]},
                    },
                    recognition_source="canonical_row_sequence",
                    status_bbox=[1, 2, 3, 4],
                    confidence=0.0,
                )
            ]
        }
    )

    row = projected["credit_account_monthly_performance"][0]

    assert "review" not in row
    assert "confidence" not in row
    assert row["status_code"] == "N"
    assert not ({"audit", "recognition_source", "status_bbox"} & set(row))


def test_resolved_deterministic_corrections_remain_semantic_only() -> None:
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                _record(
                    "account:resolved-currency",
                    account_id="account:resolved-currency",
                    account_type="credit_card",
                    account_identifier="ACCOUNT123456",
                    management_institution="中国银行股份有限公司",
                    open_date="2024-01-01",
                    currency="CNY",
                )
            ],
            "personal_detail_extraction_issues": [
                _record(
                    "issue:resolved-currency",
                    extraction_issue_id="issue:resolved-currency",
                    category="ocr_cell_level_error",
                    issue_code="candidate_b_currency_token_residue_corrected",
                    severity="info",
                    status="resolved",
                    target_dataset="credit_accounts",
                    target_record_id="account:resolved-currency",
                    field_name="currency",
                    observed_value={"raw": "CNY 江", "residue": "江"},
                    candidate_value={"currency": "CNY"},
                )
            ],
        }
    )

    assert projected.get("extraction_issues") in (None, [])
    assert projected.get("extraction_issue_evidence") in (None, [])
    assert not any(
        row.get("dataset_name") == "credit_accounts"
        for row in projected.get("dataset_status") or ()
    )


def test_actionable_issue_does_not_downgrade_source_extraction_failure() -> None:
    projected = project_personal_detail_datasets(
        {
            "personal_detail_dataset_status": [
                _record(
                    "status:accounts",
                    dataset_status_id="status:accounts",
                    dataset_name="credit_accounts",
                    applicability="applicable",
                    presence_status="extraction_failed",
                    observed_row_count=0,
                    expected_row_count=1,
                    reason="source_extraction_failed",
                )
            ],
            "personal_detail_extraction_issues": [
                _record(
                    "issue:account-population",
                    extraction_issue_id="issue:account-population",
                    category="schema_incompleteness",
                    issue_code="candidate_b_account_table_missing",
                    severity="error",
                    status="requires_review",
                    target_dataset="credit_accounts",
                    target_record_id="account:missing",
                    field_name="account_identifier",
                    observed_value=None,
                )
            ],
        }
    )

    status = next(
        row
        for row in projected["dataset_status"]
        if row["dataset_name"] == "credit_accounts"
    )
    assert status["presence_status"] == "extraction_failed"
    assert status["observed_row_count"] == 0
    assert status["expected_row_count"] == 1
    assert status["reason"] == "source_extraction_failed"


def test_v2_preserves_canonical_slash_monthly_status_silently() -> None:
    projected = project_personal_detail_datasets(
        {
            "repayment_records": [
                _record(
                    "monthly:slash",
                    repayment_id="monthly:slash",
                    account_id="account:1",
                    year=2025,
                    month=1,
                    status="/",
                    overdue_amount=0,
                )
            ]
        }
    )

    row = projected["credit_account_monthly_performance"][0]
    assert row["status_code"] == "/"
    assert "review" not in row
    assert not any(
        issue.get("field_name") == "status_code"
        for issue in projected.get("extraction_issues", [])
    )


def test_v2_monthly_issue_marks_numeric_overdue_row_for_review() -> None:
    projected = project_personal_detail_datasets(
        {
            "repayment_records": [
                _record(
                    "monthly:numeric",
                    repayment_id="monthly:numeric",
                    account_id="account:1",
                    year=2025,
                    month=1,
                    status="6",
                )
            ],
            "personal_detail_extraction_issues": [
                _record(
                    "issue:amount",
                    extraction_issue_id="issue:amount",
                    category="schema_incompleteness",
                    issue_code="pboc_cell_contract_unresolved",
                    severity="warning",
                    status="requires_review",
                    target_dataset="repayment_records",
                    target_record_id="monthly:numeric",
                    field_name="overdue_amount",
                )
            ],
        }
    )

    row = projected["credit_account_monthly_performance"][0]
    issue = projected["extraction_issues"][0]
    assert row["status_code"] == "6"
    assert row["extraction_status"] == "review"
    assert row["review"]["status"] == "requires_review"
    assert issue["field_name"] == "status_amount"


def test_v2_monthly_rows_are_sorted_within_first_seen_grid() -> None:
    projected = project_personal_detail_datasets(
        {
            "repayment_records": [
                _record(
                    f"grid:1:2025-{month:02d}",
                    repayment_id=f"grid:1:2025-{month:02d}",
                    grid_id="grid:1",
                    account_id="account:1",
                    year=2025,
                    month=month,
                    status="N",
                )
                for month in (3, 1, 2)
            ]
        }
    )

    rows = projected["credit_account_monthly_performance"]
    assert [row["performance_month"] for row in rows] == [
        "2025-01",
        "2025-02",
        "2025-03",
    ]


def test_v2_corrected_monthly_row_drops_stale_source_review() -> None:
    projected = project_personal_detail_datasets(
        {
            "repayment_records": [
                {
                    "record_id": "monthly:corrected",
                    "repayment_id": "monthly:corrected",
                    "account_id": "account:1",
                    "year": 2025,
                    "month": 2,
                    "status": "N",
                    "overdue_amount": 0,
                    "extraction_status": "review",
                    "canonical_raw": {"status": "N", "extraction_status": "review"},
                }
            ]
        }
    )

    row = projected["credit_account_monthly_performance"][0]

    assert row["status_code"] == "N"
    assert "extraction_status" not in row
    assert "review" not in row
    assert "extraction_status" not in row.get("canonical_raw", {})


def test_v2_monthly_null_overlay_is_withheld_with_explicit_grid_issue() -> None:
    projected = project_personal_detail_datasets(
        {
            "repayment_records": [
                _record(
                    "monthly:unknown",
                    repayment_id="monthly:unknown",
                    account_id=None,
                    year=2025,
                    month=2,
                    status="unknown",
                    status_code=None,
                    extraction_status="review",
                )
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
    issue = next(
        row
        for row in projected["extraction_issues"]
        if row.get("issue_code") == "candidate_b_monthly_status_grid_unresolved"
    )
    assert issue["target_dataset"] == "credit_account_monthly_performance"
    assert issue["field_name"] == "status_code"


def test_v2_hash_monthly_marker_is_a_canonical_business_status() -> None:
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                _record("account:1", account_id="account:1", account_type="credit_card")
            ],
            "repayment_records": [
                _record(
                    "monthly:hash",
                    repayment_id="monthly:hash",
                    grid_id="grid:1",
                    account_id="account:1",
                    year=2025,
                    month=2,
                    status="#",
                    overdue_amount=0,
                    extraction_status="review",
                )
            ],
        }
    )

    row = projected["credit_account_monthly_performance"][0]
    assert row["status_code"] == "#"
    assert "extraction_issues" not in projected


def test_v2_repayment_responsibility_removes_combined_field_from_all_pools() -> None:
    source_values = {
        "liability_id": "responsibility:organization",
        "related_party_id_type": "统一社会信用代码",
        "related_party_id_number": "91110000123456789X",
        "repayment_status_code": "N",
    }
    projected = project_personal_detail_datasets(
        {
            "repayment_liability_records": [
                {
                    "record_id": "responsibility:organization",
                    "normalized": dict(source_values),
                    "canonical_raw": dict(source_values),
                    "raw": dict(source_values),
                }
            ]
        }
    )

    row = projected["repayment_responsibilities"][0]
    assert row["normalized"]["repayment_status_code"] == "N"
    assert "source_status_value" not in row["normalized"]
    assert all(
        "overdue_months_or_repayment_status" not in row.get(pool_name, {})
        for pool_name in ("normalized", "canonical_raw", "raw")
    )


def test_v2_repayment_responsibility_types_normalized_overdue_months_only() -> None:
    projected = project_personal_detail_datasets(
        {
            "repayment_liability_records": [
                {
                    "record_id": "responsibility:typed-overdue",
                    "normalized": {
                        "liability_id": "responsibility:typed-overdue",
                        "related_party_id_type": "统一社会信用代码",
                        "related_party_id_number": "91110000123456789X",
                        "overdue_months": "0",
                    },
                    "canonical_raw": {"overdue_months": "0"},
                    "raw": {"overdue_months": "0"},
                }
            ]
        }
    )

    row = projected["repayment_responsibilities"][0]
    assert row["normalized"]["overdue_months"] == 0
    assert type(row["normalized"]["overdue_months"]) is int
    assert row["canonical_raw"]["overdue_months"] == "0"
    assert row["raw"]["overdue_months"] == "0"


def test_control_field_names_use_only_canonical_dataset_fields() -> None:
    assert (
        _canonical_field_name("credit_account_monthly_performance", "status")
        == "status_code"
    )
    assert _canonical_field_name("subject_employment", "employment_components") is None
    assert _canonical_field_name("subject_profile", "mobile_phone") is None
    assert (
        _canonical_field_name("credit_accounts", "account_status_raw")
        == "account_lifecycle_state"
    )
    assert (
        _canonical_field_name("credit_accounts", "account_status_resolution")
        == "account_lifecycle_state"
    )
    assert (
        _canonical_field_name("credit_business_overview", "最近2年内的查询次数")
        == "numeric_value"
    )


def test_transformed_control_field_names_land_in_the_final_catalog() -> None:
    aliases = (
        (
            "postpaid_payment_history",
            "postpaid_monthly_performance",
            "postpaid_payment_history_id",
            "postpaid_monthly_performance_id",
        ),
        (
            "postpaid_payment_history",
            "postpaid_monthly_performance",
            "year",
            "performance_month",
        ),
        (
            "postpaid_payment_history",
            "postpaid_monthly_performance",
            "month",
            "performance_month",
        ),
        (
            "postpaid_payment_history",
            "postpaid_monthly_performance",
            "status",
            "status_code",
        ),
        (
            "personal_housing_fund_records",
            "housing_fund_records",
            "personal_housing_fund_id",
            "public_record_id",
        ),
        (
            "personal_housing_fund_records",
            "housing_fund_records",
            "personal_contribution_ratio",
            "personal_contribution_ratio_percent",
        ),
        (
            "personal_housing_fund_records",
            "housing_fund_records",
            "employer_contribution_ratio",
            "employer_contribution_ratio_percent",
        ),
        (
            "tax_arrears_records",
            "tax_arrears_records",
            "authority",
            "tax_authority",
        ),
        (
            "civil_judgment_records",
            "civil_judgment_records",
            "authority",
            "filing_court",
        ),
        (
            "civil_judgment_records",
            "civil_judgment_records",
            "start_date",
            "filing_date",
        ),
        (
            "civil_judgment_records",
            "civil_judgment_records",
            "effective_date",
            "judgment_effective_date",
        ),
        (
            "enforcement_records",
            "enforcement_records",
            "authority",
            "court",
        ),
        (
            "enforcement_records",
            "enforcement_records",
            "start_date",
            "filing_date",
        ),
        (
            "enforcement_records",
            "enforcement_records",
            "end_date",
            "closure_date",
        ),
        (
            "administrative_penalty_records",
            "administrative_penalty_records",
            "start_date",
            "effective_month",
        ),
        (
            "professional_qualification_records",
            "professional_qualification_records",
            "authority",
            "issuing_authority",
        ),
        (
            "award_records",
            "administrative_award_records",
            "start_date",
            "effective_month",
        ),
        (
            "administrative_penalty_records",
            "administrative_penalty_records",
            "effective_date",
            "effective_month",
        ),
        (
            "administrative_penalty_records",
            "administrative_penalty_records",
            "end_date",
            "end_month",
        ),
        (
            "professional_qualification_records",
            "professional_qualification_records",
            "obtained_date",
            "obtained_month",
        ),
        (
            "professional_qualification_records",
            "professional_qualification_records",
            "expiry_date",
            "expiry_month",
        ),
        (
            "professional_qualification_records",
            "professional_qualification_records",
            "revocation_date",
            "revocation_month",
        ),
        (
            "award_records",
            "administrative_award_records",
            "effective_date",
            "effective_month",
        ),
        (
            "award_records",
            "administrative_award_records",
            "end_date",
            "end_month",
        ),
    )
    source_refs = {
        index: [
            {
                "ref_type": "source_cell",
                "ref_id": f"cell:{index}",
                "field_name": source_field,
                "page": index,
            }
        ]
        for index, (_source, _target, source_field, _canonical) in enumerate(
            aliases, start=1
        )
    }
    projected = project_personal_detail_datasets(
        {
            "personal_detail_field_observations": [
                _record(
                    f"observation:{index}",
                    field_observation_id=f"observation:{index}",
                    dataset_name=source_dataset,
                    business_record_id=f"business:{index}",
                    field_name=source_field,
                    observation_status="unreadable",
                    raw_value="unreadable",
                    source_refs=source_refs[index],
                )
                for index, (source_dataset, _target, source_field, _canonical) in enumerate(
                    aliases, start=1
                )
            ],
            "personal_detail_extraction_issues": [
                _record(
                    f"issue:{index}",
                    extraction_issue_id=f"issue:{index}",
                    category="schema_incompleteness",
                    issue_code="transformed_control_field_probe",
                    target_dataset=source_dataset,
                    target_record_id=f"business:{index}",
                    field_name=source_field,
                    observed_value="unreadable",
                    source_refs=source_refs[index],
                )
                for index, (source_dataset, _target, source_field, _canonical) in enumerate(
                    aliases, start=1
                )
            ],
        }
    )
    catalog = personal_detail_data_dictionary()["datasets"]
    observations = projected["field_observations"]
    issues = projected["extraction_issues"]

    for index, (_source, target, source_field, canonical_field) in enumerate(
        aliases, start=1
    ):
        observation = next(
            row
            for row in observations
            if row["field_observation_id"] == f"observation:{index}"
        )
        issue = next(
            row for row in issues if row["extraction_issue_id"] == f"issue:{index}"
        )
        assert observation["record_id"] == f"observation:{index}"
        assert observation["dataset_name"] == target
        assert observation["business_record_id"] == f"business:{index}"
        assert observation["field_name"] == canonical_field
        assert observation["field_name"] in catalog[target]["columns"]
        assert observation["source_refs"] == source_refs[index]
        assert issue["record_id"] == f"issue:{index}"
        assert issue["target_dataset"] == target
        assert issue["target_record_id"] == f"business:{index}"
        assert issue["field_name"] == canonical_field
        assert issue["field_name"] in catalog[target]["columns"]
        assert issue["source_refs"] == source_refs[index]


def test_public_structural_control_labels_are_not_canonical_fields() -> None:
    public_datasets = (
        "tax_arrears_records",
        "civil_judgment_records",
        "enforcement_records",
        "administrative_penalty_records",
        "housing_fund_records",
        "social_assistance_records",
        "professional_qualification_records",
        "administrative_award_records",
    )

    assert all(
        _canonical_field_name(dataset_name, source_field) is None
        for dataset_name in public_datasets
        for source_field in ("content", "record_type", "unmapped_content")
    )
    assert (
        _canonical_field_name("tax_arrears_records", "arrears_amount")
        == "arrears_amount"
    )


def test_multifield_control_alias_stays_explicit_without_fake_field_observation() -> None:
    projected = project_personal_detail_datasets(
        {
            "personal_detail_field_observations": [
                _record(
                    "observation:employment-components",
                    field_observation_id="observation:employment-components",
                    dataset_name="subject_employment",
                    business_record_id="employment:1",
                    field_name="employment_components",
                    observation_status="unreadable",
                    raw_value="工作单位=示例公司;职业=职员",
                )
            ],
            "personal_detail_extraction_issues": [
                _record(
                    "issue:employment-components",
                    extraction_issue_id="issue:employment-components",
                    category="schema_incompleteness",
                    issue_code="unstructured_multifield_blob",
                    severity="warning",
                    status="requires_review",
                    target_dataset="employment_records",
                    target_record_id="employment:1",
                    field_name="employment_components",
                    observed_value="工作单位=示例公司;职业=职员",
                )
            ],
        }
    )

    assert projected.get("field_observations", []) == []
    [issue] = projected["extraction_issues"]
    assert issue["target_dataset"] == "subject_employment"
    assert "field_name" not in issue
    assert issue["observed_value"] == "工作单位=示例公司;职业=职员"


def test_unlabeled_legacy_liability_status_is_not_guessed_from_party_category() -> None:
    projected = project_personal_detail_datasets(
        {
            "repayment_liability_records": [
                _record(
                    "responsibility:legacy",
                    liability_id="responsibility:legacy",
                    related_party_id_type="身份证",
                    related_party_id_number="110101198108151111",
                    overdue_months_or_repayment_status="0",
                )
            ]
        }
    )

    row = projected["repayment_responsibilities"][0]
    assert row["source_status_value"] == "0"
    assert row["extraction_status"] == "review"
    assert "overdue_months" not in row
    assert "repayment_status_code" not in row


def test_v2_neighbor_consensus_is_promoted_to_explicit_review() -> None:
    projected = project_personal_detail_datasets(
        {
            "repayment_records": [
                _record(
                    "monthly:neighbor",
                    repayment_id="monthly:neighbor",
                    account_id="account:1",
                    year=2025,
                    month=3,
                    status="N",
                    overdue_amount="0",
                    audit={"reason": "unreadable_cell_with_matching_adjacent_statuses"},
                    recognition_source="row_neighbor_consensus",
                )
            ]
        }
    )

    row = projected["credit_account_monthly_performance"][0]

    assert row["status_code"] == "N"
    assert row["extraction_status"] == "review"
    assert row["review"]["status"] == "requires_review"
    assert row["review"]["recognition_source"] == "row_neighbor_consensus"


def test_v2_successful_business_row_drops_redundant_evidence_snapshots() -> None:
    normalized = {
        "inquiry_id": "inquiry:clean",
        "inquiry_date": "2025-01-02",
        "institution": "示例银行股份有限公司",
        "reason": "本人查询",
        "source_refs": [{"logical_page": 8}],
        "page": 8,
    }
    projected = project_personal_detail_datasets(
        {
            "inquiry_records": [
                {
                    "record_id": "inquiry:clean",
                    "normalized": dict(normalized),
                    "canonical_raw": dict(normalized),
                    "raw": dict(normalized),
                }
            ]
        }
    )

    row = projected["inquiries"][0]
    assert row["normalized"] == {
        "inquiry_id": "inquiry:clean",
        "inquiry_date": "2025-01-02",
        "institution": "示例银行股份有限公司",
        "reason": "本人查询",
    }
    assert "canonical_raw" not in row
    assert "raw" not in row


def test_v2_reconciles_only_fully_matched_targetless_inquiry_issues() -> None:
    def row_ref(owner: str) -> dict[str, Any]:
        return {
            "source": "candidate_b_canonical_inquiry_line",
            "logical_page": 26,
            "source_page": 13,
            "bbox": [50.0, 100.0, 390.0, 112.0],
            "geometry_scope": "row",
            "evidence_ids": [f"personal_detail_page_reocr:{owner}"],
        }

    def inquiry(
        sequence: int,
        inquiry_date: str,
        institution: str,
        reason: str,
        *,
        channel: str = "institution",
        suffix: str = "",
        owner: str | None = None,
    ) -> dict[str, Any]:
        inquiry_id = f"inquiry:{sequence}{suffix}"
        record = _record(
            inquiry_id,
            inquiry_id=inquiry_id,
            sequence=sequence,
            inquiry_date=inquiry_date,
            institution=institution,
            reason=reason,
            query_channel=channel,
            inquiry_type=channel,
        )
        if owner is not None:
            record["source_refs"] = [row_ref(owner)]
        return record

    def unresolved(
        issue_id: str,
        sequence: int,
        row: list[str],
        *,
        owner: str | None = None,
    ) -> dict[str, Any]:
        issue = _record(
            issue_id,
            extraction_issue_id=issue_id,
            issue_code="candidate_b_inquiry_row_cells_unresolved",
            category="ocr_structure_correction",
            status="requires_review",
            severity="warning",
            parser_stage="candidate_b_inquiry_schema",
            target_dataset="inquiry_records",
            target_record_id=None,
            field_name="inquiry_date",
            observed_value={"sequence": sequence, "row": row},
            candidate_value={"missing_fields": ["inquiry_date"]},
        )
        if owner is not None:
            issue["source_refs"] = [row_ref(owner)]
        return issue

    projected = project_personal_detail_datasets(
        {
            "inquiry_records": [
                inquiry(
                    3,
                    "2023-01-03",
                    "广发银行股份有限公司",
                    "贷后管理",
                    owner="row:3",
                ),
                inquiry(
                    82,
                    "2021-04-25",
                    "中国光大银行股份有限公司",
                    "贷后管理",
                    owner="row:82",
                ),
                inquiry(90, "2021-01-01", "中信银行股份有限公司", "贷后管理"),
                inquiry(
                    90,
                    "2021-01-01",
                    "中信银行股份有限公司",
                    "贷后管理",
                    suffix=":duplicate",
                ),
                inquiry(
                    91,
                    "2021-01-02",
                    "本人",
                    "本人查询",
                    channel="personal",
                ),
                inquiry(
                    92,
                    "2021-01-03",
                    "中信银行股份有限公司",
                    "贷后管理",
                    owner="row:92",
                ),
                inquiry(
                    93,
                    "2021-01-04",
                    "中信银行股份有限公司",
                    "贷后管理",
                    owner="final-row:93",
                ),
                inquiry(
                    94,
                    "2021-01-05",
                    "中信银行股份有限公司",
                    "贷后管理",
                    owner="row:94",
                ),
                inquiry(
                    95,
                    "2021-01-06",
                    "中信银行股份有限公司",
                    "贷后管理",
                ),
            ],
            "personal_detail_extraction_issues": [
                unresolved(
                    "issue:matched-date-noise",
                    3,
                    ["3", "2023.01.03 20", "广发银行股份有限公司", "贷后管理"],
                    owner="row:3",
                ),
                unresolved(
                    "issue:matched-edge-noise",
                    82,
                    [
                        "82",
                        "202104.25",
                        "多 中国光大银行股份有限公司",
                        "贷后管理 司 %5",
                    ],
                    owner="row:82",
                ),
                unresolved(
                    "issue:missing-sequence",
                    23,
                    ["23", "2022.08:25", "平安银行股份有限公司", "贷后管理"],
                ),
                unresolved(
                    "issue:mismatched-reason",
                    3,
                    ["3", "2023.01.03", "广发银行股份有限公司", "贷款审批"],
                    owner="row:3",
                ),
                unresolved(
                    "issue:duplicate-sequence",
                    90,
                    ["90", "2021.01.01", "中信银行股份有限公司", "贷后管理"],
                ),
                unresolved(
                    "issue:personal-channel",
                    91,
                    ["91", "2021.01.02", "本人", "本人查询"],
                ),
                unresolved(
                    "issue:incomplete-fingerprint",
                    3,
                    ["3", "2023.01.03", "广发银行股份有限公司"],
                ),
                unresolved(
                    "issue:matched-whitespace",
                    92,
                    ["92", "2021.01.03", "中 信银行股份有限公司", "贷后管理"],
                    owner="row:92",
                ),
                unresolved(
                    "issue:different-owner",
                    93,
                    ["93", "2021.01.04", "中信银行股份有限公司", "贷后管理"],
                    owner="issue-row:93",
                ),
                unresolved(
                    "issue:leading-han-residue",
                    94,
                    ["94", "2021.01.05", "福中信银行股份有限公司", "贷后管理"],
                    owner="row:94",
                ),
                unresolved(
                    "issue:no-owner-proof",
                    95,
                    ["95", "2021.01.06", "中信银行股份有限公司", "贷后管理"],
                ),
            ],
        }
    )

    issue_ids = {
        row.get("extraction_issue_id")
        for row in projected.get("extraction_issues", [])
    }
    assert "issue:matched-date-noise" not in issue_ids
    assert "issue:matched-whitespace" not in issue_ids
    assert {
        "issue:matched-edge-noise",
        "issue:missing-sequence",
        "issue:mismatched-reason",
        "issue:duplicate-sequence",
        "issue:personal-channel",
        "issue:incomplete-fingerprint",
        "issue:different-owner",
        "issue:leading-han-residue",
        "issue:no-owner-proof",
    } <= issue_ids


def test_v2_targetless_inquiry_does_not_trust_textless_repair_geometry() -> None:
    raw_institution = "福 中信银行股份有限公司"
    clean_institution = "中信银行股份有限公司"
    institution_ref = {
        "source": "native_detail_table_cell",
        "logical_page": 26,
        "source_page": 13,
        "table_id": "pt_26_0",
        "row": 4,
        "column": 2,
        "bbox": [130.0, 100.0, 330.0, 112.0],
        "coordinate_system": "pdf_points_top_left",
        "geometry_scope": "cell",
        "evidence_ids": ["ocr:sp0013:lp0026:r4:c2"],
        "field_name": "institution",
        "binding": "canonical_header_column",
        "binding_quality": "canonical_header_column",
    }
    repair_ref = {
        "source": "personal_detail_installed_page_evidence",
        "logical_page": 26,
        "source_page": 13,
        "bbox": [140.0, 101.0, 320.0, 111.0],
        "coordinate_system": "pdf_points_top_left",
        "geometry_scope": "token_band",
        "geometry_status": "exact",
        "evidence_ids": ["personal_detail_page_reocr:source:13:w94"],
        "binding": "canonical_field_slot_repair_candidate",
        "binding_quality": "exact_source_bound_candidate",
    }
    inquiry = _record(
        "inquiry:94",
        inquiry_id="inquiry:94",
        sequence=94,
        inquiry_date="2021-01-05",
        institution=clean_institution,
        reason="贷后管理",
        query_channel="institution",
        inquiry_type="institution",
    )
    inquiry["canonical_raw"] = {"institution": raw_institution}
    inquiry["source_cell_refs"] = [institution_ref, repair_ref]
    issue = _record(
        "issue:source-bound-correction",
        extraction_issue_id="issue:source-bound-correction",
        issue_code="candidate_b_inquiry_row_cells_unresolved",
        category="ocr_structure_correction",
        status="requires_review",
        severity="warning",
        parser_stage="candidate_b_inquiry_schema",
        target_dataset="inquiry_records",
        target_record_id=None,
        field_name="inquiry_date",
        observed_value={
            "sequence": 94,
            "row": ["94", "2021.01.05", raw_institution, "贷后管理"],
        },
        candidate_value={"missing_fields": ["inquiry_date"]},
    )
    issue["source_refs"] = [institution_ref]

    projected = project_personal_detail_datasets(
        {
            "inquiry_records": [inquiry],
            "personal_detail_extraction_issues": [issue],
        }
    )

    assert {
        row.get("extraction_issue_id")
        for row in projected.get("extraction_issues", [])
    } == {"issue:source-bound-correction"}


def test_v2_summary_issue_targets_follow_metric_types_when_values_are_withheld() -> None:
    metrics = [
        _record(
            f"metric:{metric_code}",
            credit_summary_metric_id=f"metric:{metric_code}",
            summary_record_id="summary:1",
            row_index=index,
            column_index=1,
            metric_code=metric_code,
            mapping_status="mapped",
            value_type="unknown",
        )
        for index, metric_code in enumerate(
            ("account_count", "business_type", "first_business_issue_month"),
            start=1,
        )
    ]
    issues = [
        _record(
            f"issue:{metric['metric_code']}",
            extraction_issue_id=f"issue:{metric['metric_code']}",
            issue_code="candidate_b_summary_scalar_unresolved",
            category="schema_incompleteness",
            status="requires_review",
            severity="warning",
            parser_stage="candidate_b_summary_schema",
            target_dataset="credit_business_overview",
            target_record_id=metric["credit_summary_metric_id"],
            field_name="value",
        )
        for metric in metrics
    ]

    projected = project_personal_detail_datasets(
        {
            "personal_detail_credit_summary_metrics": metrics,
            "personal_detail_extraction_issues": issues,
        }
    )
    fields = {
        row["extraction_issue_id"].removeprefix("issue:"): row["field_name"]
        for row in projected["extraction_issues"]
        if row.get("issue_code") == "candidate_b_summary_scalar_unresolved"
    }

    assert fields == {
        "account_count": "numeric_value",
        "business_type": "text_value",
        "first_business_issue_month": "date_value",
    }


def test_v2_source_sentinel_is_silent_absence_but_monthly_dash_is_reported() -> None:
    projected = project_personal_detail_datasets(
        {
            "residence_records": [
                _record(
                    "residence:sentinel",
                    residence_id="residence:sentinel",
                    residential_phone="-",
                )
            ],
            "spouse_records": [
                _record(
                    "spouse:sentinel",
                    spouse_record_id="spouse:sentinel",
                    name="--",
                )
            ],
            "repayment_records": [
                _record(
                    "monthly:dash",
                    repayment_id="monthly:dash",
                    account_id="account:1",
                    year=2025,
                    month=1,
                    status="-",
                )
            ],
        }
    )

    residence = projected["subject_residences"][0]
    assert residence["residential_phone"] is None
    assert residence.get("canonical_raw", {}).get("residential_phone") is None
    spouse = projected["subject_spouse"][0]
    assert spouse["name"] is None
    assert spouse.get("canonical_raw", {}).get("name") is None
    assert projected.get("credit_account_monthly_performance", []) == []
    monthly_status = next(
        row
        for row in projected["dataset_status"]
        if row["dataset_name"] == "credit_account_monthly_performance"
    )
    assert monthly_status["observed_row_count"] == 0
    assert "expected_row_count" not in monthly_status
    assert not any(
        row.get("target_record_id") in {"residence:sentinel", "spouse:sentinel"}
        and row.get("field_name") in {"residential_phone", "name"}
        for row in projected["extraction_issues"]
    )
    assert any(
        row.get("issue_code") == "candidate_b_monthly_status_grid_unresolved"
        for row in projected["extraction_issues"]
    )


def test_v2_unknown_account_event_is_withheld_into_issue_evidence() -> None:
    projected = project_personal_detail_datasets(
        {
            "personal_detail_account_events": [
                _record(
                    "event:unknown",
                    event_type="unknown_future_event",
                    field_a="alpha",
                    field_b="beta",
                )
            ]
        }
    )

    assert "pboc_extension_fields" not in projected
    issue = next(
        row
        for row in projected["extraction_issues"]
        if row.get("issue_code") == "canonical_account_event_type_unresolved"
    )
    assert issue["target_record_id"] == "event:unknown"
    assert "target_dataset" not in issue
    evidence = projected["extraction_issue_evidence"]
    assert {row["string_value"] for row in evidence if row.get("evidence_kind") == "observed"} >= {
        "alpha",
        "beta",
    }


def test_v2_projection_corrects_business_semantics_and_reports_unmapped_values() -> None:
    source = {
        "personal_report_metadata": [
            _record(
                "metadata:1",
                personal_report_metadata_id="metadata:1",
                report_number="2025052510051518624525",
                report_time="2025-05-25T10:05:15+08:00",
                query_institution="中国人民银行征信中心",
                query_reason="本人查询",
            )
        ],
        "personal_profile": [
            _record("profile:1", personal_profile_id="profile:1", birth_date="1981-08-15")
        ],
        "personal_detail_field_observations": [
            _record(
                "field:1",
                field_observation_id="field:1",
                dataset_name="personal_profile",
                business_record_id="profile:1",
                field_name="work_phone",
                observation_status="not_observed",
            )
        ],
        "credit_accounts": [
            _record(
                "account:loan",
                account_id="account:loan",
                account_type="non_revolving_loan",
            ),
            _record(
                "account:quasi",
                account_id="account:quasi",
                account_type="quasi_credit_card",
            ),
        ],
        "repayment_records": [
            _record(
                "monthly:loan",
                repayment_id="monthly:loan",
                account_id="account:loan",
                year=2025,
                month=5,
                status="1",
                overdue_amount=100,
            ),
            _record(
                "monthly:quasi",
                repayment_id="monthly:quasi",
                account_id="account:quasi",
                year="2025",
                month="05",
                status=1,
                overdue_amount="1,000.00",
            ),
        ],
        "repayment_liability_records": [
            _record(
                "responsibility:person",
                liability_id="responsibility:person",
                related_party_id_type="身份证",
                related_party_id_number="110101198108151111",
                overdue_months=2,
            ),
            _record(
                "responsibility:organization",
                liability_id="responsibility:organization",
                related_party_id_type="统一社会信用代码",
                related_party_id_number="91110000123456789X",
                repayment_status_code="N",
            ),
        ],
        "personal_housing_fund_records": [
            _record(
                "housing:1",
                personal_housing_fund_id="housing:1",
                first_contribution_month="2020-01",
                personal_contribution_ratio="12%",
                employer_contribution_ratio="12%",
            )
        ],
        "administrative_penalty_records": [
            _record(
                "penalty:1",
                administrative_penalty_id="penalty:1",
                effective_date="2021-08-31",
                end_date="2024/7/15",
            )
        ],
        "statements": [_record("statement:1", statement_id="statement:1", text="本人声明")],
        "personal_detail_credit_summary_metrics": [
            _record(
                "metric:1",
                credit_summary_metric_id="metric:1",
                summary_record_id="summary:1",
                row_index=1,
                column_index=1,
                metric_code="account_count",
                value_type="integer",
                integer_value=2,
            )
        ],
        "personal_detail_summary_cells": [
            _record(
                "cell:1",
                summary_cell_id="cell:1",
                summary_record_id="summary:1",
                row_index=1,
                column_index=1,
                column_label="账户数",
                value="2",
            ),
            _record(
                "cell:2",
                summary_cell_id="cell:2",
                summary_record_id="summary:1",
                row_index=1,
                column_index=2,
                column_label="未映射指标",
                value="业务值",
            ),
        ],
    }

    projected = project_personal_detail_datasets(source)

    assert set(projected) <= set(PBOC_DATASET_ORDER)
    assert "repayment_records" not in projected
    accounts = {row["account_id"]: row for row in projected["credit_accounts"]}
    assert accounts["account:loan"]["pboc_account_type_code"] == "D1"
    assert accounts["account:quasi"]["pboc_account_type_code"] == "R4"
    monthly = {
        row["account_id"]: row
        for row in projected["credit_account_monthly_performance"]
    }
    assert monthly["account:loan"]["performance_month"] == "2025-05"
    assert monthly["account:loan"]["status_amount"] == "100"
    assert monthly["account:loan"]["status_amount_semantics"] == "delinquent_amount"
    assert monthly["account:quasi"]["performance_month"] == "2025-05"
    assert monthly["account:quasi"]["status_code"] == "1"
    assert monthly["account:quasi"]["status_amount"] == "1000"
    assert monthly["account:quasi"]["status_amount_semantics"] == "overdraft_balance"
    assert "overdue_amount" not in monthly["account:quasi"]
    responsibilities = {
        row["related_party_category"]: row
        for row in projected["repayment_responsibilities"]
    }
    assert responsibilities["person"]["overdue_months"] == 2
    assert responsibilities["organization"]["repayment_status_code"] == "N"
    assert all(
        "overdue_months_or_repayment_status" not in row
        for row in responsibilities.values()
    )
    housing = projected["housing_fund_records"][0]
    assert housing["public_record_id"] == "housing:1"
    assert "housing_fund_record_id" not in housing
    assert housing["personal_contribution_ratio_percent"] == 12
    assert housing["employer_contribution_ratio_percent"] == 12
    penalty = projected["administrative_penalty_records"][0]
    assert penalty["effective_month"] == "2021-08"
    assert penalty["end_month"] == "2024-07"
    assert projected["annotation_statements"][0]["annotation_statement_group_id"]
    assert "pboc_extension_fields" not in projected
    assert any(
        row["issue_code"] == "canonical_summary_cell_unmapped"
        and row.get("target_record_id") is None
        for row in projected["extraction_issues"]
    )


def test_v2_routes_account_events_and_builds_schema_native_dataset_status() -> None:
    source = {
        "personal_detail_account_events": [
            _record(
                "event:latest",
                account_event_id="event:latest",
                account_id="account:1",
                event_type="latest_repayment",
            ),
            _record(
                "event:special",
                account_event_id="event:special",
                account_id="account:1",
                event_type="special_event_note",
                details="sample event",
            ),
            _record(
                "event:transaction",
                account_event_id="event:transaction",
                account_id="account:1",
                event_type="special_transaction",
                transaction_type="debt restructuring",
            ),
            _record(
                "event:unknown",
                account_event_id="event:unknown",
                account_id="account:1",
                event_type="future_event_type",
                details="future extension",
            ),
        ],
        "personal_detail_dataset_status": [
            _record(
                "status:events",
                dataset_status_id="status:events",
                dataset_name="personal_detail_account_events",
                applicability="applicable",
                presence_status="observed_nonempty",
                observed_row_count=4,
                reason="records_projected",
            ),
            _record(
                "status:statements",
                dataset_status_id="status:statements",
                dataset_name="statements",
                applicability="applicable",
                presence_status="explicitly_empty",
                observed_row_count=0,
                reason="source_explicitly_empty",
            ),
            _record(
                "status:annotations",
                dataset_status_id="status:annotations",
                dataset_name="annotations",
                applicability="applicable",
                presence_status="explicitly_empty",
                observed_row_count=0,
                reason="source_explicitly_empty",
            ),
        ],
    }

    projected = project_personal_detail_datasets(source)

    assert len(projected["credit_account_latest_repayments"]) == 1
    assert len(projected["credit_account_special_events"]) == 1
    assert len(projected["credit_account_special_transactions"]) == 1
    assert "pboc_extension_fields" not in projected
    assert any(
        row["issue_code"] == "canonical_account_event_type_unresolved"
        and row["target_record_id"] == "event:unknown"
        for row in projected["extraction_issues"]
    )

    statuses = {row["dataset_name"]: row for row in projected["dataset_status"]}
    assert set(statuses) == set(PBOC_DATASET_ORDER) - {
        "dataset_status",
        "field_observations",
        "extraction_issues",
        "extraction_issue_evidence",
        "annotation_statements",
        "annotation_statement_groups",
        "pboc_extension_fields",
    }
    assert len({row["dataset_status_record_id"] for row in statuses.values()}) == len(statuses)
    for dataset_name in (
        "credit_account_latest_repayments",
        "credit_account_special_events",
        "credit_account_special_transactions",
    ):
        assert statuses[dataset_name]["presence_status"] == "partial"
    assert all(
        row["presence_status"] == "unknown"
        for name, row in statuses.items()
        if name not in {
            "credit_account_latest_repayments",
            "credit_account_special_events",
            "credit_account_special_transactions",
        }
    )
    assert {
        row["target_dataset"]
        for row in projected["extraction_issues"]
        if row["issue_code"] == "unresolved_account_event_link"
    } == {
        "credit_account_latest_repayments",
        "credit_account_special_events",
        "credit_account_special_transactions",
    }
    assert all(
        row.get("account_id") is None
        for name in (
            "credit_account_latest_repayments",
            "credit_account_special_events",
            "credit_account_special_transactions",
        )
        for row in projected[name]
    )
    assert statuses["credit_card_large_installments"]["presence_status"] == "unknown"
    assert "source_dataset_name" not in statuses["fraud_warnings"]
    assert not any(row["dataset_name"].startswith("personal_detail_") for row in statuses.values())


def test_v2_does_not_treat_wrapper_record_id_as_emitted_account_foreign_key() -> None:
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                _record(
                    "account:wrapper-only",
                    sequence=1,
                    account_type="non_revolving_loan",
                )
            ],
            "personal_detail_account_events": [
                _record(
                    "event:1",
                    account_event_id="event:1",
                    account_id="account:wrapper-only",
                    event_type="special_transaction",
                    canonical_raw={"account_id": "account:wrapper-only"},
                )
            ],
        }
    )

    event = projected["credit_account_special_transactions"][0]
    assert event["account_id"] is None
    assert event["normalized"]["account_id"] is None
    assert event["canonical_raw"]["account_id"] == "account:wrapper-only"
    issue = next(
        row
        for row in projected["extraction_issues"]
        if row.get("issue_code") == "unresolved_account_event_link"
    )
    assert issue["target_record_id"] == "event:1"


def _assert_linked_issue_targets_and_fields_are_addressable(
    projected: dict[str, list[dict[str, Any]]],
) -> None:
    records_by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset_name, records in projected.items():
        if not isinstance(records, list):
            continue
        record_index: dict[str, dict[str, Any]] = {}
        for record in records:
            values = (
                record["normalized"]
                if isinstance(record.get("normalized"), dict)
                else record
            )
            identities = {str(record.get("record_id") or "")}
            identities.update(
                str(value)
                for key, value in values.items()
                if key.endswith("_id") and value not in (None, "")
            )
            for identity in identities - {""}:
                record_index[identity] = values
        records_by_dataset[dataset_name] = record_index

    for issue in projected.get("extraction_issues") or []:
        values = (
            issue["normalized"]
            if isinstance(issue.get("normalized"), dict)
            else issue
        )
        target_record_id = str(values.get("target_record_id") or "")
        if not target_record_id:
            continue
        target_dataset = str(values.get("target_dataset") or "")
        assert target_record_id in records_by_dataset.get(target_dataset, {})
        field_name = str(values.get("field_name") or "")
        if field_name:
            assert field_name in records_by_dataset[target_dataset][target_record_id]


def test_v2_account_event_identifier_linkage_is_silent_or_localized_to_account_id() -> None:
    parent_identifier = "B10211000H0001350220190204838"
    conflicting_identifier = "D10053310H00012022052901021012089466554314"
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                _record(
                    "account:1",
                    account_id="account:1",
                    sequence=1,
                    account_type="credit_card",
                    account_identifier=parent_identifier,
                )
            ],
            "personal_detail_account_events": [
                _record(
                    "event:matching",
                    account_event_id="event:matching",
                    event_type="large_installment",
                    account_id="account:1",
                    account_identifier=parent_identifier,
                ),
                _record(
                    "event:conflicting",
                    account_event_id="event:conflicting",
                    event_type="large_installment",
                    account_id="account:1",
                    account_identifier=conflicting_identifier,
                ),
            ],
        }
    )

    events = {
        row["record_id"]: row
        for row in projected["credit_card_large_installments"]
    }
    matching = events["event:matching"]
    conflicting = events["event:conflicting"]
    issues = [
        row.get("normalized", row) for row in projected["extraction_issues"]
    ]

    assert matching["account_id"] == "account:1"
    assert "account_identifier" not in matching
    assert "account_identifier" not in matching.get("normalized", {})
    assert not any(
        issue.get("target_record_id") == "event:matching" for issue in issues
    )
    assert conflicting["account_id"] is None
    assert "account_identifier" not in conflicting
    assert "account_identifier" not in conflicting.get("normalized", {})
    conflict_issue = next(
        issue
        for issue in issues
        if issue.get("target_record_id") == "event:conflicting"
    )
    assert conflict_issue["issue_code"] == "account_event_parent_identifier_conflict"
    assert conflict_issue["target_dataset"] == "credit_card_large_installments"
    assert conflict_issue["field_name"] == "account_id"
    assert not any(
        issue.get("field_name") == "account_identifier" for issue in issues
    )
    _assert_linked_issue_targets_and_fields_are_addressable(projected)


def test_v2_preserves_invalid_month_source_without_emitting_invalid_month() -> None:
    projected = project_personal_detail_datasets(
        {
            "administrative_penalty_records": [
                _record(
                    "penalty:invalid",
                    administrative_penalty_id="penalty:invalid",
                    effective_date="2021-19-01",
                )
            ],
            "postpaid_payment_history": [
                _record(
                    "postpaid:1",
                    postpaid_payment_history_id="postpaid:1",
                    year="2024",
                    month="7",
                    status=1,
                )
            ],
        }
    )

    penalty = projected["administrative_penalty_records"][0]
    assert "effective_month" not in penalty
    assert penalty["source_effective_date"] == "2021-19-01"
    postpaid = projected["postpaid_monthly_performance"][0]
    assert postpaid["performance_month"] == "2024-07"
    assert postpaid["status_code"] == "1"


def test_v2_preserves_sparse_uncertainty_as_typed_control_datasets() -> None:
    projected = project_personal_detail_datasets(
        {
            "personal_detail_field_observations": [
                _record(
                    "field:1",
                    field_observation_id="field:1",
                    dataset_name="personal_profile",
                    business_record_id="personal_profile:primary",
                    field_name="work_phone",
                    observation_status="not_observed",
                ),
                _record(
                    "field:2",
                    field_observation_id="field:2",
                    dataset_name="datasets",
                    business_record_id="unresolved_record",
                    field_name="balance",
                    observation_status="ambiguous",
                ),
            ],
            "personal_detail_extraction_issues": [
                _record(
                    "issue:1",
                    extraction_issue_id="issue:1",
                    category="ocr_cell_level_error",
                    issue_code="pboc_cell_contract_unresolved",
                    severity="warning",
                    status="open",
                    target_dataset="credit_lines",
                    field_name="facility_limit",
                )
            ],
        }
    )

    observation = next(
        row
        for row in projected["field_observations"]
        if row["dataset_name"] == "subject_employment"
        and row["field_name"] == "employer_phone"
    )
    unresolved = next(
        row for row in projected["field_observations"] if row["field_observation_id"] == "field:2"
    )
    issue = next(
        row
        for row in projected["extraction_issues"]
        if row.get("target_dataset") == "credit_agreements"
    )
    assert observation["observation_status"] == "unreadable"
    assert observation["business_record_id"] == "unresolved_record"
    assert unresolved["dataset_name"] == "unknown"
    assert unresolved["source_dataset_name"] == "datasets"
    assert issue["target_dataset"] == "credit_agreements"
    assert any(
        row["dataset_name"] == "credit_agreements" and row["field_name"] == "facility_limit"
        for row in projected["field_observations"]
    )
    assert any(
        row.get("issue_code") == "canonical_profile_phone_relation_unresolved"
        and row.get("target_dataset") == "subject_employment"
        and row.get("field_name") == "employer_phone"
        and "target_record_id" not in row
        for row in projected["extraction_issues"]
    )
    assert "pboc_extension_fields" not in projected


def test_v2_dictionary_covers_pboc_catalog_and_logical_types() -> None:
    dictionary = personal_detail_data_dictionary()

    assert dictionary["version"] == "2.0.0"
    assert set(PBOC_DATASET_ORDER) <= set(dictionary["datasets"])
    assert dictionary["logical_types"]["Month"]["json_type"] == "string"
    assert dictionary["logical_types"]["Long"]["json_type"] == "string"
    assert dictionary["datasets"]["housing_fund_records"]["columns"][
        "personal_contribution_ratio_percent"
    ]["logical_type"] == "Short"


def test_canonical_sample_business_field_catalog_is_frozen() -> None:
    """Keep broader official-spec additions outside the sample-vetted API."""

    controls = {
        "field_observations",
        "extraction_issues",
        "extraction_issue_evidence",
        "pboc_extension_fields",
        "dataset_status",
    }
    catalog = personal_detail_data_dictionary()["datasets"]
    sample_business_fields = {
        dataset_name: sorted(catalog[dataset_name]["columns"])
        for dataset_name in PBOC_DATASET_ORDER
        if dataset_name not in controls
    }
    digest = hashlib.sha256(
        json.dumps(
            sample_business_fields,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert len(sample_business_fields) == 47
    assert sum(map(len, sample_business_fields.values())) == 441
    assert digest == "65cd98fd649baade7bef99d8c51215e687bd2f919614295079f0dd7c400fa0d7"


def test_v2_projects_credit_line_validity_under_public_field_name() -> None:
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                _record(
                    "account:perpetual",
                    account_id="account:perpetual",
                    account_type="revolving_loan_account",
                    validity_type="perpetual",
                    canonical_raw={"due_date": "长期"},
                )
            ],
            "personal_detail_field_observations": [
                _record(
                    "field:validity",
                    field_observation_id="field:validity",
                    dataset_name="credit_accounts",
                    business_record_id="account:perpetual",
                    field_name="validity_type",
                    observation_status="ambiguous",
                    raw_value="长期",
                )
            ],
        }
    )

    account = projected["credit_accounts"][0]
    assert account["credit_line_validity_type"] == "perpetual"
    assert "validity_type" not in account
    observation = projected["field_observations"][0]
    assert observation["field_name"] == "credit_line_validity_type"
    columns = personal_detail_data_dictionary()["datasets"]["credit_accounts"][
        "columns"
    ]
    assert "credit_line_validity_type" in columns
    assert "validity_type" not in columns


def _canonical_dataset(name: str, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": f"ds_{name}",
        "name": name,
        "label": name,
        "type": name,
        "section_id": "sec_personal_basic",
        "csv": f"001_datasets/{name}.csv",
        "row_count": 1,
        "grain": f"one row per {name}",
        "primary_key": "record_id",
        "schema_version": "1.0",
        "storage_role": "canonical",
        "record_path": "rows",
        "status": "complete",
        "columns": [],
        "completeness": {
            "expected_row_count": 1,
            "emitted_row_count": 1,
            "omitted_row_count": 0,
            "verified": True,
            "basis": "unit_test",
        },
        "rows": [
            {
                "record_id": str(next(value for key, value in row.items() if key.endswith("_id"))),
                "normalized": row,
                "canonical_raw": dict(row),
                "raw": dict(row),
                "source": {},
            }
        ],
    }


def test_v2_schema_is_the_only_registered_detailed_personal_contract() -> None:
    assert get_projection_schema("personal_credit_report_detailed").version == "2.0.0"
    assert get_projection_schema("personal_credit_report_detailed_v2") is None
    payload = {
        "schema": {
            "name": "docmirror.community",
            "version": "3.0.0",
            "edition": "community",
            "domain": "personal_credit_report_detailed",
            "support_level": "beta",
        },
        "document": {
            "id": "document:1",
            "type": "personal_credit_report_detailed",
            "title": "个人信用报告",
            "page_count": 1,
            "language": ["zh-CN"],
            "source_file": {
                "name": "report.pdf",
                "mime_type": "application/pdf",
                "sha256": f"sha256:{'0' * 64}",
            },
            "units": {},
            "domain_schema": {
                "id": "personal_credit_report_detailed",
                "version": "2.0.0",
                "contract_uri": (
                    "https://valuemapglobal.github.io/DocMirror/schemas/"
                    "personal_credit_report_detailed.schema.json"
                ),
                "compatibility": "canonical-v2; detailed-report-only",
                "dataset_status_semantics": {
                    "mode": "potentially_flawed_only",
                    "present_dataset_without_status": "silently_trusted_complete",
                    "absent_dataset_without_status": "silently_trusted_empty_or_not_applicable",
                    "status_row_present": "partial_unknown_or_failed_extraction",
                },
            },
        },
        "sections": [],
        "datasets": [
            _canonical_dataset(
                "report_metadata",
                {
                    "report_metadata_id": "metadata:1",
                    "report_number": "2025052510051518624525",
                    "report_time": "2025-05-25T10:05:15+08:00",
                },
            ),
            _canonical_dataset(
                "report_query",
                {
                    "report_query_id": "query:1",
                    "report_number": "2025052510051518624525",
                    "query_institution": "中国人民银行征信中心",
                    "query_reason": "本人查询",
                },
            ),
            _canonical_dataset(
                "subject_profile",
                {"subject_profile_id": "profile:1", "birth_date": "1981-08-15"},
            ),
        ],
        "reading": {},
        "files": {},
        "warnings": [],
    }

    validation = validate_projection_payload("personal_credit_report_detailed", payload)
    assert validation.valid, validation.errors


def test_real_scanned_personal_detail_facts_activate_compact_community_projection(
    tmp_path: Path,
) -> None:
    source = {
        "personal_report_metadata": [
            _record(
                "metadata:1",
                personal_report_metadata_id="metadata:1",
                report_number="2025052510051518624525",
                report_time="2025-05-25T10:05:15+08:00",
                query_institution="中国人民银行征信中心",
                query_reason="本人查询",
            )
        ],
        "personal_profile": [
            {
                "record_id": "profile:1",
                "normalized": {
                    "personal_profile_id": "profile:1",
                    "birth_date": "1981-08-15",
                    "employment_status": None,
                },
                "canonical_raw": {
                    "birth_date": "1981.08.15",
                    "employment_status": "职贝",
                },
                "raw": {
                    "birth_date": "1981.08.15",
                    "employment_status": "职贝",
                },
            }
        ],
        "personal_detail_field_observations": [
            _record(
                "field:1",
                field_observation_id="field:1",
                dataset_name="personal_profile",
                business_record_id="profile:1",
                field_name="employment_status",
                observation_status="not_observed",
                raw_value="职贝",
            )
        ],
        "credit_accounts": [
            _record(
                "account:1",
                account_id="account:1",
                category_sequence=1,
                account_type="quasi_credit_card",
                institution="示例银行",
                currency="CNY",
                account_status="active",
                account_identifier_source="internal_anchor_diagnostic",
            )
        ],
        "credit_lines": [
            _record(
                "agreement:1",
                credit_line_id="agreement:1",
                sequence=2,
                limit_identifier="LIMIT-02",
                institution="示例银行",
                facility_type="循环贷款额度",
            )
        ],
        "repayment_records": [
            _record(
                "monthly:1",
                repayment_id="monthly:1",
                account_id="account:1",
                year=2025,
                month=5,
                status="1",
                overdue_amount=1000,
            )
        ],
        "personal_detail_dataset_status": [
            _record(
                "status:metadata",
                dataset_status_id="status:metadata",
                dataset_name="personal_report_metadata",
                presence_status="observed_nonempty",
                observed_row_count=1,
            ),
            _record(
                "status:accounts",
                dataset_status_id="status:accounts",
                dataset_name="credit_accounts",
                presence_status="observed_nonempty",
                observed_row_count=1,
            ),
            _record(
                "status:repayments",
                dataset_status_id="status:repayments",
                dataset_name="repayment_records",
                presence_status="observed_nonempty",
                observed_row_count=1,
            ),
        ],
        "personal_detail_extraction_issues": [
            _record(
                "issue:profile-employment",
                extraction_issue_id="issue:profile-employment",
                category="ocr_cell_level_error",
                issue_code="pboc_cell_contract_unresolved",
                severity="warning",
                status="requires_review",
                target_dataset="personal_profile",
                target_record_id="profile:1",
                field_name="employment_status",
                observed_value="职贝",
            ),
            _record(
                "issue:community-structured",
                extraction_issue_id="issue:community-structured",
                category="page_continuation",
                issue_code="source_sequence_or_count_gap",
                severity="warning",
                status="requires_review",
                target_dataset="employment_records",
                observed_value={"observed_row_count": 1},
                candidate_value={"missing_sequences": [2]},
                reason_codes=["dataset_incomplete"],
            )
        ],
    }
    semantic = personal_detail_semantic_extensions()
    projected_datasets = project_personal_detail_datasets(source)
    projection = {
        "projector_id": "credit_report",
        "document_type": "personal_credit_report_detailed",
        "domain_facts": {
            "document_label": "个人信用报告",
            # These are the exact stable facts emitted by the real variant
            # router for scanned detailed reports.
            "report_subtype": "personal_detail",
            "content_mode": "scanned_ocr",
            "data_dictionary": personal_detail_data_dictionary(),
            **{
                f"personal_detail_v2_expected_{name}_count": len(rows)
                for name, rows in projected_datasets.items()
            },
        },
        "semantic": semantic,
        "datasets": projected_datasets,
        "sections": [
            {
                "id": "sec_personal_basic",
                "title": "个人基本信息",
                "type": "basic_information",
                "page_start": 1,
                "page_end": 1,
            }
        ],
    }
    result = ParseResult(
        entities=DocumentEntities(document_type="personal_credit_report_detailed"),
        pages=[PageContent(page_number=1)],
    )
    source_pdf = tmp_path / "report.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")

    projected = project_community_bundle(
        seal_parse_result(result),
        file_path=str(source_pdf),
        projection_data=projection,
        projection_policy=dict(semantic["community_projection_overrides"]),
    )
    bundle = _CreditReportCommunityBundle(
        schema=projected.schema,
        document=projected.document,
        sections=projected.sections,
        datasets=projected.datasets,
        files=projected.files,
        warnings=projected.warnings,
        result=projected.result,
        source_fingerprint=projected.source_fingerprint,
        parse_result_schema=projected.parse_result_schema,
        classification=projected.classification,
        domain=projected.domain,
        diagnostics=projected.diagnostics,
        content_markdown_override=projected.content_markdown_override,
    )
    payload = bundle.json_payload()

    assert payload["document"]["domain_schema"]["version"] == "2.0.0"
    assert validate_projection_payload("community", payload).valid
    v2_validation = validate_projection_payload("personal_credit_report_detailed", payload)
    assert v2_validation.valid, v2_validation.errors
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    dictionary_datasets = personal_detail_data_dictionary()["datasets"]
    for name, dataset in datasets.items():
        declared_fields = set(dictionary_datasets[name]["columns"])
        assert all(
            set(row["normalized"]) == declared_fields for row in dataset["rows"]
        )
        assert all(set(row["raw"]) <= declared_fields for row in dataset["rows"])
        assert all(
            set(row["canonical_raw"]) <= declared_fields
            for row in dataset["rows"]
        )

    profile = datasets["subject_profile"]["rows"][0]
    assert profile["canonical_raw"] == {"employment_status": "职贝"}
    assert profile["raw"] == {"employment_status": "职贝"}
    assert set(profile["canonical_raw"]) == set(profile["raw"]) == {"employment_status"}
    assert "mobile_phone" not in profile["normalized"]
    assert "work_phone" not in profile["normalized"]
    assert "residence_phone" not in profile["normalized"]
    account = datasets["credit_accounts"]["rows"][0]
    assert account["canonical_raw"] == {}
    assert account["raw"] == {}
    assert account["source"] == {}
    assert "confidence" not in account
    assert "review" not in account
    assert account["normalized"]["sequence"] == 1
    assert account["normalized"]["management_institution"] == "示例银行"
    assert account["normalized"]["account_currency"] == "CNY"
    assert account["normalized"]["account_lifecycle_state"] == "open"
    for private_field in (
        "category_sequence",
        "institution",
        "currency",
        "account_status",
        "account_identifier_source",
    ):
        assert private_field not in account["normalized"]
    agreement = datasets["credit_agreements"]["rows"][0]["normalized"]
    assert agreement["sequence"] == 2
    assert agreement["limit_identifier"] == "LIMIT-02"
    profile_columns = {
        column["key"]: column for column in datasets["subject_profile"]["columns"]
    }
    assert profile_columns["employment_status"]["raw_available"] is True
    assert profile_columns["birth_date"]["raw_available"] is False
    assert payload["document"]["domain_schema"]["dataset_status_semantics"] == {
        "mode": "potentially_flawed_only",
        "present_dataset_without_status": "silently_trusted_complete",
        "absent_dataset_without_status": "silently_trusted_empty_or_not_applicable",
        "status_row_present": "partial_unknown_or_failed_extraction",
    }
    assert datasets["dataset_status"]["sparse_status_semantics"] == payload["document"][
        "domain_schema"
    ]["dataset_status_semantics"]
    profile_status = next(
        row["normalized"]
        for row in datasets["dataset_status"]["rows"]
        if row["normalized"]["dataset_name"] == "subject_profile"
    )
    assert profile_status["presence_status"] == "partial"
    assert datasets["subject_profile"]["status"] == profile_status["presence_status"]
    assert datasets["subject_profile"]["completeness"]["verified"] is False
    for key in ("expected_row_count", "emitted_row_count", "omitted_row_count"):
        assert isinstance(datasets["subject_profile"]["completeness"][key], int)

    issue = next(
        row["normalized"]
        for row in datasets["extraction_issues"]["rows"]
        if row["record_id"] == "issue:community-structured"
    )
    evidence = [row["normalized"] for row in datasets["extraction_issue_evidence"]["rows"]]
    assert issue["observed_value_type"] == "object"
    assert issue["candidate_value_type"] == "object"
    assert any(
        row["evidence_kind"] == "observed"
        and row["evidence_path"] == "observed_row_count"
        and row["integer_value"] == 1
        for row in evidence
    )
    assert any(
        row["evidence_kind"] == "candidate"
        and row["evidence_path"] == "missing_sequences[0]"
        and row["integer_value"] == 2
        for row in evidence
    )
    assert any(
        row["evidence_kind"] == "reason" and row["string_value"] == "dataset_incomplete"
        for row in evidence
    )
    assert not any(
        isinstance(value, str) and value.lstrip().startswith(("{", "["))
        for value in issue.values()
    )
    observation = datasets["field_observations"]["rows"][0]["normalized"]
    assert observation["dataset_name"] == "subject_profile"
    assert "source_dataset_name" not in observation
    statuses = [row["normalized"] for row in datasets["dataset_status"]["rows"]]
    assert all(
        row["presence_status"] in {"not_observed", "partial", "extraction_failed", "unknown"}
        for row in statuses
    )
    # Sparse status rows name only sections that need attention: the uncertain
    # profile is present, while the silently trusted account relation is absent.
    status_names = {row["dataset_name"] for row in statuses}
    assert "subject_profile" in status_names
    assert "credit_accounts" not in status_names
    assert not {
        warning["code"]
        for warning in payload["warnings"]
        if warning["code"] in {"DATASET_COMPLETENESS_UNVERIFIED", "DATASET_ROW_COUNT_MISMATCH"}
    }


def test_wrapped_page_one_issues_target_final_community_rows_once(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Exercise the real metadata wrapper before v2 query derivation."""

    monkeypatch.setattr(PBOCPersonalDetailNativeParser, "records", lambda self, name: [])
    parse_context = SimpleNamespace(
        pages=[SimpleNamespace(page_number=1, source_page_number=1, tables=[])],
        _personal_detail_extraction_issues=[],
    )
    source = _extract_header_datasets(parse_context, "")
    source["personal_report_metadata"] = _records(
        "personal_report_metadata",
        source["personal_report_metadata"],
    )
    source["personal_detail_extraction_issues"] = _records(
        "personal_detail_extraction_issues",
        parse_context._personal_detail_extraction_issues,
    )
    source["personal_profile"] = [
        {
            "record_id": "personal_profile:primary",
            "personal_profile_id": "personal_profile:primary",
        }
    ]
    projected_datasets = project_personal_detail_datasets(source)
    semantic = personal_detail_semantic_extensions()
    projection = {
        "projector_id": "credit_report",
        "document_type": "personal_credit_report_detailed",
        "domain_facts": {
            "document_label": "个人信用报告",
            "report_subtype": "personal_detail",
            "content_mode": "scanned_ocr",
            "data_dictionary": personal_detail_data_dictionary(),
            **{
                f"personal_detail_v2_expected_{name}_count": len(rows)
                for name, rows in projected_datasets.items()
            },
        },
        "semantic": semantic,
        "datasets": projected_datasets,
        "sections": [],
    }
    parse_result = ParseResult(
        entities=DocumentEntities(document_type="personal_credit_report_detailed"),
        pages=[PageContent(page_number=1)],
    )
    source_pdf = tmp_path / "wrapped-page-one.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    projected = project_community_bundle(
        seal_parse_result(parse_result),
        file_path=str(source_pdf),
        projection_data=projection,
        projection_policy=dict(semantic["community_projection_overrides"]),
    )
    payload = _CreditReportCommunityBundle(
        schema=projected.schema,
        document=projected.document,
        sections=projected.sections,
        datasets=projected.datasets,
        files=projected.files,
        warnings=projected.warnings,
        result=projected.result,
        source_fingerprint=projected.source_fingerprint,
        parse_result_schema=projected.parse_result_schema,
        classification=projected.classification,
        domain=projected.domain,
        diagnostics=projected.diagnostics,
        content_markdown_override=projected.content_markdown_override,
    ).json_payload()

    assert validate_projection_payload("community", payload).valid
    v2_validation = validate_projection_payload("personal_credit_report_detailed", payload)
    assert v2_validation.valid, v2_validation.errors
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    final_ids = {
        name: {str(row["record_id"]) for row in datasets[name]["rows"]}
        for name in ("report_metadata", "report_query")
    }
    query_row = datasets["report_query"]["rows"][0]
    assert query_row["record_id"] == query_row["normalized"]["report_query_id"]

    page_one_issues = [
        row["normalized"]
        for row in datasets["extraction_issues"]["rows"]
        if row["normalized"].get("target_dataset")
        in {"report_metadata", "report_query"}
    ]
    keys = [
        (
            str(issue["target_dataset"]),
            str(issue["target_record_id"]),
            str(issue["field_name"]),
        )
        for issue in page_one_issues
    ]
    assert len(page_one_issues) == 12
    assert len(keys) == len(set(keys))
    assert all(target_id in final_ids[dataset_name] for dataset_name, target_id, _ in keys)
    assert sum(issue["target_dataset"] == "report_query" for issue in page_one_issues) == 7


def test_public_projection_never_attaches_raw_evidence_from_a_different_record_id() -> None:
    payload = {
        "datasets": [
            {
                "name": "subject_profile",
                "rows": [
                    {
                        "record_id": "profile:public",
                        "normalized": {"subject_profile_id": "profile:public"},
                        "raw": {},
                        "canonical_raw": {},
                        "source": {"logical_page": 1},
                    }
                ],
                "columns": [
                    {"key": "subject_profile_id", "raw_available": False},
                    {"key": "gender", "raw_available": True},
                ],
            },
            {
                "name": "extraction_issues",
                "rows": [
                    {
                        "record_id": "issue:gender",
                        "normalized": {
                            "status": "requires_review",
                            "target_dataset": "subject_profile",
                            "target_record_id": "profile:public",
                            "field_name": "gender",
                        },
                    }
                ],
                "columns": [],
            },
        ]
    }
    source_datasets = [
        SimpleNamespace(
            public={"name": "subject_profile"},
            rows=[
                {
                    "record_id": "profile:different",
                    "raw": {"gender": "女"},
                    "canonical_raw": {"gender": "女"},
                }
            ],
        )
    ]

    _compact_personal_detail_public_projection(
        payload,
        source_datasets=source_datasets,
    )

    profile = payload["datasets"][0]
    assert profile["rows"][0]["raw"] == {}
    assert profile["rows"][0]["canonical_raw"] == {}
    assert next(column for column in profile["columns"] if column["key"] == "gender")[
        "raw_available"
    ] is False


def test_success_row_drops_nonrequired_review_metadata() -> None:
    payload = {
        "datasets": [
            {
                "name": "subject_profile",
                "rows": [
                    {
                        "record_id": "profile:clean",
                        "normalized": {"subject_profile_id": "profile:clean", "gender": "男"},
                        "raw": {"gender": "男"},
                        "canonical_raw": {"gender": "男"},
                        "source": {"logical_page": 1},
                        "confidence": 0.99,
                        "review": {"required": False},
                    }
                ],
                "columns": [
                    {"key": "subject_profile_id", "raw_available": True},
                    {"key": "gender", "raw_available": True},
                ],
            }
        ]
    }

    _compact_personal_detail_public_projection(payload, source_datasets=[])

    row = payload["datasets"][0]["rows"][0]
    assert row["raw"] == {}
    assert row["canonical_raw"] == {}
    assert row["source"] == {}
    assert "confidence" not in row
    assert "review" not in row


def _lin_card_20_native_currency_slot_ref() -> dict[str, Any]:
    return {
        "source": "native_detail_table_cell",
        "logical_page": 23,
        "source_page": 12,
        "table_id": "pt_23_1",
        "coordinate_system": "pdf_points_top_left",
        "row": 1,
        "column": 5,
        "geometry_scope": "cell",
        "evidence_ids": [],
        "binding": "canonical_field_slot",
        "binding_quality": "canonical_header_column",
        "canonical_row": 1,
        "canonical_column": 5,
        "bbox": [263.5, 113.0, 307.5, 144.0],
        "field_slot_role": "value",
        "canonical_label_row": 0,
        "canonical_value_row": 1,
        "field_name": "currency",
    }


def _lin_card_20_corrected_currency_ref(
    field_name: str = "currency",
) -> dict[str, Any]:
    return {
        "source": "personal_detail_corrected_page_cell",
        "logical_page": 23,
        "source_page": 12,
        "bbox": [
            273.203125,
            124.56275727087709,
            300.2232201057983,
            133.86744477087709,
        ],
        "geometry_scope": "cell",
        "evidence_ids": [
            "personal_detail_page_reocr:source:12:crop:0.0000:0.0000:"
            "403.5000:595.5000:rotation:0:w44"
        ],
        "binding": "canonical_field_slot",
        "binding_quality": "canonical_field_slot",
        "field_slot_role": "value",
        "evidence_plane": "business_repair",
        "coordinate_system": "pdf_points_top_left",
        "field_name": field_name,
    }


def _project_lin_card_20_currency_record() -> dict[str, Any]:
    record_id = "credit_account:credit_card:20"
    return project_personal_detail_datasets(
        {
            "credit_accounts": [
                {
                    "record_id": record_id,
                    "account_id": record_id,
                    "sequence": 43,
                    "account_type": "credit_card",
                    "currency": "CNY",
                    "account_currency": "CNY",
                    "reporting_amount_currency": "CNY",
                    "canonical_raw": {
                        "currency": "人民币元",
                        "account_currency": "人民币元",
                        "reporting_amount_currency": "人民币元",
                    },
                    "source_refs": [
                        {
                            "source": "native_detail_table",
                            "logical_page": 22,
                            "source_page": 11,
                            "table_id": "pt_22_9",
                        },
                        {
                            "source": "candidate_b_account_anchor",
                            "logical_page": 23,
                            "source_page": 12,
                            "evidence_ids": ["native:account:20"],
                        },
                    ],
                    "source_refs_by_field": {
                        "currency": [
                            _lin_card_20_native_currency_slot_ref(),
                            _lin_card_20_corrected_currency_ref("currency"),
                        ],
                        "account_currency": [
                            _lin_card_20_corrected_currency_ref(
                                "account_currency"
                            )
                        ],
                        "reporting_amount_currency": [
                            _lin_card_20_corrected_currency_ref(
                                "reporting_amount_currency"
                            )
                        ],
                    },
                }
            ]
        }
    )["credit_accounts"][0]


def _compact_single_currency_source_row(
    source_row: dict[str, Any],
) -> dict[str, Any]:
    payload = {
        "datasets": [
            {
                "name": "credit_accounts",
                "rows": [
                    {
                        "record_id": source_row["record_id"],
                        "normalized": dict(source_row["normalized"]),
                        "canonical_raw": {},
                        "raw": {},
                        "source": {"page_range": [22, 23]},
                    }
                ],
                "columns": [
                    {"key": "account_currency", "raw_available": False},
                    {
                        "key": "reporting_amount_currency",
                        "raw_available": False,
                    },
                ],
            }
        ]
    }
    _compact_personal_detail_public_projection(
        payload,
        source_datasets=[
            SimpleNamespace(
                public={"name": "credit_accounts"},
                rows=[source_row],
            )
        ],
    )
    return payload["datasets"][0]


def test_account_currency_ref_promotion_accepts_wrapped_live_envelope() -> None:
    record_id = "credit_account:credit_card:20"
    source_row = project_personal_detail_datasets(
        {
            "credit_accounts": [
                {
                    "record_id": record_id,
                    "normalized": {
                        "account_id": record_id,
                        "account_type": "credit_card",
                        "account_currency": "CNY",
                        "reporting_amount_currency": "CNY",
                    },
                    "canonical_raw": {
                        "account_currency": "人民币元",
                        "reporting_amount_currency": "人民币元",
                    },
                    "source_refs_by_field": {
                        "currency": [
                            _lin_card_20_native_currency_slot_ref(),
                            _lin_card_20_corrected_currency_ref("currency"),
                        ],
                        "account_currency": [
                            _lin_card_20_corrected_currency_ref(
                                "account_currency"
                            )
                        ],
                        "reporting_amount_currency": [
                            _lin_card_20_corrected_currency_ref(
                                "reporting_amount_currency"
                            )
                        ],
                    },
                }
            ]
        }
    )["credit_accounts"][0]

    assert "source_refs_by_field" not in source_row
    assert {
        ref["field_name"]
        for ref in source_row["source_refs"]
        if ref.get("source") == "personal_detail_corrected_page_cell"
    } == {"account_currency", "reporting_amount_currency"}


def test_account_currency_raw_alias_survives_schema_and_community_compaction(
    tmp_path: Path,
) -> None:
    record_id = "credit_account:credit_card:20"
    source_row = _project_lin_card_20_currency_record()
    projected = {"credit_accounts": [source_row]}
    assert "source_refs_by_field" not in source_row
    assert {
        ref["field_name"]
        for ref in source_row["source_refs"]
        if ref.get("source") == "personal_detail_corrected_page_cell"
    } == {"account_currency", "reporting_amount_currency"}

    semantic = personal_detail_semantic_extensions()
    projection = {
        "projector_id": "credit_report",
        "document_type": "personal_credit_report_detailed",
        "domain_facts": {
            "document_label": "个人信用报告",
            "report_subtype": "personal_detail",
            "content_mode": "scanned_ocr",
            "data_dictionary": personal_detail_data_dictionary(),
            **{
                f"personal_detail_v2_expected_{name}_count": len(rows)
                for name, rows in projected.items()
            },
        },
        "semantic": semantic,
        "datasets": projected,
        "sections": [],
    }
    parse_result = ParseResult(
        entities=DocumentEntities(document_type="personal_credit_report_detailed"),
        pages=[PageContent(page_number=1)],
    )
    source_pdf = tmp_path / "currency-raw-envelope.pdf"
    source_pdf.write_bytes(b"%PDF-1.4\n%%EOF\n")
    projected_bundle = project_community_bundle(
        seal_parse_result(parse_result),
        file_path=str(source_pdf),
        projection_data=projection,
        projection_policy=dict(semantic["community_projection_overrides"]),
    )
    payload = _CreditReportCommunityBundle(
        schema=projected_bundle.schema,
        document=projected_bundle.document,
        sections=projected_bundle.sections,
        datasets=projected_bundle.datasets,
        files=projected_bundle.files,
        warnings=projected_bundle.warnings,
        result=projected_bundle.result,
        source_fingerprint=projected_bundle.source_fingerprint,
        parse_result_schema=projected_bundle.parse_result_schema,
        classification=projected_bundle.classification,
        domain=projected_bundle.domain,
        diagnostics=projected_bundle.diagnostics,
        content_markdown_override=projected_bundle.content_markdown_override,
    ).json_payload()

    assert validate_projection_payload("community", payload).valid
    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    dataset = datasets["credit_accounts"]
    row = dataset["rows"][0]
    assert row["normalized"]["account_currency"] == "CNY"
    assert row["normalized"]["reporting_amount_currency"] == "CNY"
    assert row["canonical_raw"]["account_currency"] == "人民币元"
    assert row["canonical_raw"]["reporting_amount_currency"] == "人民币元"
    assert row["raw"]["account_currency"] == "人民币元"
    assert row["raw"]["reporting_amount_currency"] == "人民币元"
    source_refs = row["source"]["source_refs"]
    assert {ref["field_name"] for ref in source_refs} == {
        "account_currency",
        "reporting_amount_currency",
    }
    assert all(
        ref["source"] == "personal_detail_corrected_page_cell"
        and ref["evidence_ids"]
        == _lin_card_20_corrected_currency_ref()["evidence_ids"]
        for ref in source_refs
    )
    assert row["source"]["page_range"] == [22, 23]
    columns = {str(column["key"]): column for column in dataset["columns"]}
    assert columns["account_currency"]["raw_available"] is True
    assert columns["reporting_amount_currency"]["raw_available"] is True
    active_currency_issues = [
        issue
        for issue_dataset in payload["datasets"]
        if issue_dataset.get("name") == "extraction_issues"
        for issue in issue_dataset.get("rows") or ()
        if str((issue.get("normalized") or {}).get("target_record_id") or "")
        == record_id
        and str((issue.get("normalized") or {}).get("field_name") or "")
        in {"currency", "account_currency", "reporting_amount_currency"}
        and str((issue.get("normalized") or {}).get("status") or "requires_review")
        not in {"resolved", "suppressed_redundant", "informational"}
    ]
    assert active_currency_issues == []


def test_account_currency_canonical_alias_overrides_stale_raw_compatibility() -> None:
    for case, stale_raw in (
        ("conflicting_alias", "美元"),
        ("null", None),
        ("normalized_code", "CNY"),
        ("missing_pool", ...),
    ):
        source_row = deepcopy(_project_lin_card_20_currency_record())
        if stale_raw is ...:
            source_row.pop("raw", None)
        else:
            source_row["raw"] = {
                "account_currency": stale_raw,
                "reporting_amount_currency": stale_raw,
            }

        dataset = _compact_single_currency_source_row(source_row)
        row = dataset["rows"][0]
        assert row["canonical_raw"] == {
            "account_currency": "人民币元",
            "reporting_amount_currency": "人民币元",
        }, case
        assert row["raw"] == row["canonical_raw"], case
        assert {
            ref["field_name"] for ref in row["source"]["source_refs"]
        } == {"account_currency", "reporting_amount_currency"}, case


def test_account_currency_trust_requires_canonical_source_alias() -> None:
    for case, canonical_raw in (
        ("missing", {}),
        (
            "conflicting",
            {
                "account_currency": "美元",
                "reporting_amount_currency": "美元",
            },
        ),
        (
            "normalized_not_printed_alias",
            {
                "account_currency": "CNY",
                "reporting_amount_currency": "CNY",
            },
        ),
    ):
        source_row = deepcopy(_project_lin_card_20_currency_record())
        source_row["canonical_raw"] = canonical_raw
        source_row["raw"] = {
            "account_currency": "人民币元",
            "reporting_amount_currency": "人民币元",
        }

        dataset = _compact_single_currency_source_row(source_row)
        row = dataset["rows"][0]
        assert row["canonical_raw"] == {}, case
        assert row["raw"] == {}, case
        assert row["source"] == {}, case
        assert not any(column["raw_available"] for column in dataset["columns"]), case


def test_account_currency_raw_alias_requires_exact_corrected_value_ref() -> None:
    record_id = "credit_account:credit_card:20"
    native_ref = _lin_card_20_native_currency_slot_ref()
    base_ref = _lin_card_20_corrected_currency_ref("account_currency")
    variants = {
        "missing_native_slot": base_ref,
        "injected_top_level": base_ref,
        "malformed_logical_page": {**base_ref, "logical_page": "not-a-page"},
        "boolean_logical_page": {**base_ref, "logical_page": True},
        "boolean_source_page": {**base_ref, "source_page": True},
        "foreign_logical_page": {**base_ref, "logical_page": 24},
        "foreign_source_page": {**base_ref, "source_page": 13},
        "foreign_account_cell": {
            **base_ref,
            "bbox": [90.0, 124.5, 120.0, 133.8],
        },
        "string_evidence_ids": {
            **base_ref,
            "evidence_ids": "personal_detail_page_reocr:source:12:w44",
        },
        "invalid_evidence_id": {
            **base_ref,
            "evidence_ids": ["ocr:sp0012:lp0023:0044"],
        },
        "not_a_value_slot": {**base_ref, "field_slot_role": "label"},
        "not_business_repair_evidence": {
            **base_ref,
            "evidence_plane": "native_static",
        },
        "wrong_binding": {**base_ref, "binding": "canonical_header_column"},
        "wrong_binding_quality": {
            **base_ref,
            "binding_quality": "canonical_header_column",
        },
        "wrong_coordinate_system": {
            **base_ref,
            "coordinate_system": "image_pixels_top_left",
        },
    }

    for case, corrected_ref in variants.items():
        native_refs = [] if case == "missing_native_slot" else [native_ref]
        top_level_refs = [corrected_ref] if case == "injected_top_level" else []
        refs_by_field = (
            {}
            if case == "injected_top_level"
            else {
                "currency": native_refs,
                "account_currency": [corrected_ref],
            }
        )
        projected = project_personal_detail_datasets(
            {
                "credit_accounts": [
                    {
                        "record_id": record_id,
                        "account_id": record_id,
                        "account_type": "credit_card",
                        "account_currency": "CNY",
                        "reporting_amount_currency": "CNY",
                        "canonical_raw": {
                            "account_currency": "人民币元",
                            "reporting_amount_currency": "人民币元",
                        },
                        "source_refs": top_level_refs,
                        "source_refs_by_field": refs_by_field,
                    }
                ]
            }
        )
        source_row = projected["credit_accounts"][0]
        assert not any(
            ref.get("source") == "personal_detail_corrected_page_cell"
            for ref in source_row.get("source_refs") or ()
        ), case

        dataset = _compact_single_currency_source_row(source_row)
        row = dataset["rows"][0]
        assert row["canonical_raw"] == {}, case
        assert row["raw"] == {}, case
        assert row["source"] == {}, case
        assert not any(column["raw_available"] for column in dataset["columns"]), case


def test_dataset_status_expected_less_than_emitted_is_an_explicit_population_conflict() -> None:
    payload = {
        "datasets": [
            {
                "name": "subject_profile",
                "status": "complete",
                "row_count": 2,
                "rows": [{}, {}],
                "completeness": {
                    "expected_row_count": 2,
                    "emitted_row_count": 2,
                    "omitted_row_count": 0,
                    "verified": True,
                    "basis": "row_conservation",
                },
            },
            {
                "name": "dataset_status",
                "rows": [
                    {
                        "normalized": {
                            "dataset_name": "subject_profile",
                            "presence_status": "observed_nonempty",
                            "expected_row_count": 1,
                        }
                    }
                ],
            },
        ]
    }

    _apply_personal_detail_dataset_status(payload)

    profile = payload["datasets"][0]
    assert profile["status"] == "partial"
    assert profile["completeness"] == {
        "expected_row_count": 2,
        "emitted_row_count": 2,
        "omitted_row_count": 0,
        "verified": False,
        "basis": (
            "personal_detail_dataset_status:observed_nonempty:"
            "expected_less_than_emitted"
        ),
    }


def test_v2_decodes_employment_mapping_blob_into_typed_fields() -> None:
    blob = json.dumps(
        {
            "编号": "3",
            "工作单位": "示例科技有限公司",
            "单位地址": "浙江省杭州市西湖区文三路一号",
            "职业": "专业技术人员",
            "进入本单位年份": "2021",
        },
        ensure_ascii=False,
    )

    projected = project_personal_detail_datasets(
        {
            "employment_records": [
                _record(
                    "employment:3",
                    normalized={"employment_record_id": None, "sequence": None},
                    canonical_raw={"values": blob},
                )
            ]
        }
    )

    wrapper = projected["subject_employment"][0]
    row = wrapper["normalized"]
    assert row["sequence"] == 3
    assert row["employer"] == "示例科技有限公司"
    assert row["occupation"] == "专业技术人员"
    assert row["entry_year"] == 2021
    assert "values" not in row
    assert "values" not in wrapper.get("canonical_raw", {})
    assert wrapper["review"]["source_values_blob"] == blob
    assert not any(
        issue["issue_code"] == "unstructured_multifield_blob"
        for issue in projected.get("extraction_issues", [])
    )


def test_v2_rejects_ambiguous_employment_blob_and_reports_it() -> None:
    blob = "编号=3;工作单位=示例科技有限公司;职业=职员"

    projected = project_personal_detail_datasets(
        {"employment_records": [_record("employment:3", values=blob)]}
    )

    row = projected["subject_employment"][0]
    assert "values" not in row
    assert "values" not in row.get("canonical_raw", {})
    assert row["review"]["source_values_blob"] == blob
    issue = next(
        issue
        for issue in projected["extraction_issues"]
        if issue["issue_code"] == "unstructured_multifield_blob"
    )
    assert issue["target_dataset"] == "subject_employment"
    status = next(
        row for row in projected["dataset_status"] if row["dataset_name"] == "subject_employment"
    )
    assert status["presence_status"] == "partial"


def test_v2_normalizes_valid_currency_and_suppresses_false_issue() -> None:
    false_issue = _record(
        "issue:currency",
        extraction_issue_id="issue:currency",
        category="ocr_cell_level_error",
        issue_code="pboc_cell_contract_unresolved",
        severity="warning",
        status="requires_review",
        target_dataset="repayment_records",
        target_record_id="monthly:1",
        field_name="reporting_amount_currency",
        observed_value="人民币元",
    )
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                _record("account:1", account_id="account:1", account_type="credit_card")
            ],
            "repayment_records": [
                _record(
                    "monthly:1",
                    repayment_id="monthly:1",
                    account_id="account:1",
                    year=2025,
                    month=1,
                    status="N",
                    overdue_amount=0,
                    reporting_amount_currency="人民币元",
                )
            ],
            "personal_detail_extraction_issues": [false_issue],
        }
    )

    assert projected["credit_account_monthly_performance"][0][
        "reporting_amount_currency"
    ] == "CNY"
    assert "extraction_issues" not in projected
    assert "field_observations" not in projected


def test_v2_normalizes_extended_currency_alias_and_declares_iso_codes() -> None:
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                _record(
                    "account:aud",
                    account_id="account:aud",
                    account_type="credit_card",
                    account_currency="澳元",
                    reporting_amount_currency="AUD",
                )
            ]
        }
    )

    row = projected["credit_accounts"][0]
    assert row["account_currency"] == "AUD"
    assert row["reporting_amount_currency"] == "AUD"
    currency_codes = personal_detail_data_dictionary()["enums"]["currency_code"]
    assert {"AUD", "CAD", "CHF", "SGD", "MOP"}.issubset(currency_codes)


def test_v2_types_account_repayment_periods_without_losing_source_absence() -> None:
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                _record(
                    "account:dash-periods",
                    normalized={
                        "account_id": "account:dash-periods",
                        "account_type": "non_revolving_loan",
                        "repayment_periods": "--",
                    },
                    repayment_periods="--",
                    canonical_raw={"repayment_periods": "--"},
                    raw={"repayment_periods": "--"},
                ),
                _record(
                    "account:numeric-periods",
                    account_id="account:numeric-periods",
                    account_type="non_revolving_loan",
                    repayment_periods="36",
                    canonical_raw={"repayment_periods": "36"},
                    raw={"repayment_periods": "36"},
                ),
                _record(
                    "account:invalid-periods",
                    account_id="account:invalid-periods",
                    account_type="non_revolving_loan",
                    repayment_periods="36?",
                    canonical_raw={"repayment_periods": "36?"},
                    raw={"repayment_periods": "36?"},
                ),
            ]
        }
    )

    rows = {
        (row.get("normalized") or row)["account_id"]: row
        for row in projected["credit_accounts"]
    }
    dash = rows["account:dash-periods"]
    assert dash["normalized"]["repayment_periods"] is None
    assert "repayment_periods" not in dash
    assert dash["canonical_raw"]["repayment_periods"] == "--"
    assert dash["raw"]["repayment_periods"] == "--"
    assert not any(
        issue.get("target_record_id") == "account:dash-periods"
        and issue.get("field_name") == "repayment_periods"
        for issue in projected.get("extraction_issues") or ()
    )
    numeric = rows["account:numeric-periods"]
    assert numeric["repayment_periods"] == 36
    assert type(numeric["repayment_periods"]) is int
    assert numeric["canonical_raw"]["repayment_periods"] == "36"
    assert numeric["raw"]["repayment_periods"] == "36"
    invalid = rows["account:invalid-periods"]
    assert invalid["repayment_periods"] is None
    assert invalid["canonical_raw"]["repayment_periods"] == "36?"
    assert invalid["raw"]["repayment_periods"] == "36?"
    invalid_issues = [
        issue
        for issue in projected["extraction_issues"]
        if issue.get("target_record_id") == "account:invalid-periods"
        and issue.get("field_name") == "repayment_periods"
    ]
    assert len(invalid_issues) == 1
    assert invalid_issues[0]["issue_code"] == "canonical_field_contract_failed"
    assert invalid_issues[0]["observed_value"] == "36?"


def test_yang_saved_community_datasets_keep_typed_repayment_absence() -> None:
    record_id = "credit_account:non_revolving_loan:7"
    source_row = {
        "record_id": record_id,
        "normalized": {
            "account_id": record_id,
            "account_type": "non_revolving_loan",
            "repayment_periods": "--",
        },
        "repayment_periods": "--",
        "canonical_raw": {"repayment_periods": "--"},
        "raw": {"repayment_periods": "--"},
    }
    payload = {
        "datasets": [
            {
                "name": "credit_accounts",
                "rows": [
                    {
                        "record_id": record_id,
                        "normalized": dict(source_row["normalized"]),
                        "repayment_periods": "--",
                        "canonical_raw": {},
                        "raw": {},
                        "source": {"page_range": [6, 6]},
                        "review": {
                            "status": "requires_review",
                            "extraction_status": "review",
                        },
                    }
                ],
                "columns": [
                    {"key": "repayment_periods", "raw_available": False}
                ],
            },
            {
                "name": "extraction_issues",
                "rows": [
                    {
                        "record_id": "issue:yang-account-identifier",
                        "normalized": {
                            "extraction_issue_id": "issue:yang-account-identifier",
                            "status": "requires_review",
                            "target_dataset": "credit_accounts",
                            "target_record_id": record_id,
                            "field_name": "account_identifier",
                        },
                    }
                ],
                "columns": [],
            },
        ]
    }

    _compact_personal_detail_public_projection(
        payload,
        source_datasets=[
            SimpleNamespace(
                public={"name": "credit_accounts"},
                rows=[source_row],
            )
        ],
    )

    datasets = {dataset["name"]: dataset for dataset in payload["datasets"]}
    account = datasets["credit_accounts"]["rows"][0]
    assert account["normalized"]["repayment_periods"] is None
    assert "repayment_periods" not in account
    assert account["canonical_raw"]["repayment_periods"] == "--"
    assert account["raw"]["repayment_periods"] == "--"
    assert datasets["credit_accounts"]["columns"][0]["raw_available"] is True
    assert not any(
        str((issue.get("normalized") or {}).get("target_record_id") or "")
        == record_id
        and str((issue.get("normalized") or {}).get("field_name") or "")
        == "repayment_periods"
        and str((issue.get("normalized") or {}).get("status") or "requires_review")
        not in {"resolved", "suppressed_redundant", "informational"}
        for issue in datasets["extraction_issues"]["rows"]
    )


def test_community_repayment_absence_requires_canonical_source_evidence() -> None:
    for case, normalized_value, canonical_raw in (
        ("conflicting_canonical", 36, {"repayment_periods": "36"}),
        ("missing_canonical", "--", {}),
    ):
        record_id = f"account:{case}"
        source_row = {
            "record_id": record_id,
            "normalized": {
                "account_id": record_id,
                "account_type": "non_revolving_loan",
                "repayment_periods": normalized_value,
            },
            "canonical_raw": canonical_raw,
            "raw": {"repayment_periods": "--"},
        }
        payload = {
            "datasets": [
                {
                    "name": "credit_accounts",
                    "rows": [
                        {
                            "record_id": record_id,
                            "normalized": dict(source_row["normalized"]),
                            "canonical_raw": {},
                            "raw": {},
                            "source": {"page_range": [6, 6]},
                        }
                    ],
                    "columns": [
                        {"key": "repayment_periods", "raw_available": False}
                    ],
                }
            ]
        }

        _compact_personal_detail_public_projection(
            payload,
            source_datasets=[
                SimpleNamespace(
                    public={"name": "credit_accounts"},
                    rows=[source_row],
                )
            ],
        )

        dataset = payload["datasets"][0]
        row = dataset["rows"][0]
        expected = 36 if case == "conflicting_canonical" else None
        assert row["normalized"]["repayment_periods"] == expected, case
        assert row["canonical_raw"] == {}, case
        assert row["raw"] == {}, case
        assert dataset["columns"][0]["raw_available"] is False, case


def test_v2_withholds_yuan_units_for_non_cny_currency_with_exact_issues() -> None:
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                _record(
                    "account:usd",
                    account_id="account:usd",
                    account_type="credit_card",
                    currency="USD",
                    account_currency="USD",
                    reporting_amount_currency="USD",
                    amount_unit="yuan",
                    reporting_amount_unit="yuan",
                )
            ]
        }
    )

    row = projected["credit_accounts"][0]
    assert "currency" not in row
    assert row["account_currency"] == "USD"
    assert row["reporting_amount_currency"] == "USD"
    assert row["amount_unit"] is None
    assert row["reporting_amount_unit"] is None
    assert row["canonical_raw"]["amount_unit"] == "yuan"
    assert row["canonical_raw"]["reporting_amount_unit"] == "yuan"
    issues = [
        issue
        for issue in projected["extraction_issues"]
        if issue["issue_code"] == "currency_amount_unit_conflict"
    ]
    assert {issue["field_name"] for issue in issues} == {
        "amount_unit",
        "reporting_amount_unit",
    }
    assert {issue["target_record_id"] for issue in issues} == {"account:usd"}


def test_v2_keeps_yuan_units_for_cny_currency() -> None:
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                _record(
                    "account:cny",
                    account_id="account:cny",
                    account_type="credit_card",
                    currency="CNY",
                    account_currency="CNY",
                    reporting_amount_currency="CNY",
                    amount_unit="yuan",
                    reporting_amount_unit="yuan",
                )
            ]
        }
    )

    row = projected["credit_accounts"][0]
    assert row["amount_unit"] == "yuan"
    assert row["reporting_amount_unit"] == "yuan"
    assert "extraction_issues" not in projected


def test_v2_withholds_invalid_header_values_in_every_canonical_copy() -> None:
    projected = project_personal_detail_datasets(
        {
            "personal_report_metadata": [
                _record(
                    "metadata:bad",
                    personal_report_metadata_id="metadata:bad",
                    report_number="20O5BAD",
                    report_time="2025-19-45 99:00",
                    subject_name="被查询者证件类型",
                    primary_id_type="身份证",
                    primary_id_number="11010519491231002X",
                    query_institution="本人",
                    query_reason="本人查询",
                )
            ]
        }
    )

    for dataset_name in ("report_metadata", "report_query"):
        row = projected[dataset_name][0]
        assert row["report_number"] is None
        assert row["report_time"] is None
        assert row["subject_name"] is None
        assert row["canonical_raw"]["report_number"] == "20O5BAD"
    affected = {
        issue["target_dataset"]
        for issue in projected["extraction_issues"]
        if issue["field_name"] == "report_number"
    }
    assert affected == {"report_metadata", "report_query"}


def test_v2_projection_policy_hides_conservation_facts_and_covers_all_datasets() -> None:
    policy = personal_detail_semantic_extensions()["community_projection_overrides"]

    assert set(policy["completeness"]) == set(PBOC_DATASET_ORDER)
    assert set(policy["internal_fields"]) == {
        f"personal_detail_v2_expected_{name}_count" for name in PBOC_DATASET_ORDER
    }


def test_v2_prefers_record_specific_schema_issue_over_unlinked_source_duplicate() -> None:
    source_issue = _record(
        "issue:unlinked",
        extraction_issue_id="issue:unlinked",
        category="ocr_cell_level_error",
        issue_code="pboc_cell_contract_unresolved",
        severity="warning",
        status="requires_review",
        target_dataset="credit_lines",
        field_name="used_limit",
        observed_value="ЁА",
    )

    projected = project_personal_detail_datasets(
        {
            "credit_lines": [
                _record(
                    "credit-line:1",
                    credit_line_id="credit-line:1",
                    used_limit="ЁА",
                )
            ],
            "personal_detail_extraction_issues": [source_issue],
        }
    )

    issues = [
        issue
        for issue in projected["extraction_issues"]
        if issue.get("field_name") == "used_limit"
    ]
    assert len(issues) == 1
    assert issues[0]["issue_code"] == "canonical_money_invalid"
    assert issues[0]["target_record_id"] == "credit-line:1"


def test_v2_credit_agreement_raw_alias_cannot_reenter_normalized_business_data() -> None:
    invalid_limit = "B10512900H00010010011135264974289"
    projected = project_personal_detail_datasets(
        {
            "credit_lines": [
                {
                    "record_id": "credit-line:1",
                    "credit_line_id": "credit-line:1",
                    "account_identifier": invalid_limit,
                    "total_limit": None,
                    "canonical_raw": {"total_limit": invalid_limit},
                    "raw": {"total_limit": invalid_limit},
                }
            ]
        }
    )

    row = projected["credit_agreements"][0]
    assert row["normalized"]["facility_limit"] is None
    assert "total_limit" not in row["normalized"]
    assert row["canonical_raw"]["facility_limit"] == invalid_limit
    assert row["raw"]["facility_limit"] == invalid_limit
    assert "total_limit" not in row["canonical_raw"]
    assert "total_limit" not in row["raw"]


def test_v2_keeps_facility_limit_and_credit_limit_as_distinct_business_fields() -> None:
    projected = project_personal_detail_datasets(
        {
            "credit_lines": [
                _record(
                    "credit-line:1",
                    credit_line_id="credit-line:1",
                    account_identifier="T10151210H0001ABC12345",
                    total_limit="100000",
                    credit_limit="25000",
                )
            ]
        }
    )

    row = projected["credit_agreements"][0]
    assert row["facility_limit"] == "100000"
    assert row["credit_limit"] == "25000"


def test_v2_links_unambiguous_source_issue_to_projected_business_record() -> None:
    projected = project_personal_detail_datasets(
        {
            "inquiry_records": [
                _record(
                    "inquiry:1",
                    inquiry_id="inquiry:1",
                    institution="本人 业 您",
                    inquiry_date="2025-01-02",
                    reason="本人查询",
                )
            ],
            "personal_detail_extraction_issues": [
                _record(
                    "issue:unlinked-inquiry",
                    extraction_issue_id="issue:unlinked-inquiry",
                    category="ocr_cell_level_error",
                    issue_code="pboc_cell_contract_unresolved",
                    severity="warning",
                    status="requires_review",
                    target_dataset="inquiries",
                    field_name="institution",
                    observed_value="本人 业 您",
                )
            ],
        }
    )

    issue = next(
        row
        for row in projected["extraction_issues"]
        if row.get("field_name") == "institution"
    )
    assert issue["target_record_id"] == "inquiry:1"


def test_v2_withholds_unknown_account_status_and_removes_raw_detail_blob() -> None:
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                _record(
                    "account:1",
                    account_id="account:1",
                    account_type="credit_card",
                    account_status="正常结清",
                    raw_detail_lines=[{"text": "账户状态 正常结清"}, {"text": "余额 100"}],
                )
            ]
        }
    )

    row = projected["credit_accounts"][0]
    assert "account_status" not in row
    assert row["account_lifecycle_state"] is None
    assert "raw_detail_lines" not in row
    assert "raw_detail_lines" not in row.get("normalized", {})
    assert any(
        issue.get("field_name") == "account_lifecycle_state"
        and issue["issue_code"] == "canonical_field_contract_failed"
        for issue in projected["extraction_issues"]
    )


def test_summary_scalar_failure_is_unknown_and_reported_without_text_fallback() -> None:
    content = prepare_personal_detail_source_collections(
        {
            "facts": {},
            "datasets": {
                "personal_detail_summary_cells": [
                    _record(
                        "cell:bad",
                        summary_cell_id="cell:bad",
                        summary_record_id="summary:1",
                        summary_type="逾期（透支）",
                        row_index=1,
                        column_index=2,
                        column_label="月份数",
                        value="二O个月",
                    )
                ]
            },
        }
    )

    metric = content["datasets"]["personal_detail_credit_summary_metrics"][0]
    assert metric["reporting_status"] == "unknown"
    assert metric["value_type"] == "unknown"
    assert "numeric_value" not in metric
    assert "text_value" not in metric
    issue = next(
        issue
        for issue in content["datasets"]["personal_detail_extraction_issues"]
        if issue["issue_code"] == "candidate_b_summary_scalar_unresolved"
    )
    assert issue["observed_value"] == "二O个月"


def test_failed_summary_scalar_is_issue_only_not_generic_extension() -> None:
    projected = project_personal_detail_datasets(
        {
            "personal_detail_summary_cells": [
                {
                    "record_id": "cell:bad",
                    "normalized": {
                        "summary_cell_id": "cell:bad",
                        "summary_record_id": "summary:1",
                        "row_index": 2,
                        "column_index": 4,
                        "column_label": "单月最高逾期/透支总额",
                        "value": None,
                    },
                    "canonical_raw": {"value": "393 我"},
                    "raw": {"value": "393 我"},
                }
            ],
            "personal_detail_extraction_issues": [
                _record(
                    "issue:bad-cell",
                    extraction_issue_id="issue:bad-cell",
                    category="ocr_cell_level_error",
                    issue_code="pboc_cell_contract_unresolved",
                    severity="warning",
                    status="requires_review",
                    target_record_id="cell:bad",
                    field_name="value",
                    observed_value="393 我",
                )
            ],
        }
    )

    assert "pboc_extension_fields" not in projected
    issue = next(
        row
        for row in projected["extraction_issues"]
        if row.get("issue_code") == "canonical_summary_cell_unmapped"
    )
    assert issue["target_dataset"] == "credit_business_overview"
    assert issue.get("target_record_id") is None


def test_source_endpoint_does_not_report_missing_population_when_row_count_is_met() -> None:
    content = prepare_personal_detail_source_collections(
        {
            "facts": {
                "personal_detail_source_completeness_ledger": {
                    "sequence_endpoints": {"residence_records": 3}
                }
            },
            "datasets": {
                "residence_records": [
                    _record("residence:1", sequence=1, address="一号"),
                    _record("residence:3a", sequence=3, address="三号甲"),
                    _record("residence:3b", sequence=3, address="三号乙"),
                ]
            },
        },
        final_dataset_counts={"residence_records": 3},
    )

    assert not any(
        issue.get("issue_code") == "source_sequence_or_count_gap"
        for issue in content["datasets"].get("personal_detail_extraction_issues", [])
    )


def test_v2_issue_evidence_stays_native_and_machine_readable() -> None:
    projected = project_personal_detail_datasets(
        {
            "personal_detail_extraction_issues": [
                _record(
                    "issue:structured",
                    extraction_issue_id="issue:structured",
                    category="page_continuation",
                    issue_code="source_sequence_or_count_gap",
                    severity="warning",
                    status="requires_review",
                    target_dataset="employment_records",
                    observed_value={"observed_row_count": 2},
                    candidate_value={"missing_sequences": [3, 4]},
                    reason_codes=["dataset_incomplete", "no_missing_row_invented"],
                )
            ]
        }
    )

    issue = projected["extraction_issues"][0]
    evidence = projected["extraction_issue_evidence"]
    assert issue["observed_value_type"] == "object"
    assert issue["candidate_value_type"] == "object"
    assert {
        (row["evidence_kind"], row["evidence_path"], row.get("integer_value"), row.get("string_value"))
        for row in evidence
    } >= {
        ("observed", "observed_row_count", 2, None),
        ("candidate", "missing_sequences[0]", 3, None),
        ("candidate", "missing_sequences[1]", 4, None),
        ("reason", "reason_codes[0]", None, "dataset_incomplete"),
    }
    assert issue["reason_code_count"] == 2
    assert "observed_value" not in issue
    assert "candidate_value" not in issue
    assert all(not isinstance(value, (dict, list)) for value in issue.values())


def test_credit_card_summary_total_maximum_and_minimum_limits_keep_distinct_metrics() -> None:
    cells = []
    for index, (label, value) in enumerate(
        (
            ("授信总额", "62000"),
            ("单家机构最高授信额", "50000"),
            ("单家机构最低授信额", "12000"),
        ),
        start=1,
    ):
        cells.append(
            _record(
                f"cell:{index}",
                summary_cell_id=f"cell:{index}",
                summary_record_id="summary:credit-card",
                summary_type="贷记卡账户",
                title="贷记卡账户信息汇总",
                row_index=1,
                column_index=index + 2,
                column_label=label,
                value=value,
            )
        )

    content = prepare_personal_detail_source_collections(
        {
            "facts": {},
            "datasets": {
            "personal_detail_summary_records": [
                _record(
                    "summary:credit-card",
                    summary_record_id="summary:credit-card",
                    summary_type="贷记卡账户",
                    title="贷记卡账户信息汇总",
                )
            ],
            "personal_detail_summary_cells": cells,
            },
        },
        final_dataset_counts={},
    )
    projected = project_personal_detail_datasets(content["datasets"])

    metrics = {
        row["metric_code"]: row["numeric_value"]
        for row in projected["credit_business_overview"]
    }
    assert metrics == {
        "total_credit_limit": "62000",
        "maximum_single_institution_limit": "50000",
        "minimum_single_institution_limit": "12000",
    }
