# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from copy import deepcopy

import pytest

from docmirror.models.schemas.registry import validate_projection_payload
from docmirror.plugins.credit_report.personal_brief_native import audit as audit_module
from docmirror.plugins.credit_report.personal_brief_native.audit import (
    append_personal_brief_observational_warnings,
    build_personal_brief_observational_findings,
)
from docmirror.plugins.credit_report.personal_brief_native.projector import (
    project_personal_brief_community_json,
)
from tests.unit.test_personal_brief_public_json_projection import (
    _dataset,
    _payload,
    _row,
)


def _valid_payload() -> dict[str, object]:
    payload = _payload()
    account = _dataset_by_name(payload, "credit_accounts")["rows"][0]["normalized"]
    account["used_amount_status"] = "not_reported"
    account["loan_amount_status"] = "not_applicable"
    metadata_dataset = next(
        dataset
        for dataset in payload["datasets"]
        if dataset["name"] == "personal_report_metadata"
    )
    metadata = metadata_dataset["rows"][0]["normalized"]
    identity = _dataset(
        "identity_documents",
        "sec_header",
        [
            _row(
                "identity:1",
                {
                    "holder_name": metadata["subject_name"],
                    "document_type": metadata["primary_id_type"],
                    "document_number": metadata["primary_id_number"],
                    "is_primary": True,
                },
            )
        ],
    )
    payload["datasets"].insert(1, identity)
    header = next(section for section in payload["sections"] if section["id"] == "sec_header")
    header["dataset_refs"].append(identity["id"])
    payload["reading"]["tables"].append(
        {
            "id": f"reading:{identity['id']}",
            "dataset_id": identity["id"],
            "section_id": identity["section_id"],
            "title": identity["label"],
            "column_keys": [column["key"] for column in identity["columns"]],
            "row_count": identity["row_count"],
        }
    )
    payload["reading"]["document_flow"].append(
        {
            "order": len(payload["reading"]["document_flow"]) + 1,
            "kind": "dataset",
            "ref_id": identity["id"],
        }
    )
    return payload


def _dataset_by_name(payload: dict[str, object], name: str) -> dict[str, object]:
    return next(dataset for dataset in payload["datasets"] if dataset["name"] == name)


def test_clean_personal_brief_audit_is_observational() -> None:
    semantic = _valid_payload()
    public = project_personal_brief_community_json(semantic)
    semantic_before = deepcopy(semantic)
    public_before = deepcopy(public)

    findings = build_personal_brief_observational_findings(semantic, public)
    audited = append_personal_brief_observational_warnings(semantic, public)

    assert findings == ()
    assert audited == public
    assert semantic == semantic_before
    assert public == public_before


def test_auditor_reports_contract_and_business_relation_conflicts_without_repair() -> None:
    semantic = _valid_payload()
    public = project_personal_brief_community_json(semantic)
    account = _dataset_by_name(public, "credit_accounts")["rows"][0]["normalized"]
    account["account_type"] = "future_account"
    account["balance_status"] = "not_reported"
    tax = _dataset_by_name(public, "tax_arrears_records")["rows"][0]["normalized"]
    tax["arrears_amount"] = "1.5"
    overdue = _dataset_by_name(public, "overdue_records")["rows"][0]["normalized"]
    overdue["account_id"] = "missing-account"
    public_before = deepcopy(public)

    findings = build_personal_brief_observational_findings(semantic, public)

    assert public == public_before
    assert {
        "PERSONAL_BRIEF_AUDIT_ENUM_CONTRACT_CONFLICT",
        "PERSONAL_BRIEF_AUDIT_AMOUNT_STATUS_CONFLICT",
        "PERSONAL_BRIEF_AUDIT_MONEY_VALUE_CONFLICT",
        "PERSONAL_BRIEF_AUDIT_ORPHAN_OVERDUE_RECORD",
        "PERSONAL_BRIEF_AUDIT_OVERDUE_STATE_CONFLICT",
    } <= {finding.code for finding in findings}
    assert account["account_type"] == "future_account"
    assert account["balance_status"] == "not_reported"
    assert tax["arrears_amount"] == "1.5"
    assert overdue["account_id"] == "missing-account"


def test_warning_adapter_changes_only_warnings_and_preserves_existing_rows() -> None:
    semantic = _valid_payload()
    account = _dataset_by_name(semantic, "credit_accounts")["rows"][0]["normalized"]
    account["balance_status"] = "not_reported"
    public = project_personal_brief_community_json(semantic)
    public["warnings"] = [
        {
            "code": "EXISTING_WARNING",
            "level": "warning",
            "message": "Existing parser warning.",
        }
    ]
    public_before = deepcopy(public)

    audited = append_personal_brief_observational_warnings(semantic, public)

    assert public == public_before
    for key in public:
        if key != "warnings":
            assert audited[key] == public[key]
    assert audited["warnings"][0] == public["warnings"][0]
    audit_warnings = [
        warning
        for warning in audited["warnings"]
        if warning["code"] == "PERSONAL_BRIEF_AUDIT_AMOUNT_STATUS_CONFLICT"
    ]
    assert len(audit_warnings) == 1
    assert audit_warnings[0]["dataset_id"] == "ds_credit_accounts"
    assert audit_warnings[0]["section_id"] == "sec_credit"
    assert audit_warnings[0]["page_range"] == [1, 2]
    assert "record_id=credit_account:1" in audit_warnings[0]["message"]
    validation = validate_projection_payload("community", audited)
    assert validation.valid, validation.errors


def test_warning_adapter_is_deterministic_and_deduplicates_repeated_runs() -> None:
    semantic = _valid_payload()
    account = _dataset_by_name(semantic, "credit_accounts")["rows"][0]["normalized"]
    account["balance_status"] = "not_reported"
    public = project_personal_brief_community_json(semantic)

    first = append_personal_brief_observational_warnings(semantic, public)
    second = append_personal_brief_observational_warnings(semantic, first)
    reversed_semantic = deepcopy(semantic)
    reversed_public = deepcopy(public)
    reversed_semantic["datasets"].reverse()
    reversed_public["datasets"].reverse()
    reversed_findings = build_personal_brief_observational_findings(
        reversed_semantic,
        reversed_public,
    )

    assert second == first
    assert reversed_findings == build_personal_brief_observational_findings(
        semantic,
        public,
    )


def test_incomplete_dataset_is_warned_without_mislabeling_counter_arithmetic() -> None:
    semantic = _valid_payload()
    public = project_personal_brief_community_json(semantic)
    accounts = _dataset_by_name(public, "credit_accounts")
    accounts["completeness"].update(
        {
            "expected_row_count": 2,
            "emitted_row_count": 1,
            "omitted_row_count": 1,
            "verified": False,
        }
    )

    findings = build_personal_brief_observational_findings(semantic, public)
    account_codes = [
        finding.code for finding in findings if finding.dataset == "credit_accounts"
    ]

    assert "PERSONAL_BRIEF_AUDIT_DATASET_INCOMPLETE" in account_codes
    assert "PERSONAL_BRIEF_AUDIT_DATASET_ENVELOPE_CONFLICT" not in account_codes


def test_duplicate_business_account_id_breaks_exact_overdue_join() -> None:
    semantic = _valid_payload()
    accounts = _dataset_by_name(semantic, "credit_accounts")
    duplicate = deepcopy(accounts["rows"][0])
    duplicate["record_id"] = "credit_account:2"
    duplicate["normalized"]["sequence"] = 2
    duplicate["normalized"]["source_sequence"] = 2
    accounts["rows"].append(duplicate)
    accounts["row_count"] = 2
    accounts["completeness"].update(
        {
            "expected_row_count": 2,
            "emitted_row_count": 2,
            "omitted_row_count": 0,
            "verified": True,
        }
    )
    public = project_personal_brief_community_json(semantic)

    findings = build_personal_brief_observational_findings(semantic, public)

    assert "PERSONAL_BRIEF_AUDIT_ACCOUNT_ID_CONFLICT" in {
        finding.code for finding in findings
    }


def test_projection_conservation_detects_a_dropped_business_field() -> None:
    semantic = _valid_payload()
    public = project_personal_brief_community_json(semantic)
    tax = _dataset_by_name(public, "tax_arrears_records")["rows"][0]["normalized"]
    tax.pop("taxpayer_identifier")

    findings = build_personal_brief_observational_findings(semantic, public)

    finding = next(
        item
        for item in findings
        if item.code == "PERSONAL_BRIEF_AUDIT_PROJECTION_CONSERVATION_CONFLICT"
        and item.dataset == "tax_arrears_records"
    )
    assert finding.fields == ("taxpayer_identifier",)


def test_audit_runtime_failure_is_fault_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic = _valid_payload()
    public = project_personal_brief_community_json(semantic)
    public_before = deepcopy(public)

    def _raise(*_args: object, **_kwargs: object) -> tuple[object, ...]:
        raise RuntimeError("synthetic audit failure")

    monkeypatch.setattr(
        audit_module,
        "build_personal_brief_observational_findings",
        _raise,
    )
    audited = audit_module.append_personal_brief_observational_warnings(
        semantic,
        public,
    )

    assert public == public_before
    for key in public:
        if key != "warnings":
            assert audited[key] == public[key]
    assert audited["warnings"] == [
        {
            "code": "PERSONAL_BRIEF_AUDIT_INTERNAL_ERROR",
            "level": "warning",
            "message": (
                "The observational auditor failed internally; extracted business data "
                "were retained unchanged (RuntimeError)."
            ),
        }
    ]
    validation = validate_projection_payload("community", audited)
    assert validation.valid, validation.errors


def test_warning_conversion_failure_is_also_fault_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    semantic = _valid_payload()
    account = _dataset_by_name(semantic, "credit_accounts")["rows"][0]["normalized"]
    account["balance_status"] = "not_reported"
    public = project_personal_brief_community_json(semantic)
    public_before = deepcopy(public)

    def _raise(*_args: object, **_kwargs: object) -> dict[str, object]:
        raise RuntimeError("synthetic warning conversion failure")

    monkeypatch.setattr(audit_module, "_warning_from_finding", _raise)
    audited = audit_module.append_personal_brief_observational_warnings(
        semantic,
        public,
    )

    assert public == public_before
    for key in public:
        if key != "warnings":
            assert audited[key] == public[key]
    assert [warning["code"] for warning in audited["warnings"]] == [
        "PERSONAL_BRIEF_AUDIT_INTERNAL_ERROR"
    ]
    assert audited["warnings"][0]["level"] == "warning"
    validation = validate_projection_payload("community", audited)
    assert validation.valid, validation.errors
