# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Source-bound cleanup contracts for final Candidate-B liability issues."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    collect_extraction_issues,
    make_issue,
    register_final_liability_issue_records,
)


def _ref(table_id: str, *, row: int = 1, column: int = 0) -> dict[str, object]:
    return {
        "source": "native_detail_tolerant_table_cell",
        "logical_page": 12,
        "source_page": 6,
        "table_id": table_id,
        "row": row,
        "column": column,
        "bbox": [20.0, 70.0 + row, 580.0, 88.0 + row],
        "geometry_scope": "cell",
    }


def _context(*issues: dict[str, object]) -> SimpleNamespace:
    return SimpleNamespace(
        _personal_detail_extraction_issues=list(issues),
        ocr_correction_audit=lambda: {"cell_anomalies": []},
        page_topology_audit=lambda: {"issues": []},
    )


def _packed_issue(
    *,
    ref: dict[str, object],
    affected_fields: list[str],
    contract_number: str = "",
    printed_sequence: str = "",
) -> dict[str, object]:
    retained = {"保证合同编号": contract_number} if contract_number else {}
    observed: dict[str, object] = {
        "header": ["管理机构 业务种类 开立日期 到期日期 责任人类型 还款责任金额 币种 保证合同编号"],
        "value": ["packed source witness"],
        "unresolved_reason": "typed_spans_with_ocr_residue",
        "retained_typed_fields": retained,
        "affected_fields": affected_fields,
    }
    if printed_sequence:
        observed["printed_sequence"] = printed_sequence
    return make_issue(
        category="ocr_cell_level_error",
        issue_code="candidate_b_packed_liability_row_unresolved",
        message="packed row unresolved",
        target_dataset="repayment_liability_records",
        observed_value=observed,
        source_refs=(ref,),
    )


def _final_record(
    liability_id: str,
    *,
    ref: dict[str, object],
    contract_number: str = "",
    printed_sequence: int | None = None,
    **fields: object,
) -> dict[str, object]:
    record: dict[str, object] = {
        "liability_id": liability_id,
        "source_refs": [ref],
        "source_refs_by_field": {
            field_name: [ref]
            for field_name, value in fields.items()
            if value not in (None, "")
        },
        **fields,
    }
    if contract_number:
        record["contract_number"] = contract_number
        record["source_refs_by_field"]["contract_number"] = [ref]
    if printed_sequence is not None:
        record["_printed_sequence"] = printed_sequence
    return record


def test_fully_recovered_packed_row_issue_is_suppressed_after_contract_link() -> None:
    ref = _ref("liability-contract")
    issue = _packed_issue(
        ref=ref,
        affected_fields=["currency"],
        contract_number="CONTRACT0001",
    )
    context = _context(issue)
    register_final_liability_issue_records(
        context,
        [
            _final_record(
                "repayment_liability:contract-1",
                ref=ref,
                contract_number="CONTRACT0001",
                currency="CNY",
            )
        ],
    )

    assert collect_extraction_issues(context) == []


def test_business_value_without_field_provenance_does_not_suppress_issue() -> None:
    ref = _ref("liability-no-field-provenance")
    issue = _packed_issue(
        ref=ref,
        affected_fields=["currency"],
        contract_number="CONTRACT-NO-FIELD-REF",
    )
    final = _final_record(
        "repayment_liability:no-field-ref",
        ref=ref,
        contract_number="CONTRACT-NO-FIELD-REF",
        currency="CNY",
    )
    final["source_refs_by_field"].pop("currency")
    context = _context(issue)
    register_final_liability_issue_records(context, [final])

    issues = collect_extraction_issues(context)

    assert len(issues) == 1
    assert issues[0]["target_record_id"] == "repayment_liability:no-field-ref"


def test_missing_packed_currency_is_preserved_and_retargeted() -> None:
    ref = _ref("liability-missing-currency")
    issue = _packed_issue(
        ref=ref,
        affected_fields=["currency"],
        contract_number="CONTRACT0002",
    )
    context = _context(issue)
    register_final_liability_issue_records(
        context,
        [
            _final_record(
                "repayment_liability:contract-2",
                ref=ref,
                contract_number="CONTRACT0002",
            )
        ],
    )

    issues = collect_extraction_issues(context)

    assert len(issues) == 1
    assert issues[0]["target_record_id"] == "repayment_liability:contract-2"
    assert "unique_final_liability_source_link" in issues[0]["reason_codes"]


@pytest.mark.parametrize("field_name", ("currency", "five_tier_class", "repayment_status_code"))
def test_genuine_liability_field_uncertainty_survives_unique_provenance_link(
    field_name: str,
) -> None:
    ref = _ref(f"liability-missing-{field_name}")
    issue = make_issue(
        category="ocr_cell_level_error",
        issue_code="pboc_cell_contract_unresolved",
        message="field contract unresolved",
        target_dataset="repayment_liability_records",
        field_name=field_name,
        observed_value="damaged",
        source_refs=(ref,),
        reason_codes=("normalized_value_withheld",),
    )
    context = _context(issue)
    register_final_liability_issue_records(
        context,
        [_final_record("repayment_liability:missing", ref=ref)],
    )

    issues = collect_extraction_issues(context)

    assert len(issues) == 1
    assert issues[0]["field_name"] == field_name
    assert issues[0]["target_record_id"] == "repayment_liability:missing"


def test_recovered_pboc_field_issue_is_suppressed_only_with_source_bound_record() -> None:
    ref = _ref("liability-recovered-pboc", column=6)
    context = SimpleNamespace(
        _personal_detail_extraction_issues=[],
        ocr_correction_audit=lambda: {
            "cell_anomalies": [
                {
                    "stage": "candidate_b_final_validation",
                    "path": "repayment_liability_records[temporary].currency",
                    "role": "currency",
                    "dataset_name": "repayment_liability_records",
                    "field_name": "currency",
                    "value": "人 民币元",
                    "normalized_value_withheld": True,
                    "source_refs": [ref],
                }
            ]
        },
        page_topology_audit=lambda: {"issues": []},
    )
    register_final_liability_issue_records(
        context,
        [_final_record("repayment_liability:recovered", ref=ref, currency="CNY")],
    )

    assert collect_extraction_issues(context) == []


def test_shared_table_provenance_never_guesses_between_two_final_liabilities() -> None:
    issue_ref = _ref("liability-shared", row=0)
    issue_ref.pop("bbox")
    issue_ref.pop("geometry_scope")
    issue_ref.pop("row")
    issue_ref.pop("column")
    issue = _packed_issue(ref=issue_ref, affected_fields=["currency"])
    context = _context(issue)
    register_final_liability_issue_records(
        context,
        [
            _final_record(
                "repayment_liability:shared-1",
                ref=_ref("liability-shared", row=1),
                currency="CNY",
            ),
            _final_record(
                "repayment_liability:shared-2",
                ref=_ref("liability-shared", row=4),
                currency="CNY",
            ),
        ],
    )

    issues = collect_extraction_issues(context)

    assert len(issues) == 1
    assert "target_record_id" not in issues[0]


def test_same_table_different_row_is_not_enough_to_link_an_orphan_issue() -> None:
    issue = _packed_issue(
        ref=_ref("liability-one-table", row=1),
        affected_fields=["currency"],
    )
    context = _context(issue)
    register_final_liability_issue_records(
        context,
        [
            _final_record(
                "repayment_liability:other-row",
                ref=_ref("liability-one-table", row=5),
                currency="CNY",
            )
        ],
    )

    issues = collect_extraction_issues(context)

    assert len(issues) == 1
    assert "target_record_id" not in issues[0]


def test_equal_cell_geometry_in_a_different_table_is_not_source_identity() -> None:
    issue = _packed_issue(
        ref=_ref("liability-issue-table"),
        affected_fields=["currency"],
    )
    context = _context(issue)
    register_final_liability_issue_records(
        context,
        [
            _final_record(
                "repayment_liability:different-table",
                ref=_ref("liability-final-table"),
                currency="CNY",
            )
        ],
    )

    issues = collect_extraction_issues(context)

    assert len(issues) == 1
    assert "target_record_id" not in issues[0]


def test_disagreeing_contract_and_provenance_leave_issue_unlinked() -> None:
    first_ref = _ref("liability-first")
    second_ref = _ref("liability-second", row=3)
    issue = _packed_issue(
        ref=second_ref,
        affected_fields=["currency"],
        contract_number="CONTRACT-FIRST",
    )
    context = _context(issue)
    register_final_liability_issue_records(
        context,
        [
            _final_record(
                "repayment_liability:first",
                ref=first_ref,
                contract_number="CONTRACT-FIRST",
                currency="CNY",
            ),
            _final_record(
                "repayment_liability:second",
                ref=second_ref,
                contract_number="CONTRACT-SECOND",
                currency="CNY",
            ),
        ],
    )

    issues = collect_extraction_issues(context)

    assert len(issues) == 1
    assert "target_record_id" not in issues[0]


def test_printed_ordinal_can_retarget_but_not_suppress_a_missing_field() -> None:
    issue = _packed_issue(
        ref={"logical_page": 12, "source_page": 6},
        affected_fields=["currency"],
        printed_sequence="2",
    )
    context = _context(issue)
    register_final_liability_issue_records(
        context,
        [
            _final_record(
                "repayment_liability:ordinal-1",
                ref=_ref("liability-ordinal-1"),
                printed_sequence=1,
                currency="CNY",
            ),
            _final_record(
                "repayment_liability:ordinal-2",
                ref=_ref("liability-ordinal-2"),
                printed_sequence=2,
            ),
        ],
    )

    issues = collect_extraction_issues(context)

    assert len(issues) == 1
    assert issues[0]["target_record_id"] == "repayment_liability:ordinal-2"
