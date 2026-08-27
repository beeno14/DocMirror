# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused Yang business-document replay without PDF or OCR access.

The fixture is a hand-transcribed, self-contained excerpt from the saved Yang
Community artifact.  These tests cover both sides of the contract: production
strategies must interpret a real-shaped source card correctly, and the public
PBOC v2 projection must conserve the exact six-row business population and the
two exact monthly-status decisions that previously needed real-PDF diagnosis.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned import (
    native_extraction,
    native_status_conflict,
)
from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b_strategy import (
    ACCOUNT_SECTION,
    CANDIDATE_B_STAGE_REGISTRY,
    LIABILITY_SECTION,
    REPORT_HEADER_SECTION,
    SECTION_TO_CANONICAL_DATASETS,
    plan_candidate_b_initial_extraction,
)
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_strategy import (
    MaterializationMode,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)

_FIXTURE_PATH = (
    Path(__file__).parent
    / "fixtures"
    / "personal_detail"
    / "yang_business_replay.json"
)


def _fixture() -> dict[str, Any]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _values(record: dict[str, Any]) -> dict[str, Any]:
    normalized = record.get("normalized")
    return normalized if isinstance(normalized, dict) else record


def _owned_liability_table(table: SimpleNamespace) -> SimpleNamespace:
    table.metadata["canonical_template_id"] = LIABILITY_SECTION
    return table


def _yang_fifth_liability_source_card() -> SimpleNamespace:
    """Reproduce Yang page 19's adjacent ordinal-4 liability source card."""

    headers = [
        "管理机构",
        "业务种类",
        "开立日期",
        "到期日期",
        "责任人类型",
        "还款责任金额",
        "币种",
        "保证合同编号",
    ]
    values = [
        "中国农业银行股份有限公司大理分行",
        "贷款",
        "2023.09.18",
        "2024.09.13",
        "共同借款人",
        "",
        "人民币元",
        "",
    ]
    column_width = 44.0
    source_cell_bboxes = [
        [
            [
                47.0 + column * column_width,
                339.5 if row == 0 else 352.5,
                47.0 + (column + 1) * column_width,
                352.5 if row == 0 else 378.0,
            ]
            for column in range(8)
        ]
        for row in range(2)
    ]
    table = _owned_liability_table(
        SimpleNamespace(
            table_id="pt_19_2",
            metadata={
                "raw_rows": [headers, values],
                "source_cell_bboxes": source_cell_bboxes,
                "cell_evidence_ids": [
                    [
                        [f"yang:pt19:{row}:{column}"]
                        if [headers, values][row][column]
                        else []
                        for column in range(8)
                    ]
                    for row in range(2)
                ],
            },
            headers=[],
            rows=[],
            bbox=[47.0, 339.5, 399.0, 444.0],
        )
    )
    page = SimpleNamespace(
        page_number=19,
        source_page_number=10,
        canonical_template_id=LIABILITY_SECTION,
        texts=[
            SimpleNamespace(
                content="账户4",
                bbox=[49.5, 329.0, 70.0, 340.5],
            )
        ],
        tables=[table],
    )
    return SimpleNamespace(
        pages=[page],
        reading_order_by_logical={19: 19},
        reading_order_resolution={
            "resolved": True,
            "authoritative": True,
        },
        tables_continue=lambda _left, _right: None,
        corrected_evidence_pages=lambda: [],
        _personal_detail_extraction_issues=[],
    )


def _strategy_audit() -> dict[str, Any]:
    sections = (
        (1, REPORT_HEADER_SECTION),
        (13, ACCOUNT_SECTION),
        (18, LIABILITY_SECTION),
        (19, LIABILITY_SECTION),
    )
    groups = (
        (REPORT_HEADER_SECTION, (1,)),
        (ACCOUNT_SECTION, (13,)),
        (LIABILITY_SECTION, (18, 19)),
    )
    return {
        "corrected_evidence_conservation": {
            "valid": True,
            "conserved_logical_pages": [page for page, _section in sections],
        },
        "canonical_subset_conservation": {"valid": True},
        "unresolved_pages": [],
        "registrations": [
            {
                "logical_page": page,
                "source_page": (page + 1) // 2,
                "status": "registered",
                "template_id": section,
                "basis": "yang_exact_heading_and_table_signature",
                "affected_source_datasets": sorted(
                    SECTION_TO_CANONICAL_DATASETS[section]
                ),
                "printed_page": page,
                "printed_total": 26,
            }
            for page, section in sections
        ],
        "fragment_groups": [
            {
                "template_id": section,
                "fragment_logical_pages": list(pages),
                "canonical_page": pages[0],
                "coverage_status": "full",
                "coverage_ratio": 1.0,
            }
            for section, pages in groups
        ],
    }


def _public_liability_inputs() -> list[dict[str, Any]]:
    rows = []
    for expected in _fixture()["liabilities"]:
        values = deepcopy(expected)
        sequence = int(values["sequence"])
        values.update(
            {
                "liability_id": f"yang:repayment_liability:{sequence}",
                "responsibility_amount_reported": (
                    values["responsibility_amount"] is not None
                ),
                "reporting_amount_currency": "CNY",
                "amount_unit": "yuan",
                "reporting_amount_unit": "yuan",
            }
        )
        rows.append(values)
    return rows


def test_yang_fixture_is_a_complete_six_row_business_oracle() -> None:
    replay = _fixture()

    assert replay["evidence_kind"] == "hand_transcribed_business_excerpt"
    assert replay["liability_count"] == len(replay["liabilities"]) == 6
    assert [row["sequence"] for row in replay["liabilities"]] == list(
        range(1, 7)
    )
    assert len(
        {
            (
                row["related_party_id_number"],
                row["institution"],
                row["open_date"],
            )
            for row in replay["liabilities"]
        }
    ) == 6
    assert {
        row["repayment_id"] for row in replay["monthly_cells"]
    } == {
        "mg_p13_repayment_1:2023-05",
        "mg_p13_repayment_1:2019-08",
    }


def test_yang_native_liability_strategy_parses_real_shaped_card_and_blanks() -> None:
    context = _yang_fifth_liability_source_card()

    rows = native_extraction._extract_liabilities(context)

    assert len(rows) == 1
    row = rows[0]
    assert row["_printed_sequence"] == 4
    assert row["institution"] == "中国农业银行股份有限公司大理分行"
    assert row["business_type"] == "贷款"
    assert row["open_date"] == "2023-09-18"
    assert row["due_date"] == "2024-09-13"
    assert row["responsibility_type"] == "共同借款人"
    assert row["currency"] == "CNY"
    assert row.get("responsibility_amount") is None
    assert row.get("contract_number") is None

    unresolved = {
        str(issue.get("field_name") or "")
        for issue in context._personal_detail_extraction_issues
        if issue.get("target_record_id") == row["liability_id"]
        and issue.get("status") != "resolved"
    }
    assert {"responsibility_amount", "contract_number"} <= unresolved

    projected = project_personal_detail_datasets(
        {
            "repayment_liability_records": rows,
            "personal_detail_extraction_issues": (
                context._personal_detail_extraction_issues
            ),
        }
    )
    public = _values(projected["repayment_responsibilities"][0])
    assert public["repayment_responsibility_id"] == row["liability_id"]
    assert public["institution"] == row["institution"]
    assert public["business_type"] == row["business_type"]
    assert public["open_date"] == row["open_date"]
    assert public["due_date"] == row["due_date"]
    assert public.get("responsibility_amount") is None
    assert public.get("contract_number") is None
    assert projected["extraction_issues"]


def test_yang_exact_six_liabilities_survive_public_canonical_projection() -> None:
    expected_rows = _fixture()["liabilities"]

    projected = project_personal_detail_datasets(
        {"repayment_liability_records": _public_liability_inputs()}
    )
    public_rows = projected["repayment_responsibilities"]

    assert len(public_rows) == 6
    by_sequence = {
        int(_values(row)["sequence"]): _values(row) for row in public_rows
    }
    for expected in expected_rows:
        actual = by_sequence[int(expected["sequence"])]
        for field_name, expected_value in expected.items():
            assert actual.get(field_name) == expected_value, (
                expected["sequence"],
                field_name,
            )

    fifth = by_sequence[5]
    assert fifth["related_party_id_number"] == "5329010002043257"
    assert fifth["related_party_category"] == "organization"
    assert fifth["responsibility_amount"] is None
    assert fifth["contract_number"] is None
    assert fifth["balance"] == "3500000"


def test_yang_month_conflict_guard_and_public_projection_are_exactly_local(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    replay_rows = _fixture()["monthly_cells"]
    records = []
    native_by_id = {}
    for replay in replay_rows:
        year, month = map(int, replay["performance_month"].split("-"))
        records.append(
            {
                "repayment_id": replay["repayment_id"],
                "account_id": replay["account_id"],
                "grid_id": "mg_p13_repayment_1",
                "year": year,
                "month": month,
                "performance_month": replay["performance_month"],
                "status": replay["corrected_status"],
                "overdue_amount": replay["overdue_amount"],
            }
        )
        native_by_id[replay["repayment_id"]] = replay["native_status"]

    context = SimpleNamespace(
        parse_result=SimpleNamespace(),
        _personal_detail_extraction_issues=[],
    )
    monkeypatch.setattr(
        native_status_conflict,
        "_exact_final_status_ref",
        lambda record, **_kwargs: {
            "field_name": "status",
            "repayment_id": record["repayment_id"],
        },
    )
    monkeypatch.setattr(
        native_status_conflict,
        "_exact_final_amount_ref",
        lambda record, **_kwargs: {
            "field_name": "overdue_amount",
            "repayment_id": record["repayment_id"],
        },
    )

    def native_candidate(
        _context: Any,
        *,
        final_ref: dict[str, Any],
        **_kwargs: Any,
    ) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
        repayment_id = str(final_ref["repayment_id"])
        return [
            (
                str(native_by_id[repayment_id]),
                {
                    "field_name": "status",
                    "repayment_id": repayment_id,
                    "evidence_plane": "sealed_native_source_cell",
                },
                {
                    "field_name": "overdue_amount",
                    "repayment_id": repayment_id,
                    "evidence_plane": "sealed_native_source_cell",
                },
            )
        ]

    monkeypatch.setattr(
        native_status_conflict,
        "_native_candidates",
        native_candidate,
    )

    audit = native_status_conflict.apply_candidate_b_native_status_conflict_guard(
        context,
        records,
        enabled=True,
    )

    by_id = {str(row["repayment_id"]): row for row in records}
    issue_target_ids = {
        str(issue["target_record_id"])
        for issue in context._personal_detail_extraction_issues
    }
    for expected in replay_rows:
        repayment_id = str(expected["repayment_id"])
        if expected["expected_issue"]:
            assert "status" not in by_id[repayment_id]
        else:
            assert by_id[repayment_id]["status"] == expected["corrected_status"]
        assert (repayment_id in issue_target_ids) is expected["expected_issue"]
    assert audit == {
        "enabled": True,
        "records_checked": 2,
        "unique_native_witnesses": 2,
        "native_numeric_witnesses_rejected_for_nonpositive_amount": 1,
        "agreements": 0,
        "conflicts_withheld": 1,
        "low_source_ocr_confidence_withheld": 0,
        "independent_monthly_field_confirmations": 0,
        "preserved_source_plane_conflicts": 0,
    }
    assert issue_target_ids == {"mg_p13_repayment_1:2019-08"}

    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                {
                    "account_id": "credit_account:credit_card:4",
                    "account_type": "credit_card",
                }
            ],
            "repayment_records": records,
            "personal_detail_extraction_issues": (
                context._personal_detail_extraction_issues
            ),
        }
    )
    public_months = {
        str(_values(row)["monthly_performance_id"]): _values(row)
        for row in projected["credit_account_monthly_performance"]
    }
    public_issues = [
        _values(issue) for issue in projected["extraction_issues"]
    ]
    for expected in replay_rows:
        repayment_id = str(expected["repayment_id"])
        assert public_months[repayment_id]["status_code"] == expected[
            "expected_status_code"
        ]
        has_public_issue = any(
            issue.get("target_record_id") == repayment_id
            and issue.get("field_name") == "status_code"
            for issue in public_issues
        )
        assert has_public_issue is expected["expected_issue"]


def test_yang_document_shape_selects_only_account_liability_and_core_stages() -> None:
    census, plan = plan_candidate_b_initial_extraction(_strategy_audit())

    assert census.complete is True
    assert plan.mode is MaterializationMode.LAZY
    assert plan.ordered_stage_names == (
        "account_inventory",
        "monthly_repayments",
        "liabilities",
        "overdue",
        "header",
        "recovery",
        "residence",
        "employment",
        "source_rows",
        "profile_details",
        "profile",
    )
    assert {
        "credit_agreements",
        "inquiries",
        "public",
        "notes",
        "summary",
        "postpaid_records",
        "postpaid_history",
    } <= set(plan.skipped_stage_names)
    assert set(plan.ordered_stage_names) | set(plan.skipped_stage_names) == set(
        CANDIDATE_B_STAGE_REGISTRY.names
    )
