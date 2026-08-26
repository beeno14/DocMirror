# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Focused Lin month corpus across strategy, producer, ledger, and schema.

The literals below are minimized excerpts of the frozen real-output artifact
``artifacts/personal_detail_six_live_iteration_20260826_linfix4/林岚挺征信.semantic.json``.
They deliberately do not read or execute the real PDF.  The page-10 fixture
preserves the production-only topology that helper-style alias tests miss:

* three source-table populations contain four physical month rows;
* four canonical rows are owned by three adjacent account segments;
* detector-local grid ids are reused by two detached source-structure grids;
* one position is canonical for account 17 and an explicit alias for account
  18, while an earlier detached position aliases account 17 at the same month;
* only the earlier detached position has the three pending field diagnostics;
* retained canonical refs have rejected geometry, while the detached alias has
  exact cell boxes and no ``source_page``.

The tests enter through the real orchestration planner, relationship producer,
source-projection ledger, and final schema projector.  They are intentionally
stronger than helper-only consumer fixtures and may remain red while production
source changes are frozen.
"""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.candidate_b_strategy import (
    ACCOUNT_SECTION,
    CANDIDATE_B_STAGE_REGISTRY,
    SECTION_TO_CANONICAL_DATASETS,
    plan_candidate_b_initial_extraction,
)
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_issues import (
    make_issue,
)
from docmirror.plugins.credit_report.personal_detail_scanned.extraction_strategy import (
    MaterializationMode,
)
from docmirror.plugins.credit_report.personal_detail_scanned.native_status_conflict import (
    apply_candidate_b_native_status_conflict_guard,
)
from docmirror.plugins.credit_report.personal_detail_scanned.relations import (
    link_candidate_b_repayments,
)
from docmirror.plugins.credit_report.personal_detail_scanned.schema import (
    project_personal_detail_datasets,
)
from docmirror.plugins.credit_report.personal_detail_scanned.source_projection import (
    prepare_personal_detail_source_collections,
)
from docmirror.plugins.credit_report.value_utils import stable_record_id


LIN_P10_GRID_0 = "mg_p10_repayment_0"
LIN_P10_GRID_1 = "mg_p10_repayment_1"
LIN_P10_GRID_2 = "mg_p10_repayment_2"
LIN_ACCOUNT_16 = "credit_account:non_revolving_loan:16"
LIN_ACCOUNT_17 = "credit_account:non_revolving_loan:17"
LIN_ACCOUNT_18 = "credit_account:non_revolving_loan:18"
LIN_P19_GRID = "mg_p19_repayment_0"
LIN_P19_ACCOUNT_11 = "credit_account:credit_card:11"
LIN_P19_ACCOUNT_12 = "credit_account:credit_card:12"

# The signed source-table audit is the independent physical oracle.  It is not
# derived from parser/grid positions: pt_10_0 has Sep, pt_10_1 has Sep+Oct, and
# pt_10_2 has Oct.  Therefore the page contributes four, not five, raw rows.
LIN_P10_SIGNED_SOURCE_TABLE_MONTHS = {
    "pt_10_0": ("2020-09",),
    "pt_10_1": ("2020-09", "2020-10"),
    "pt_10_2": ("2020-10",),
}

_DETACHED_REASON_CODES = (
    "detached_source_structure_exact_key",
    "canonical_deduplicated_key_missing",
    "source_structure_is_audit_only",
    "account_month_owner_reconciliation_pending",
    "dataset_incomplete",
    "exact_grid_month_source_position",
    "normalized_value_withheld",
    "owner_or_status_value_not_invented",
)


def _source_account_month_target(account_id: str, performance_month: str) -> str:
    owner_key = stable_record_id(
        "source_account_month_owner", account_id
    ).split(":", 1)[-1]
    return f"source_account_month:{owner_key}:{performance_month}"


def _lin_account(
    account_id: str,
    *,
    minimum: float,
    maximum: float,
) -> dict[str, object]:
    return {
        "record_id": account_id,
        "account_id": account_id,
        "account_type": "non_revolving_loan",
        "_canonical_segment": {
            "ownership_basis": "printed_anchor_to_next_anchor",
            "anchor_logical_page": 10,
            "anchor_bbox": [42.0, minimum + 4.0, 196.0, minimum + 14.0],
            "pages": [
                {
                    "logical_page": 10,
                    "min_y": minimum,
                    "max_y": maximum,
                }
            ],
        },
    }


def _lin_month_grid(
    grid_id: str,
    *,
    top: float,
    start_month: int,
    end_month: int,
    page: int = 10,
) -> dict[str, object]:
    return {
        "grid_id": grid_id,
        "page": page,
        "bbox": [42.0, top, 397.5, top + 62.0],
        "coordinate_system": "pdf_points_top_left",
        "audit": {
            "date_range": {
                "start_year": 2020,
                "start_month": start_month,
                "end_year": 2020,
                "end_month": end_month,
            }
        },
    }


def _lin_exact_cell_refs(
    grid_id: str,
    *,
    performance_month: str,
    status_bbox: list[float],
    amount_bbox: list[float],
    status_row: int = 2,
    page: int = 10,
) -> list[dict[str, object]]:
    month = int(performance_month[-2:])
    return [
        {
            "page": page,
            "logical_page": page,
            "geometry_scope": "cell",
            "coordinate_system": "pdf_points_top_left",
            "grid_id": grid_id,
            "row": status_row,
            "col": month,
            "field_name": "status",
            "performance_month": performance_month,
            "bbox": list(status_bbox),
        },
        {
            "page": page,
            "logical_page": page,
            "geometry_scope": "cell",
            "coordinate_system": "pdf_points_top_left",
            "grid_id": grid_id,
            "row": status_row + 1,
            "col": month,
            "field_name": "overdue_amount",
            "performance_month": performance_month,
            "bbox": list(amount_bbox),
        },
    ]


def _lin_p10_detached_cell_refs(
    grid_id: str,
    performance_month: str,
) -> list[dict[str, object]]:
    if (grid_id, performance_month) == (LIN_P10_GRID_0, "2020-09"):
        return _lin_exact_cell_refs(
            grid_id,
            performance_month=performance_month,
            status_bbox=[291.3333333333333, 374.5, 316.375, 396.0],
            amount_bbox=[291.3333333333333, 389.5, 316.375, 412.5],
        )
    if (grid_id, performance_month) == (LIN_P10_GRID_0, "2020-10"):
        # Exact Lin p10 cells from the frozen artifact.
        return _lin_exact_cell_refs(
            grid_id,
            performance_month=performance_month,
            status_bbox=[316.375, 374.5, 341.4166666666667, 396.0],
            amount_bbox=[316.375, 389.5, 341.4166666666667, 412.5],
        )
    if (grid_id, performance_month) == (LIN_P10_GRID_1, "2020-10"):
        return _lin_exact_cell_refs(
            grid_id,
            performance_month=performance_month,
            status_bbox=[316.375, 474.5, 341.4166666666667, 496.0],
            amount_bbox=[316.375, 489.5, 341.4166666666667, 512.5],
        )
    raise AssertionError((grid_id, performance_month))


def _lin_rejected_canonical_ref(
    grid_id: str,
    performance_month: str,
) -> dict[str, object]:
    month = int(performance_month[-2:])
    return {
        "page": 10,
        "logical_page": 10,
        "geometry_scope": "logical_page",
        "geometry_status": "unresolved",
        "coordinate_system": "pdf_points_top_left",
        "grid_id": grid_id,
        "row": 0,
        "col": month,
        "field_name": "status",
        "geometry_rejection": {
            "source": "rejected_month_geometry",
            "reason": "source_table_month_ownership_required",
            "logical_page": 10,
            "value_inputs_used": False,
        },
    }


def _lin_canonical_month_row(
    grid_id: str,
    performance_month: str,
) -> dict[str, object]:
    year, month = (int(value) for value in performance_month.split("-"))
    return {
        "record_id": f"{grid_id}:{performance_month}",
        "repayment_id": f"{grid_id}:{performance_month}",
        "grid_id": grid_id,
        "year": year,
        "month": month,
        "status": "N",
        "overdue_amount": "0",
        "source_cell_refs": [
            _lin_rejected_canonical_ref(grid_id, performance_month)
        ],
    }


def _lin_source_structure_month_row(
    grid_id: str,
    performance_month: str,
) -> dict[str, object]:
    year, month = (int(value) for value in performance_month.split("-"))
    return {
        "record_id": f"{grid_id}:{performance_month}",
        "repayment_id": f"{grid_id}:{performance_month}",
        "grid_id": grid_id,
        "year": year,
        "month": month,
        "status": "N",
        "overdue_amount": "0",
        "source_cell_refs": _lin_p10_detached_cell_refs(
            grid_id, performance_month
        ),
    }


def _lin_detached_issue_trio() -> list[dict[str, object]]:
    grid_id = LIN_P10_GRID_0
    performance_month = "2020-10"
    raw_refs = _lin_p10_detached_cell_refs(grid_id, performance_month)

    def localized_ref(index: int, field_name: str) -> dict[str, object]:
        ref = deepcopy(raw_refs[index])
        ref["source_field_name"] = ref["field_name"]
        ref["field_name"] = field_name
        return ref

    common = {
        "category": "ocr_structure_correction",
        "issue_code": "canonical_monthly_source_structure_missing_field",
        "message": (
            "A detached source-structure grid/month position was absent from "
            "the deduplicated canonical monthly population."
        ),
        "parser_stage": "canonical_monthly_grid_materialization",
        "target_dataset": "repayment_records",
        "target_record_id": f"{grid_id}:{performance_month}",
        "candidate_value": {"resolution": "withheld_pending_review"},
        "reason_codes": _DETACHED_REASON_CODES,
    }

    def issue(
        field_name: str,
        refs: list[dict[str, object]],
        source_observations: list[object],
    ) -> dict[str, object]:
        return make_issue(
            **common,
            field_name=field_name,
            observed_value={
                "grid_id": grid_id,
                "performance_month": performance_month,
                "field_state": "source_position_withheld",
                "source_observations": source_observations,
                "source_structure_key_count": 1,
            },
            source_refs=refs,
        )

    return [
        issue(
            "performance_month",
            [
                localized_ref(0, "performance_month"),
                localized_ref(1, "performance_month"),
            ],
            [performance_month],
        ),
        issue("status_code", [localized_ref(0, "status_code")], ["N"]),
        issue("status_amount", [localized_ref(1, "status_amount")], ["0"]),
    ]


def _lin_p10_producer_result() -> tuple[
    list[dict[str, object]], SimpleNamespace
]:
    accounts = [
        _lin_account(LIN_ACCOUNT_16, minimum=60.0, maximum=160.0),
        _lin_account(LIN_ACCOUNT_17, minimum=160.0, maximum=270.0),
        _lin_account(LIN_ACCOUNT_18, minimum=270.0, maximum=380.0),
    ]
    canonical_grids = [
        _lin_month_grid(
            LIN_P10_GRID_0,
            top=90.0,
            start_month=9,
            end_month=9,
        ),
        _lin_month_grid(
            LIN_P10_GRID_1,
            top=180.0,
            start_month=9,
            end_month=10,
        ),
        _lin_month_grid(
            LIN_P10_GRID_2,
            top=280.0,
            start_month=10,
            end_month=10,
        ),
    ]
    canonical_rows = [
        _lin_canonical_month_row(LIN_P10_GRID_0, "2020-09"),
        _lin_canonical_month_row(LIN_P10_GRID_1, "2020-09"),
        _lin_canonical_month_row(LIN_P10_GRID_1, "2020-10"),
        _lin_canonical_month_row(LIN_P10_GRID_2, "2020-10"),
    ]

    # The same detector-local ids are replayed one account segment lower.  This
    # is the shape that makes a global set of alias positions unsafe.
    detached_grids = [
        _lin_month_grid(
            LIN_P10_GRID_0,
            top=180.0,
            start_month=9,
            end_month=10,
        ),
        _lin_month_grid(
            LIN_P10_GRID_1,
            top=280.0,
            start_month=10,
            end_month=10,
        ),
    ]
    detached_rows = [
        _lin_source_structure_month_row(LIN_P10_GRID_0, "2020-09"),
        _lin_source_structure_month_row(LIN_P10_GRID_0, "2020-10"),
        _lin_source_structure_month_row(LIN_P10_GRID_1, "2020-10"),
    ]
    context = SimpleNamespace(
        _personal_detail_extraction_issues=_lin_detached_issue_trio(),
        _candidate_b_monthly_source_structure_grids=detached_grids,
        _candidate_b_monthly_source_structure_records=detached_rows,
    )

    linked = link_candidate_b_repayments(
        canonical_rows,
        accounts,
        canonical_grids,
        issue_context=context,
    )
    return linked, context


def _lin_p10_content() -> dict[str, object]:
    linked, context = _lin_p10_producer_result()
    return {
        "facts": {},
        "datasets": {
            "repayment_records": linked,
            "personal_detail_extraction_issues": deepcopy(
                context._personal_detail_extraction_issues
            ),
        },
    }


def _lin_p10_alias_issue(
    issues: list[dict[str, object]],
    *,
    grid_id: str = LIN_P10_GRID_0,
    performance_month: str = "2020-10",
) -> dict[str, object]:
    return next(
        issue
        for issue in issues
        if issue.get("issue_code")
        == "candidate_b_monthly_source_position_alias_reconciled"
        and (issue.get("observed_value") or {}).get("grid_id") == grid_id
        and (issue.get("observed_value") or {}).get("performance_month")
        == performance_month
    )


def _lin_p10_detached_siblings(
    issues: list[dict[str, object]],
) -> list[dict[str, object]]:
    return [
        issue
        for issue in issues
        if issue.get("issue_code")
        == "canonical_monthly_source_structure_missing_field"
        and issue.get("target_record_id")
        == f"{LIN_P10_GRID_0}:2020-10"
    ]


def test_lin_p10_real_topology_conserves_four_physical_rows_and_four_business_rows() -> None:
    content = _lin_p10_content()
    datasets = content["datasets"]
    linked = datasets["repayment_records"]
    issues = datasets["personal_detail_extraction_issues"]

    expected_identities = {
        (LIN_ACCOUNT_16, "2020-09"),
        (LIN_ACCOUNT_17, "2020-09"),
        (LIN_ACCOUNT_17, "2020-10"),
        (LIN_ACCOUNT_18, "2020-10"),
    }
    assert {
        (
            row["account_id"],
            f"{int(row['year']):04d}-{int(row['month']):02d}",
        )
        for row in linked
    } == expected_identities
    assert len(linked) == sum(
        len(months) for months in LIN_P10_SIGNED_SOURCE_TABLE_MONTHS.values()
    ) == 4

    aliases = [
        issue
        for issue in issues
        if issue.get("issue_code")
        == "candidate_b_monthly_source_position_alias_reconciled"
    ]
    assert {
        (
            issue["observed_value"]["account_id"],
            issue["observed_value"]["grid_id"],
            issue["observed_value"]["performance_month"],
        )
        for issue in aliases
    } == {
        (LIN_ACCOUNT_17, LIN_P10_GRID_0, "2020-09"),
        (LIN_ACCOUNT_17, LIN_P10_GRID_0, "2020-10"),
        (LIN_ACCOUNT_18, LIN_P10_GRID_1, "2020-10"),
    }
    exact_alias = _lin_p10_alias_issue(issues)
    assert exact_alias["target_record_id"] == _source_account_month_target(
        LIN_ACCOUNT_17, "2020-10"
    )
    assert len(exact_alias["source_refs"]) == 2
    assert all(
        ref["geometry_scope"] == "cell"
        and "bbox" in ref
        and "source_page" not in ref
        for ref in exact_alias["source_refs"]
    )
    retained_october = next(
        row
        for row in linked
        if row["account_id"] == LIN_ACCOUNT_17 and row["month"] == 10
    )
    assert retained_october["grid_id"] == LIN_P10_GRID_1
    assert retained_october["source_cell_refs"][0]["geometry_scope"] == "logical_page"
    assert retained_october["source_cell_refs"][0]["geometry_status"] == "unresolved"
    assert "bbox" not in retained_october["source_cell_refs"][0]

    prepared = prepare_personal_detail_source_collections(content)
    closure = prepared["facts"]["personal_detail_account_month_closure"]
    assert closure["candidate_identity_count"] == 4
    assert closure["expected_identity_count"] == 4
    assert closure["source_month_position_observations"] == 5
    assert closure["raw_source_month_positions"] == 4
    assert closure["unique_physical_source_month_positions"] == 4
    assert closure["owner_bound_account_months"] == 4
    assert closure["owner_unresolved_positions"] == 0
    assert closure["alias_source_month_positions"] == 3
    assert closure["physical_alias_source_month_observations"] == 1
    assert closure["physical_owner_conflict_free"] is True
    assert closure["source_position_balance_valid"] is True
    assert closure["status"] == "identity_closed"
    assert prepared["facts"]["personal_detail_dataset_states"][
        "repayment_records"
    ] == {
        "presence_status": "present",
        "reason": "account_month_identity_closure",
        "observed_row_count": 4,
        "expected_row_count": 4,
    }

    projected = project_personal_detail_datasets(prepared["datasets"])
    business_rows = projected["credit_account_monthly_performance"]
    assert len(business_rows) == 4
    assert {
        (row["account_id"], row["performance_month"])
        for row in business_rows
    } == expected_identities
    assert not any(
        row["monthly_performance_id"]
        == f"{LIN_P10_GRID_0}:2020-10"
        for row in business_rows
    )


def test_lin_p10_producer_does_not_leave_resolved_alias_as_active_business_omission() -> None:
    _linked, context = _lin_p10_producer_result()
    active = [
        issue
        for issue in _lin_p10_detached_siblings(
            context._personal_detail_extraction_issues
        )
        if issue.get("status") == "requires_review"
    ]

    # Once the same producer emits an exact account/month alias, the earlier
    # detached trio may remain as audit history but must no longer claim that a
    # business month is missing.  This is expected to be red until the producer
    # and consumer contracts are changed together.
    assert active == []


def test_lin_p10_geometry_fallback_survives_when_exact_alias_issue_is_absent() -> None:
    content = _lin_p10_content()
    datasets = content["datasets"]
    issues = datasets["personal_detail_extraction_issues"]
    issues[:] = [
        issue
        for issue in issues
        if not (
            issue.get("issue_code")
            == "candidate_b_monthly_source_position_alias_reconciled"
            and (issue.get("observed_value") or {}).get("grid_id")
            == LIN_P10_GRID_0
            and (issue.get("observed_value") or {}).get("performance_month")
            == "2020-10"
        )
    ]
    retained = next(
        row
        for row in datasets["repayment_records"]
        if row["account_id"] == LIN_ACCOUNT_17 and row["month"] == 10
    )
    retained_refs = _lin_p10_detached_cell_refs(
        LIN_P10_GRID_0, "2020-10"
    )
    for ref in retained_refs:
        ref["grid_id"] = LIN_P10_GRID_1
        ref["source_page"] = 5
    retained["source_cell_refs"] = retained_refs

    prepared = prepare_personal_detail_source_collections(content)
    closure = prepared["facts"]["personal_detail_account_month_closure"]
    assert closure["source_month_position_observations"] == 5
    assert closure["raw_source_month_positions"] == 4
    assert closure["owner_bound_account_months"] == 4
    assert closure["owner_unresolved_positions"] == 0
    assert closure["alias_source_month_positions"] == 2
    assert closure["reconciled_detached_diagnostic_positions"] == 1
    assert closure["physical_alias_source_month_observations"] == 1
    assert closure["source_position_balance_valid"] is True


@pytest.mark.parametrize(
    "mutation",
    (
        "identity",
        "account",
        "month",
        "bbox",
        "row",
        "col",
        "sibling_count",
        "reason",
        "duplicate",
    ),
)
def test_lin_p10_exact_alias_tamper_matrix_fails_closed(mutation: str) -> None:
    content = _lin_p10_content()
    issues = content["datasets"]["personal_detail_extraction_issues"]
    alias = _lin_p10_alias_issue(issues)
    siblings = _lin_p10_detached_siblings(issues)
    performance_sibling = next(
        issue for issue in siblings if issue["field_name"] == "performance_month"
    )

    if mutation == "identity":
        alias["target_record_id"] = "source_account_month:tampered:2020-10"
    elif mutation == "account":
        alias["observed_value"]["account_id"] = LIN_ACCOUNT_18
    elif mutation == "month":
        alias["observed_value"]["performance_month"] = "2020-11"
    elif mutation == "bbox":
        alias["source_refs"][0]["bbox"][0] += 1.0
    elif mutation == "row":
        alias["source_refs"][0]["row"] += 1
    elif mutation == "col":
        alias["source_refs"][0]["col"] = 9
    elif mutation == "sibling_count":
        issues.remove(
            next(issue for issue in siblings if issue["field_name"] == "status_amount")
        )
    elif mutation == "reason":
        performance_sibling["reason_codes"].remove(
            "canonical_deduplicated_key_missing"
        )
    elif mutation == "duplicate":
        issues.append(deepcopy(alias))
    else:  # pragma: no cover - parametrization is the closed mutation catalog.
        raise AssertionError(mutation)

    prepared = prepare_personal_detail_source_collections(content)
    closure = prepared["facts"]["personal_detail_account_month_closure"]
    assert closure["source_position_balance_valid"] is True
    assert closure["raw_source_month_positions"] >= 5
    assert closure["physical_alias_source_month_observations"] == 0
    assert closure["reconciled_detached_diagnostic_positions"] == 0


def test_lin_p10_competing_physical_owner_fails_closed_and_withholds_proof() -> None:
    content = _lin_p10_content()
    datasets = content["datasets"]
    alias = _lin_p10_alias_issue(
        datasets["personal_detail_extraction_issues"]
    )
    competing_refs = deepcopy(alias["source_refs"])
    for ref in competing_refs:
        ref.pop("account_id")
        ref.pop("binding")
        ref.pop("binding_quality")
        ref["field_name"] = ref["source_field_name"]
    datasets["repayment_records"].append(
        {
            "record_id": "tampered:competing-owner:2020-10",
            "repayment_id": "tampered:competing-owner:2020-10",
            "grid_id": LIN_P10_GRID_0,
            "account_id": LIN_ACCOUNT_18,
            "year": 2020,
            "month": 10,
            "status": "N",
            "overdue_amount": "0",
            "source_cell_refs": competing_refs,
            "_account_month_identity_proof": {
                "account_id": LIN_ACCOUNT_18,
                "performance_month": "2020-10",
                "grid_id": LIN_P10_GRID_0,
                "owner_basis": "canonical_account_segment",
                "account_anchor_exact": True,
                "printed_month_range_exact": True,
                "grid_geometry_exact": True,
                "unique_owner": True,
            },
            "_account_month_identity_proof_status": "exact",
        }
    )

    prepared = prepare_personal_detail_source_collections(content)
    closure = prepared["facts"]["personal_detail_account_month_closure"]
    assert closure["physical_owner_conflict_free"] is False
    assert closure["cross_owner_physical_conflict_count"] >= 1
    assert closure["status"] == "physical_owner_conflict"
    assert closure["source_position_balance_valid"] is True
    assert "personal_detail_account_month_closure_proof" not in prepared["datasets"]
    assert any(
        issue.get("issue_code") == "account_month_physical_owner_conflict"
        for issue in prepared["datasets"]["personal_detail_extraction_issues"]
    )


def _lin_p19_geometry_provenance() -> dict[str, object]:
    return {
        "selection_basis": "source_table_year_plus_twelve_ownership",
        "source": "source_table_geometry",
        "reason": "exact_source_table_month_lattice_calibration",
        "table_id": "pt_19_0",
        "vertical_rule_count": 14,
        "rule_count": 14,
        "horizontal_rule_count": 18,
        "column_count": 13,
        "month_column_count": 12,
        "status_row_index": 9,
        "amount_row_index": 10,
        "year_anchor_row_index": 9,
        "year_anchor_mode": "spanning_year_cell",
        "year_row_span": 2,
        "active_cell_geometry_exact": True,
        "active_cell_rule_derived_count": 0,
        "coordinate_system": "pdf_points_top_left",
        "value_inputs_used": False,
        "corroborated_by_source_table_geometry": True,
        "ambiguous_visual_geometry_superseded": False,
        "source_table_comparison": "agree",
        "calibrated_from_source_table_geometry": True,
        "visual_selection_basis": "year_plus_twelve_rule_ownership",
        "visual_owned_month_rule_hits": 13,
        "visual_residual_shift_months": 0.0687,
        "logical_page": 19,
    }


def _lin_p19_source_refs() -> list[dict[str, object]]:
    return [
        {
            "page": 19,
            "logical_page": 19,
            "geometry_scope": "cell",
            "geometry_status": "exact",
            "coordinate_system": "pdf_points_top_left",
            "grid_id": LIN_P19_GRID,
            "row": 2,
            "col": 8,
            "field_name": "status",
            "geometry_provenance": _lin_p19_geometry_provenance(),
            "bbox": [262.0, 170.5, 289.0, 183.5],
            "performance_month": "2022-08",
        },
        {
            "page": 19,
            "logical_page": 19,
            "geometry_scope": "cell",
            "geometry_status": "exact",
            "coordinate_system": "pdf_points_top_left",
            "grid_id": LIN_P19_GRID,
            "row": 3,
            "col": 8,
            "field_name": "overdue_amount",
            "geometry_provenance": _lin_p19_geometry_provenance(),
            "bbox": [262.0, 183.5, 289.0, 196.5],
            "performance_month": "2022-08",
        },
    ]


def _lin_p19_accounts() -> list[dict[str, object]]:
    """Replay the real account-11 continuation and the next p19 anchor."""

    # The frozen anchor inventory puts card 11 at lp18 y=535.5 and card 12 at
    # lp19 y=288.5.  Native table pt_19_0 ends at y=275.7315, so it belongs to
    # card 11's continuation segment and cannot belong to card 12.
    return [
        {
            "record_id": LIN_P19_ACCOUNT_11,
            "account_id": LIN_P19_ACCOUNT_11,
            "account_type": "credit_card",
            "category_sequence": 11,
            "source_refs": [
                {
                    "source": "candidate_b_account_anchor",
                    "logical_page": 18,
                    "source_page": 9,
                    "geometry_scope": "line",
                    "binding": "printed_account_ordinal",
                    "binding_quality": "printed_account_ordinal",
                    "account_type": "credit_card",
                    "category_sequence": 11,
                    "bbox": [53.0, 535.5, 303.0, 548.0],
                    "evidence_ids": [
                        "ocr:sp0009:lp0018:0339",
                        "ocr:sp0009:lp0018:0340",
                    ],
                }
            ],
            "_canonical_segment": {
                "ownership_basis": "printed_anchor_to_next_anchor",
                "anchor_logical_page": 18,
                "anchor_bbox": [53.0, 535.5, 303.0, 548.0],
                "pages": [
                    {"logical_page": 18, "min_y": 535.5, "max_y": None},
                    {"logical_page": 19, "min_y": 0.0, "max_y": 288.5},
                ],
            },
        },
        {
            "record_id": LIN_P19_ACCOUNT_12,
            "account_id": LIN_P19_ACCOUNT_12,
            "account_type": "credit_card",
            "category_sequence": 12,
            "source_refs": [
                {
                    "source": "candidate_b_account_anchor",
                    "logical_page": 19,
                    "source_page": 10,
                    "geometry_scope": "line",
                    "binding": "printed_account_ordinal",
                    "binding_quality": "printed_account_ordinal",
                    "account_type": "credit_card",
                    "category_sequence": 12,
                    "bbox": [48.0, 288.5, 123.0, 300.5],
                    "evidence_ids": ["ocr:sp0010:lp0019:0160"],
                }
            ],
            "_canonical_segment": {
                "ownership_basis": "printed_anchor_to_next_anchor",
                "anchor_logical_page": 19,
                "anchor_bbox": [48.0, 288.5, 123.0, 300.5],
                "pages": [
                    {"logical_page": 19, "min_y": 288.5, "max_y": None}
                ],
            },
        },
    ]


def _lin_p19_grid() -> dict[str, object]:
    return {
        "grid_id": LIN_P19_GRID,
        "page": 19,
        "bbox": [
            45.05555555555556,
            35.52983193277311,
            399.49259259259264,
            275.73151260504204,
        ],
        "coordinate_system": "pdf_points_top_left",
        "audit": {
            "date_range": {
                "start_year": 2019,
                "start_month": 5,
                "end_year": 2022,
                "end_month": 12,
            }
        },
    }


def _lin_p19_rows() -> list[dict[str, object]]:
    # The frozen artifact contains exactly these 44 record ids for this grid.
    months = [
        (year, month)
        for year in range(2019, 2023)
        for month in range(1, 13)
        if (year, month) >= (2019, 5)
    ]
    rows: list[dict[str, object]] = []
    for year, month in months:
        performance_month = f"{year:04d}-{month:02d}"
        record_id = f"{LIN_P19_GRID}:{performance_month}"
        source_refs = (
            _lin_p19_source_refs()
            if performance_month == "2022-08"
            else [
                {
                    "page": 19,
                    "logical_page": 19,
                    "geometry_scope": "grid",
                    "coordinate_system": "pdf_points_top_left",
                    "grid_id": LIN_P19_GRID,
                    "field_name": "performance_month",
                    "performance_month": performance_month,
                    "bbox": list(_lin_p19_grid()["bbox"]),
                }
            ]
        )
        rows.append(
            {
                "record_id": record_id,
                "repayment_id": record_id,
                "grid_id": LIN_P19_GRID,
                "year": year,
                "month": month,
                "status": "M" if performance_month == "2022-08" else "N",
                "overdue_amount": "0",
                "source_cell_refs": source_refs,
            }
        )
    return rows


def test_lin_p19_unresolved_canonical_page_forces_monthly_strategy_fallback() -> None:
    audit = {
        "corrected_evidence_conservation": {
            "valid": True,
            "conserved_logical_pages": [19],
        },
        "canonical_subset_conservation": {"valid": True},
        "unresolved_pages": [19],
        "registrations": [
            {
                "logical_page": 19,
                "source_page": 19,
                "status": "registered",
                "template_id": ACCOUNT_SECTION,
                "basis": "printed_heading_and_table_signature",
                "affected_source_datasets": sorted(
                    SECTION_TO_CANONICAL_DATASETS[ACCOUNT_SECTION]
                ),
                "printed_page": 19,
                "printed_total": 23,
            }
        ],
        "fragment_groups": [
            {
                "template_id": ACCOUNT_SECTION,
                "fragment_logical_pages": [19],
                "canonical_page": 19,
                "coverage_status": "full",
                "coverage_ratio": 1.0,
            }
        ],
    }

    census, plan = plan_candidate_b_initial_extraction(audit)

    assert census.complete is False
    assert census.fallback_reason == "canonical_unresolved_pages:19"
    assert plan.mode is MaterializationMode.EAGER_FALLBACK
    assert plan.ordered_stage_names == CANDIDATE_B_STAGE_REGISTRY.ordered()
    assert "monthly_repayments" in plan.ordered_stage_names
    assert plan.skipped_stage_names == ()


def test_lin_p19_unowned_august_cells_remain_source_conserved_but_not_business_emitted() -> None:
    context = SimpleNamespace(_personal_detail_extraction_issues=[])
    grid = {
        "grid_id": LIN_P19_GRID,
        "page": 19,
        "bbox": [42.0, 150.0, 397.5, 210.0],
        "coordinate_system": "pdf_points_top_left",
        "audit": {
            "date_range": {
                "start_year": 2022,
                "start_month": 8,
                "end_year": 2022,
                "end_month": 8,
            }
        },
    }
    source_row = {
        "record_id": f"{LIN_P19_GRID}:2022-08",
        "repayment_id": f"{LIN_P19_GRID}:2022-08",
        "grid_id": LIN_P19_GRID,
        "year": 2022,
        "month": 8,
        "status": "M",
        "overdue_amount": "0",
        "source_cell_refs": _lin_p19_source_refs(),
    }

    linked = link_candidate_b_repayments(
        [source_row],
        [],
        [grid],
        issue_context=context,
    )

    assert linked == []
    owner_issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code") == "candidate_b_monthly_grid_owner_unresolved"
    ]
    assert len(owner_issues) == 1
    field_issues = [
        issue
        for issue in context._personal_detail_extraction_issues
        if issue.get("issue_code")
        == "candidate_b_monthly_grid_owner_unresolved_field"
        and issue.get("target_record_id")
        == f"{LIN_P19_GRID}:2022-08"
    ]
    assert {issue["field_name"] for issue in field_issues} == {
        "performance_month",
        "status_code",
        "status_amount",
    }
    performance_issue = next(
        issue for issue in field_issues if issue["field_name"] == "performance_month"
    )
    assert len(performance_issue["source_refs"]) == 2
    assert {
        (
            ref["row"],
            ref["col"],
            tuple(ref["bbox"]),
            ref["geometry_provenance"]["table_id"],
        )
        for ref in performance_issue["source_refs"]
    } == {
        (2, 8, (262.0, 170.5, 289.0, 183.5), "pt_19_0"),
        (3, 8, (262.0, 183.5, 289.0, 196.5), "pt_19_0"),
    }
    assert all(
        "relation_withheld" in issue["reason_codes"] for issue in field_issues
    )

    content = {
        "facts": {},
        "datasets": {
            "repayment_records": linked,
            "personal_detail_extraction_issues": deepcopy(
                context._personal_detail_extraction_issues
            ),
        },
    }
    prepared = prepare_personal_detail_source_collections(content)
    closure = prepared["facts"]["personal_detail_account_month_closure"]
    assert closure["candidate_identity_count"] == 0
    assert closure["source_month_position_observations"] == 1
    assert closure["raw_source_month_positions"] == 1
    assert closure["owner_bound_account_months"] == 0
    assert closure["owner_unresolved_positions"] == 1
    assert closure["unresolved_source_position_count"] == 1
    assert closure["source_position_balance_valid"] is True
    assert closure["status"] == "partial_owner_unresolved"
    state = prepared["facts"]["personal_detail_dataset_states"][
        "repayment_records"
    ]
    assert state["presence_status"] == "partial"
    assert state["observed_row_count"] == 0

    projected = project_personal_detail_datasets(prepared["datasets"])
    assert projected.get("credit_account_monthly_performance", []) == []
    assert any(
        issue.get("issue_code")
        == "candidate_b_monthly_grid_owner_unresolved_field"
        and issue.get("target_record_id") == f"{LIN_P19_GRID}:2022-08"
        for issue in projected["extraction_issues"]
    )


def _lin_p19_native_cell(
    text: str,
    bbox: list[float] | None,
    *,
    row: int,
    col: int,
    row_span: int = 1,
    exact: bool = True,
) -> SimpleNamespace:
    evidence_ids = [f"ocr:p19:r{row}:c{col}"] if text and bbox else []
    return SimpleNamespace(
        text=text,
        bbox=bbox,
        row_index=row,
        col_index=col,
        row_span=row_span,
        col_span=1,
        geometry_status="exact" if exact else "derived",
        geometry_source="scanned_image_line_grid",
        evidence_ids=evidence_ids,
        token_ids=list(evidence_ids),
    )


def _lin_p19_native_table() -> SimpleNamespace:
    """Exact p19 row-8 header plus row-9/10 native source lattice."""

    x_edges = [45.0, 73.0, *[100.0 + 27.0 * index for index in range(12)]]
    y_edges = [53.5 + 13.0 * index for index in range(18)]
    header_text = [
        "",
        "28",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "T.8",
        "9",
        "10",
        "11",
        "12",
    ]
    header_cells = [
        _lin_p19_native_cell(
            header_text[col],
            [x_edges[col], y_edges[8], x_edges[col + 1], y_edges[9]],
            row=8,
            col=col,
        )
        for col in range(13)
    ]
    status_cells = [
        _lin_p19_native_cell(
            "2022 搜",
            [x_edges[0], y_edges[9], x_edges[1], y_edges[11]],
            row=9,
            col=0,
            row_span=2,
        )
    ] + [
        _lin_p19_native_cell(
            "W" if col == 1 else "" if col == 7 else "N",
            [x_edges[col], y_edges[9], x_edges[col + 1], y_edges[10]],
            row=9,
            col=col,
        )
        for col in range(1, 13)
    ]
    amount_cells = [
        _lin_p19_native_cell("", None, row=10, col=0, exact=False)
    ] + [
        _lin_p19_native_cell(
            "0",
            [x_edges[col], y_edges[10], x_edges[col + 1], y_edges[11]],
            row=10,
            col=col,
        )
        for col in range(1, 13)
    ]
    leading_rows = [SimpleNamespace(cells=[]) for _ in range(8)]
    trailing_rows = [SimpleNamespace(cells=[]) for _ in range(6)]
    missing_boxes = [[None] * 13 for _ in range(8)]
    trailing_boxes = [[None] * 13 for _ in range(6)]
    missing_statuses = [["missing"] * 13 for _ in range(8)]
    trailing_statuses = [["missing"] * 13 for _ in range(6)]
    geometry = {
        "geometry_source": "scanned_image_line_grid",
        "coordinate_system": "pdf_points_top_left",
        "cell_bboxes": [
            *missing_boxes,
            [cell.bbox for cell in header_cells],
            [cell.bbox for cell in status_cells],
            [cell.bbox for cell in amount_cells],
            *trailing_boxes,
        ],
        "cell_geometry_status": [
            *missing_statuses,
            [cell.geometry_status for cell in header_cells],
            [cell.geometry_status for cell in status_cells],
            [cell.geometry_status for cell in amount_cells],
            *trailing_statuses,
        ],
        "cell_spans": [
            {"row": 9, "col": 0, "row_span": 2, "col_span": 1}
        ],
        "row_bands": [
            {"index": index, "y0": y_edges[index], "y1": y_edges[index + 1]}
            for index in range(17)
        ],
        "col_bands": [
            {"index": col, "x0": x_edges[col], "x1": x_edges[col + 1]}
            for col in range(13)
        ],
        "vertical_lines": x_edges,
        "horizontal_lines": y_edges,
    }
    return SimpleNamespace(
        table_id="pt_19_0",
        bbox=[x_edges[0], y_edges[0], x_edges[-1], y_edges[-1]],
        extraction_layer="scanned_image_line_grid",
        metadata={"geometry": geometry},
        rows=[
            *leading_rows,
            SimpleNamespace(cells=header_cells),
            SimpleNamespace(cells=status_cells),
            SimpleNamespace(cells=amount_cells),
            *trailing_rows,
        ],
    )


def test_lin_p19_august_m_vs_native_n_withholds_status_but_retains_zero() -> None:
    page = SimpleNamespace(
        page_number=19,
        source_page_number=10,
        tables=[_lin_p19_native_table()],
    )
    context = SimpleNamespace(
        parse_result=SimpleNamespace(pages=[page]),
        source_page_by_logical={19: 10},
    )
    record_id = f"{LIN_P19_GRID}:2022-08"
    records = link_candidate_b_repayments(
        _lin_p19_rows(),
        _lin_p19_accounts(),
        [_lin_p19_grid()],
        issue_context=context,
    )

    assert len(records) == 44
    assert {
        f"{int(row['year']):04d}-{int(row['month']):02d}" for row in records
    } == {
        f"{year:04d}-{month:02d}"
        for year in range(2019, 2023)
        for month in range(1, 13)
        if (year, month) >= (2019, 5)
    }
    assert {row["account_id"] for row in records} == {LIN_P19_ACCOUNT_11}
    record = next(row for row in records if row["record_id"] == record_id)
    assert record["account_id"] == LIN_P19_ACCOUNT_11
    assert {
        (ref["row"], ref["col"], tuple(ref["bbox"]))
        for ref in record["source_cell_refs"]
    } == {
        (2, 8, (262.0, 170.5, 289.0, 183.5)),
        (3, 8, (262.0, 183.5, 289.0, 196.5)),
    }
    assert record["_account_month_identity_proof"] == {
        "account_id": LIN_P19_ACCOUNT_11,
        "performance_month": "2022-08",
        "grid_id": LIN_P19_GRID,
        "owner_basis": "canonical_account_segment",
        "account_anchor_exact": True,
        "printed_month_range_exact": True,
        "grid_geometry_exact": True,
        "unique_owner": True,
    }
    assert not any(
        issue.get("issue_code") == "candidate_b_monthly_grid_owner_unresolved"
        for issue in getattr(context, "_personal_detail_extraction_issues", ())
    )

    audit = apply_candidate_b_native_status_conflict_guard(
        context,
        records,
        enabled=True,
    )

    assert "status" not in record
    assert record["overdue_amount"] == "0"
    assert record["canonical_raw"]["status"] == ["M", "N"]
    assert record["_unresolved_fields"] == ["status"]
    assert record["extraction_status"] == "review"
    assert audit["unique_native_witnesses"] == 1
    assert audit["conflicts_withheld"] == 1
    assert len(context._personal_detail_extraction_issues) == 1
    conflict = context._personal_detail_extraction_issues[0]
    assert conflict["issue_code"] == (
        "candidate_b_native_source_cell_repayment_status_conflict"
    )
    assert conflict["target_record_id"] == record_id
    assert conflict["field_name"] == "status_code"
    assert conflict["observed_value"] == {
        "corrected_final": "M",
        "paired_status_amount": "0",
        "sealed_native_source_cell": "N",
    }
    assert "normalized_value_withheld" in conflict["reason_codes"]
    assert [
        (ref["evidence_plane"], ref["field_name"])
        for ref in conflict["source_refs"]
    ] == [
        ("corrected_final", "status"),
        ("corrected_final", "overdue_amount"),
        ("sealed_native_source_table", "status"),
        ("sealed_native_source_table", "overdue_amount"),
    ]
    assert {
        (ref["row"], ref["col"])
        for ref in conflict["source_refs"]
        if ref["evidence_plane"] == "sealed_native_source_table"
    } == {(9, 8), (10, 8)}

    projected = project_personal_detail_datasets(
        {
            "repayment_records": records,
            "personal_detail_extraction_issues": deepcopy(
                context._personal_detail_extraction_issues
            ),
        }
    )
    monthly = projected["credit_account_monthly_performance"]
    assert len(monthly) == 44
    target_rows = [
        row for row in monthly if row["monthly_performance_id"] == record_id
    ]
    assert len(target_rows) == 1
    target = target_rows[0]
    assert target["account_id"] == LIN_P19_ACCOUNT_11
    assert target["status_code"] is None
    assert target["status_amount"] == "0"
    assert target["extraction_status"] == "review"
    projected_conflict = next(
        issue
        for issue in projected["extraction_issues"]
        if issue.get("issue_code")
        == "candidate_b_native_source_cell_repayment_status_conflict"
    )
    assert projected_conflict["target_record_id"] == record_id
    assert projected_conflict["field_name"] == "status_code"


def _lin_p13_source_refs() -> list[dict[str, object]]:
    provenance = {
        "selection_basis": "source_table_year_plus_twelve_ownership",
        "source": "source_table_geometry",
        "reason": "exact_source_table_month_lattice_calibration",
        "table_id": "pt_13_1",
        "vertical_rule_count": 14,
        "rule_count": 14,
        "horizontal_rule_count": 18,
        "column_count": 13,
        "month_column_count": 12,
        "status_row_index": 9,
        "amount_row_index": 10,
        "year_anchor_row_index": 9,
        "year_anchor_mode": "boundary_straddling_singleton_year_cells",
        "year_row_span": 2,
        "active_cell_geometry_exact": True,
        "active_cell_rule_derived_count": 0,
        "coordinate_system": "pdf_points_top_left",
        "value_inputs_used": False,
        "corroborated_by_source_table_geometry": True,
        "ambiguous_visual_geometry_superseded": False,
        "source_table_comparison": "agree",
        "calibrated_from_source_table_geometry": True,
        "visual_selection_basis": "year_plus_twelve_rule_ownership",
        "visual_owned_month_rule_hits": 13,
        "visual_residual_shift_months": 0.1719,
        "logical_page": 13,
    }
    return [
        {
            "page": 13,
            "logical_page": 13,
            "geometry_scope": "cell",
            "coordinate_system": "pdf_points_top_left",
            "grid_id": "mg_p13_repayment_1",
            "row": 2,
            "col": 7,
            "field_name": "status",
            "geometry_provenance": deepcopy(provenance),
            "bbox": [226.0, 455.0, 253.0, 468.0],
        },
        {
            "page": 13,
            "logical_page": 13,
            "geometry_scope": "cell",
            "coordinate_system": "pdf_points_top_left",
            "grid_id": "mg_p13_repayment_1",
            "row": 3,
            "col": 7,
            "field_name": "overdue_amount",
            "geometry_provenance": deepcopy(provenance),
            "bbox": [226.0, 468.0, 253.0, 481.0],
        },
    ]


def test_lin_p13_july_n_with_explicit_ten_withholds_only_amount_and_conserves_evidence() -> None:
    account_id = "credit_account:credit_card:2"
    record_id = "mg_p13_repayment_1:2022-07"
    projected = project_personal_detail_datasets(
        {
            "credit_accounts": [
                {
                    "record_id": account_id,
                    "account_id": account_id,
                    "sequence": 2,
                    "account_type": "credit_card",
                }
            ],
            "repayment_records": [
                {
                    "record_id": record_id,
                    "repayment_id": record_id,
                    "grid_id": "mg_p13_repayment_1",
                    "account_id": account_id,
                    "year": 2022,
                    "month": 7,
                    "status": "N",
                    "overdue_amount": "10",
                    "status_amount_semantics": "delinquent_amount",
                    "source_cell_refs": _lin_p13_source_refs(),
                }
            ],
        }
    )

    rows = projected["credit_account_monthly_performance"]
    assert len(rows) == 1
    row = rows[0]
    assert row["monthly_performance_id"] == record_id
    assert row["account_id"] == account_id
    assert row["performance_month"] == "2022-07"
    assert row["status_code"] == "N"
    assert row.get("status_amount") is None
    assert row.get("status_amount_semantics") is None
    assert row["canonical_raw"]["status_amount"] == "10"

    conflicts = [
        issue
        for issue in projected["extraction_issues"]
        if issue.get("issue_code")
        == "candidate_b_monthly_zero_status_amount_conflict"
        and issue.get("target_record_id") == record_id
    ]
    assert len(conflicts) == 1
    conflict = conflicts[0]
    assert conflict["field_name"] == "status_amount"
    assert conflict["observed_value"] == "10"
    assert "normalized_value_withheld" in conflict["reason_codes"]
    # There is no safe replacement amount: public output must not advertise a
    # candidate zero merely because N normally pairs with zero.
    assert conflict["candidate_value"] is None
    assert {
        (ref["row"], ref["col"], tuple(ref["bbox"]))
        for ref in conflict["source_refs"]
    } == {
        (2, 7, (226.0, 455.0, 253.0, 468.0)),
        (3, 7, (226.0, 468.0, 253.0, 481.0)),
    }
    assert all(
        ref["geometry_provenance"]["table_id"] == "pt_13_1"
        and ref["geometry_provenance"]["value_inputs_used"] is False
        for ref in conflict["source_refs"]
    )
    assert not any(
        issue.get("issue_code")
        == "candidate_b_monthly_status_amount_unresolved"
        and issue.get("target_record_id") == record_id
        for issue in projected["extraction_issues"]
    )
