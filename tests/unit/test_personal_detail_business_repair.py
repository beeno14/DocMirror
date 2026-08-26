# Copyright (c) 2026 ValueMap Global and contributors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from docmirror.plugins.credit_report.personal_detail_scanned.business_repair import (
    BusinessUncertaintyRepairCoordinator,
)
from docmirror.plugins.credit_report.personal_detail_scanned.context import (
    PersonalDetailExtractionContext,
)
from docmirror.plugins.credit_report.personal_detail_scanned.ocr_correction import (
    PersonalDetailOCRCorrectionOverlay,
)


def _bad_count(record_id: str, bbox: list[float]) -> dict[str, object]:
    row = int(record_id.rpartition(":")[2])
    return {
        "record_id": record_id,
        "column_label": "月份数",
        "value": "无法识别",
        "source_refs": [
            {
                "source": "native_detail_table_cell",
                "logical_page": 1,
                "source_page": 1,
                "table_id": "pt_1_0",
                "row": row,
                "column": 1,
                "bbox": bbox,
                "evidence_ids": [f"native:{record_id}"],
                "geometry_scope": "cell",
                "field_name": "value",
                "binding": "canonical_field_slot",
            }
        ],
    }


def test_existing_complete_page_evidence_cannot_replace_failed_independent_reocr() -> None:
    coordinator = BusinessUncertaintyRepairCoordinator(SimpleNamespace())
    payload = {"personal_detail_summary_cells": [_bad_count("cell:1", [10, 10, 40, 30])]}
    plan = coordinator.plan(payload, canonical_audit={"unresolved_pages": []})
    calls = 0

    def page_ocr_loader(_pages, *, reason):
        nonlocal calls
        calls += 1
        return []

    plan = coordinator.resolve_page_evidence(
        plan,
        source_pages=[
            {
                "page": 1,
                "source_page": 1,
                "lines": [
                    {
                        "text": "2",
                        "confidence": 0.98,
                        "bbox": [10, 10, 40, 30],
                        "evidence_ids": ["repair:cell:1"],
                    }
                ],
            }
        ],
        page_ocr_loader=page_ocr_loader,
    )

    assert calls == 1
    assert plan.requires_second_pass is True
    assert plan.page_decisions[0]["mode"] == (
        "page_ocr_failed_existing_evidence_not_used_for_field_repair"
    )
    assert plan.page_decisions[0]["ocr_invocations"] == 1
    assert plan.page_decisions[0]["page_reconstruction"] is False
    assert plan.page_evidence == {}
    assert plan.reconstruction_evidence == {}

    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    overlay.install_business_repair_evidence(plan.page_evidence.values(), affected_pages=plan.affected_pages)
    corrected = overlay.correct_business_candidates(payload, stage="candidate_b_final_validation")
    cell = corrected["personal_detail_summary_cells"][0]
    assert cell["value"] is None
    assert cell["canonical_raw"]["value"] == "无法识别"
    assert overlay.audit()["applied_count"] == 0


def test_business_uncertainties_are_grouped_into_one_page_ocr_request() -> None:
    coordinator = BusinessUncertaintyRepairCoordinator(SimpleNamespace())
    payload = {
        "personal_detail_summary_cells": [
            _bad_count("cell:1", [10, 10, 40, 30]),
            _bad_count("cell:2", [50, 10, 80, 30]),
        ]
    }
    plan = coordinator.plan(payload, canonical_audit={"unresolved_pages": []})
    calls: list[tuple[set[int], str]] = []

    def page_ocr_loader(pages, *, reason):
        calls.append((set(pages), reason))
        return [{"page": 1, "lines": [{"text": "2", "confidence": 0.98, "bbox": [10, 10, 40, 30]}]}]

    plan = coordinator.resolve_page_evidence(
        plan,
        source_pages=[{"page": 1, "lines": []}],
        page_ocr_loader=page_ocr_loader,
    )

    assert calls == [({1}, "business_field_context_rich_reocr_required")]
    assert len(plan.page_decisions) == 1
    assert plan.page_decisions[0]["target_count"] == 2
    assert plan.page_decisions[0]["page_reconstruction"] is False
    assert plan.audit()["field_triggered_ocr_requests"] == 1


def test_overlapping_but_role_invalid_text_still_requests_one_page_ocr() -> None:
    coordinator = BusinessUncertaintyRepairCoordinator(SimpleNamespace())
    payload = {"personal_detail_summary_cells": [_bad_count("cell:1", [10, 10, 40, 30])]}
    plan = coordinator.plan(payload, canonical_audit={"unresolved_pages": []})
    calls: list[set[int]] = []

    def page_ocr_loader(pages, *, reason):
        assert reason == "business_field_context_rich_reocr_required"
        calls.append(set(pages))
        return [{"page": 1, "lines": [{"text": "3", "bbox": [10, 10, 40, 30]}]}]

    plan = coordinator.resolve_page_evidence(
        plan,
        source_pages=[{"page": 1, "lines": [{"text": "月份数", "bbox": [10, 10, 40, 30]}]}],
        page_ocr_loader=page_ocr_loader,
    )

    assert calls == [{1}]
    assert plan.page_decisions[0]["ocr_invocations"] == 1


def _exact_field_ref(
    *,
    field_name: str,
    bbox: list[float],
    row: int = 1,
    column: int = 1,
) -> dict[str, object]:
    return {
        "source": "native_detail_table_cell",
        "logical_page": 1,
        "source_page": 1,
        "table_id": "pt_1_0",
        "row": row,
        "column": column,
        "bbox": bbox,
        "evidence_ids": [f"native:{field_name}:{row}:{column}"],
        "geometry_scope": "cell",
        "field_name": field_name,
        "binding": "canonical_header_column",
    }


def _field_issue(
    *,
    field_name: str,
    observed_value: object,
    source_ref: dict[str, object],
) -> dict[str, object]:
    return {
        "issue_code": "candidate_b_inquiry_row_cells_unresolved",
        "target_dataset": "inquiry_records",
        "target_record_id": "credit_inquiry:institution:25",
        "field_name": field_name,
        "observed_value": observed_value,
        "source_refs": [source_ref],
        "reason_codes": ["exact_header_column_binding_failed"],
    }


def test_safe_inquiry_date_residue_is_deterministic_and_starts_no_ocr() -> None:
    coordinator = BusinessUncertaintyRepairCoordinator(SimpleNamespace())
    ref = _exact_field_ref(
        field_name="inquiry_date",
        bbox=[10, 10, 40, 30],
    )
    plan = coordinator.plan(
        {},
        canonical_audit={"unresolved_pages": []},
        extraction_issues=[
            _field_issue(
                field_name="inquiry_date",
                observed_value="2022.06.16 广",
                source_ref=ref,
            )
        ],
    )
    calls = 0

    def page_ocr_loader(_pages, *, reason):
        nonlocal calls
        calls += 1
        raise AssertionError(reason)

    coordinator.resolve_page_evidence(
        plan,
        source_pages=[{"page": 1, "source_page": 1, "lines": []}],
        page_ocr_loader=page_ocr_loader,
    )

    assert calls == 0
    assert len(plan.field_repairs) == 1
    assert plan.field_repairs[0].mode == "deterministic"
    assert plan.field_repairs[0].candidate_value == "2022-06-16"
    assert plan.page_evidence == {}
    assert plan.reconstruction_evidence == {}
    assert plan.page_decisions[0]["acquisition_scope"] == "none"


def test_numeric_date_residue_gets_one_page_acquisition_but_no_page_replacement() -> None:
    coordinator = BusinessUncertaintyRepairCoordinator(SimpleNamespace())
    ref = _exact_field_ref(
        field_name="inquiry_date",
        bbox=[10, 10, 40, 30],
    )
    plan = coordinator.plan(
        {},
        canonical_audit={"unresolved_pages": []},
        extraction_issues=[
            _field_issue(
                field_name="inquiry_date",
                observed_value="2023.01.03 20",
                source_ref=ref,
            )
        ],
    )
    calls: list[tuple[set[int], str]] = []
    reocr_page = {
        "page": 1,
        "source_page": 1,
        "page_key": "numeric-date",
        "lines": [],
    }

    def page_ocr_loader(pages, *, reason):
        calls.append((set(pages), reason))
        return [reocr_page]

    coordinator.resolve_page_evidence(
        plan,
        source_pages=[{"page": 1, "source_page": 1, "lines": []}],
        page_ocr_loader=page_ocr_loader,
    )

    assert calls == [({1}, "business_field_context_rich_reocr_required")]
    assert plan.field_repairs[0].mode == "context_rich_reocr"
    assert plan.field_repairs[0].candidate_value is None
    assert plan.page_evidence == {1: reocr_page}
    assert plan.reconstruction_evidence == {}
    assert plan.page_decisions[0]["mode"] == (
        "one_shot_context_rich_page_ocr_field_overlay_only"
    )
    assert plan.page_decisions[0]["page_reconstruction"] is False


def test_exact_cell_does_not_turn_structural_issue_code_into_field_contract() -> None:
    coordinator = BusinessUncertaintyRepairCoordinator(SimpleNamespace())
    ref = _exact_field_ref(
        field_name="unknown_structure",
        bbox=[10, 10, 40, 30],
    )
    plan = coordinator.plan(
        {},
        canonical_audit={"unresolved_pages": []},
        extraction_issues=[
            {
                "issue_code": "business_structure_uncertain",
                "target_dataset": "unknown_business_rows",
                "target_record_id": "unknown:1",
                "field_name": "unknown_structure",
                "observed_value": "damaged",
                "source_refs": [ref],
            }
        ],
    )

    assert plan.field_repairs == ()

    coordinator.resolve_page_evidence(
        plan,
        source_pages=[{"page": 1, "source_page": 1, "lines": []}],
        page_ocr_loader=lambda _pages, *, reason: [
            {"page": 1, "source_page": 1, "lines": [], "reason": reason}
        ],
    )

    assert set(plan.reconstruction_evidence) == {1}
    assert plan.page_decisions[0]["page_reconstruction"] is True


def test_field_issue_without_exact_owner_stays_unrepaired_and_starts_no_ocr() -> None:
    coordinator = BusinessUncertaintyRepairCoordinator(SimpleNamespace())
    nonowned_ref = {
        "source": "native_detail_table_row",
        "logical_page": 1,
        "source_page": 1,
        "bbox": [10, 10, 80, 30],
        "evidence_ids": ["native:row:1"],
        "geometry_scope": "row",
    }
    plan = coordinator.plan(
        {},
        canonical_audit={"unresolved_pages": []},
        extraction_issues=[
            _field_issue(
                field_name="inquiry_date",
                observed_value="2023.01.03 20",
                source_ref=nonowned_ref,
            )
        ],
    )

    assert plan.field_repairs == ()
    assert plan.affected_pages == ()
    assert plan.requires_second_pass is False

    def page_ocr_loader(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
        raise AssertionError("unowned field uncertainty must not start OCR")

    coordinator.resolve_page_evidence(
        plan,
        source_pages=[{"page": 1, "source_page": 1, "lines": []}],
        page_ocr_loader=page_ocr_loader,
    )

    assert plan.page_decisions == []


def test_liability_policy_separates_deterministic_and_context_rich_fields() -> None:
    coordinator = BusinessUncertaintyRepairCoordinator(SimpleNamespace())
    payload = {
        "repayment_liability_records": [
            {
                "liability_id": "liability:2",
                "business_type": "爱贷款",
                "related_party_name": "密厦门雯明轩商贸有限公司",
                "canonical_raw": {
                    "business_type": "爱 贷款",
                    "related_party_name": "密 厦门雯明轩商贸有限公司",
                },
                "source_refs_by_field": {
                    "business_type": [
                        _exact_field_ref(
                            field_name="business_type",
                            bbox=[10, 10, 40, 30],
                            column=1,
                        )
                    ],
                    "related_party_name": [
                        _exact_field_ref(
                            field_name="related_party_name",
                            bbox=[10, 40, 80, 60],
                            row=3,
                            column=0,
                        )
                    ],
                },
            }
        ]
    }

    plan = coordinator.plan(
        payload,
        canonical_audit={"unresolved_pages": []},
    )
    by_field = {repair.field_name: repair for repair in plan.field_repairs}

    assert by_field["business_type"].mode == "deterministic"
    assert by_field["business_type"].candidate_value == "贷款"
    assert by_field["related_party_name"].mode == "context_rich_reocr"
    assert by_field["related_party_name"].candidate_value is None


def test_unresolved_business_template_can_trigger_repair_but_topology_cannot() -> None:
    coordinator = BusinessUncertaintyRepairCoordinator(SimpleNamespace())
    plan = coordinator.plan({}, canonical_audit={"unresolved_pages": [3]})
    calls: list[tuple[set[int], str]] = []

    def page_ocr_loader(pages, *, reason):
        calls.append((set(pages), reason))
        return []

    plan = coordinator.resolve_page_evidence(
        plan,
        source_pages=[{"page": 3, "lines": [{"text": "unknown", "bbox": [1, 1, 10, 10]}]}],
        page_ocr_loader=page_ocr_loader,
    )

    assert calls == [({3}, "business_schema_template_unresolved")]
    assert plan.audit()["topology_ocr_requests"] == 0
    assert plan.audit()["field_triggered_ocr_requests"] == 1


def test_structurally_missing_business_record_enters_the_page_repair_plan() -> None:
    coordinator = BusinessUncertaintyRepairCoordinator(SimpleNamespace())
    plan = coordinator.plan(
        {},
        canonical_audit={"unresolved_pages": []},
        extraction_issues=[
            {
                "issue_code": "candidate_b_unmatched_account_table_suppressed",
                "target_dataset": "credit_accounts",
                "target_record_id": "credit_account:credit_card:4",
                "source_refs": [{"logical_page": 20, "bbox": [10, 20, 300, 400]}],
                "reason_codes": ["printed_anchor_missing"],
            }
        ],
    )

    assert plan.affected_pages == (20,)
    assert plan.uncertainties[0].dataset_name == "credit_accounts"
    assert "candidate_b_unmatched_account_table_suppressed" in plan.uncertainties[0].reason_codes


def test_second_source_pass_invalidates_discovery_account_anchor_skeleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repaired_page = {
        "page": 5,
        "lines": [
            {
                "text": "\u8d26\u62376",
                "bbox": [53.0, 229.0, 73.5, 240.0],
                "evidence_ids": ["ocr:sp0003:lp0005:0044"],
            }
        ],
    }
    plan = SimpleNamespace(
        affected_pages=(5,),
        page_evidence={5: repaired_page},
        reconstruction_evidence={5: repaired_page},
        requires_second_pass=True,
    )

    class _Coordinator:
        def __init__(self, _parse_result: object) -> None:
            pass

        def plan(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return plan

        def resolve_page_evidence(
            self, candidate: SimpleNamespace, **_kwargs: object
        ) -> SimpleNamespace:
            return candidate

    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.business_repair."
        "BusinessUncertaintyRepairCoordinator",
        _Coordinator,
    )
    context = object.__new__(PersonalDetailExtractionContext)
    context.parse_result = SimpleNamespace()
    context._personal_detail_extraction_issues = []
    context._initial_personal_detail_extraction_issues = []
    context._candidate_b_account_anchor_skeleton_cache = [
        {"account_id": "credit_account:non_revolving_loan:5"}
    ]
    context._cache = {"account_collections": ([{"account_id": "stale"}], [], [])}
    context._canonical_layout_projection_cache = object()
    context._pboc_layout_profile_cache = object()
    context._canonical_entity_context_ready = True
    context._source_evidence_pages = lambda: []
    context.full_page_ocr_evidence = lambda *_args, **_kwargs: []
    context.canonical_layout_audit = lambda: {"unresolved_pages": []}

    assert context.prepare_candidate_b_business_repair({}) is True
    assert "_candidate_b_account_anchor_skeleton_cache" not in context.__dict__
    assert context._candidate_b_pre_repair_account_anchor_inventory == (
        {"account_id": "credit_account:non_revolving_loan:5"},
    )
    assert context._cache == {}
    assert context._canonical_layout_projection_cache is None
    assert context._pboc_layout_profile_cache is None
    assert context._business_repair_evidence_by_page == {5: repaired_page}


def test_field_only_repair_keeps_discovery_projection_and_account_skeleton(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reocr_page = {
        "page": 5,
        "source_page": 3,
        "page_key": "field-only",
        "lines": [],
    }
    plan = SimpleNamespace(
        affected_pages=(5,),
        page_evidence={5: reocr_page},
        reconstruction_evidence={},
        requires_second_pass=True,
    )

    class _Coordinator:
        def __init__(self, _parse_result: object) -> None:
            pass

        def plan(self, *_args: object, **_kwargs: object) -> SimpleNamespace:
            return plan

        def resolve_page_evidence(
            self, candidate: SimpleNamespace, **_kwargs: object
        ) -> SimpleNamespace:
            return candidate

    monkeypatch.setattr(
        "docmirror.plugins.credit_report.personal_detail_scanned.business_repair."
        "BusinessUncertaintyRepairCoordinator",
        _Coordinator,
    )
    context = object.__new__(PersonalDetailExtractionContext)
    context.parse_result = SimpleNamespace()
    context._personal_detail_extraction_issues = []
    context._initial_personal_detail_extraction_issues = []
    context._candidate_b_account_anchor_skeleton_cache = [
        {"account_id": "credit_account:credit_card:1"}
    ]
    context._cache = {"inquiries": [{"inquiry_id": "stale"}]}
    projection = object()
    profile = object()
    context._canonical_layout_projection_cache = projection
    context._pboc_layout_profile_cache = profile
    context._canonical_entity_context_ready = True
    context._source_evidence_pages = lambda: []
    context.full_page_ocr_evidence = lambda *_args, **_kwargs: []
    context.canonical_layout_audit = lambda: {"unresolved_pages": []}

    assert context.prepare_candidate_b_business_repair({}) is True
    assert context._business_repair_evidence_by_page == {}
    assert context._canonical_layout_projection_cache is projection
    assert context._pboc_layout_profile_cache is profile
    assert context._canonical_entity_context_ready is True
    assert context._candidate_b_account_anchor_skeleton_cache == [
        {"account_id": "credit_account:credit_card:1"}
    ]


def test_numeric_monthly_status_without_amount_is_withheld_and_explicitly_reported() -> None:
    overlay = PersonalDetailOCRCorrectionOverlay(SimpleNamespace())
    payload = {
        "repayment_records": [
            {
                "repayment_id": "repayment:2024-01",
                "year": 2024,
                "month": 1,
                "status": "2",
                "overdue_amount": None,
                "source_refs": [{"logical_page": 4, "geometry_scope": "table"}],
                "source_cell_refs": [
                    {
                        "logical_page": 4,
                        "bbox": [10, 20, 30, 40],
                        "geometry_scope": "cell",
                        "field_name": "status",
                    },
                    {
                        "logical_page": 4,
                        "bbox": [10, 40, 30, 60],
                        "geometry_scope": "cell",
                        "field_name": "overdue_amount",
                    },
                ],
            }
        ]
    }

    plan = BusinessUncertaintyRepairCoordinator(SimpleNamespace()).plan(
        payload,
        canonical_audit={"unresolved_pages": []},
    )
    corrected = overlay.correct_business_candidates(payload, stage="candidate_b_final_validation")
    anomalies = overlay.audit()["cell_anomalies"]

    assert corrected["repayment_records"][0]["status"] == "unknown"
    assert corrected["repayment_records"][0]["canonical_raw"]["status"] == "2"
    assert corrected["repayment_records"][0]["overdue_amount"] is None
    status_issue = next(
        anomaly
        for anomaly in anomalies
        if anomaly["field_name"] == "status_code"
        and "monthly_status_amount_unresolved" in anomaly["reason_codes"]
    )
    assert status_issue["value"] == "2"
    assert status_issue["normalized_value_withheld"] is True
    assert status_issue["source_refs"][0]["field_name"] == "status"
    amount_issue = next(
        anomaly
        for anomaly in anomalies
        if anomaly["field_name"] == "overdue_amount"
        and "monthly_status_amount_unresolved" in anomaly["reason_codes"]
    )
    assert amount_issue["source_refs"][0]["field_name"] == "overdue_amount"
    amount_uncertainty = next(item for item in plan.uncertainties if item.field_name == "overdue_amount")
    assert amount_uncertainty.source_refs[0]["bbox"] == [10, 40, 30, 60]
